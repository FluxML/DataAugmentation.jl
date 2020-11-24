# DataAugmentation.jl

This library provides data transformations for machine and deep learning. At the moment, it focuses on spatial data (think images, keypoint data and masks), but that is owed only to my current work. The extensible abstractions should fit other domains as well.

For the most part, the transformations themselves are not very complex. The challenge this library tackles is to reconcile an easy-to-use, composable interface with performant execution.


The key abstractions are `Transform`s, the transformation to apply, and `Item`s which contain the data to be transformed. [`apply`](#)`(tfm, item)`, as the name gives away, applies a transformation to an item.  For example, given an [`Image`](#) item, we can resize it with the [`CenterResizeCrop`](#) transformation.

```julia
item = Image(image)
tfm = CenterResizeCrop((128, 128))
apply(tfm, item) |> showitem
```

The image data is wrapped in an the item type `Image` to give it meaning.  Different kinds of data may need to be transformed differently, but could be represented by the same Julia type, so this resolves ambiguities. The data can be "unwrapped" using [`itemdata`](#)`(item)`.

## A note on this documentation

The documentation is composed of prose and commented source code. The prose introduces concepts and motivates the chosen abstractions. The accompanying source files contain the implementation. You are encouraged to reference the source files for a better understanding and documentation of implementation details.

## Requirements

The above example is simple, but there are more requirements of data augmentation pipelines that this library adresses. They serve as a motivation to the interface I've arrived at for defining transformations.

### Stochasticity

A transformation is stochastic (as opposed to deterministic) if it produces different outputs based on some random state.
This randomness can become a problem when applying an transformation to an aligned pair of input and target. If we have an image and a corresponding segmentation mask, using different scaling factors results in misalignment of the two; the segmentation no longer matches up with the image pixels.

To handle this, the random state is explicitly passed to the transformations, rendering them deterministic. A generator for the random state can be defined with [`getrandstate`](#)`(tfm)` and passed to `apply` with the `randstate` keyword argument.

### Composition

Most data augmentation pipelines are made up of multiple steps: augmenting an image can mean resizing, randomly rotating, cropping and then normalizing the values. So applying transformations one after another – sequencing – is one way to compose transformations. But some operations, like affine transformations, can also be *fused*, resulting in a single transformation that is more performant and produces more accurate results.

### Buffering

Since data augmentation pipelines often run on large amounts of data, performance can often be improved by using prealloacted output buffers for the transformations. This results in fewer memory allocations and less garbage collection which both take time. 

## `Transform` interface

- `abstract type `[`Transform`](#)

    Every transformation needs to be a subtype of `Transform`. We'll introduce other abstract transformations later.

- [`apply`](#)`(tfm, item; randstate)`

    Applies the transformation `tfm` to item `item`. Implemented methods can of course dispatch on the type of `item`. `randstate` encapsulates the random state needed for stochastic transformations. The `apply` method implementation itself should be deterministic.

- [`getrandstate`](#)`(tfm)`

    Generates random state for stochastic transformations.
    Calling `apply(tfm, item)` is equivalent to
    `apply(tfm, item; randstate = getrandstate(tfm))`. It defaults to `nothing`, so we need not implement it for deterministic transformations.

- [`apply!`](#)`(bufitem, tfm, item; randstate)`

    Buffered version of `apply` that mutates `bufitem`. If not implemented,
    falls back to regular `apply`.

- [`compose`](#)`(tfm1, tfm2)` | `tfm1 |> tfm2`

    Composes transformations. By default, returns a [`Sequence`](#) transformation that applies the transformations one after the other.


### Example

The implementation of the [`MapElem`](#) transformation illustrates this interface well. It transforms any item with array data by mapping a function over the array's elements, just like `Base.map`.

```julia
struct MapElem <: Transform
    f
end
```

The `apply` implementation dispatches on [`AbstractArrayItem`](#), an abstract item type for items that wrap arrays. Note that the `randstate` keyword argument needs to be given even for implementations of deterministic transformations. We also make use of the [`setdata`](#) helper to update the item data.

```julia
function apply(tfm::MapElem, item::AbstractArrayItem; randstate = nothing)
    a = itemdata(item)
    a_ = map(tfm.f, a)
    return setdata(item, a_)
end
```

The buffered version applies the function inplace using `Base.map!`:

```julia
function apply!(
        bufitem::I,
        tfm::MapElem,
        item::I;
        randstate = nothing) where I <: AbstractArrayItem
    map!(tfm.f, itemdata(bufitem), itemdata(item))
    return bufitem
end
```

Finally, a `MapElem` can also be composed nicely with other `MapElem`s. Instead of applying them sequentially, the functions are fused and applied once.

```julia
compose(tfm1::MapElem, tfm2::MapElem2) = MapElem(tfm2.f ∘ tfm1.f)
```