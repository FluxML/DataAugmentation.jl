# Defining transformations

The transformation interface is the centrepiece of this library. Beside straightforward transform application it also enables stochasticity, composition and buffering (see [Design](design.md)).

A transformation is a type that subtypes [`Transform`](@ref). The only *required* function to implement for your transformation type `T` is

- [`apply`](@ref)`(tfm::T, item::I; randstate)`

    Calling `tfm(item)` forwards to this method. `randstate` encapsulates the random state needed for stochastic transformations. The method should be deterministic for a given `randstate`.

    You may dispatch on a specific item type `I` or use the abstract `Item` if one implementation works for all item types.

You may additionally also implement:

- [`DataAugmentation.getrandstate`](@ref)`(tfm)` for *stochastic* transformations

    Generates the default random state for `tfm(item)`. The fallback returns `nothing`.

- [`apply!`](@ref)`(bufitem, tfm::T, item; randstate)` to support *buffering*

    Buffered version of `apply` that mutates `bufitem`. If not implemented,
    falls back to regular `apply`. See [Supporting buffering](@ref) below.

- [`DataAugmentation.compose`](@ref)`(tfm1, tfm2)` for custom *composition* with other transformations

    Composes transformations. By default, returns a [`Sequence`](@ref) transformation that applies the transformations one after the other.

Spatial transformations that should be fused and cropped efficiently are better defined as [projective transformations](projinterface.md).

## Example

The implementation of the [`MapElem`](@ref) transformation illustrates this interface well. It transforms any item with array data by mapping a function over the array's elements, just like `Base.map`.

```julia
struct MapElem <: Transform
    f
end
```

The `apply` implementation dispatches on [`DataAugmentation.AbstractArrayItem`](@ref), an abstract item type for items that wrap arrays. Note that the `randstate` keyword argument needs to be given even for implementations of deterministic transformations. We also make use of the [`DataAugmentation.setdata`](@ref) helper to update the item data.

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

Finally, `compose` could be customized so that two `MapElem`s are fused: instead of applying them sequentially, the composed function is applied once. `MapElem` does not do this at the moment, but it would look like this:

```julia
compose(tfm1::MapElem, tfm2::MapElem) = MapElem(tfm2.f ∘ tfm1.f)
```

## Supporting buffering

Usually, `buffer = tfm(item)` gives you an item that you can reuse with `apply!(buffer, tfm, item)`.

However, for some transformations, a different buffer is needed. [`Sequence`](@ref), for example, needs to reuse all intermediate results. That is why the buffer creation can be customized:

- [`DataAugmentation.makebuffer`](@ref)`(tfm, item)` creates a buffer `buf` that can be used in an `apply!` call: `apply!(buf, tfm, item)`.

Users do not have to handle buffers themselves, see [Buffering](buffering.md).
