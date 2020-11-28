# DataAugmentation.jl

This library provides data transformations for machine and deep learning. At the moment, it focuses on spatial data (think images, keypoint data and masks), but that is owed only to my current work. The extensible abstractions should fit other domains as well.

For the most part, the transformations themselves are not very complex. The challenge this library tackles is to reconcile an easy-to-use, composable interface with performant execution.

The key abstractions are `Transform`s, the transformation to apply, and `Item`s which contain the data to be transformed. [`apply`](#)`(tfm, item)`, as the name gives away, applies a transformation to an item.  For example, given an [`Image`](#) item, we can resize it with the [`CenterResizeCrop`](#) transformation.

```julia
item = Image(image)
tfm = CenterResizeCrop((128, 128))
apply(tfm, item)
```

!!! info "A note on this documentation"

    The documentation is composed of prose affixed with links to commented source code. The prose introduces concepts and motivates the chosen abstractions while the accompanying source files contain the implementation. You are encouraged to reference the source files for better understanding and documentation of implementation details.

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

---

Let's next see how these requirements are reflected in the [item](./iteminterface.md) and [transformation](./tfminterface.md) interfaces.