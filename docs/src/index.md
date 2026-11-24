# DataAugmentation.jl

Efficient, composable data augmentation for machine and deep learning, with support for n-dimensional images, categorical masks, keypoints, and bounding boxes.

An [`Item`](@ref) wraps data, and a [`Transform`](@ref) operates on the item. To resize an [`Image`](@ref), create a [`CenterResizeCrop`](@ref) transform and call it on the item.

```julia
item = Image(image)
tfm = CenterResizeCrop((128, 128))
tfm(item)
```

For the most part, the transformations themselves are not very complex. The challenge this library tackles is to reconcile an easy-to-use, composable interface with performant execution; the [Design](design.md) page explains how.

At the moment, the library focuses on spatial data (images, keypoints and masks), and also supports rows of tabular data. The extensible abstractions should fit other domains as well.

## Installation

```julia
using Pkg
Pkg.add("DataAugmentation")
```

## Where to go next

- [Quickstart](quickstart.md): load an image, transform it and get the data back.
- **Guide**: using the library.
    - [Items](items.md): the containers for your data.
    - [Transformations](transformations.md): composing and randomizing transformations, and an overview of those that are available.
    - [Gallery](gallery.md): what the spatial transformations look like.
    - [Pipelines](pipelines.md) walks through classification, segmentation, landmarks, object detection, and 3D images.
    - [Preprocessing](preprocessing.md): getting data ready to be put into a model.
    - [Using with Flux](flux.md): data loading, batching and training with augmented data.
    - [Buffering](buffering.md): reusing output buffers for speed.
- **Concepts**: why the library looks the way it does.
    - [Design](design.md): stochasticity, composition and buffering.
    - [Projective transformations](projective.md): fusing and cropping spatial transformations.
- **Extending**: writing your own [items](iteminterface.md), [transformations](tfminterface.md) and [projective transformations](projinterface.md).
- [API reference](ref.md) with an alphabetical [symbol index](@ref reference-index).
