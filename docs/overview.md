# Overview

DataAugmentation.jl makes it easy to create efficient, composable, stochastic data transformation pipelines.  Let's unravel that.

- *efficient*: The most performant way of applying a transformation is selected based on the context
- *composable*: Transformations can be sequenced and combined in a sensible way
- *stochastic*: Parameters of the transformations can be random variables. Care is taken to ensure that all transformations are
  target-preserving.

The library also aims to be extensible and provides [interfaces](./interface.md) for implementing new transformations and custom kinds of data.

In DataAugmentation.jl, you use transforms to transform [items](items.md) which represent your data. This means you have to wrap your data in an [`Item`](#) type before applying transformations to it. This might seem tedious but has some benefits:

- It resolves type ambiguities that arise because the same machine type can represent different data. For example, `Array{Float, 3}` could represent an RGB image or a segmentation mask.
- Transformations can dispatch on the `Item` type.
- We can attach metadata to the data.

