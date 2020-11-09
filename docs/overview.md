# Overview

DataAugmentation.jl makes it easy to create efficient, composable, stochastic data transformation pipelines.  Let's unravel that.

- *efficient*: The most performant way of applying a transformation is selected based on the context
- *composable*: Transformations can be sequenced and combined in a sensible way
- *stochastic*: Parameters of the transformations can be random variables. Care is taken to ensure that all transformations are
  target-preserving

The library also aims to be extensible and provides [interfaces](./interface.md) for implementing new transformations and custom kinds of data.


