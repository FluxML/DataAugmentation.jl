# Defining projective transformations

See [Projective transformations](projective.md) for what projective transformations are and why they are treated specially.

## Transformations

The abstract type [`DataAugmentation.ProjectiveTransform`](@ref) represents a projective transformation.
A `ProjectiveTransform` needs to implement [`DataAugmentation.getprojection`](@ref)`(tfm, bounds; randstate)` that should return a `Transformation` from [CoordinateTransformations.jl](https://github.com/JuliaGeometry/CoordinateTransformations.jl).

To apply a fixed projection `P` that does not depend on the bounds, wrap it in [`Project`](@ref)`(P)`.

## Items

To add support for projective transformations to an item `I`, you need to implement

- [`DataAugmentation.getbounds`](@ref)`(item::I)` returns the spatial bounds of the item; and
- [`DataAugmentation.project`](@ref)`(P, item::I, indices)` applies the projective transformation `P` and crops to `indices`

To support `apply!`-ing projective transformations, [`DataAugmentation.project!`](@ref)`(bufitem, P, item, indices)` can also be implemented.
