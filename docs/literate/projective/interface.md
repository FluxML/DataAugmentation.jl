
## Projective transformations interface


The abstract type [`ProjectiveTransform`](#) represents a projective transformation.
A `ProjectiveTransform` needs to implement [`getprojection`](#)`(tfm, bounds; randstate)` that should return a `Transformation` from [CoordinateTransformations.jl](https://github.com/JuliaGeometry/CoordinateTransformations.jl).

To add support for projective transformations to an item `I`, you need to implement

- `getbounds(item::I)` returns the spatial bounds of the item; and
- `project(P, item::I, indices)` applies the projective transformation `P` and crops to `indices`

To support `apply!`-ing projective transformations, `project!(bufitem, P, item)` can also be implemented.
