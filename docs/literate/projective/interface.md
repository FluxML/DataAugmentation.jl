

# Projective transformations

DataAugmentation.jl has great support for transforming spatial data like images and keypoints. Most of these transformations are projective transformations. For our purposes, a projection means a mapping between two coordinate spaces. In computer vision, these are frequently used for preprocessing and augmenting image data: images are randomly scaled, maybe flipped horizontally and finally cropped to the same size.

This library generalizes projective transformations for different kinds of image and keypoint data in an N-dimensional Euclidean space. It also uses composition for performance improvements like fusing affine transformations.

Unlike mathematical objects, the spatial data we want to transform has *spatial bounds*. For an image, these bounds are akin to the array size. But keypoint data aligned with an image has the same bounds even if they are not explicitly encoded in the representation of the data.
These spatial bounds can be used to dynamically create useful transformations. For example, a rotation around the center or a horizontal flip of keypoint annotations can be calculated from the bounds.

Often, we also want to *crop* an area from the projected results. By evaluating only the parts of a projection that fall inside the cropped area, a lot of unnecessary computation can be avoided.

That in mind, let's see how projective transformations are implemented.

## Interface


The abstract type [`ProjectiveTransform`](#) represents a projective transformation.
A `ProjectiveTransform` needs to implement [`getprojection`](#)`(tfm, bounds; randstate)` that should return a `Transformation` from [CoordinateTransformations.jl](https://github.com/JuliaGeometry/CoordinateTransformations.jl).

To add support for projective transformations to an item `I`, you need to implement

- `getbounds(item::I)` returns the spatial bounds of the item; and
- `project(P, item::I, indices)` applies the projective transformation `P` and crops to `indices`

To support `apply!`-ing projective transformations, `project!(bufitem, P, item)` can also be implemented.
