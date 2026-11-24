# Items

Items are containers for data: an [`Image`](@ref) represents an image, and [`Keypoints`](@ref) some keypoints. Transformations are applied to items with [`apply`](@ref), and [`itemdata`](@ref) unwraps the data of the result.

## Why wrap data in an item?

For one, the item `struct`s may contain metadata that is useful for some transformations. More importantly, though, by wrapping data in an item type, the *meaning* of the data is separated from the *representation*, that is, the concrete type.

An `Array{Integer, 2}` could represent an image, but also a multi-class segmentation mask. Is `Array{Float32, 3}` a 3-dimensional image or a 2-dimensional image with the color channels expanded?

Separating the representation from the data's meaning resolves those ambiguities.

## Spatial items

There are three kinds of spatial data currently supported: images, keypoints and segmentation masks. Both 2D and 3D data is supported (and technically, higher dimensions).

- [`Image`](@ref)`{N, T}` represents an `N`-dimensional image. `T` refers to the element type of the array that `Image` wraps, usually a color. When projecting images, proper interpolation methods are used to reduce artifacts like aliasing.

- [`MaskBinary`](@ref)`{N}` and [`MaskMulti`](@ref)`{N, T}` likewise represent `N`-dimensional segmentation masks. Unlike images, nearest-neighbor interpolation is used for projecting masks.

- Lastly, [`Keypoints`](@ref)`{N}` represent keypoint data. The data should be an array of `SVector{N}`. Since there are many interpretations of keypoint data, there are also wrapper items for convenience: [`BoundingBox`](@ref) and [`Polygon`](@ref).

Spatial items have spatial bounds, which [projective transformations](projective.md) use. The [Gallery](gallery.md) shows how to apply transformations to images and keypoints together.

## Other items

- [`ArrayItem`](@ref) wraps any array. It is what the [preprocessing](preprocessing.md) transformations produce, for example [`ImageToTensor`](@ref) turns an `Image` into an `ArrayItem`.

- [`TabularItem`](@ref) represents one row of a table, to be used with the row transformations described under [Preprocessing](preprocessing.md#Tabular-data).

## Visualizing items

[`showitems`](@ref) visualizes spatial items, and [`showgrid`](@ref) arranges the visualizations of many items in a grid. The [Gallery](gallery.md) uses both throughout.

Use `showitems((image, mask))` to check alignment after a transformation.
Set `maskalpha` to adjust the default opacity of `0.5` and `maskbackground`
to skip a label such as `1` for a multiclass mask or `false` for a binary mask.
`showgrid` accepts the same options for grids of image and mask pairs.

To define your own item, see [Defining items](iteminterface.md).
