# Spatial data

Before introducing various projective transformations, we have a look at the data that can be projected. There are three kinds of data currently supported: images, keypoints and segmentation masks. Both 2D and 3D data is supported (and technically, higher dimensions, but I've yet to find a dataset with 4 spatial dimensions).

- [`Image`](#)`{N, T}` represents an `N`-dimensional image. `T` refers to the element type of the array that `Image` wraps, usually a color. When projecting images, proper interpolation methods are used to reduce artifacts like aliasing. See [`image.jl`](../../../src/items/image.jl)

- [`MaskBinary`](#)`{N}` and [`MaskMulti`](#)`{N, T}` likewise represents `N`-dimensional segmentation masks. Unlike images, nearest-neighbor interpolation is used for projecting masks. See [`mask.jl`](../../src/items/mask.jl)

- Lastly, [`Keypoints`](#)`{N}` represent keypoint data. The data should be an array of `SVector{N}`. Since there are many interpretations of keypoint data, there are also wrapper items for convenience: [`BoundingBox`](#) and [`Polygon`](#).See [`keypoints.jl`](../../src/items/keypoints.jl)
