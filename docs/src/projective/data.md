# Spatial data

Before introducing various projective transformations, we have a look at the data that can be projected. There are three kinds of data currently supported: images, keypoints and segmentation masks. Both 2D and 3D data is supported (and technically, higher dimensions, but I've yet to find a dataset with 4 spatial dimensions).

- [`Image`](@ref)`{N, T}` represents an `N`-dimensional image. `T` refers to the element type of the array that `Image` wraps, usually a color. When projecting images, proper interpolation methods are used to reduce artifacts like aliasing. See `src/items/image.jl`

- [`MaskBinary`](@ref)`{N}` and [`MaskMulti`](@ref)`{N, T}` likewise represents `N`-dimensional segmentation masks. Unlike images, nearest-neighbor interpolation is used for projecting masks. See `src/items/mask.jl`

- Lastly, [`Keypoints`](@ref)`{N}` represent keypoint data. The data should be an array of `SVector{N}`. Since there are many interpretations of keypoint data, there are also wrapper items for convenience: [`BoundingBox`](@ref) and [`Polygon`](@ref).See `src/items/keypoints.jl`
