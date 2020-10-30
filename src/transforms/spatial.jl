
# SpatialItem interface

# for data where affine transformations can be applied, the bounds are important
# for calculating the transformation matrix

getbounds(item::Image) = item.bounds
getbounds(item::Keypoints) = item.bounds
getbounds(a::AbstractMatrix) = makebounds(a)


# Crops

abstract type Crop <: Transform end

struct CropExact <: Crop
    crop::Tuple
end

struct CropFactor <: Crop
    factors::Tuple
end

getcrop(t::CropExact, ::Item) = t.crop
getcrop(t::CropFactor, item::Item) = Int.(ceil.(getbounds(item) .* t.factors))
Crop(crop) = CropExact(crop)
Crop(; factors = (1, 1)) = CropFactor(factors)


# TODO: Implement
apply(::Crop, ::Image) = error("Not implemented")

# TODO: Remove keypoints that are not in new bounds
apply(t::Crop, keypoints::Keypoints) = Keypoints(keypoints.data, getcrop(t, keypoints))


# FlipTransform

# TODO: implement affine version!

"""
    FlipX()

Flips an `item` along the x-axis
"""
struct FlipX <: Transform end

apply(::FlipX, image::Image, param = nothing) = Image(reverse(image.data; dims = 2))

function apply(::FlipX, keypoints::Keypoints, param)
    _, w = keypoints.bounds
    data = map(k -> fmap((p) -> SVector(p[1], w - p[2]), k), keypoints.data)
    return Keypoints(
        data,
        keypoints.bounds,
    )
end

"""
    FlipY()

Flips an `Item` along the y-axis
"""
struct FlipY <: Transform end

apply(::FlipY, image::Image, param = nothing) = Image(reverse(image.data; dims = 1))

function apply(::FlipY, keypoints::Keypoints, param = nothing)
    h, _ = keypoints.bounds
    data = map(k -> fmap((p) -> SVector(h - p[1], p[2]), k), keypoints.data)
    return Keypoints(
        data,
        keypoints.bounds
    )
end


# Rotations

"""
    Rotate90()

Rotate clock-wise by 90 degress

Faster than [`Rotate`](@ref) for array types.
"""
struct Rotate90 <: Transform end
(t::Rotate90)(image::Image) = Image(rotl90(image.data))
(t::Rotate90)(keypoints::Keypoints) = error("Not implemented")

"""
    Rotate180()

Rotate clock-wise by 180 degress

Faster than [`Rotate`](@ref) for array types.
"""
struct Rotate180 <: Transform end
(t::Rotate180)(image::Image) = Image(rot180(image.data))
(t::Rotate180)(keypoints::Keypoints) = error("Not implemented")

"""
    Rotate270()

Rotate clock-wise by 270 degress

Faster than [`Rotate`](@ref) for array types.
"""
struct Rotate270 <: Transform end
(t::Rotate270)(image::Image) = Image(rotr90(image.data))
(t::Rotate270)(keypoints::Keypoints) = error("Not implemented")
