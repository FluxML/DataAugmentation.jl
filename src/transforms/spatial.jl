
# SpatialItem interface

# for data where affine transformations can be applied, the bounds are important
# for calculating the transformation matrix

getbounds(item::Image)::Tuple = size(item.data)
getbounds(item::Keypoints)::Tuple = item.bounds


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
apply(t::Crop, item::Image) = error("Not implemented")

# TODO: Remove keypoints that are not in new bounds
apply(t::Crop, keypoints::Keypoints) = Keypoints(keypoints.data, getcrop(t, keypoints))


# FlipTransform

"""
    FlipX()

Flips an `item` along the x-axis
"""
struct FlipX <: Transform end

(t::FlipX)(image::Image, param = nothing) = Image(reverse(image.data; dims = 2))

function (t::FlipX)(keypoints::Keypoints, param)
    _, w = keypoints.bounds
    data = map(k -> fmap((p) -> (p[1], w - p[2]), k), keypoints.data)
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

(t::FlipY)(image::Image, param = nothing) = Image(reverse(image.data; dims = 1))

function (t::FlipY)(keypoints::Keypoints, param = nothing)
    h, _ = keypoints.bounds
    data = map(k -> fmap((p) -> (h - p[1], p[2]), k), keypoints.data),
    return Keypoints(
        keypoints.bounds
    )
end


# Rotations

struct Rotate90 <: Transform end
(t::Rotate90)(image::Image) = Image(rotl90(image.data))
(t::Rotate90)(keypoints::Keypoints) = error("Not implemented")

struct Rotate180 <: Transform end
(t::Rotate180)(image::Image) = Image(rot180(image.data))
(t::Rotate180)(keypoints::Keypoints) = error("Not implemented")

struct Rotate270 <: Transform end
(t::Rotate270)(image::Image) = Image(rotr90(image.data))
(t::Rotate270)(keypoints::Keypoints) = error("Not implemented")
