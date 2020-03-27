
# SpatialItem interface

# for data where affine transformations can be applied, the bounds are important
# for calculating the transformation matrix
getbounds(itemw::ItemWrapper) = getbounds(getwrapped(itemw))
getbounds(item::Image)::Tuple = size(item.data)
getbounds(item::Keypoints)::Tuple = item.bounds



# Crops

abstract type AbstractCropTransform <: AbstractTransform end

struct CropTransformExact <: AbstractCropTransform
    crop::Tuple
end

struct CropTransformFactor <: AbstractCropTransform
    factors::Tuple
end

getcrop(t::CropTransformExact, _::Item) = t.crop
getcrop(t::CropTransformFactor, item::Item) = Int.(ceil.(getbounds(item) .* t.factors))
CropTransform(crop) = CropTransformExact(crop)
CropTransform(; factors = (1, 1)) = CropTransformFactor(factors)


# TODO: Implement
function (t::AbstractCropTransform)(item::Image)
    error("Not implemented")
end

# TODO: Remove keypoints that are not in new bounds
(t::AbstractCropTransform)(item::Keypoints) = Keypoints(itemdata(item), getcrop(t, item))


# FlipTransform

"""
    FlipX
"""
struct FlipX <: AbstractTransform end

(t::FlipX)(item::Image, param = nothing) = Image(reverse(itemdata(item); dims = 2))

function (t::FlipX)(item::Keypoints, param = nothing)
    _, w = getbounds(item)
    return Keypoints(
        map(k -> applykeypoint((p) -> (p[1], w - p[2]), k), itemdata(item)),
        item.bounds
    )
end

"""
    FlipY
"""
struct FlipY <: AbstractTransform end

(t::FlipY)(item::Image, param = nothing) = Image(reverse(itemdata(item); dims = 1))

function (t::FlipY)(item::Keypoints, param = nothing)
    h, _ = getbounds(item)
    return Keypoints(
        map(k -> applykeypoint((p) -> (h - p[1], p[2]), k), itemdata(item)),
        item.bounds
    )
end


# Rotations

struct Rotate90 <: AbstractTransform end
(t::Rotate90)(item::Image) = Image(rotl90(itemdata(item)))
(t::Rotate90)(item::Keypoints) = error("Not implemented")

struct Rotate180 <: AbstractTransform end
(t::Rotate180)(item::Image) = Image(rot180(itemdata(item)))
(t::Rotate180)(item::Keypoints) = error("Not implemented")

struct Rotate270 <: AbstractTransform end
(t::Rotate270)(item::Image) = Image(rotr90(itemdata(item)))
(t::Rotate270)(item::Keypoints) = error("Not implemented")



# Utils

applykeypoint(f, ::Nothing) = nothing
applykeypoint(f, k::Tuple) = f(k)
applykeypoint(f, k::Tuple{Tuple, Tuple}) = (f(k[1]), f(k[2]))