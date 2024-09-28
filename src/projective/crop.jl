abstract type CropFrom end

struct FromOrigin <: CropFrom end
struct FromCenter <: CropFrom end
struct FromRandom <: CropFrom end


abstract type AbstractCrop <: Transform end


struct Crop{N, F<:CropFrom} <: AbstractCrop
    size::NTuple{N, Int}
    from::F
end

"""
Crop(sz, FromOrigin())
"""
Crop(sz) = Crop(sz, FromOrigin())



function apply(crop::Crop, item::Item; randstate = getrandstate(crop))
    return apply(
        Project(CoordinateTransformations.IdentityTransformation()) |> crop,
        item;
        randstate = (nothing, (randstate,)))
end


"""
Crop(sz, FromCenter())
"""
CenterCrop(sz) = Crop(sz, FromCenter())
"""
Crop(sz, FromRandom())
"""
RandomCrop(sz) = Crop(sz, FromRandom())

# The random state of a [`Crop`](@ref) consists of offsets from the origin.

getrandstate(crop::Crop{N, FromOrigin}) where N = Tuple(0. for _ in 1:N)
getrandstate(crop::Crop{N, FromCenter}) where N = Tuple(0.5 for _ in 1:N)
getrandstate(crop::Crop{N, FromRandom}) where N = Tuple(rand() for _ in 1:N)


struct PadDivisible <: AbstractCrop
    by::Int
end



struct CroppedProjectiveTransform{P<:ProjectiveTransform, C<:Tuple} <: ProjectiveTransform
    tfm::P
    crops::C
end


function getrandstate(cropped::CroppedProjectiveTransform)
    return (getrandstate(cropped.tfm), getrandstate.(cropped.crops))
end


function getprojection(
        cropped::CroppedProjectiveTransform,
        bounds;
        randstate = getrandstate(cropped))
    tfmstate, cropstate = randstate
    return getprojection(cropped.tfm, bounds; randstate = tfmstate)
end

function projectionbounds(
        cropped::CroppedProjectiveTransform,
        P,
        bounds;
        randstate = getrandstate(cropped))
    tfmstate, cropstates = randstate
    bounds_ = projectionbounds(cropped.tfm, P, bounds; randstate = tfmstate)
    for (crop, cropstate) in zip(cropped.crops, cropstates)
        bounds_ = cropbounds(crop, bounds_; randstate = cropstate)
    end
    return bounds_
end

compose(tfm::ProjectiveTransform, crop::AbstractCrop) = CroppedProjectiveTransform(tfm, (crop,))
compose(tfm::ProjectiveTransform, cropped::CroppedProjectiveTransform) =
    CroppedProjectiveTransform(tfm |> cropped.tfm, cropped.crops)

function compose(composed::ComposedProjectiveTransform, cropped::CroppedProjectiveTransform)
    return CroppedProjectiveTransform(composed |> cropped.tfm, cropped.crops)
end

function compose(cropped::CroppedProjectiveTransform, crop::AbstractCrop)
    return CroppedProjectiveTransform(cropped.tfm, (cropped.crops..., crop))
end

function compose(cropped::CroppedProjectiveTransform, projective::ProjectiveTransform)
    return Sequence(cropped, projective)
end

function compose(cropped::CroppedProjectiveTransform, composed::ComposedProjectiveTransform)
    return Sequence(cropped, composed)
end


cropbounds(crop::Crop, bounds::Bounds; randstate=getrandstate(crop)) = offsetcropbounds(crop.size, bounds, randstate)

function cropbounds(
        crop::PadDivisible,
        bounds::Bounds;
        randstate = getrandstate(crop))
    ranges = bounds.rs

    sz = length.(ranges)
    pad = (crop.by .- (sz .% crop.by)) .% crop.by

    start = minimum.(ranges)
    end_ = start .+ sz .+ pad .- 1
    rs = UnitRange.(start, end_)
    return Bounds(rs)
end


"""
    offsetcropbounds(sz, bounds, offsets)

Calculate offset bounds for a crop of size `sz`.

For every dimension i where `sz[i] < length(indices[i])`, offsets
the crop by `offsets[i]` times the difference between the two.
"""
function offsetcropbounds(
        sz::NTuple{N, Int},
        bounds::Bounds{N},
        offsets::NTuple{N, T}) where {N, T<:AbstractFloat}
    offsets = map(o -> o == one(T) ? one(T) - eps(T) : o, offsets)
    indices = bounds.rs
    mins = getindex.(indices, 1)
    diffs = length.(indices) .- sz .+ 1

    startindices = floor.(Int, mins .+ (diffs .* offsets))
    endindices = startindices .+ sz .- 1

    bs = Bounds(UnitRange.(startindices, endindices))
    return bs
end
