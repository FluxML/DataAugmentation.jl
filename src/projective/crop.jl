abstract type CropFrom end

struct FromOrigin <: CropFrom end
struct FromCenter <: CropFrom end
struct FromRandom <: CropFrom end


abstract type AbstractCrop <: Transform end


struct Crop{N, F<:CropFrom} <: AbstractCrop
    size::NTuple{N, Int}
    from::F
end

Crop(sz) = Crop(sz, FromOrigin())



function apply(crop::Crop, item::Item; randstate = getrandstate(crop))
    return apply(
        Project(CoordinateTransformations.IdentityTransformation()) |> crop,
        item;
        randstate = (nothing, randstate))
end


CenterCrop(sz) = Crop(sz, FromCenter())
RandomCrop(sz) = Crop(sz, FromRandom())

# The random state of a [`Crop`](#) consists of offsets from the origin.

getrandstate(crop::Crop{N, FromOrigin}) where N = Tuple(0. for _ in 1:N)
getrandstate(crop::Crop{N, FromCenter}) where N = Tuple(0.5 for _ in 1:N)
getrandstate(crop::Crop{N, FromRandom}) where N = Tuple(rand() for _ in 1:N)


struct PadDivisible <: AbstractCrop
    by::Int
end



struct CroppedProjectiveTransform{P<:ProjectiveTransform, C<:AbstractCrop} <: ProjectiveTransform
    tfm::P
    crop::C
end


function getrandstate(cropped::CroppedProjectiveTransform)
    return (getrandstate(cropped.tfm), getrandstate(cropped.crop))
end


function getprojection(
        cropped::CroppedProjectiveTransform,
        bounds;
        randstate = getrandstate(cropped))
    tfmstate, cropstate = randstate
    return getprojection(cropped.tfm, bounds; randstate = tfmstate)
end


function projectionbounds(
        cropped::CroppedProjectiveTransform{PT, C},
        P,
        bounds;
        randstate = getrandstate(cropped)) where {PT, C<:Crop}
    tfmstate, cropstate = randstate
    bounds_ = projectionbounds(cropped.tfm, P, bounds; randstate = tfmstate)
    return offsetcropbounds(cropped.crop.size, bounds_, cropstate)
end


function projectionbounds(
        cropped::CroppedProjectiveTransform{PT, PadDivisible},
        P,
        bounds;
        randstate = getrandstate(cropped)) where {PT}
    tfmstate, cropstate = randstate
    bounds_ = projectionbounds(cropped.tfm, P, bounds; randstate = tfmstate)
    ranges = bounds_.rs

    sz = length.(ranges)
    pad = (cropped.crop.by .- (sz .% cropped.crop.by)) .% cropped.crop.by

    start = minimum.(ranges)
    end_ = start .+ sz .+ pad .- 1
    rs = UnitRange.(start, end_)
    return Bounds(rs)
end



compose(tfm::ProjectiveTransform, crop::AbstractCrop) = CroppedProjectiveTransform(tfm, crop)
compose(tfm::ProjectiveTransform, crop::CroppedProjectiveTransform) =
    CroppedProjectiveTransform(tfm |> crop.tfm, crop.crop)

function compose(composed::ComposedProjectiveTransform, cropped::CroppedProjectiveTransform)
    return CroppedProjectiveTransform(composed |> cropped.tfm, cropped.crop)

end

function compose(cropped::CroppedProjectiveTransform, projective::ProjectiveTransform)
    return Sequence(cropped, projective)
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
        offsets::NTuple{N, <:Number}) where N
    indices = bounds.rs
    mins = getindex.(indices, 1)
    diffs = length.(indices) .- sz .+ 1

    startindices = floor.(Int, mins .+ (diffs .* offsets))
    endindices = startindices .+ sz .- 1

    bs = Bounds(UnitRange.(startindices, endindices))
    return bs
end
