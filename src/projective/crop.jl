abstract type CropFrom end

struct FromOrigin <: CropFrom end
struct FromCenter <: CropFrom end
struct FromRandom <: CropFrom end


abstract type AbstractCrop <: Transform end


struct Crop{N, F<:CropFrom} <: AbstractCrop
    size::NTuple{N, Int}
    from::F
end


function apply(crop::Crop, items; randstate = getrandstate(crop))
    return apply(
        Project(CoordinateTransformations.IdentityTransformation()) |> crop,
        items;
        randstate = (nothing, randstate))
end


CenterCrop(sz) = Crop(sz, FromCenter())
RandomCrop(sz) = Crop(sz, FromRandom())

# The random state of a [`Crop`](#) consists of offsets from the origin.

getrandstate(crop::Crop{N, FromOrigin}) where N = Tuple(0. for _ in 1:N)
getrandstate(crop::Crop{N, FromCenter}) where N = Tuple(0.5 for _ in 1:N)
getrandstate(crop::Crop{N, FromRandom}) where N = Tuple(rand() for _ in 1:N)


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


function cropindices(
        cropped::CroppedProjectiveTransform,
        P,
        bounds;
        randstate = getrandstate(cropped))
    tfmstate, cropstate = randstate
    bounds_ = P.(bounds)
    return offsetcropindices(cropped.crop.size, boundsranges(bounds_), cropstate)
end


compose(tfm::ProjectiveTransform, crop::AbstractCrop) = CroppedProjectiveTransform(tfm, crop)



"""
    offsetcropindices(sz, indices, offsets)

Calculate indices for a crop of size `sz` out of `indices`.
Input and output indices are represented as tuples of ranges.

For every dimension i where `sz[i] < length(indices[i])`, offsets
the crop by `offsets[i]` times the difference between the two.
"""
function offsetcropindices(
        sz::NTuple{N, Int},
        indices::NTuple{N, <:AbstractRange},
        offsets::NTuple{N, <:Number}) where N
    mins = getindex.(indices, 1)
    diffs = length.(indices) .- sz

    startindices = floor.(Int, mins .+ (diffs .* offsets))
    endindices = startindices .+ sz .- 1

    return UnitRange.(startindices, endindices)
end
