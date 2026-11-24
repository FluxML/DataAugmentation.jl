abstract type CropFrom end

struct FromOrigin <: CropFrom end
struct FromCenter <: CropFrom end
struct FromRandom <: CropFrom end


abstract type AbstractCrop <: Transform end


struct Crop{N, F<:CropFrom} <: AbstractCrop
    size::NTuple{N, Int}
    from::F
    padding::NTuple{N, Int}
end

struct PaddedCrop{C<:Crop, P, F} <: Transform
    crop::C
    padding::P
    pad_if_needed::Bool
    fill::F
    padding_mode::Symbol
end

"""
    Crop(sz[, from]; padding = 0, pad_if_needed = false, fill = nothing, padding_mode = :constant)

Crop a region of size `sz` from the bounds of an item. `from` is `FromOrigin()` (default),
`FromCenter()` or `FromRandom()`; see also [`CenterCrop`](@ref) and [`RandomCrop`](@ref).

With symmetric `padding` (one integer for all axes or one per axis), the bounds are
extended on both sides before cropping. See [`Pad`](@ref) for asymmetric padding and
the available padding modes. By default, padded pixels use the item's extrapolation
setting, including when `sz` exceeds the bounds. With `pad_if_needed = true`, each
side of an undersized axis is padded by the shortfall before choosing the crop.

## Examples

```julia
tfm = Crop((32, 32), FromRandom(); padding = 4) |> PinOrigin()
apply(tfm, Image(rand(RGB, 32, 32)))
```
"""
function Crop(sz::NTuple{N, Int}, from::CropFrom = FromOrigin(); padding = 0,
              pad_if_needed::Bool = false, fill = nothing, padding_mode::Symbol = :constant) where N
    if !pad_if_needed && isnothing(fill) && padding_mode === :constant
        padding isa Integer && return Crop(sz, from, ntuple(_ -> Int(padding), N))
        padding isa NTuple{N, Integer} && return Crop(sz, from, map(Int, padding))
    end
    _validate_padding_mode(padding_mode)
    crop = Crop(sz, from, ntuple(_ -> 0, N))
    return PaddedCrop(crop, _padparams(padding, N), pad_if_needed, fill, padding_mode)
end


function apply(crop::Crop, item::Item; randstate = getrandstate(crop))
    return apply(
        Project(CoordinateTransformations.IdentityTransformation()) |> crop,
        item;
        randstate = (nothing, (randstate,)))
end

getrandstate(tfm::PaddedCrop) = getrandstate(tfm.crop)

function apply(tfm::PaddedCrop, item::Item; randstate = getrandstate(tfm))
    bounds = getbounds(item)
    padding = if tfm.pad_if_needed
        map(tfm.padding, length.(bounds.rs), tfm.crop.size) do sides, n, size
            extra = max(0, size - n - sum(sides))
            return sides .+ extra
        end
    else
        tfm.padding
    end
    padded = _padded_bounds(bounds, padding)
    target = cropbounds(tfm.crop, padded; randstate)
    materialized = map(bounds.rs, padded.rs, target.rs) do axis, padded_axis, crop_axis
        return (first(axis) - min(first(padded_axis), first(crop_axis)),
                max(last(padded_axis), last(crop_axis)) - last(axis))
    end
    source = apply(Pad(materialized; tfm.fill, tfm.padding_mode), item)
    return project(IdentityTransformation(), source, target)
end


"""
    CenterCrop(sz; padding = 0, kwargs...)

Crop a region of size `sz` from the center. Same as `Crop(sz, FromCenter(); padding)`,
see [`Crop`](@ref).
"""
CenterCrop(sz; kwargs...) = Crop(sz, FromCenter(); kwargs...)

"""
    RandomCrop(sz; padding = 0, pad_if_needed = false, fill = nothing, padding_mode = :constant)

Crop a region of size `sz` from a random location. For example,
`RandomCrop((32, 32); padding = 4)` pads a 32×32 image by 4 on each side and crops
a random 32×32 window from the result. An integer requests a square crop.
See [`Crop`](@ref) for padding behavior.
"""
RandomCrop(sz; kwargs...) = Crop(sz, FromRandom(); kwargs...)
RandomCrop(sz::Integer; kwargs...) = RandomCrop((Int(sz), Int(sz)); kwargs...)

# The random state of a [`Crop`](@ref) consists of offsets from the origin.

getrandstate(crop::Crop{N, FromOrigin}) where N = Tuple(0. for _ in 1:N)
getrandstate(crop::Crop{N, FromCenter}) where N = Tuple(0.5 for _ in 1:N)
getrandstate(crop::Crop{N, FromRandom}) where N = Tuple(rand() for _ in 1:N)


"""
    PadDivisible(by)

Pads the bounds of a projected item so that every side length is divisible by `by`.
"""
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
    return _apply_crop_bounds(cropped.crops, bounds_, cropstates)
end

function _projection_geometry(cropped::CroppedProjectiveTransform, bounds, randstate)
    tfmstate, cropstates = randstate
    P, bounds_ = _projection_geometry(cropped.tfm, bounds, tfmstate)
    return P, _apply_crop_bounds(cropped.crops, bounds_, cropstates)
end

function _apply_crop_bounds(crops, bounds, cropstates)
    for (crop, cropstate) in zip(crops, cropstates)
        bounds = cropbounds(crop, bounds; randstate = cropstate)
    end
    return bounds
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

function compose(cropped1::CroppedProjectiveTransform, cropped2::CroppedProjectiveTransform)
    return Sequence(cropped1, cropped2)
end

function compose(cropped::CroppedProjectiveTransform, composed::ComposedProjectiveTransform)
    return Sequence(cropped, composed)
end


function cropbounds(crop::Crop, bounds::Bounds; randstate = getrandstate(crop))
    return offsetcropbounds(crop.size, padbounds(bounds, crop.padding), randstate)
end

padbounds(bounds::Bounds, padding) = Bounds(map((r, p) -> first(r) - p : last(r) + p, bounds.rs, padding))

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
        offsets::NTuple{N, <:AbstractFloat}) where N
    offsets = map(o -> o == one(o) ? one(o) - eps(typeof(o)) : o, offsets)
    indices = bounds.rs
    mins = getindex.(indices, 1)
    diffs = length.(indices) .- sz .+ 1

    startindices = floor.(Int, mins .+ (diffs .* offsets))
    endindices = startindices .+ sz .- 1

    bs = Bounds(UnitRange.(startindices, endindices))
    return bs
end
