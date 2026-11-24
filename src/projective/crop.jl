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

`sz` is a tuple with one size per dimension. An integer `sz` is the same size along every
dimension, so that `Crop(32, FromRandom())` crops a square from an image and a cube from a
volume. An integer `padding` is the same for all dimensions, too; any other form of `padding`
fixes the dimensionality.

With symmetric `padding` (one integer for all axes or one per axis), the bounds are
extended on both sides before cropping. See [`Pad`](@ref) for asymmetric padding and
the available padding modes. By default, padded pixels use the item's extrapolation
setting, including when `sz` exceeds the bounds. With `pad_if_needed = true`, each
side of an undersized axis is padded by the shortfall before choosing the crop.
`fill` applies to [`Image`](@ref) items only; masks use their own extrapolation.
With `pad_if_needed`, `fill` or a non-constant `padding_mode`, the result is a
plain transform that is not fused with neighboring projective transforms.

## Examples

```julia
tfm = Crop((32, 32), FromRandom(); padding = 4) |> PinOrigin()
tfm(Image(rand(RGB, 32, 32)))
```
"""
function Crop(sz::NTuple{N, Int}, from::CropFrom = FromOrigin(); padding = 0,
              pad_if_needed::Bool = false, fill = nothing, padding_mode::Symbol = :constant) where N
    sides = _padparams(padding, N)
    if !pad_if_needed && isnothing(fill) && padding_mode === :constant &&
            (padding isa Integer || padding isa NTuple{N, Integer})
        return Crop(sz, from, map(first, sides))
    end
    _validate_padding_mode(padding_mode)
    crop = Crop(sz, from, ntuple(_ -> 0, N))
    return PaddedCrop(crop, sides, pad_if_needed, fill, padding_mode)
end

# A crop given by a single size has the same size along every dimension. Its random
# state is a seed that is expanded to one offset per dimension once the item, and with it
# the dimensionality, is known. Items that are applied together still get the same crop.
struct IsotropicCrop{F<:CropFrom} <: AbstractCrop
    size::Int
    from::F
    padding::Int
end

struct IsotropicPaddedCrop{F<:CropFrom, X} <: Transform
    size::Int
    from::F
    padding::Int
    pad_if_needed::Bool
    fill::X
    padding_mode::Symbol
end

function Crop(sz::Integer, from::CropFrom = FromOrigin(); padding = 0,
              pad_if_needed::Bool = false, fill = nothing, padding_mode::Symbol = :constant)
    n = _padding_ndims(padding)
    if !isnothing(n)
        return Crop(ntuple(_ -> Int(sz), n), from; padding, pad_if_needed, fill, padding_mode)
    end
    pad = only(_padparams(padding, 1))[1]
    _validate_padding_mode(padding_mode)
    if !pad_if_needed && isnothing(fill) && padding_mode === :constant
        return IsotropicCrop(Int(sz), from, pad)
    end
    return IsotropicPaddedCrop(Int(sz), from, pad, pad_if_needed, fill, padding_mode)
end

# The dimensionality implied by `padding`, if any.
_padding_ndims(::Integer) = nothing
_padding_ndims(padding::NTuple{4, Integer}) = 2
_padding_ndims(padding::Tuple) = length(padding)
_padding_ndims(padding) =
    throw(ArgumentError("padding must be an integer, one integer or pair per axis, or a 2D four-tuple"))

_isotropic_randstate(::FromRandom) = rand(UInt64)
_isotropic_randstate(::CropFrom) = nothing

getrandstate(crop::IsotropicCrop) = _isotropic_randstate(crop.from)
getrandstate(crop::IsotropicPaddedCrop) = _isotropic_randstate(crop.from)

# Offsets given explicitly, as the random state of a crop with a size per dimension,
# are used as they are.
_cropoffsets(::FromOrigin, ::Nothing, ::Val{N}) where N = ntuple(_ -> 0., Val(N))
_cropoffsets(::FromCenter, ::Nothing, ::Val{N}) where N = ntuple(_ -> 0.5, Val(N))

function _cropoffsets(::FromRandom, seed::Integer, ::Val{N}) where N
    rng = Xoshiro(seed)
    return ntuple(_ -> rand(rng), Val(N))
end

function _cropoffsets(::CropFrom, offsets::Tuple, ::Val{N}) where N
    length(offsets) == N ||
        throw(ArgumentError("expected $N crop offsets for $N-dimensional data, got $(length(offsets))"))
    return offsets
end

_resolve(crop::IsotropicCrop, ::Val{N}) where N =
    Crop(ntuple(_ -> crop.size, Val(N)), crop.from, ntuple(_ -> crop.padding, Val(N)))

function cropbounds(crop::IsotropicCrop, bounds::Bounds{N}; randstate = getrandstate(crop)) where N
    offsets = _cropoffsets(crop.from, randstate, Val(N))
    return cropbounds(_resolve(crop, Val(N)), bounds; randstate = offsets)
end

function _resolve(tfm::IsotropicPaddedCrop, ::Val{N}) where N
    return Crop(ntuple(_ -> tfm.size, Val(N)), tfm.from; tfm.padding, tfm.pad_if_needed,
                tfm.fill, tfm.padding_mode)
end


function apply(crop::Union{Crop, IsotropicCrop}, item::Item; randstate = getrandstate(crop))
    return apply(
        Project(CoordinateTransformations.IdentityTransformation()) |> crop,
        item;
        randstate = (nothing, (randstate,)))
end

getrandstate(tfm::PaddedCrop) = getrandstate(tfm.crop)

function apply(tfm::IsotropicPaddedCrop, item::Item; randstate = getrandstate(tfm))
    return _apply_padded(tfm, item, getbounds(item), randstate)
end

function _apply_padded(tfm::IsotropicPaddedCrop, item, ::Bounds{N}, randstate) where N
    offsets = _cropoffsets(tfm.from, randstate, Val(N))
    return apply(_resolve(tfm, Val(N)), item; randstate = offsets)
end

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
see [`Crop`](@ref). An integer `sz` is the same size along every dimension: a square
crop of an image, a cube crop of a volume.
"""
CenterCrop(sz; kwargs...) = Crop(sz, FromCenter(); kwargs...)

"""
    RandomCrop(sz; padding = 0, pad_if_needed = false, fill = nothing, padding_mode = :constant)

Crop a region of size `sz` from a random location. For example,
`RandomCrop((32, 32); padding = 4)` pads a 32×32 image by 4 on each side and crops
a random 32×32 window from the result. An integer `sz` is the same size along every
dimension: `RandomCrop(32)` crops a square from an image and a cube from a volume.
See [`Crop`](@ref) for padding behavior.
"""
RandomCrop(sz; kwargs...) = Crop(sz, FromRandom(); kwargs...)

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
