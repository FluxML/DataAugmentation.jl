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

"""
    Crop(sz[, from]; padding = 0)

Crop a region of size `sz` from the bounds of an item. `from` is `FromOrigin()` (default),
`FromCenter()` or `FromRandom()`; see also [`CenterCrop`](@ref) and [`RandomCrop`](@ref).

With `padding`, the bounds are first extended by `padding` on both sides of every dimension
(`padding` is an `Int` or one `Int` per dimension) and the crop is taken from the padded
bounds. This is the equivalent of `padding` in torchvision's `RandomCrop`.

The padded region is filled according to the `extrapolate` option of the item: black for
an [`Image`](@ref) by default, replicated edge labels for a [`MaskMulti`](@ref). Pass for
example `extrapolate = Reflect()` or `extrapolate = 0` when constructing the item to change
this. A `sz` larger than the bounds is padded in the same way.

## Examples

```julia
tfm = Crop((32, 32), FromRandom(); padding = 4) |> PinOrigin()
apply(tfm, Image(rand(RGB, 32, 32)))
```
"""
function Crop(sz::NTuple{N, Int}, from::CropFrom = FromOrigin(); padding = 0) where N
    return Crop(sz, from, _paddingtuple(padding, Val(N)))
end

_paddingtuple(padding::Integer, ::Val{N}) where N = ntuple(_ -> Int(padding), N)
function _paddingtuple(padding::NTuple{N, Integer}, ::Val{N}) where N
    return map(Int, padding)
end
function _paddingtuple(padding::Tuple, ::Val{N}) where N
    throw(ArgumentError("`padding` must be an integer or a tuple of $N integers, got $padding"))
end


function apply(crop::Crop, item::Item; randstate = getrandstate(crop))
    return apply(
        Project(CoordinateTransformations.IdentityTransformation()) |> crop,
        item;
        randstate = (nothing, (randstate,)))
end


"""
    CenterCrop(sz; padding = 0)

Crop a region of size `sz` from the center. Same as `Crop(sz, FromCenter(); padding)`,
see [`Crop`](@ref).
"""
CenterCrop(sz; padding = 0) = Crop(sz, FromCenter(); padding)

"""
    RandomCrop(sz; padding = 0)

Crop a region of size `sz` from a random location. Same as `Crop(sz, FromRandom(); padding)`,
see [`Crop`](@ref). For example, `RandomCrop((32, 32); padding = 4)` pads a 32×32 image by 4
on each side and crops a random 32×32 window from the result.
"""
RandomCrop(sz; padding = 0) = Crop(sz, FromRandom(); padding)

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
