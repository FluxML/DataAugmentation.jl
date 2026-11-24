struct _SingletonInverse{P,N} <: CoordinateTransformations.Transformation
    projection::P
    centers::SVector{N, Float32}
    axes::NTuple{N, Bool}
end

function (t::_SingletonInverse{P,N})(x::AbstractVector) where {P,N}
    y = t.projection(x)
    return SVector{N}(ntuple(i -> t.axes[i] && abs(y[i] - t.centers[i]) <= 0.5 ? t.centers[i] : y[i], N))
end

struct _SingletonProjection{P,N} <: CoordinateTransformations.Transformation
    projection::P
    centers::SVector{N, Float32}
    axes::NTuple{N, Bool}
end

(t::_SingletonProjection)(x) = t.projection(x)
Base.inv(t::_SingletonProjection) = _SingletonInverse(inv(t.projection), t.centers, t.axes)
Base.inv(t::_SingletonInverse) = _SingletonProjection(inv(t.projection), t.centers, t.axes)

_prefilter_image(data, inverse::_SingletonInverse) = _prefilter_image(data, inverse.projection)

# The inverse samples an expanded singleton axis at its only pixel center.
function _singletonprojection(P, bounds::Bounds{N}, sizes::NTuple{N,Int}) where N
    lengths = length.(bounds.rs)
    axes = map((n, s) -> n == 1 && s > 1, lengths, sizes)
    any(axes) || return P
    return _SingletonProjection(P, SVector{N, Float32}(mean.(bounds.rs)), axes)
end

abstract type ResizeTransform <: ProjectiveTransform end

function getprojection(tfm::ResizeTransform, bounds::Bounds{2}; randstate = getrandstate(tfm))
    return getprojection(ScaleFixed(_resized_size(tfm, bounds, randstate)), bounds)
end

function projectionbounds(tfm::ResizeTransform, P, bounds::Bounds{2}; randstate = getrandstate(tfm))
    return projectionbounds(ScaleFixed(_resized_size(tfm, bounds, randstate)), P, bounds)
end

"""
    Resize(size; max_size = nothing)

An integer `size` sets the shorter side of 2D spatial data while preserving
aspect ratio. `max_size` caps the longer side and may override `size`.
Use `Resize(nothing; max_size)` to set the longer side, or provide a
`(height, width)` tuple to set both sides explicitly.
"""
struct Resize <: ResizeTransform
    size::Union{Int, NTuple{2, Int}, Nothing}
    max_size::Union{Int, Nothing}
end

function Resize(size::Union{Integer, Tuple{Integer, Integer}, Nothing}; max_size::Union{Integer, Nothing} = nothing)
    if size isa Tuple
        all(>(0), size) || throw(ArgumentError("size entries must be positive"))
        isnothing(max_size) || throw(ArgumentError("max_size requires a scalar size or size = nothing"))
        return Resize(Int.(size), nothing)
    end
    isnothing(size) || size > 0 || throw(ArgumentError("size must be positive"))
    isnothing(max_size) || max_size > 0 || throw(ArgumentError("max_size must be positive"))
    isnothing(size) && isnothing(max_size) && throw(ArgumentError("size = nothing requires max_size"))
    return Resize(size, max_size)
end

function _size_by_short_side(bounds, size, max_size)
    h, w = length.(bounds.rs)
    short, long = minmax(h, w)
    short > 0 || throw(ArgumentError("resize requires nonempty spatial bounds"))
    newlong = isnothing(size) ? max_size : widemul(size, long) ÷ short
    if isnothing(size) || (!isnothing(max_size) && newlong > max_size)
        newlong = max_size
        size = max(1, widemul(max_size, short) ÷ long)
    end
    return h <= w ? (Int(size), Int(newlong)) : (Int(newlong), Int(size))
end

function _resized_size(tfm::Resize, bounds, randstate)
    tfm.size isa Tuple && return tfm.size
    return _size_by_short_side(bounds, tfm.size, tfm.max_size)
end

"""
    ScaleJitter(target_size; scale_range = (0.1, 2.0))

Randomly resize 2D spatial data while preserving aspect ratio. Multiply the
scale needed to fit inside `target_size = (height, width)` by a uniform factor
from `scale_range`.
"""
struct ScaleJitter <: ResizeTransform
    target_size::NTuple{2, Int}
    scale_range::Tuple{Float64, Float64}
end

function ScaleJitter(target_size::Tuple{Integer, Integer}; scale_range::Tuple{Real, Real} = (0.1, 2.0))
    all(>(0), target_size) || throw(ArgumentError("target_size entries must be positive"))
    lo, hi = Float64.(scale_range)
    0 < lo <= hi < Inf || throw(ArgumentError("scale_range must be positive, finite, and ordered"))
    return ScaleJitter(Int.(target_size), (lo, hi))
end

function getrandstate(tfm::ScaleJitter)
    lo, hi = tfm.scale_range
    return lo + rand() * (hi - lo)
end

function _resized_size(tfm::ScaleJitter, bounds, scale)
    sz = length.(bounds.rs)
    all(>(0), sz) || throw(ArgumentError("resize requires nonempty spatial bounds"))
    factor = minimum(tfm.target_size ./ sz) * scale
    return max.(1, floor.(Int, sz .* factor))
end

"""
    RandomShortestSize(min_size; max_size = nothing)

Choose a shorter-side target from an integer or a collection of integers.
Resize as in [`Resize`](@ref), optionally capping the longer side at `max_size`.
"""
struct RandomShortestSize{S<:Tuple} <: ResizeTransform
    min_size::S
    max_size::Union{Int, Nothing}
end

function RandomShortestSize(min_size; max_size::Union{Integer, Nothing} = nothing)
    sizes = min_size isa Integer ? (min_size,) : Tuple(min_size)
    !isempty(sizes) && all(x -> x isa Integer && x > 0, sizes) ||
        throw(ArgumentError("min_size must supply positive integers"))
    isnothing(max_size) || max_size > 0 || throw(ArgumentError("max_size must be positive"))
    return RandomShortestSize(Int.(sizes), isnothing(max_size) ? nothing : Int(max_size))
end

getrandstate(tfm::RandomShortestSize) = rand(tfm.min_size)
_resized_size(tfm::RandomShortestSize, bounds, size) = _size_by_short_side(bounds, size, tfm.max_size)

"""
    RandomResize(min_size, max_size)

Resize with aspect ratio preserved. Sample the shorter-side size uniformly
from `min_size` up to but excluding `max_size`. Equal limits give a fixed size.
"""
struct RandomResize <: ResizeTransform
    min_size::Int
    max_size::Int

    function RandomResize(min_size::Integer, max_size::Integer)
        0 < min_size <= max_size || throw(ArgumentError("expected 0 < min_size <= max_size"))
        return new(min_size, max_size)
    end
end

getrandstate(tfm::RandomResize) = rand(tfm.min_size:max(tfm.min_size, tfm.max_size - 1))
_resized_size(::RandomResize, bounds, size) = _size_by_short_side(bounds, size, nothing)

"""
    RandomResizedCrop(size; scale = (0.08, 1.0), ratio = (0.75, 4/3), attempts = 10)

Sample a crop area as a fraction of the input area and sample the width-to-height
ratio uniformly in log space. Resize the selected region to `size` (an integer
or a `(height, width)` tuple). If no sampled crop fits, use a center crop with
the closest allowed aspect ratio.
"""
struct RandomResizedCrop <: ProjectiveTransform
    size::NTuple{2, Int}
    scale::Tuple{Float64, Float64}
    ratio::Tuple{Float64, Float64}
    attempts::Int
end

function RandomResizedCrop(size::Tuple{Integer, Integer}; scale::Tuple{Real, Real} = (0.08, 1.0),
                          ratio::Tuple{Real, Real} = (0.75, 4/3), attempts::Integer = 10)
    all(>(0), size) || throw(ArgumentError("size entries must be positive"))
    lo, hi = Float64.(scale)
    rlo, rhi = Float64.(ratio)
    0 < lo <= hi <= 1 || throw(ArgumentError("scale must satisfy 0 < low <= high <= 1"))
    0 < rlo <= rhi < Inf || throw(ArgumentError("ratio must be positive, finite, and ordered"))
    attempts > 0 || throw(ArgumentError("attempts must be positive"))
    return RandomResizedCrop(Int.(size), (lo, hi), (rlo, rhi), Int(attempts))
end

RandomResizedCrop(size::Integer; kwargs...) = RandomResizedCrop((size, size); kwargs...)
getrandstate(::RandomResizedCrop) = rand(UInt64)

function _resized_crop_bounds(tfm::RandomResizedCrop, bounds::Bounds{2}, state)
    h, w = length.(bounds.rs)
    min(h, w) > 0 || throw(ArgumentError("resize requires nonempty spatial bounds"))
    r0, c0 = first.(bounds.rs)
    rng = Xoshiro(state)
    loglo, loghi = log.(tfm.ratio)
    for _ in 1:tfm.attempts
        side = sqrt(float(h) * w * _sample_between(rng, tfm.scale...))
        aspect = exp(_sample_between(rng, loglo, loghi) / 2)
        ch, cw = side / aspect, side * aspect
        ch <= h + 0.5 && cw <= w + 0.5 || continue
        rows, cols = round(Int, ch), round(Int, cw)
        0 < rows <= h && 0 < cols <= w || continue
        top, left = r0 + rand(rng, 0:(h - rows)), c0 + rand(rng, 0:(w - cols))
        return Bounds((top:(top + rows - 1), left:(left + cols - 1)))
    end

    input_ratio = w / h
    lo, hi = tfm.ratio
    rows = input_ratio < lo ? max(1, round(Int, w / lo)) : h
    cols = input_ratio > hi ? max(1, round(Int, h * hi)) : w
    top, left = r0 + (h - rows) ÷ 2, c0 + (w - cols) ÷ 2
    return Bounds((top:(top + rows - 1), left:(left + cols - 1)))
end

function getprojection(tfm::RandomResizedCrop, bounds::Bounds{2}; randstate = getrandstate(tfm))
    crop = _resized_crop_bounds(tfm, bounds, randstate)
    P = getprojection(ScaleFixed(tfm.size), crop)
    P isa IdentityTransformation && return Translation(SVector(1 .- first.(crop.rs)))
    return P
end

projectionbounds(tfm::RandomResizedCrop, P, bounds::Bounds{2}; randstate = nothing) = Bounds(tfm.size)
