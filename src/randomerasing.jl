"""
    RandomErasing(; p = 0.5, scale = (0.02, 0.33), ratio = (0.3, 3.3), value = 0)

With probability `p`, erase a random box in an [`Image`](@ref) with any number of
spatial dimensions, a rectangle in 2D. Sample its volume fraction uniformly from
`scale ⊆ (0, 1]`. Sample the ratio of each side to the last one, except for the last
one, log-uniformly and independently from `ratio`; in 2D, this is the height-to-width
ratio. If no box fits after ten attempts, leave the image pixels unchanged.

Fill the box with a real number or color converted to the pixel type.
With `value = :random`, draw each color channel uniformly between zero and one
before conversion, preserving alpha. Machine integer pixels use their full
range. Masks and keypoints passed alongside the image are unchanged.
"""
struct RandomErasing{S, R, V} <: PhotometricTransform
    p::Float64
    scale::S
    logratio::R
    value::V

    function RandomErasing(; p::Real = 0.5, scale = (0.02, 0.33), ratio = (0.3, 3.3), value = 0)
        0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1]"))
        scales = _erasing_bounds(scale, "scale")
        ratios = _erasing_bounds(ratio, "ratio")
        last(scales) <= 1 || throw(ArgumentError("scale must be at most one"))
        value === :random || value isa Union{Real, Colorant} ||
            throw(ArgumentError("value must be a real number, color, or :random"))
        area = _photometric_uniform(scales...)
        logratio = _photometric_uniform(map(log, ratios)...)
        return new{typeof(area), typeof(logratio), typeof(value)}(Float64(p), area, logratio, value)
    end
end

function _erasing_bounds(bounds, name)
    bounds isa Tuple{Real, Real} || throw(ArgumentError("$name must be a two-element tuple of real numbers"))
    lo, hi = bounds
    0 < lo <= hi && isfinite(hi) || throw(ArgumentError("$name bounds must be positive, finite, and ordered"))
    return bounds
end

getrandstate(::RandomErasing) = (probability = rand(), seed = rand(UInt64))

function photometric!(dst, tfm::RandomErasing, src, randstate)
    probability, seed = randstate
    copyto!(dst, src)
    probability < tfm.p || return dst
    rng = Xoshiro(seed)
    box = _erasing_box(tfm, rng, axes(src))
    isnothing(box) && return dst
    region = view(dst, box...)
    if tfm.value === :random
        map!(pixel -> _erasing_noise(rng, pixel), region, region)
    else
        fill!(region, convert(eltype(dst), tfm.value))
    end
    return dst
end

function _erasing_box(tfm, rng, ranges::NTuple{N, AbstractUnitRange}) where N
    lengths = length.(ranges)
    for _ in 1:10
        volume = prod(float, lengths) * _sample_erasing(rng, tfm.scale)
        ratios = ntuple(_ -> exp(_sample_erasing(rng, tfm.logratio)), Val(N - 1))
        last_side = (volume / prod(ratios))^(1 / N)
        sides = (map(ratio -> ratio * last_side, ratios)..., last_side)
        all(side -> 0.5 <= side, sides) && all(sides .<= lengths .+ 0.5) || continue
        extent = round.(Int, sides)
        all(1 .<= extent .<= lengths) || continue
        starts = map((r, n) -> rand(rng, first(r):(last(r) - n + 1)), ranges, extent)
        return map((start, n) -> start:(start + n - 1), starts, extent)
    end
    return nothing
end

_sample_erasing(rng, x::Real) = x
_sample_erasing(rng, x::Sampleable) = rand(rng, x)

_erasing_noise(rng, pixel::Union{Base.BitInteger, Bool}) = rand(rng, typeof(pixel))

function _erasing_noise(rng, pixel)
    color = _photometric_color(pixel)
    return _restore_photometric_color(pixel, _map_channels(x -> rand(rng, typeof(x)), color))
end
