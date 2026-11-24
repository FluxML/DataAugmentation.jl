"""
    RandomErasing(; p = 0.5, scale = (0.02, 0.33), ratio = (0.3, 3.3), value = 0)

With probability `p`, erase a random rectangle in a two-dimensional [`Image`](@ref).
Sample its area fraction uniformly from `scale ⊆ (0, 1]` and its height-to-width
ratio log-uniformly from `ratio`. If no rectangle fits after ten attempts, leave
the image pixels unchanged.

Fill the rectangle with a real number or color converted to the pixel type.
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
    ndims(src) == 2 || throw(ArgumentError("RandomErasing requires a two-dimensional image"))
    probability, seed = randstate
    copyto!(dst, src)
    probability < tfm.p || return dst
    rng = Xoshiro(seed)
    rectangle = _erasing_rectangle(tfm, rng, axes(src))
    isnothing(rectangle) && return dst
    region = view(dst, rectangle...)
    if tfm.value === :random
        map!(pixel -> _erasing_noise(rng, pixel), region, region)
    else
        fill!(region, convert(eltype(dst), tfm.value))
    end
    return dst
end

function _erasing_rectangle(tfm, rng, (rows, cols))
    height, width = length(rows), length(cols)
    for _ in 1:10
        area = height * width * _sample_erasing(rng, tfm.scale)
        aspect = exp(_sample_erasing(rng, tfm.logratio))
        h, w = sqrt(area * aspect), sqrt(area / aspect)
        0.5 <= h <= height + 0.5 && 0.5 <= w <= width + 0.5 || continue
        erase_height, erase_width = round(Int, h), round(Int, w)
        1 <= erase_height <= height && 1 <= erase_width <= width || continue
        top = rand(rng, first(rows):(last(rows) - erase_height + 1))
        left = rand(rng, first(cols):(last(cols) - erase_width + 1))
        return top:(top + erase_height - 1), left:(left + erase_width - 1)
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
