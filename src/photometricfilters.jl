function _gaussian_sigma(sigma::Real)
    return _gaussian_sigma((sigma, sigma))
end

function _gaussian_sigma((lo, hi)::Tuple{Real, Real})
    isfinite(lo) && isfinite(hi) && 0 <= lo <= hi ||
        throw(ArgumentError("sigma must be nonnegative and finite, with ordered range endpoints"))
    return _photometric_uniform(lo, hi)
end


"""
    GaussianBlur(sigma = 1; kernel_size = 3)
    GaussianBlur(sigma, kernel_size)

Apply a separable Gaussian blur to a 2D image. `sigma` is positive or a
`(min, max)` pair of positive values to sample uniformly. `kernel_size` is
an odd positive integer or a pair of odd positive sizes in array axis order.
The filter reflects the image at its edges.
"""
struct GaussianBlur{S} <: PhotometricTransform
    sigma::S
    kernel_size::NTuple{2, Int}

    function GaussianBlur(sigma::Union{Real, Tuple{Real, Real}},
                          kernel_size::Union{Integer, Tuple{Integer, Integer}})
        s = _gaussian_sigma(sigma)
        min_sigma = s isa Real ? s : minimum(s)
        min_sigma > 0 || throw(ArgumentError("GaussianBlur requires a positive sigma"))
        sizes = kernel_size isa Integer ? (kernel_size, kernel_size) : kernel_size
        all(k -> k > 0 && isodd(k), sizes) ||
            throw(ArgumentError("kernel_size must contain positive odd sizes"))
        return new{typeof(s)}(s, sizes)
    end
end

GaussianBlur(sigma = 1; kernel_size = 3) = GaussianBlur(sigma, kernel_size)

getrandstate(tfm::GaussianBlur) = _sample_photometric(tfm.sigma)

function photometric!(dst, tfm::GaussianBlur, src, sigma)
    ndims(src) == 2 || throw(ArgumentError("GaussianBlur requires a two-dimensional image"))
    (isempty(src) || tfm.kernel_size == (1, 1)) && return copyto!(dst, src)
    colors = map(_photometric_color, src)
    temp = similar(colors)
    _gaussian_pass!(temp, colors, 1, sigma, tfm.kernel_size[1])
    _gaussian_pass!(colors, temp, 2, sigma, tfm.kernel_size[2])
    return map!(_restore_photometric_color, dst, src, colors)
end

function _reflect_index(i, ax)
    n = length(ax)
    n == 1 && return first(ax)
    j = mod(i - first(ax), 2n - 2)
    return first(ax) + (j < n ? j : 2n - 2 - j)
end

function _gaussian_pass!(dst, src, dim, sigma, kernel_size)
    kernel_size == 1 && return copyto!(dst, src)
    ax = axes(src, dim)
    radius = kernel_size ÷ 2
    weights = [exp(-(j / sigma)^2 / 2) for j in -radius:radius]
    weights ./= sum(weights)
    for i in CartesianIndices(src)
        value = zero(src[i] * first(weights))
        for (offset, weight) in zip(-radius:radius, weights)
            j = _reflect_index(i[dim] + offset, ax)
            value += (dim == 1 ? src[j, i[2]] : src[i[1], j]) * weight
        end
        dst[i] = value
    end
    return dst
end


"""
    GaussianNoise(sigma = 0.1; mean = 0, clamp = true)

Add independent Gaussian noise with the given `mean` and standard deviation `sigma`
to each color channel. Supply a `(min, max)` pair to sample `sigma` uniformly.
Pixels are clamped to `[0, 1]` by default. The random state holds the sampled `sigma`
and a seed so it can be replayed across images.
"""
struct GaussianNoise{S, M<:Real} <: PhotometricTransform
    sigma::S
    mean::M
    clamp::Bool
end

function GaussianNoise(sigma = 0.1; mean::Real = 0, clamp::Bool = true)
    isfinite(mean) || throw(ArgumentError("mean must be finite"))
    return GaussianNoise(_gaussian_sigma(sigma), mean, clamp)
end

getrandstate(tfm::GaussianNoise) = (sigma = _sample_photometric(tfm.sigma), seed = rand(UInt64))

function photometric!(dst, tfm::GaussianNoise, src, randstate)
    rng = Xoshiro(randstate.seed)
    return _map_photometric!(dst, src) do c
        _map_channels(c) do x
            y = x + tfm.mean + randstate.sigma * randn(rng)
            _clip_photometric(y, tfm.clamp)
        end
    end
end


"""
    AdjustSharpness(factor = 1)

Blend a 2D image with a weighted 3×3 smoothing filter. `factor = 0` gives the
smoothed image, `factor = 1` preserves the input, and larger values sharpen it.
The one-pixel border is preserved.
"""
struct AdjustSharpness{T<:Real} <: PhotometricTransform
    factor::T

    function AdjustSharpness(factor::T = 1) where {T<:Real}
        isfinite(factor) && factor >= 0 || throw(ArgumentError("factor must be nonnegative and finite"))
        return new{T}(factor)
    end
end

function photometric!(dst, tfm::AdjustSharpness, src, randstate)
    ndims(src) == 2 || throw(ArgumentError("AdjustSharpness requires a two-dimensional image"))
    (isone(tfm.factor) || minimum(size(src)) < 3) && return copyto!(dst, src)
    input = dst === src ? copy(src) : src
    copyto!(dst, input)
    interior = map(ax -> (first(ax) + 1):(last(ax) - 1), axes(src))
    for i in CartesianIndices(interior)
        original = _photometric_color(input[i])
        smoothed = 4 * original
        for j in CartesianIndices((-1:1, -1:1))
            smoothed += _photometric_color(input[i + j])
        end
        smoothed /= 13
        value = smoothed + tfm.factor * (original - smoothed)
        dst[i] = _restore_photometric_color(input[i], _map_channels(clamp01, value))
    end
    return dst
end
