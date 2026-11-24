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

Apply a separable Gaussian blur to an image with any number of spatial dimensions.
`sigma` is positive or a `(min, max)` pair of positive values to sample uniformly.
`kernel_size` is an odd positive integer, the same for every dimension, or a tuple of
odd positive sizes in array axis order, one per dimension.
The filter reflects the image at its edges.
"""
struct GaussianBlur{S, K<:Union{Int, Tuple{Vararg{Int}}}} <: PhotometricTransform
    sigma::S
    kernel_size::K

    function GaussianBlur(sigma::Union{Real, Tuple{Real, Real}},
                          kernel_size::Union{Integer, Tuple{Vararg{Integer}}})
        s = _gaussian_sigma(sigma)
        min_sigma = s isa Real ? s : minimum(s)
        min_sigma > 0 || throw(ArgumentError("GaussianBlur requires a positive sigma"))
        sizes = kernel_size isa Integer ? Int(kernel_size) : map(Int, kernel_size)
        all(k -> k > 0 && isodd(k), sizes) ||
            throw(ArgumentError("kernel_size must contain positive odd sizes"))
        return new{typeof(s), typeof(sizes)}(s, sizes)
    end
end

GaussianBlur(sigma = 1; kernel_size = 3) = GaussianBlur(sigma, kernel_size)

getrandstate(tfm::GaussianBlur) = _sample_photometric(tfm.sigma)

_kernel_sizes(k::Int, ::Val{N}) where N = ntuple(_ -> k, Val(N))

function _kernel_sizes(sizes::Tuple, ::Val{N}) where N
    length(sizes) == N ||
        throw(ArgumentError("kernel_size has $(length(sizes)) entries but the image has $N dimensions"))
    return sizes
end

function photometric!(dst, tfm::GaussianBlur, src, sigma)
    sizes = _kernel_sizes(tfm.kernel_size, Val(ndims(src)))
    (isempty(src) || all(isone, sizes)) && return copyto!(dst, src)
    colors = map(_photometric_color, src)
    temp = similar(colors)
    for (dim, k) in enumerate(sizes)
        _gaussian_pass!(temp, colors, dim, sigma, k)
        colors, temp = temp, colors
    end
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
            value += src[CartesianIndex(Base.setindex(Tuple(i), j, dim))] * weight
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

Blend an image with a weighted smoothing filter over the 3×3 neighborhood of each
pixel, or over its 3×3×3 neighborhood in 3D, and so on for other numbers of spatial
dimensions. The center has weight 5 and every other pixel of the neighborhood weight 1.
`factor = 0` gives the smoothed image, `factor = 1` preserves the input, and larger values
sharpen it. The one-pixel border is preserved.
"""
struct AdjustSharpness{T<:Real} <: PhotometricTransform
    factor::T

    function AdjustSharpness(factor::T = 1) where {T<:Real}
        isfinite(factor) && factor >= 0 || throw(ArgumentError("factor must be nonnegative and finite"))
        return new{T}(factor)
    end
end

function photometric!(dst, tfm::AdjustSharpness, src, randstate)
    (isone(tfm.factor) || minimum(size(src)) < 3) && return copyto!(dst, src)
    input = dst === src ? copy(src) : src
    copyto!(dst, input)
    interior = map(ax -> (first(ax) + 1):(last(ax) - 1), axes(src))
    neighborhood = CartesianIndices(map(_ -> -1:1, interior))
    for i in CartesianIndices(interior)
        original = _photometric_color(input[i])
        smoothed = 4 * original
        for j in neighborhood
            smoothed += _photometric_color(input[i + j])
        end
        smoothed /= length(neighborhood) + 4
        value = smoothed + tfm.factor * (original - smoothed)
        dst[i] = _restore_photometric_color(input[i], _map_channels(clamp01, value))
    end
    return dst
end
