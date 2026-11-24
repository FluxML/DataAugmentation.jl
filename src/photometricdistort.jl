"""
    RandomChannelPermutation()

Uniformly permute the three RGB channels. Other color spaces are converted to RGB
for the permutation and converted back afterward. Grayscale and alpha are preserved.
"""
struct RandomChannelPermutation <: PhotometricTransform end

getrandstate(::RandomChannelPermutation) = Tuple(randperm(3))

function photometric!(dst, ::RandomChannelPermutation, src, order)
    if order == (1, 2, 3)
        return copyto!(dst, src)
    end
    return map!(dst, src) do pixel
        _permute_rgb(pixel, _photometric_color(pixel), order)
    end
end

_permute_rgb(pixel, ::Union{Real, AbstractGray}, order) = pixel

function _permute_rgb(pixel, value::AbstractRGB, order)
    channels = (red(value), green(value), blue(value))
    return _restore_photometric_color(pixel, RGB(map(i -> channels[i], order)...))
end


"""
    RandomPhotometricDistort(; brightness = (0.875, 1.125), contrast = (0.5, 1.5),
                              saturation = (0.5, 1.5), hue = (-0.05, 0.05), p = 0.5)

Apply brightness, saturation and hue adjustments with factors sampled from the given ranges.
Apply contrast randomly before or after the saturation and hue adjustments, then permute the
RGB channels. Each of the five operations is selected independently with probability `p`.
Hue is measured in turns.
"""
struct RandomPhotometricDistort{T<:Tuple, P<:Real} <: PhotometricTransform
    transforms::T
    p::P
end

function RandomPhotometricDistort(;
        brightness = (0.875, 1.125), contrast = (0.5, 1.5),
        saturation = (0.5, 1.5), hue = (-0.05, 0.05), p::Real = 0.5)
    0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1]"))
    transforms = (AdjustBrightness(brightness), AdjustContrast(contrast),
                  AdjustSaturation(saturation), AdjustHue(hue), RandomChannelPermutation())
    return RandomPhotometricDistort(transforms, p)
end

function getrandstate(tfm::RandomPhotometricDistort)
    order = rand(Bool) ? (1, 2, 3, 4, 5) : (1, 3, 4, 2, 5)
    states = map(tfm.transforms) do transform
        rand() < tfm.p ? getrandstate(transform) : nothing
    end
    return order, states
end

function photometric!(dst, tfm::RandomPhotometricDistort, src, randstate)
    order, states = randstate
    return _apply_photometric_steps!(dst, src, tfm.transforms, states, order)
end


"""
    JPEG(quality = 75)
    JPEG((min_quality, max_quality))
    JPEG(min_quality:max_quality)

Compress and decompress a two-dimensional image in memory using JPEG. Quality is an integer
from 1 to 100, or an inclusive range of qualities to sample uniformly. Lower quality produces
stronger compression. Pixel intensities are clamped to `[0, 1]` and quantized to 8 bits for
the codec, then converted back to the original element type with alpha preserved.

Load `JpegTurbo` with `using JpegTurbo` before applying this transform.
"""
struct JPEG <: PhotometricTransform
    quality::UnitRange{Int}

    function JPEG(quality::AbstractUnitRange{<:Integer})
        1 <= first(quality) <= last(quality) <= 100 ||
            throw(ArgumentError("JPEG quality must be in [1, 100]"))
        return new(Int(first(quality)):Int(last(quality)))
    end
end

JPEG(quality::Integer = 75) = JPEG(quality:quality)
JPEG(quality::Tuple{Integer, Integer}) = JPEG(quality[1]:quality[2])

getrandstate(tfm::JPEG) = rand(tfm.quality)

function photometric!(dst, ::JPEG, src, quality)
    throw(ArgumentError("JPEG requires JpegTurbo. Load it with `using JpegTurbo`."))
end
