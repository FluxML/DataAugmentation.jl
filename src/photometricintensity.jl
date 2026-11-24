"""
    AdjustGamma(gamma; gain = 1, clamp = true)

Apply `gain * x^gamma` independently to image channels. Pixels are clamped to
`[0, 1]` by default.
"""
struct AdjustGamma{T<:Real} <: PhotometricTransform
    gamma::T
    gain::T
    clamp::Bool
    function AdjustGamma(gamma::Real; gain::Real=1, clamp::Bool=true)
        isfinite(gamma) && gamma >= 0 || throw(ArgumentError("gamma must be nonnegative and finite"))
        isfinite(gain) && gain >= 0 || throw(ArgumentError("gain must be nonnegative and finite"))
        T = promote_type(typeof(gamma), typeof(gain))
        new{T}(gamma, gain, clamp)
    end
end

function photometric!(dst, tfm::AdjustGamma, src, randstate)
    return _map_photometric!(dst, src) do c
        _map_channels(x -> _clip_photometric(tfm.gain * x^tfm.gamma, tfm.clamp), c)
    end
end

"""
    Solarize(threshold=0.5)

Invert image channel values greater than or equal to `threshold`.
"""
struct Solarize{T<:Real} <: PhotometricTransform
    threshold::T
end

Solarize() = Solarize(0.5)

function photometric!(dst, tfm::Solarize, src, randstate)
    return _map_photometric!(dst, src) do c
        _map_channels(x -> x >= tfm.threshold ? one(x) - x : x, c)
    end
end

"""
    Posterize(bits=4)

Quantize unit-range channels to 8 bits, then keep the `bits` most significant bits
and set the other bits to zero.
"""
struct Posterize <: PhotometricTransform
    bits::Int
    function Posterize(bits::Integer=4)
        1 <= bits <= 8 || throw(ArgumentError("bits must be between 1 and 8"))
        new(bits)
    end
end

function photometric!(dst, tfm::Posterize, src, randstate)
    mask = 0xff << (8 - tfm.bits)
    return _map_photometric!(dst, src) do c
        _map_channels(x -> (UInt8(_intensity_bin(x) - 1) & mask) / 255, c)
    end
end

"""
    Invert()

Replace each color channel value `x` with `1 - x`.
"""
struct Invert <: PhotometricTransform end

function photometric!(dst, ::Invert, src, randstate)
    return _map_photometric!(dst, src) do c
        _map_channels(x -> one(x) - x, c)
    end
end

_intensity_bin(x) = round(Int, clamp01(x) * 255) + 1

_channels(c::Real) = (c,)
_channels(c::AbstractGray) = (gray(c),)
_channels(c::AbstractRGB) = (red(c), green(c), blue(c))

_from_channels(::Real, values) = only(values)
_from_channels(::AbstractGray, values) = Gray(only(values))
_from_channels(::AbstractRGB, values) = RGB(values...)

_equalized_value(::Nothing, x) = x
_equalized_value(lut, x) = lut[_intensity_bin(x)]

"""
    Equalize()

Equalize each image channel using a 256-bin histogram over `[0, 1]`.
Channels whose values all fall in one bin are preserved.
"""
struct Equalize <: PhotometricTransform end

function photometric!(dst, ::Equalize, src, randstate)
    isempty(src) && return dst
    hists = map(_ -> zeros(Int, 256), _channels(_photometric_color(first(src))))
    for pixel in src
        for (hist, x) in zip(hists, _channels(_photometric_color(pixel)))
            hist[_intensity_bin(x)] += 1
        end
    end
    luts = map(hists) do hist
        first_count = hist[findfirst(!iszero, hist)]
        first_count == length(src) && return nothing
        clamp01.((cumsum(hist) .- first_count) ./ (length(src) - first_count))
    end
    all(isnothing, luts) && return copyto!(dst, src)
    return _map_photometric!(dst, src) do c
        _from_channels(c, map(_equalized_value, luts, _channels(c)))
    end
end

"""
    Autocontrast()

Stretch each image channel's observed range to `[0, 1]`, preserving constant channels.
"""
struct Autocontrast <: PhotometricTransform end

function photometric!(dst, ::Autocontrast, src, randstate)
    isempty(src) && return dst
    mins = maxs = _channels(_photometric_color(first(src)))
    for pixel in src
        values = _channels(_photometric_color(pixel))
        mins = map(min, mins, values)
        maxs = map(max, maxs, values)
    end
    mins == maxs && return copyto!(dst, src)
    return _map_photometric!(dst, src) do c
        stretched = map(_channels(c), mins, maxs) do x, lo, hi
            lo == hi ? x : (x - lo) / (hi - lo)
        end
        _from_channels(c, stretched)
    end
end
