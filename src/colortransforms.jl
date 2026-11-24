function _factor_distribution(amount::Real)
    isfinite(amount) && amount >= 0 || throw(ArgumentError("amount must be nonnegative and finite"))
    δ = float(amount)
    return _photometric_uniform(max(0, 1 - δ), 1 + δ)
end

function _factor_distribution((lo, hi)::Tuple{Real, Real})
    isfinite(lo) && isfinite(hi) && 0 <= lo <= hi ||
        throw(ArgumentError("factor endpoints must be finite, nonnegative, and ordered"))
    return _photometric_uniform(lo, hi)
end

_factor_distribution(distribution::Sampleable) = distribution

const RGBOrGray = Union{AbstractRGB, AbstractGray, AbstractFloat, FixedPoint}


"""
    AdjustBrightness(δ = 0.2; clamp = true)
    AdjustBrightness((min, max); clamp = true)
    AdjustBrightness(distribution; clamp = true)

Multiply each image color channel by a factor sampled uniformly from
`[max(0, 1-δ), 1+δ]`, from the specified interval, or from a `Distributions.Sampleable`.
Pixels are clamped to `[0, 1]` by default. Alpha, masks, and keypoints are preserved.

## Example

```julia
using DataAugmentation, TestImages

item = Image(testimage("lighthouse"))
tfm = AdjustBrightness(0.2)
titems = [tfm(item) for _ in 1:8]
showgrid(titems; ncol = 4, npad = 16)
```
"""
struct AdjustBrightness{S<:Union{Real, Sampleable}} <: PhotometricTransform
    dist::S
    clamp::Bool
end

AdjustBrightness(amount = 0.2; clamp::Bool = true) = AdjustBrightness(_factor_distribution(amount), clamp)

getrandstate(tfm::AdjustBrightness) = _sample_photometric(tfm.dist)

photometric!(dst, tfm::AdjustBrightness, src, factor) = adjustbrightness!(dst, src, factor, tfm.clamp)

adjustbrightness(img, factor, clamp) = adjustbrightness!(similar(img), img, factor, clamp)
adjustbrightness!(img, factor, clamp) = adjustbrightness!(img, img, factor, clamp)

function adjustbrightness!(dst, src::AbstractArray{<:RGBOrGray}, factor, clamp)
    return map!(x -> _clip_photometric(x * factor, clamp), dst, src)
end

function adjustbrightness!(dst, src, factor, clamp)
    return _map_photometric!(dst, src) do c
        _map_channels(x -> _clip_photometric(x * factor, clamp), c)
    end
end


"""
    AdjustContrast(δ = 0.2; clamp = true)
    AdjustContrast((min, max); clamp = true)
    AdjustContrast(distribution; clamp = true)

Transform each color `c` to `c + μ*(1-f)`, where `μ` is the mean color of the image.
Sample `f` uniformly from `[max(0, 1-δ), 1+δ]`, from the specified interval, or from a
`Distributions.Sampleable`. Pixels are clamped to `[0, 1]` by default. Alpha, masks,
and keypoints are preserved.

## Example

```julia
using DataAugmentation, TestImages

item = Image(testimage("lighthouse"))
tfm = AdjustContrast(0.2)
titems = [tfm(item) for _ in 1:8]
showgrid(titems; ncol = 4, npad = 16)
```
"""
struct AdjustContrast{S<:Union{Real, Sampleable}} <: PhotometricTransform
    dist::S
    clamp::Bool
end

AdjustContrast(amount = 0.2; clamp::Bool = true) = AdjustContrast(_factor_distribution(amount), clamp)

getrandstate(tfm::AdjustContrast) = _sample_photometric(tfm.dist)

photometric!(dst, tfm::AdjustContrast, src, factor) = adjustcontrast!(dst, src, factor, tfm.clamp)

adjustcontrast(img, factor, clamp) = adjustcontrast!(similar(img), img, factor, clamp)
adjustcontrast!(img, factor, clamp) = adjustcontrast!(img, img, factor, clamp)

function adjustcontrast!(dst, src::AbstractArray{<:RGBOrGray}, factor, clamp)
    isempty(src) && return dst
    shift = mean(src) * (1 - factor)
    return map!(x -> _clip_photometric(x + shift, clamp), dst, src)
end

function adjustcontrast!(dst, src, factor, clamp)
    isempty(src) && return dst
    shift = mean(_photometric_color, src) * (1 - factor)
    return _map_photometric!(dst, src) do c
        _map_channels(x -> _clip_photometric(x, clamp), c + shift)
    end
end


"""
    AdjustSaturation(δ = 0.2)
    AdjustSaturation((min, max))
    AdjustSaturation(distribution)

Blend each color with its grayscale luminance. A factor of zero gives grayscale and a
factor of one preserves the color. The factor is sampled uniformly from
`[max(0, 1-δ), 1+δ]`, from the specified interval, or from a `Distributions.Sampleable`.
"""
struct AdjustSaturation{S<:Union{Real, Sampleable}} <: PhotometricTransform
    dist::S

    function AdjustSaturation(amount = 0.2)
        dist = _factor_distribution(amount)
        return new{typeof(dist)}(dist)
    end
end

getrandstate(tfm::AdjustSaturation) = _sample_photometric(tfm.dist)

function photometric!(dst, ::AdjustSaturation, src, factor)
    isone(factor) && return copyto!(dst, src)
    return _map_photometric!(dst, src) do c
        _adjust_saturation(c, factor)
    end
end

_adjust_saturation(c::Union{Real, AbstractGray}, factor) = c

function _adjust_saturation(c::AbstractRGB, factor)
    luminance = gray(Gray(c))
    return _map_channels(x -> clamp01(luminance + factor * (x - luminance)), c)
end


function _hue_distribution(amount::Real)
    0 <= amount <= 0.5 || throw(ArgumentError("hue amount must be in [0, 0.5]"))
    return _hue_distribution((-float(amount), float(amount)))
end

function _hue_distribution((lo, hi)::Tuple{Real, Real})
    -0.5 <= lo <= hi <= 0.5 || throw(ArgumentError("hue endpoints must be ordered within [-0.5, 0.5]"))
    return _photometric_uniform(lo, hi)
end

_hue_distribution(distribution::Sampleable) = distribution


"""
    AdjustHue(δ = 0.1)
    AdjustHue((min, max))
    AdjustHue(distribution)

Shift the hue in HSV space by a sampled fraction of a turn. Sample uniformly from
`[-δ, δ]`, from the specified interval, or from a `Distributions.Sampleable`.
Specify shifts between `-0.5` and `0.5` using nonnegative RGB inputs.
"""
struct AdjustHue{S<:Union{Real, Sampleable}} <: PhotometricTransform
    dist::S

    function AdjustHue(amount = 0.1)
        dist = _hue_distribution(amount)
        return new{typeof(dist)}(dist)
    end
end

getrandstate(tfm::AdjustHue) = _sample_photometric(tfm.dist)

function photometric!(dst, ::AdjustHue, src, shift)
    iszero(shift) && return copyto!(dst, src)
    return _map_photometric!(dst, src) do c
        _adjust_hue(c, shift)
    end
end

_adjust_hue(c::Union{Real, AbstractGray}, shift) = c

function _adjust_hue(c::AbstractRGB, shift)
    hsv = HSV(c)
    return RGB(HSV(mod(hsv.h + 360 * shift, 360), hsv.s, hsv.v))
end


"""
    ColorJitter(; brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1)

Sample brightness, contrast, saturation, and hue adjustments independently and apply them
in a uniformly random order. Each keyword accepts the same parameters as its respective
adjustment transform. Set a keyword to `0` or `nothing` to skip that adjustment.
"""
struct ColorJitter{T<:Tuple} <: PhotometricTransform
    transforms::T
end

function ColorJitter(; brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1)
    types = (AdjustBrightness, AdjustContrast, AdjustSaturation, AdjustHue)
    return ColorJitter(map(_jitter_stage, types, (brightness, contrast, saturation, hue)))
end

_jitter_stage(T, ::Nothing) = nothing
_jitter_stage(T, amount::Real) = iszero(amount) ? nothing : T(amount)
_jitter_stage(T, amount) = T(amount)

function getrandstate(tfm::ColorJitter)
    states = map(t -> isnothing(t) ? nothing : getrandstate(t), tfm.transforms)
    return Tuple(randperm(4)), states
end

function photometric!(dst, tfm::ColorJitter, src, randstate)
    order, states = randstate
    return _apply_photometric_steps!(dst, src, tfm.transforms, states, order)
end

function _apply_photometric_steps!(dst, src, transforms, states, order)
    input = src
    for i in order
        state = states[i]
        isnothing(state) && continue
        photometric!(dst, transforms[i], input, state)
        input = dst
    end
    return input === dst ? dst : copyto!(dst, src)
end


"""
    Grayscale(num_output_channels = 1)

Convert the image to grayscale. With `num_output_channels = 3`, repeat the luminance
in the three RGB channels. Preserve the channel precision and alpha.
"""
struct Grayscale{N} <: PhotometricTransform
    function Grayscale(num_output_channels::Integer = 1)
        num_output_channels in (1, 3) || throw(ArgumentError("num_output_channels must be 1 or 3"))
        return new{Int(num_output_channels)}()
    end
end

"""
    ToRGB()

Convert an image to RGB, replicating grayscale values across the three channels.
Preserve alpha when present.
"""
struct ToRGB <: PhotometricTransform end

const ColorConversion = Union{Grayscale, ToRGB}

_conversion_eltype(::Grayscale{1}, ::Type{T}) where {T<:Real} = T
_conversion_eltype(::Grayscale{1}, ::Type{T}) where {T<:Colorant} = Gray{eltype(T)}
_conversion_eltype(::Grayscale{1}, ::Type{T}) where {T<:TransparentColor} = GrayA{eltype(T)}

_rgb_channel_type(::Type{T}) where {T<:Union{AbstractFloat, FixedPoint}} = T
_rgb_channel_type(::Type{T}) where {T<:Real} = floattype(T)

_conversion_eltype(::Union{Grayscale{3}, ToRGB}, ::Type{T}) where {T<:Real} = RGB{_rgb_channel_type(T)}
_conversion_eltype(::Union{Grayscale{3}, ToRGB}, ::Type{T}) where {T<:Colorant} = RGB{_rgb_channel_type(eltype(T))}
_conversion_eltype(::Union{Grayscale{3}, ToRGB}, ::Type{T}) where {T<:TransparentColor} = RGBA{_rgb_channel_type(eltype(T))}

_conversion_color(::ToRGB, c) = c
_conversion_color(::Grayscale, c::Union{Real, AbstractGray}) = c
_conversion_color(::Grayscale, c::AbstractRGB) = Gray(c)

_convert_image_pixel(T, pixel, c) = convert(T, c)
_convert_image_pixel(::Type{T}, pixel::TransparentColor, c::Color) where {T<:TransparentColor} = convert(T, c, alpha(pixel))
_convert_image_pixel(::Type{T}, pixel, c::AbstractGray) where {T<:Real} = convert(T, gray(c))

function _convert_image!(dst, tfm::ColorConversion, item::Image)
    src = itemdata(item)
    if dst !== src && Base.mightalias(dst, src)
        src = copy(src)
    end
    convert_pixel(p) = _convert_image_pixel(eltype(dst), p, _conversion_color(tfm, _photometric_color(p)))
    map!(convert_pixel, dst, src)
    return _photometric_image(item, dst, convert_pixel)
end

function apply(tfm::ColorConversion, item::Image; randstate = nothing)
    src = itemdata(item)
    T = _conversion_eltype(tfm, eltype(src))
    return _convert_image!(similar(src, T), tfm, item)
end

function apply!(buf::Image, tfm::ColorConversion, item::Image; randstate = nothing)
    src, dst = itemdata(item), itemdata(buf)
    if eltype(dst) != _conversion_eltype(tfm, eltype(src)) || axes(dst) != axes(src)
        return apply(tfm, item)
    end
    return _convert_image!(dst, tfm, item)
end


"""
    RandomGrayscale(p = 0.1)

With probability `p`, turn colors into grayscale while keeping the original element type
and alpha. In RGB, the three output channels all contain the luminance.
"""
struct RandomGrayscale{P<:Real} <: PhotometricTransform
    p::P

    function RandomGrayscale(p::P = 0.1) where {P<:Real}
        0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1]"))
        return new{P}(p)
    end
end

getrandstate(tfm::RandomGrayscale) = rand() < tfm.p

function photometric!(dst, ::RandomGrayscale, src, grayscale)
    grayscale || return copyto!(dst, src)
    return map!(dst, src) do p
        _grayscale_pixel(p, _photometric_color(p))
    end
end

_grayscale_pixel(pixel, ::Union{Real, AbstractGray}) = pixel
_grayscale_pixel(pixel, c::AbstractRGB) = _restore_photometric_color(pixel, RGB(Gray(c)))
