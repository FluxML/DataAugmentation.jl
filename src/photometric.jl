abstract type PhotometricTransform <: Transform end

const NonColorItem = Union{MaskMulti, MaskBinary, Keypoints}

_photometric_uniform(lo, hi) = lo == hi ? float(lo) : Uniform(float(lo), float(hi))
_sample_photometric(value::Real) = value
_sample_photometric(distribution::Sampleable) = rand(distribution)
_clip_photometric(x, enabled) = enabled ? clamp01(x) : x

apply(::PhotometricTransform, item::NonColorItem; randstate = nothing) = item
apply!(buf, ::PhotometricTransform, item::NonColorItem; randstate = nothing) = item

function apply(tfm::PhotometricTransform, item::Image; randstate = getrandstate(tfm))
    src = itemdata(item)
    dst = similar(src)
    photometric!(dst, tfm, src, randstate)
    return setdata(item, dst)
end

function apply!(buf::Image, tfm::PhotometricTransform, item::Image; randstate = getrandstate(tfm))
    src = itemdata(item)
    dst = itemdata(buf)
    if axes(dst) != axes(src)
        return apply(tfm, item; randstate)
    end
    if dst !== src && Base.mightalias(dst, src)
        src = copy(src)
    end
    photometric!(dst, tfm, src, randstate)
    return setdata(item, dst)
end

function _photometric_image(item::Image, data, convert_pixel)
    fill = item.extrapolate
    if fill isa Union{Real, Colorant}
        fill = convert_pixel(fill)
    end
    return Image(data, item.bounds; interpolate = item.interpolate, extrapolate = fill, antialias = item.antialias)
end

_photometric_color(p::Real) = convert(promote_type(Float32, typeof(float(p))), p)
_photometric_color(p::AbstractGray) = Gray(_photometric_color(gray(p)))
_photometric_color(p::Color) = convert(RGB{promote_type(Float32, floattype(eltype(p)))}, p)
_photometric_color(p::TransparentColor) = _photometric_color(color(p))

_restore_photometric_color(p::T, value) where {T<:Union{Real, Color}} = convert(T, value)
_restore_photometric_color(p::T, value::Real) where {T<:Integer} = round(T, value)
_restore_photometric_color(p::Real, value::AbstractGray) = _restore_photometric_color(p, gray(value))
_restore_photometric_color(p::T, value::Color) where {T<:TransparentColor} = convert(T, value, alpha(p))

_map_channels(f, x::Color) = mapc(f, x)
_map_channels(f, x::Real) = f(x)

function _map_photometric!(f, dst, src)
    return map!(dst, src) do pixel
        _restore_photometric_color(pixel, f(_photometric_color(pixel)))
    end
end
