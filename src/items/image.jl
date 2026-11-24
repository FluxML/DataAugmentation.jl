# We first define the [`Image`](@ref) item. Since we need to keep
# track of the spatial bounds for projective transformations
# we add them as a field. By default, they will simply
# correspond to the image axes.

"""
    Image(image[, bounds]; interpolate=BSpline(Linear()), extrapolate=zero(T), antialias=false)

Item representing an N-dimensional image with element type T. Optionally, the
interpolation and extrapolation method can be provided. Interpolation here
refers to how the values of projected pixels that fall into the transformed
content bounds are calculated. Extrapolation refers to how to assign values
that fall outside the projected content bounds. The default is linear
interpolation and to fill new regions with zero.

Set `antialias = true` to apply a box prefilter when an affine projection
downsamples the image. This setting stays with the image through transforms.

!!! info
    The `Interpolations` package provides numerous methods for use with
    the `interpolate` and `extrapolate` keyword arguments.  For instance,
    `BSpline(Linear())` and `BSpline(Constant())` provide linear and nearest
    neighbor interpolation, respectively. In addition `Flat()`, `Reflect()` and
    `Periodic()` boundary conditions are available for extrapolation.

## Examples

```julia
using DataAugmentation, Images

imagedata = rand(RGB, 100, 100)
item = Image(imagedata)
showitems(item)
```

If `T` is not a color, the image will be interpreted as grayscale:

```julia
imagedata = rand(Float32, 100, 100)
item = Image(imagedata)
showitems(item)
```

"""
struct Image{N,T} <: AbstractArrayItem{N,T}
    data::AbstractArray{T,N}
    bounds::Bounds{N}
    interpolate::Interpolations.InterpolationType
    extrapolate::ImageTransformations.FillType
    antialias::Bool
end

Image(data::AbstractArray{T,N}, bounds::Bounds{N}, interpolate, extrapolate) where {T,N} =
    Image(data, bounds, interpolate, extrapolate, false)

Image{N,T}(data, bounds, interpolate, extrapolate) where {N,T} =
    Image{N,T}(data, bounds, interpolate, extrapolate, false)

function Image(
    data::AbstractArray{T,N},
    bounds::Bounds{N};
    interpolate::Interpolations.InterpolationType=BSpline(Linear()),
    extrapolate::ImageTransformations.FillType=zero(T),
    antialias::Bool=false,
) where {T,N}
    return Image(data, bounds, interpolate, extrapolate, antialias)
end

Image(data; kwargs...) = Image(data, Bounds(axes(data)); kwargs...)

function Image(data::AbstractArray{T,N}, sz::NTuple{N,Int}; kwargs...) where {T,N}
    return Image(data, Bounds(sz); kwargs...)
end

Base.show(io::IO, item::Image{N,T}) where {N,T} =
    print(io, "Image{$N, $T}() with bounds $(item.bounds)")


function showitem!(img, image::Image{2, <:Colorant})
    showimage!(img, itemdata(image))
end

function showitem!(img, image::Image{2, <:AbstractFloat})
    return showimage!(img, colorview(Gray, itemdata(image)))
end


# To support projective transformations, we need to implement [`getbounds`](@ref)
# and [`project`](@ref).

getbounds(image::Image) = image.bounds

# For the projection, we use `warp` from
# [ImageTransformations.jl](https://github.com/JuliaImages/ImageTransformations.jl).
# We have to pass the inverse of the projection `P` as it uses backward
# mode warping.

function project(P, image::Image, bounds::Bounds)
    data = itemdata(image)
    inverse = inv(P)
    source = image.antialias ? _prefilter_image(data, inverse) : data
    itp = box_extrapolation(data; method = image.interpolate, fillvalue = image.extrapolate)
    result = if source === data
        warp(itp, inverse, bounds.rs)
    else
        filtered = box_extrapolation(source; method = image.interpolate, fillvalue = image.extrapolate)
        dst = similar(Array{eltype(itp)}, bounds.rs)
        warp!(dst, filtered, inverse)
    end
    return Image(result, bounds; antialias = image.antialias)
end

function project!(bufimage::Image, P, image::Image, bounds::Bounds)
    size(itemdata(bufimage)) == length.(bounds.rs) || return project(P, image, bounds)
    inverse = inv(P)
    data = itemdata(image)
    source = image.antialias ? _prefilter_image(data, inverse) : data
    a = OffsetArray(parent(itemdata(bufimage)), bounds.rs)
    res = warp!(
        a,
        box_extrapolation(source; method = image.interpolate, fillvalue = image.extrapolate),
        inverse,
    )
    return Image(res, bounds; antialias = image.antialias)
end

_prefilter_image(data, inverse) = data

function _prefilter_image(data::AbstractArray{T,N}, inverse::Union{AffineMap, LinearMap}) where {T,N}
    isempty(data) && return data
    filtered = data
    for dim in 1:N
        width = min(sqrt(sum(abs2, view(inverse.linear, dim, :))), size(data, dim))
        if width > 1 + eps(Float32)
            filtered = _box_prefilter(filtered, dim, width)
        end
    end
    return filtered
end

function _box_prefilter(data::AbstractArray{T,N}, dim, width) where {T,N}
    dst = similar(data, typeof(float(first(data))))
    axis = axes(data, dim)
    start = first(axis)
    endpoint = length(axis) - 1
    radius = floor(Int, (width - 1) / 2)
    edge_weight = (width - 2radius - 1) / 2
    step = CartesianIndex(ntuple(i -> Int(i == dim), N))
    cross = CartesianIndices(Base.setindex(axes(data), start:start, dim))
    for origin in cross
        # A rolling sum keeps the cost independent of the filter width.
        total = ((radius + 1) * 1.0) * float(data[origin])
        for j in 1:radius
            total += float(data[origin + j * step])
        end
        for i in 0:endpoint
            left = float(data[origin + max(0, i - radius - 1) * step])
            right = float(data[origin + min(endpoint, i + radius + 1) * step])
            dst[origin + i * step] = (total + edge_weight * (left + right)) / width
            total += right - float(data[origin + max(0, i - radius) * step])
        end
    end
    return dst
end
