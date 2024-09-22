# We first define the [`Image`](@ref) item. Since we need to keep
# track of the spatial bounds for projective transformations
# we add them as a field. By default, they will simply
# correspond to the image axes.

"""
    Image(image[, bounds]; interpolate=BSpline(Linear()), extrapolate=zero(T))

Item representing an N-dimensional image with element type T. Optionally, the
interpolation and extrapolation method can be provided. Interpolation here
refers to how the values of projected pixels that fall into the transformed
content bounds are calculated. Extrapolation refers to how to assign values
that fall outside the projected content bounds. The default is linear
interpolation and to fill new regions with zero.

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
end

function Image(
    data::AbstractArray{T,N},
    bounds::Bounds{N};
    interpolate::Interpolations.InterpolationType=BSpline(Linear()),
    extrapolate::ImageTransformations.FillType=zero(T),
) where {T,N}
    return Image(data, bounds, interpolate, extrapolate)
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

function project(P, image::Image{N, T}, bounds::Bounds) where {N, T}
    # TODO: make interpolation scheme and boundary conditions configurable
    data_ = warp(
        itemdata(image),
        inv(P),
        bounds.rs;
        method=image.interpolate,
        fillvalue=image.extrapolate)
    return Image(data_, bounds)
end

# The inplace version `project!` is quite similar. Note `indices` are not needed
# as they are implicitly given by the buffer.

function project!(bufimage::Image, P, image::Image{N, T}, bounds::Bounds{N}) where {N, T}
    a = OffsetArray(parent(itemdata(bufimage)), bounds.rs)
    res = warp!(
        a,
        box_extrapolation(itemdata(image); method=image.interpolate, fillvalue=image.extrapolate),
        inv(P),
    )
    return Image(res, bounds)
end
