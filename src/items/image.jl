# We first define the [`Image`](#) item. Since we need to keep
# track of the spatial bounds for projective transformations
# we add them as a field. By default, they will simply
# correspond to the image axes.

"""
    Image(image[, bounds])

Item representing an N-dimensional image with element type T.

## Examples

{cell=image}
```julia
using DataAugmentation, Images

imagedata = rand(RGB, 100, 100)
item = Image(imagedata)
showitems(item)
```

If `T` is not a color, the image will be interpreted as grayscale:

{cell=image}
```julia
imagedata = rand(Float32, 100, 100)
item = Image(imagedata)
showitems(item)
```

"""
struct Image{N,T} <: AbstractArrayItem{N,T}
    data::AbstractArray{T,N}
    bounds::Bounds{N}
end

Image(data) = Image(data, Bounds(axes(data)))

function Image(data::AbstractArray{T,N}, sz::NTuple{N,Int}) where {T,N}
    return Image(data, Bounds(sz))
end


Base.show(io::IO, item::Image{N,T}) where {N,T} =
    print(io, "Image{$N, $T}() with bounds $(item.bounds)")


function showitem!(img, image::Image{2, <:Colorant})
    showimage!(img, itemdata(image))
end

function showitem!(img, image::Image{2, <:AbstractFloat})
    return showimage!(img, colorview(Gray, itemdata(image)))
end


# To support projective transformations, we need to implement [`getbounds`](#)
# and [`project`](#).

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
        bounds.rs,
        zero(T))
    return Image(data_, bounds)
end

# The inplace version `project!` is quite similar. Note `indices` are not needed
# as they are implicitly given by the buffer.

function project!(bufimage::Image, P, image::Image{N, T}, bounds::Bounds{N}) where {N, T}
    a = OffsetArray(parent(itemdata(bufimage)), bounds.rs)
    res = warp!(
        a,
        box_extrapolation(itemdata(image); fillvalue=zero(T)),
        inv(P),
    )
    return Image(res, bounds)
end
