# ## `items/image.jl`
#
# ### Item
#
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
showitem(item)
```

If `T` is not a color, the image will be interpreted as grayscale:

{cell=image}
```julia
imagedata = rand(Float32, 100, 100)
item = Image(imagedata)
showitem(item)
```

"""
struct Image{N,T,B} <: AbstractArrayItem{N,T}
    data::AbstractArray{T,N}
    bounds::AbstractArray{<:SVector{N,B},N}
end

Image(data) = Image(data, size(data))

function Image(data::AbstractArray{T,N}, sz::NTuple{N,Int}) where {T,N}
    bounds = makebounds(sz)
    return Image(data, bounds)
end

Base.show(io::IO, item::Image{N,T}) where {N,T} =
    print(io, "Image{$N, $T}() with size $(size(itemdata(item)))")


function showitem(image::Image{2, <:Colorant})
    return image.data
end


function showitem(image::Image{2, <:AbstractFloat})
    return colorview(Gray, image.data)
end

# ### Projective transformations
#
# To support projective transformations, we need to implement [`getbounds`](#)
# and [`project`](#).

getbounds(image::Image) = image.bounds

# For the projection, we use `warp` from
# [ImageTransformations.jl](https://github.com/JuliaImages/ImageTransformations.jl).
# We have to pass the inverse of the projection `P` as it uses backward
# mode warping.

function project(P, image::Image{N, T}, indices) where {N, T}
    ## Transform the bounds along with the image
    bounds_ = P.(getbounds(image))
    data_ = warp(itemdata(image), inv(P), indices, zero(T))
    return Image(data_, makebounds(indices))
end

# The inplace version `project!` is quite similar. Note `indices` are not needed
# as they are implicitly given by the buffer.

function project!(bufimage::Image, P, image::Image{N, T}, indices) where {N, T}
    warp!(
        itemdata(bufimage),
        box_extrapolation(itemdata(image), zero(T)),
        inv(P),
    )
    return Image(itemdata(bufimage), P.(getbounds(image)))
end
