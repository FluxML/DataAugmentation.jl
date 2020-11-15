using StaticArrays
using Setfield

abstract type AbstractItem end
abstract type Item <: AbstractItem end
abstract type ItemWrapper{Item} <: AbstractItem end

itemfield(wrapped::ItemWrapper) = :item
getwrapped(wrapped::ItemWrapper) = getfield(wrapped, itemfield(wrapped))
function setwrapped(wrapped::ItemWrapper, item)
    wrapped = Setfield.@set wrapped.item = item
    return wrapped
end


function setdata(item::Item, data)
    item = Setfield.@set item.data = data
    return item
end

struct Many{I} <: AbstractItem
    items::AbstractArray{I}
end

itemdata(item::Item) = item.data
itemdata(wrapper::ItemWrapper) = itemdata(getwrapped(wrapper))
itemdata(items) = itemdata.(items)
itemdata(many::Many) = itemdata.(many.items)


"""
    abstract type AbstractArrayItem{N, T}

Abstract type for all [`Item`]s that wrap a `N`-dimensional
array with element type `T`.
"""
abstract type AbstractArrayItem{N, T} <: Item end


struct ArrayItem{N, T} <: AbstractArrayItem{N, T}
    data::AbstractArray{T, N}
end


"""
    Image(image[, bounds])

Item representing an N-dimensional image with element type T.

Supported `Transform`s:

- all [`AbstractAffine`](#)s

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
struct Image{N, T, B} <: AbstractArrayItem{N, T}
    data::AbstractArray{T, N}
    bounds::AbstractArray{<:SVector{N, B}, N}
end

Image(data) = Image(data, size(data))

function Image(data::AbstractArray{T, N}, sz::NTuple{N, Int}) where {T, N}
    bounds = makebounds(sz)
    return Image(data, bounds)
end

Base.show(io::IO, item::Image{N, T}) where {N, T} =
    print(io, "Image{$N, $T}() with size $(size(itemdata(item)))")


"""
    Keypoints(points, sz)
    Keypoints{N, T, M}(points, bounds)

`N`-dimensional keypoints represented as `SVector{N, T}`.

Spatial bounds are given by the polygon `bounds::Vector{SVector{N, T}}`
or `sz::NTuple{N, Int}`.

## Examples
{cell=Keypoints}
```julia
using DataAugmentation, StaticArrays
points = [SVector(y, x) for (y, x) in zip(4:5:80, 10:6:90)]
item = Keypoints(points, (100, 100))
```

{cell=Keypoints}
```julia
showitem(item)
```

"""
struct Keypoints{N, T, S<:Union{SVector{N, T}, Nothing}, M} <: AbstractArrayItem{M, S}
    data::AbstractArray{S, M}
    bounds::AbstractArray{<:SVector{N, T}, N}
end

Base.show(io::IO, item::Keypoints{N, T, M}) where {N, T, M} =
    print(io, "Keypoints{$N, $T, $M}() with $(length(item.data)) elements")

function Keypoints(data::AbstractArray{S, M}, sz::NTuple{N, Int}) where {T, N, S<:Union{SVector{N, T}, Nothing}, M}
    return Keypoints{N, T, S, M}(data, makebounds(sz, T))
end

"""
    Polygon(points, sz)
    Polygon{N, T, M}(points, bounds)

Item wrapper around [`Keypoints`](#).

## Examples

{cell=Polygon}
```julia
using DataAugmentation, StaticArrays
points = [SVector(10., 10.), SVector(80., 20.), SVector(90., 70.), SVector(20., 90.)]
item = Polygon(points, (100, 100))
```

{cell=Polygon}
```julia
showitem(item)
```
"""
struct Polygon{N, T, S, M} <: ItemWrapper{Keypoints{N, T, S, M}}
    item::Keypoints{N, T, S, M}
end

Polygon(data, bounds) = Polygon(Keypoints(data, bounds))

Base.show(io::IO, item::Polygon{N, T, M}) where {N, T, M} =
    print(io, "Polygon{$N, $T}() with $(length(item.item.data)) elements")

"""
    Polygon(points, sz)
    Polygon{N, T, M}(points, bounds)

Item wrapper around [`Keypoints`](#).

## Examples

{cell=BoundingBox}
```julia
using DataAugmentation, StaticArrays
points = [SVector(10., 10.), SVector(80., 60.)]
item = BoundingBox(points, (100, 100))
```

{cell=BoundingBox}
```julia
showitem(item)
```
"""
struct BoundingBox{N, T, S} <: ItemWrapper{Keypoints{N, T, S, 1}}
    item::Keypoints{N, T, S, 1}
end

function BoundingBox(data::AbstractVector{<:SVector{N}}, bounds) where N
    length(data) == N || error("Give $N corner points for an $N-dimensional bounding box")
    BoundingBox(Keypoints(data, bounds))
end


Base.show(io::IO, item::BoundingBox{N, T}) where {N, T} =
    print(io, "BoundingBox{$N, $T}()")

# TODO: add cropping to `BoundingBox` and `Polygon` so the area they enclose is in bounds

struct Category{N} <: Item
    data::Int
end


"""
    MaskMulti(a, [classes])

An `N`-dimensional multilabel mask with labels `classes`.
"""
struct MaskMulti{N, T, B} <: AbstractArrayItem{N, T}
    data::AbstractArray{T, N}
    classes::AbstractVector{T}
    bounds::AbstractArray{<:SVector{N, B}, N}
end


function MaskMulti(a::AbstractArray, classes = unique(a), bounds = makebounds(size(a)))
    return MaskMulti(a, classes = bounds)
end

"""
    MaskBinary(a)

An `N`-dimensional binary mask with labels `classes`.
"""
struct MaskBinary{N, B} <: AbstractArrayItem{N, Bool}
    data::AbstractArray{Bool, N}
    bounds::AbstractArray{<:SVector{N, B}, N}
end

function MaskBinary(a::AbstractArray{Bool, N}, bounds = makebounds(size(a))) where N
    return MaskBinary(a, bounds)
end



# ## Bounds helpers


function boundsextrema(bounds::AbstractArray{<:SVector{N}}) where N
    mins = Tuple(floor(Int, minimum(getindex.(bounds, i))) for i = 1:N)
    maxs = Tuple(ceil(Int, maximum(getindex.(bounds, i))) for i = 1:N)
    return mins, maxs

end

function boundsranges(bounds)
    mins, maxs = boundsextrema(bounds)
    return UnitRange.(mins .+ 1, maxs)
end

"""
    boundssize(bounds)

`(100, 100) |> makebounds |> boundssize == (100, 100)`
"""
boundssize(bounds) = length.(boundsranges(bounds))


"""
    makebounds(sz[, T])
    makebounds(ranges[, T])

Helper for creating spatial bounds.

## Examples

{cell=makebounds}
```julia
using DataAugmentation: makebounds, showbounds
makebounds((100, 100), Float32)
```
{cell=makebounds}
```julia
makebounds((100, 100)) == makebounds((1:100, 1:100))
```

{cell=makebounds}
```julia
bounds = makebounds((100, 100))
showbounds(bounds)
```
"""
function makebounds(sz::NTuple{N, Int}, T = Float32) where N
    return makebounds(Tuple(1:a for a in sz), T)
end

function makebounds(ranges::NTuple{N, R}, T = Float32) where {N, R<:AbstractUnitRange}
    return collect(SVector{N, T}, Iterators.product(((r[begin]-1, r[end]) for r in ranges)...))
end
