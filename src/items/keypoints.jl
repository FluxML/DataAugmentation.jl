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
    bounds::AbstractArray{<:SVector{N, Float32}, N}
end


function Keypoints(data::AbstractArray{S, M}, sz::NTuple{N, Int}) where {T, N, S<:Union{SVector{N, T}, Nothing}, M}
    return Keypoints{N, T, S, M}(data, makebounds(sz, Float32))
end


Base.show(io::IO, item::Keypoints{N, T, M}) where {N, T, M} =
    print(io, "Keypoints{$N, $T, $M}() with $(length(item.data)) elements")


getbounds(keypoints::Keypoints) = keypoints.bounds


function project(P, keypoints::Keypoints{N, T}, indices) where {N, T}
    return Keypoints(
        map(fmap(P), keypoints.data),
        makebounds(indices),
    )
end


function showitem!(img, keypoints::Keypoints{N}) where N
    for point in filter(!isnothing, keypoints.data)
        showkeypoint!(img, point, RGBA(1, 0, 0, 1))
    end
    return img
end

# ## Wrappers
#
# We also define some wrappers for `Keypoints` which have the same representation,
# but a different meaning.

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


function showitem!(img, polygon::Polygon)
    showpolygon!(img, itemdata(polygon), RGBA(1, 0, 0, 1))
    return img
end


"""
    BoundingBox(points, sz)
    BoundingBox{N, T, M}(points, bounds)

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


function showitem!(img, bbox::BoundingBox{2})
    ul, br = itemdata(bbox)
    points = [
        ul,
        SVector(br[1], ul[2]),
        br,
        SVector(ul[1], br[2]),
    ]
    showpolygon!(img, points, RGBA(1, 0, 0, 1))
    return img
end


# ### Helpers

fmap(f) = x -> fmap(f, x)
fmap(f, x) = f(x)
fmap(f, x::Nothing) = x
