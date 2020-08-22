using StaticArrays
using Setfield

abstract type AbstractItem end
abstract type Item <: AbstractItem end
abstract type ItemWrapper <: AbstractItem end

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

# Keypoint data

"""
    Keypoints{N, T>:SVector{N}}

`n`-dimensionalKeypoints represented as SVector{N}.
Spatial bounds are given by the polygon `bounds`.

Example:

Keypoints{2, SVector{2, Float32}}(
    [SVector(1.f, 1.f), SVector()]
)

"""

abstract type AbstractArrayItem{T} <: Item end

struct ArrayItem{T} <: AbstractArrayItem{T}
    data::AbstractArray{T}
end

struct Keypoints{N, T} <: AbstractArrayItem{T}
    data::AbstractArray{>:SVector{N, T}}
    bounds::AbstractVector{<:SVector{N}}
end

struct Polygon{N, T} <: ItemWrapper
    item::Keypoints{N, T}
end
Polygon(data, bounds) = Polygon(Keypoints(data, bounds))

"""

"""
struct BoundingBox{N, T} <: ItemWrapper
    item::Keypoints{N, T}
end
BoundingBox(data, bounds) = BoundingBox(Keypoints(data, bounds))

struct Image{C<:Colorant} <: AbstractArrayItem{C}
    data::AbstractMatrix{C}
    bounds::AbstractVector{<:SVector{2}}
end

Image(imdata::AbstractMatrix{C}, bounds = makebounds(imdata)) where C<:Colorant = Image(imdata, bounds)


struct Category{N} <: Item
    data::Int
end


function index_ranges(bounds::AbstractVector{<:SVector{N}}) where N
    ext = [extrema([b[i] for b in bounds]) for i in 1:N]
    return Tuple(floor(Int, mi+1):ceil(Int, ma) for (mi, ma) in ext)
end


boundssize(item::AbstractItem) = boundssize(getbounds(item))
boundssize(bounds) = length.(index_ranges(bounds))


function makebounds(a::AbstractArray)
    idxs = CartesianIndices(a)
    return makebounds(Tuple(idxs[begin]), Tuple(idxs[end]))
end
makebounds(h::Int, w::Int) = makebounds((1, 1), (h, w))
makebounds(r1::R, r2::R) where R<:AbstractRange = makebounds((r1[begin], r2[begin]), (r1[end], r2[end]))
function makebounds(idx1, idx2)
    y1, x1 = idx1 .- 1
    y2, x2 = idx2
    return [
        SVector(y1, x1),
        SVector(y1, x2),
        SVector(y2, x2),
        SVector(y2, x1),
    ]
end
