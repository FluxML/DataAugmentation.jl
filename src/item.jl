using StaticArrays
using Setfield

abstract type AbstractItem end
abstract type Item <: AbstractItem end
abstract type ItemWrapper <: AbstractItem end

itemfield(wrapped::ItemWrapper) = :item
getwrapped(wrapped::ItemWrapper) = getfield(wrapped, itemfield(wrapped))
function setwrapped(wrapped::ItemWrapper, item)
    Setfield.@set wrapped.item = item
    wrapped
end

itemdata(item::Item) = item.data
itemdata(wrapper::ItemWrapper) = itemdata(getwrapped(wrapper))
itemdata(items) = itemdata.(items)

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
struct Keypoints{N, T} <: Item
    data::AbstractArray{>:SVector{N, T}}
    bounds::AbstractVector{<:SVector{N}}
end

struct Polygon{N, T} <: ItemWrapper
    item::Keypoints{N, T}
end

struct BoundingBox{N, T} <: ItemWrapper
    item::Keypoints{N, T}
end

struct Image{C<:Colorant} <: Item
    data::AbstractMatrix{C}
    bounds::AbstractVector{<:SVector{2}}
end

Image(imdata::AbstractMatrix{C}, bounds = makebounds(imdata)) where C<:Colorant = Image(imdata, bounds)


struct Category{N} <: Item
    data::Int
end


# TODO: bounds should begin at 0 so that they're transformed properly
#=
makebounds(a::AbstractMatrix) = makebounds(indices_spatial(a)...)
makebounds(h::Int, w::Int) = makebounds(1:h, 1:w)
function makebounds(r1::AbstractRange, r2::AbstractRange)
    y1, y2 = r1[begin], r1[end]
    x1, x2 = r2[begin], r2[end]
    return [
        SVector(y1, x1) .- 1,
        SVector(y1, x2) .- 1,
        SVector(y2, x2) .- 1,
        SVector(y2, x1) .- 1,
    ]
end
=#


function index_ranges(bounds::AbstractVector{<:SVector{N}}) where N
    ext = [extrema([b[i] for b in bounds]) for i in 1:N]
    return Tuple(floor(Int, mi+1):ceil(Int, ma) for (mi, ma) in ext)
end


boundssize(bounds) = length.(index_ranges(bounds))

# Experimental

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

function makesizes(bounds)
    p1, p2, p3, p4 = bounds


end
