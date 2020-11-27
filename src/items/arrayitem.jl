
"""
    abstract type AbstractArrayItem{N, T}

Abstract type for all [`Item`]s that wrap an `N`-dimensional
array with element type `T`.
"""
abstract type AbstractArrayItem{N, T} <: Item end


"""
    ArrayItem(a)

An item that contains an array.
"""
struct ArrayItem{N, T} <: AbstractArrayItem{N, T}
    data::AbstractArray{T, N}
end

Base.show(io::IO, item::ArrayItem{N, T}) where {N, T} =
    print(io, "ArrayItem{$N, $T}() of size $(size(itemdata(item)))")


"""
    MapElem(f)

Applies `f` to every element in an [`AbstractArrayItem`].
"""
struct MapElem <: Transform
    f
end

function apply(tfm::MapElem, item::AbstractArrayItem; randstate = nothing)
    return setdata(item, map(tfm.f, itemdata(item)))
end

function apply!(
        bufitem::I,
        tfm::MapElem,
        item::I;
        randstate = nothing) where I <: AbstractArrayItem
    map!(tfm.f, itemdata(bufitem), itemdata(item))
    return bufitem
end
