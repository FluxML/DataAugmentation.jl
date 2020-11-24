
"""
    abstract type AbstractArrayItem{N, T}

Abstract type for all [`Item`]s that wrap an `N`-dimensional
array with element type `T`.
"""
abstract type AbstractArrayItem{N, T} <: Item end


"""
    MapElem(f)

Applies `f` to every element in an [`AbstractArrayItem`].
"""
struct MapElem <: Transform
    fn
end

function apply(tfm::MapElem, item::AbstractArrayItem; randstate = nothing)
    return setdata(item, map(tfm.fn, itemdata(item)))
end

function apply!(
        bufitem::I,
        tfm::MapElem,
        item::I;
        randstate = nothing) where I <: AbstractArrayItem
    map!(tfm.f, itemdata(bufitem), itemdata(item))
    return bufitem
end
