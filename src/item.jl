abstract type AbstractItem end
"""
    `Item`

Abstract data container

# TODO: document `Item` interface
"""
abstract type Item <: AbstractItem end

"""
    `ItemWrapper`

Abstract type for wrapper items

# TODO: document `ItemWrapper` interface
"""
abstract type ItemWrapper <: AbstractItem end
getwrapped(itemw::ItemWrapper) = error("Not implemented!")


# Item implementations

"""
    Image

Item for an Image
"""
struct Image{C<:Colorant} <: Item
    data::NamedDimsArray{(:y, :x), C, 2}
end
Image(a) = Image(NamedDimsArray(a, (:y, :x)))


"""
    Label
"""
struct Label <: Item
    data::Integer
end

"""
    Keypoints
"""
struct Keypoints <: Item
    data::AbstractArray
    bounds::Tuple
end

"""
    Tensor
"""
struct Tensor <: Item
    data
end

# interfaces

# TODO: add `itemdata` interface

"""
    itemdata(item::Item)

TODO: document
"""
itemdata(item::Item) = item.data
itemdata(item::ItemWrapper) = itemdata(getwrapped(item))
