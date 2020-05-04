# types
abstract type AbstractItem end
abstract type Item <: AbstractItem end
abstract type ItemWrapper <: AbstractItem end


# implementations
struct Image{C<:Colorant} <: Item
    data::AbstractArray{C, 2}
end


struct Label <: Item
    data::Integer
end


struct Keypoints <: Item
    data::AbstractArray
    bounds::Tuple
end


struct Tensor <: Item
    data::AbstractArray
end
