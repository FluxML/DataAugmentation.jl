
struct ToEltype{T} <: AbstractTransform end
ToEltype(T::Type) = ToEltype{T}()

(t::ToEltype{T})(item::Image{<:T}) where T = Image(parent(item.data))
(t::ToEltype{T})(item::Image{U}) where {T, U} = Image(parent(map(x -> convert(T, x), item.data)))


struct Normalize <: AbstractTransform
    means
    stds
    inplace::Bool
end

Normalize(means, stds; inplace = true) = Normalize(means, stds, inplace)

(t::Normalize)(item::Tensor) = Tensor(normalize!(t.inplace ? item.data : copy(item.data), t.means, t.stds))


struct ToTensor <: AbstractTransform end
(::ToTensor)(item::Image) = Tensor(imagetotensor(itemdata(item)))
(::ToTensor)(item::Keypoints) = Tensor(parent(itemdata(item)))


struct OneHot <: AbstractTransform
    nclasses::Int
end

(t::OneHot)(item::Label) = Tensor(onehot(itemdata(item), 1:t.nclasses))


function onehot(T, x::Int, labels::AbstractVector)
    v = fill(zero(T), length(labels))
    v[x] = one(T)
    return v
end
onehot(x, labels) = onehot(Float32, x, labels)


# helper functions

function normalize!(a, means, stds)
    for i = 1:3
        a[:,:,i] .-= means[i]
        a[:,:,i] ./= stds[i]
    end
    a
end
normalize(a, means, stds) = normalize!(copy(a), means, stds)

function denormalize!(a, means, stds)
    for i = 1:3
        a[:,:,i] .*= stds[i]
        a[:,:,i] .+= means[i]
    end
    a
end
denormalize(a, means, stds) = denormalize!(copy(a), means, stds)


imagetotensor(image::AbstractArray{<:AbstractRGB, 2}) = float.(permuteddimsview(channelview(image), (2, 3, 1))) |> parent
tensortoimage(tensor::AbstractArray{T, 3}) where T = colorview(RGB, permuteddimsview(tensor, (3, 1, 2)))
tensortoimage(tensor::AbstractArray{T, 2}) where T = colorview(Gray, permuteddimsview(tensor, (3, 1, 2)))