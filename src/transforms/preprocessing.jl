
struct ToEltype{T} <: Transform end
ToEltype(T::Type) = ToEltype{T}()

apply(t::ToEltype{T}, item::Image{<:T}) where T = Image(parent(item.data))
apply(t::ToEltype{T}, item::Image{U}) where {T, U} = Image(parent(map(x -> convert(T, x), item.data)))


struct Normalize <: Transform
    means
    stds
    inplace::Bool
end

Normalize(means, stds; inplace = true) = Normalize(means, stds, inplace)

apply(t::Normalize, item::Tensor) = Tensor(normalize!(t.inplace ? item.data : copy(item.data), t.means, t.stds))


struct ToTensor <: Transform end
apply(::ToTensor, image::Image) = Tensor(imagetotensor(image.data))


struct OneHot <: Transform
    nclasses::Int
end

apply(t::OneHot, label::Label) = Tensor(onehot(label.data, 1:t.nclasses))


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
