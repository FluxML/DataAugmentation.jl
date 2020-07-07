
struct ToEltype{T} <: Transform end
ToEltype(T::Type) = ToEltype{T}()

apply(::ToEltype{T}, item::AbstractArrayItem{<:T}; randstate = nothing) where T = item
function apply(::ToEltype{T}, item::AbstractArrayItem{U}; randstate = nothing) where {T, U}
    newdata = map(x -> convert(T, x), itemdata(item))
    item = setdata(item, newdata)
    return item
end


struct Normalize <: Transform
    means
    stds
    inplace::Bool
end

Normalize(means, stds; inplace = true) = Normalize(means, stds, inplace)

apply(t::Normalize, item::ArrayItem; randstate = nothing) = ArrayItem(
    normalize!(t.inplace ? itemdata(item) : copy(itemdata(item)), t.means, t.stds))


struct SplitChannels <: Transform end
apply(::SplitChannels, image::Image; randstate = nothing) =
    ArrayItem(imagetotensor(itemdata(image)))


struct OneHotEncode <: Transform
    nclasses::Int
end

apply(t::OneHotEncode, category::Category) =
    ArrayItem(onehot(itemdata(category), 1:t.nclasses))


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


# TODO: check if this always allocates and if it can be done without
# TODO: is `parent` necessary?
imagetotensor(image::AbstractArray{<:AbstractRGB, 2}) = float.(permuteddimsview(channelview(image), (2, 3, 1))) |> parent
tensortoimage(tensor::AbstractArray{T, 3}) where T = colorview(RGB, permuteddimsview(tensor, (3, 1, 2)))
tensortoimage(tensor::AbstractArray{T, 2}) where T = colorview(Gray, permuteddimsview(tensor, (3, 1, 2)))


# TODO: implement in-place version
