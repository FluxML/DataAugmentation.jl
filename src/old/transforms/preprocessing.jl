
# ToEltype

struct ToEltype{T} <: Transform end
ToEltype(T::Type) = ToEltype{T}()

apply(::ToEltype{T}, item::AbstractArrayItem{N, <:T}; randstate = nothing) where {N, T} = item
function apply(::ToEltype{T1}, item::AbstractArrayItem{N, T2}; randstate = nothing) where {N, T1, T2}
    newdata = map(x -> convert(T1, x), itemdata(item))
    item = setdata(item, newdata)
    return item
end

function apply!(buf, ::ToEltype, item::AbstractArrayItem; randstate = nothing)
    # copy! does type conversion under the hood
    copy!(itemdata(buf), itemdata(item))
    return buf
end


"""
    Normalize(means, stds)

Normalizes a 3D `AbstractArrayItem` of size (h, w, c)
using `means` and `stds`.
"""
struct Normalize{N} <: Transform
    means::SArray{Tuple{1, 1, N}}
    stds::SArray{Tuple{1, 1, N}}
end

function Normalize(means, stds)
    length(means) == length(stds) || error("`means` and `stds` must have same length")
    N = length(means)
    return Normalize{N}(SArray{Tuple{1, 1, N}}(means), SArray{Tuple{1, 1, N}}(stds))
end

function apply(tfm::Normalize, item::ArrayItem; randstate = nothing)
    return ArrayItem(normalize(itemdata(item), tfm.means, tfm.stds))
end

function apply!(buf, tfm::Normalize, item::ArrayItem; randstate = nothing)
    copy!(itemdata(buf), itemdata(item))
    normalize!(itemdata(buf), tfm.means, tfm.stds)
    return buf
end


"""
    ImageToTensor()

Converts a 2D [`Image`](#) to a 3D array item of size `(h, w, ch)`.
"""
struct ImageToTensor{T} <: Transform end

ImageToTensor(T::Type{<:Number} = Float32) = ImageToTensor{T}()


function apply(::ImageToTensor{T}, image::Image; randstate = nothing) where T
    return ArrayItem(imagetotensor(itemdata(image), T))
end

# TODO: inplace-version is much slower
function apply!(buf, ::ImageToTensor, image::Image; randstate = nothing)
    imagetotensor!(buf.data, image.data)
    return buf
end


struct OneHotEncode <: Transform
    nclasses::Int
end

apply(t::OneHotEncode, category::Category) =
    ArrayItem(onehot(itemdata(category), 1:t.nclasses))


function onehot(T, x::Int, n::Int)
    v = fill(zero(T), n)
    v[x] = one(T)
    return v
end
onehot(x, n) = onehot(Float32, x, n)


# helper functions

function normalize!(a, means, stds)
    a .-= means
    a ./= stds
    return a
end

normalize(a, means, stds) = normalize!(copy(a), means, stds)

function denormalize!(a, means, stds)
    a .*= stds
    a .+= means
    return a
end

denormalize(a, means, stds) = denormalize!(copy(a), means, stds)


imagetotensor(image::AbstractArray{<:AbstractRGB, 2}, T = Float32) = T.(permuteddimsview(channelview(image), (2, 3, 1)))

imagetotensor!(buf, image::AbstractArray{<:AbstractRGB, 2}) = permutedims!(
    buf,
    channelview(image),
    (2, 3, 1))
tensortoimage(a::AbstractArray{T, 3}) where T = colorview(RGB, permuteddimsview(a, (3, 1, 2)))
tensortoimage(a::AbstractArray{T, 2}) where T = colorview(Gray, a)
