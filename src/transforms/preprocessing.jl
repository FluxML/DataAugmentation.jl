
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

# TODO: inplace-version is much slower
function apply!(buf, ::SplitChannels, image::Image; randstate = nothing)
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
    # TODO: use vectorized ops for generality
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


# TODO: is `parent` necessary?
imagetotensor(image::AbstractArray{<:AbstractRGB, 2}, T = Float32) = T.(permuteddimsview(channelview(image), (2, 3, 1))) |> parent
# improve performance, this takes double the time of `imagetotensor`
imagetotensor!(buf, image::AbstractArray{<:AbstractRGB, 2}) = permutedims!(
    buf,
    channelview(image),
    (2, 3, 1))
tensortoimage(a::AbstractArray{T, 3}) where T = colorview(RGB, permuteddimsview(a, (3, 1, 2)))
tensortoimage(a::AbstractArray{T, 2}) where T = colorview(Gray, a)

#=
T = Normed{UInt8,8}
img = zeros(RGB{T}, 224, 224)

using BenchmarkTools
@btime begin
    imagetotensor(img)
end;  # 676.103 μs (11 allocations: 588.47 KiB)

@btime begin
    imagetotensor!(b2, img)
    nothing
end;  #


b1 = collect(channelview(img))
b2 = imagetotensor(img)
b3 = permutedims(b1, (2, 3, 1))

@btime begin
    channels!(b1, img)
    permutedims!(b2, b1, (2, 3, 1))
end  # 839.176 μs (9 allocations: 336 bytes)

@btime begin
    channels!(b1, img)
    permutedims!(b3, b1, (2, 3, 1))
end  # 707.046 μs (2 allocations: 112 bytes)

=#
