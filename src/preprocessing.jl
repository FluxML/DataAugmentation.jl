# ### [`ToEltype`](@ref)

"""
    ToEltype(T)

Converts any `AbstractArrayItem` to an `AbstractArrayItem{N, T}`.

Supports `apply!`.

## Examples

{cell=ToEltype}
```julia
using DataAugmentation

tfm = ToEltype(Float32)
item = ArrayItem(rand(Int, 10))
apply(tfm, item)
```
"""
struct ToEltype{T} <: Transform end
ToEltype(T::Type) = ToEltype{T}()

apply(::ToEltype{T}, item::AbstractArrayItem{N, <:T}; randstate = nothing) where {N, T} = item
function apply(::ToEltype{T1}, item::AbstractArrayItem{N, T2}; randstate = nothing) where {N, T1, T2}
    newdata = map(x -> convert(T1, x), itemdata(item))
    item = setdata(item, newdata)
    return item
end

function apply!(buf, ::ToEltype, item::AbstractArrayItem; randstate = nothing)
    ## copy! does type conversion under the hood
    copy!(itemdata(buf), itemdata(item))
    return buf
end

# ### [`Normalize`](@ref)

"""
    Normalize(means, stds)

Normalizes the last dimension of an `AbstractArrayItem{N}`.

Supports `apply!`.

## Examples

Preprocessing a 3D image with 3 color channels.

{cell=Normalize}
```julia
using DataAugmentation, Images
image = Image(rand(RGB, 20, 20, 20))
tfms = ImageToTensor() |> Normalize((0.1, -0.2, -0.1), (1,1,1.))
apply(tfms, image)
```

"""
struct Normalize{N} <: Transform
    means::SVector{N}
    stds::SVector{N}
end

function Normalize(means, stds)
    length(means) == length(stds) || error("`means` and `stds` must have same length")
    N = length(means)
    return Normalize{N}(SVector{N}(means), SVector{N}(stds))
end

function apply(tfm::Normalize, item::ArrayItem{N, T}; randstate = nothing) where {N, T}
    return ArrayItem(normalize(itemdata(item), tfm.means, tfm.stds))
end

function apply!(
        buf,
        tfm::Normalize,
        item::ArrayItem;
        randstate = nothing)
    copy!(itemdata(buf), itemdata(item))
    normalize!(itemdata(buf), tfm.means, tfm.stds)
    return buf
end


function normalize!(
        a::AbstractArray{T, N},
        means::SVector{M},
        stds::SVector{M}) where {N, T, M}
    means = reshape(convert.(T, means), (1 for _ = 2:N)..., M)
    stds = reshape(convert.(T, stds), (1 for _ = 2:N)..., M)
    a .-= means
    a ./= stds
    return a
end

normalize(a, means, stds) = normalize!(copy(a), means, stds)

function denormalize!(
        a::AbstractArray{T, N},
        means::SVector{M},
        stds::SVector{M}) where {N, T, M}
    means = reshape(convert.(T, means), (1 for _ = 2:N)..., M)
    stds = reshape(convert.(T, stds), (1 for _ = 2:N)..., M)
    a .*= stds
    a .+= means
    return a
end

denormalize(a, means, stds) = denormalize!(copy(a), means, stds)


# ### [`NormalizeIntensity`]

"""
    NormalizeIntensity()

Normalizes the pixels of an array based on calculated mean and std.
"""
struct NormalizeIntensity <: Transform end

function apply(tfm::NormalizeIntensity, item::ArrayItem; randstate = nothing)
    array = itemdata(item)
    slices = ones(Bool, size(array))
    means = mean(array[slices])
    stds = std(array[slices])
    array[slices] = (array[slices] .- means) / stds
    return ArrayItem(array)
end

# ### [`LinearTransformation`](@ref)

"""
    LinearTransformation(matrix; mean = zeros(eltype(matrix), size(matrix, 1)))

Flattens an [`ArrayItem`](@ref) of length `D`, subtracts `mean`, multiplies the result by
the `D × D` `matrix` and restores the original size, i.e. `matrix * (vec(x) - mean)`.
Use it for whitening with a matrix computed offline from a dataset.

The result has the promoted element type of the data, `matrix` and `mean`. This is the
transformation of torchvision's `LinearTransformation` for a symmetric `matrix`, such as
the ZCA whitening matrix; torchvision multiplies the row vector from the right, so pass
the transpose of a non-symmetric torchvision matrix.

Supports `apply!`.

## Examples

ZCA whitening of 4×4 single-channel samples:

```julia
using DataAugmentation, LinearAlgebra, Statistics

samples = randn(Float32, 16, 1000)
μ = vec(mean(samples; dims = 2))
eig = eigen(Symmetric(cov(samples; dims = 2)))
W = eig.vectors * Diagonal(1 ./ sqrt.(eig.values .+ 1f-5)) * eig.vectors'

tfm = LinearTransformation(W; mean = μ)
apply(tfm, ArrayItem(randn(Float32, 4, 4)))
```
"""
struct LinearTransformation{M<:AbstractMatrix, V<:AbstractVector} <: Transform
    matrix::M
    mean::V
    function LinearTransformation(matrix::AbstractMatrix, mean::AbstractVector)
        n = size(matrix, 1)
        n == size(matrix, 2) ||
            throw(DimensionMismatch("matrix must be square, got size $(size(matrix))"))
        length(mean) == n ||
            throw(DimensionMismatch("mean must have length $n, got length $(length(mean))"))
        return new{typeof(matrix), typeof(mean)}(matrix, mean)
    end
end

LinearTransformation(matrix::AbstractMatrix; mean::AbstractVector = zeros(eltype(matrix), size(matrix, 1))) =
    LinearTransformation(matrix, mean)

_lineartype(tfm::LinearTransformation, data) =
    promote_type(eltype(tfm.matrix), eltype(tfm.mean), eltype(data))

function _linear!(dst, tfm::LinearTransformation, src)
    length(src) == size(tfm.matrix, 1) || throw(DimensionMismatch(
        "item has $(length(src)) elements but the transformation matrix has size $(size(tfm.matrix))"))
    mul!(vec(dst), tfm.matrix, vec(src) .- tfm.mean)
    return dst
end

function apply(tfm::LinearTransformation, item::ArrayItem; randstate = nothing)
    src = itemdata(item)
    return ArrayItem(_linear!(Array{_lineartype(tfm, src)}(undef, size(src)), tfm, src))
end

function apply!(buf::ArrayItem, tfm::LinearTransformation, item::ArrayItem; randstate = nothing)
    src = itemdata(item)
    dst = itemdata(buf)
    if size(dst) != size(src) || eltype(dst) != _lineartype(tfm, src) || Base.mightalias(dst, src)
        return apply(tfm, item)
    end
    _linear!(dst, tfm, src)
    return buf
end

# ### [`ImageToTensor`]

"""
    ImageToTensor()

Expands an `Image{N, T}` of size `(height, width, ...)` to an `ArrayItem{N+1}` with
size `(width, height, ..., ch)` where `ch` is the number of color channels of `T`.

Supports `apply!`.

## Examples

{cell=ImageToTensor}
```julia
using DataAugmentation, Images

h, w = 40, 50
image = Image(rand(RGB, h, w))
tfm = ImageToTensor()
apply(tfm, image) # ArrayItem in WHC format of size (50, 40, 3)
```

"""
struct ImageToTensor{T} <: Transform end

ImageToTensor(T::Type{<:Number} = Float32) = ImageToTensor{T}()


function apply(::ImageToTensor{T}, image::Image; randstate = nothing) where T
    return ArrayItem(imagetotensor(itemdata(image), T))
end


function apply!(buf, ::ImageToTensor, image::Image; randstate = nothing)
    imagetotensor!(buf.data, image.data)
    return buf
end

function imagetotensor(image::AbstractArray{C, N}, T = Float32) where {C<:Colorant, N}
    T.(PermutedDimsArray(_channelview(image), (3, 2, 4:N+1..., 1)))
end

function imagetotensor!(buf, image::AbstractArray{<:Colorant, N}) where N
    permutedims!(buf, _channelview(image), (3, 2, 4:N+1..., 1))
end

function tensortoimage(a::AbstractArray)
    nchannels = size(a)[end]
    if nchannels == 3
        return tensortoimage(RGB, a)
    elseif nchannels == 1
        return tensortoimage(Gray, a)
    else
        error("Found image tensor with $nchannels color channels. Pass in color type
                explicitly.")
    end
end

function tensortoimage(C::Type{<:Colorant}, a::AbstractArray{T, N}) where {T, N}
    perm = (N, 2, 1, 3:N-1...)
    return _colorview(C, PermutedDimsArray(a, perm))
end

# For single-channel colors, expand the color dimension anyway
# such that the output is always of size (channels, height, width, ...)
_channelview(img::AbstractArray{<:Colorant{T, N}}) where {T, N} = channelview(img)
function _channelview(img::AbstractArray{<:Colorant{T, 1}}) where T
    cv = channelview(img)
    return reshape(cv, 1, size(cv)...)
end

function _colorview(C::Type{<:Colorant}, img)
    if size(img, 1) == 1
        img = reshape(img, size(img)[2:end])
    end
    return colorview(C, img)
end

# ### [`PermuteDims`](#)

"""
    PermuteDims(perm)

Permute the dimensions of an `ArrayItem`.
`perm` is a vector or a tuple specifying the permutation, whose length has to match the dimensionality of the `ArrayItem`s data.

Refer to the `permutedims` documentation for examples of permutation vectors `perm`.

Supports `apply!`.

## Examples

Preprocessing an image with 3 color channels.

{cell=PermuteDims}
```julia
using DataAugmentation, Images
image = Image(rand(RGB, 20, 20))

# Turn image to tensor and permute dimensions 2 and 1
# to convert HWC (height, width, channel) array to WHC (width, height, channel)
tfms = ImageToTensor() |> PermuteDims(2, 1, 3)
apply(tfms, image)
```

"""
struct PermuteDims{N} <: Transform
    perm::NTuple{N, Int}
end
PermuteDims(perm...) = PermuteDims(perm)

function apply(tfm::PermuteDims, item::ArrayItem; randstate = nothing)
    data = PermutedDimsArray(itemdata(item), tfm.perm)
    return ArrayItem(data)
end

function apply!(buf, tfm::PermuteDims, item::ArrayItem; randstate = nothing)
    permutedims!(itemdata(buf), itemdata(item), tfm.perm)
    return buf
end

# OneHot encoding

"""
    OneHot([T = Float32])

One-hot encodes a `MaskMulti` with `n` classes and size `sz` into
an array item of size `(sz..., n)` with element type `T`. Supports [`apply!`](@ref).

```julia
item = MaskMulti(rand(1:4, 100, 100), 1:4)
apply(OneHot(), item)
```
"""
struct OneHot{T} <: Transform end
OneHot() = OneHot{Float32}()

function apply(tfm::OneHot{T}, item::MaskMulti; randstate = nothing) where T
    mask = parent(itemdata(item))
    a = zeros(T, size(mask)..., length(item.classes))
    for I in CartesianIndices(mask)
        a[I, mask[I]] = one(T)
    end

    return ArrayItem(a)
end


function apply!(buf, tfm::OneHot{T}, item::MaskMulti; randstate = nothing) where T
    mask = parent(itemdata(item))
    a = itemdata(buf)
    fill!(a, zero(T))

    for I in CartesianIndices(mask)
        a[I, mask[I]] = one(T)
    end

    return buf
end


"""
    onehot([T = Float32], x, n)

One-hot encode the class index `x` in `1:n` as a vector of length `n` with element
type `T`.
"""
function onehot(T, x::Int, n::Int)
    v = fill(zero(T), n)
    v[x] = one(T)
    return v
end
onehot(x, n) = onehot(Float32, x, n)
