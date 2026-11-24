"""
    MixUp(alpha)
    MixUp(; alpha = 1)

Blend each image in an [`ImageBatch`](@ref) with its partner in a random permutation
of the batch. The images and their label columns use the same weight, drawn
from `Beta(alpha, alpha)` with positive finite `alpha`. Images and labels are
returned as floating-point arrays.
"""
struct MixUp <: Transform
    alpha::Float64
    function MixUp(alpha::Real)
        a = Float64(alpha)
        isfinite(a) && a > 0 || throw(ArgumentError("alpha must be positive and finite"))
        return new(a)
    end
end

MixUp(; alpha = 1) = MixUp(alpha)

"""
    CutMix(alpha)
    CutMix(; alpha = 1)

Copy a shared random rectangle into each image in an [`ImageBatch`](@ref) from
its partner in a random permutation of the batch. The target area fraction is
drawn from `Beta(alpha, alpha)` with positive finite `alpha`. The label weight
uses the fraction of pixels actually replaced after rounding and clipping the
rectangle. Images keep their element type and labels are returned as floats.
"""
struct CutMix <: Transform
    alpha::Float64
    function CutMix(alpha::Real)
        a = Float64(alpha)
        isfinite(a) && a > 0 || throw(ArgumentError("alpha must be positive and finite"))
        return new(a)
    end
end

CutMix(; alpha = 1) = CutMix(alpha)

const BatchMixingTransform = Union{MixUp, CutMix}

getrandstate(tfm::BatchMixingTransform) = (weight = rand(Beta(tfm.alpha, tfm.alpha)), seed = rand(UInt64))

_mixed_image_type(::MixUp, T) = float(T)
_mixed_image_type(::CutMix, T) = T

function makebuffer(tfm::BatchMixingTransform, batch::ImageBatch)
    images, labels = itemdata(batch)
    return ImageBatch(similar(images, _mixed_image_type(tfm, eltype(images))), similar(labels, float(eltype(labels))))
end

function apply(tfm::BatchMixingTransform, batch::ImageBatch; randstate = getrandstate(tfm))
    return _mix_batch!(makebuffer(tfm, batch), tfm, batch, randstate)
end

function apply!(buf::ImageBatch, tfm::BatchMixingTransform, batch::ImageBatch; randstate = getrandstate(tfm))
    dst = itemdata(buf)
    src = itemdata(batch)
    types = (_mixed_image_type(tfm, eltype(src[1])), float(eltype(src[2])))
    if axes.(dst) != axes.(src) || eltype.(dst) != types ||
        any(Base.mightalias(a, b) for a in dst for b in src) || Base.mightalias(dst...)
        buf = makebuffer(tfm, batch)
    end
    return _mix_batch!(buf, tfm, batch, randstate)
end

function _mix_batch!(buf, tfm, batch, randstate)
    images, labels = itemdata(batch)
    out_images, out_labels = itemdata(buf)
    rng = Xoshiro(randstate.seed)
    partners = randperm(rng, length(batch))
    weight = _mix_images!(out_images, tfm, images, partners, randstate.weight, rng)
    _blend_batch!(out_labels, labels, partners, weight)
    return buf
end

function _blend_batch!(dst, src, partners, weight)
    dim = ndims(src)
    for (i, j) in enumerate(partners)
        dest = selectdim(dst, dim, i)
        if isone(weight) || i == j
            copyto!(dest, selectdim(src, dim, i))
        elseif iszero(weight)
            copyto!(dest, selectdim(src, dim, j))
        else
            dest .= weight .* selectdim(src, dim, i) .+ (1 - weight) .* selectdim(src, dim, j)
        end
    end
    return dst
end

function _mix_images!(dst, ::MixUp, src, partners, weight, rng)
    _blend_batch!(dst, src, partners, weight)
    return weight
end

function _mix_images!(dst, ::CutMix, src, partners, weight, rng)
    copyto!(dst, src)
    ratio = sqrt(1 - weight)
    width, height = size(src, 1), size(src, 2)
    x = _cutmix_range(rng, width, ratio)
    y = _cutmix_range(rng, height, ratio)
    if isempty(x) || isempty(y)
        return 1.0
    end
    trailing_axes = ntuple(_ -> Colon(), ndims(src) - 2)
    dst_patch = view(dst, x, y, trailing_axes...)
    src_patch = view(src, x, y, trailing_axes...)
    for (i, j) in enumerate(partners)
        copyto!(selectdim(dst_patch, ndims(dst_patch), i), selectdim(src_patch, ndims(src_patch), j))
    end
    return 1 - length(x) * length(y) / (width * height)
end

function _cutmix_range(rng, axis_size, ratio)
    extent = round(Int, axis_size * ratio)
    start = rand(rng, 1:axis_size) - extent ÷ 2
    return max(1, start):min(axis_size, start + extent - 1)
end
