"""
    UniformTemporalSubsample(num_samples; dim = 3)

Keep `num_samples` frames of a video, evenly spaced along its temporal dimension `dim`.
Frame `i` of the result is frame `1 + fld((i - 1) * (T - 1), num_samples - 1)` of the
input, for `T` input frames. The first and last frame are always kept, and a single
sample is the first frame. If `num_samples` exceeds `T`, frames repeat, picking the
nearest preceding one.

Videos are N-dimensional items with the temporal dimension next to the spatial ones, so
the default `dim = 3` fits an `Image{3}` of size `(height, width, time)` as well as the
`(width, height, time, channels)` tensor that [`ImageToTensor`](@ref) makes from it.
It applies to [`ArrayItem`](@ref)s, [`Image`](@ref)s, [`MaskMulti`](@ref)s and
[`MaskBinary`](@ref)s, which are subsampled with the same frames when passed together.
The bounds of the result are those of its new axes.

Supports `apply!`.

## Examples

```julia
using DataAugmentation, Images

video = Image(rand(RGB{N0f8}, 32, 32, 16))
tfm = UniformTemporalSubsample(8)
apply(tfm, video)  # Image{3, RGB{N0f8}} with 8 frames
```
"""
struct UniformTemporalSubsample <: Transform
    num_samples::Int
    dim::Int
    function UniformTemporalSubsample(num_samples::Integer; dim::Integer = 3)
        num_samples > 0 || throw(ArgumentError("num_samples must be positive"))
        dim > 0 || throw(ArgumentError("dim must be positive"))
        return new(num_samples, dim)
    end
end

const TemporalItem = Union{ArrayItem, Image, MaskMulti, MaskBinary}

function _temporalindices(tfm::UniformTemporalSubsample, data)
    tfm.dim <= ndims(data) || throw(ArgumentError(
        "cannot subsample dimension $(tfm.dim) of an item with $(ndims(data)) dimensions"))
    frames = axes(data, tfm.dim)
    isempty(frames) && throw(ArgumentError("cannot subsample an empty temporal dimension"))
    n = tfm.num_samples
    return [first(frames) + fld((i - 1) * (length(frames) - 1), max(n - 1, 1)) for i in 1:n]
end

_temporalitem(item::ArrayItem, data) = setdata(item, data)
_temporalitem(item::Union{Image, MaskMulti, MaskBinary}, data) =
    setdata(@set(item.bounds = Bounds(axes(data))), data)

function apply(tfm::UniformTemporalSubsample, item::TemporalItem; randstate = nothing)
    data = itemdata(item)
    return _temporalitem(item, copy(selectdim(data, tfm.dim, _temporalindices(tfm, data))))
end

function apply!(buf::I, tfm::UniformTemporalSubsample, item::I; randstate = nothing) where {I<:TemporalItem}
    src = itemdata(item)
    dst = itemdata(buf)
    indices = _temporalindices(tfm, src)
    if size(dst) != Base.setindex(size(src), length(indices), tfm.dim) || Base.mightalias(dst, src)
        return apply(tfm, item)
    end
    copyto!(dst, selectdim(src, tfm.dim, indices))
    return buf
end
