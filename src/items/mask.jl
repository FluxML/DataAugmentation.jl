"""
    MaskMulti(a, [classes])

An `N`-dimensional multilabel mask with labels `classes`.

## Examples

{cell=MaskMulti}
```julia
using DataAugmentation

mask = MaskMulti(rand(1:3, 100, 100))
```
{cell=MaskMulti}
```julia
showitems(mask)
```
"""
struct MaskMulti{N, T<:Integer, U, B} <: AbstractArrayItem{N, T}
    data::AbstractArray{T, N}
    classes::AbstractVector{U}
    bounds::AbstractArray{<:SVector{N, B}, N}
end


function MaskMulti(a::AbstractArray, classes = unique(a))
    bounds = makebounds(size(a))
    minimum(a) >= 1 || error("Class values must start at 1")
    return MaskMulti(a, classes, bounds)
end

MaskMulti(a::AbstractArray{<:Gray{T}}, args...) where T = MaskMulti(reinterpret(T, a), args...)
MaskMulti(a::AbstractArray{<:Normed{T}}, args...) where T = MaskMulti(reinterpret(T, a), args...)

Base.show(io::IO, mask::MaskMulti{N, T}) where {N, T} =
    print(io, "MaskMulti{$N, $T}() with size $(size(itemdata(mask))) and $(length(mask.classes)) classes")


getbounds(mask::MaskMulti) = mask.bounds


function project(P, mask::MaskMulti, indices)
    a = itemdata(mask)
    etp = mask_extrapolation(a)
    return MaskMulti(
        warp(etp, inv(P), indices),
        mask.classes,
        P.(mask.bounds)
    )
end


function project!(bufmask::MaskMulti, P, mask::MaskMulti, indices)
    a = OffsetArray(parent(itemdata(bufmask)), indices)
    bounds_ = P.(getbounds(mask))
    res = warp!(
        a,
        mask_extrapolation(itemdata(mask)),
        inv(P),
    )
    return MaskMulti(a, mask.classes, P.(getbounds(mask)))
end


function showitem!(img, mask::MaskMulti)
    colors = distinguishable_colors(length(mask.classes))
    maskimg = map(itemdata(mask)) do val
        colors[findfirst(==(val), mask.classes)]
    end
    showimage!(img, maskimg)
end


# ## Binary masks

"""
    MaskBinary(a)

An `N`-dimensional binary mask.

## Examples

{cell=MaskMulti}
```julia
using DataAugmentation

mask = MaskBinary(rand(Bool, 100, 100))
```
{cell=MaskMulti}
```julia
showitems(mask)
```
"""
struct MaskBinary{N, B} <: AbstractArrayItem{N, Bool}
    data::AbstractArray{Bool, N}
    bounds::AbstractArray{<:SVector{N, B}, N}
end

function MaskBinary(a::AbstractArray{Bool, N}, bounds = makebounds(size(a))) where N
    return MaskBinary(a, bounds)
end

Base.show(io::IO, mask::MaskBinary{N}) where {N} =
    print(io, "MaskBinary{$N}() with size $(size(itemdata(mask)))")

getbounds(mask::MaskBinary) = mask.bounds

function project(P, mask::MaskBinary, indices)
    etp = mask_extrapolation(itemdata(mask))
    return MaskBinary(
        warp(etp, inv(P), indices),
        P.(getbounds(mask)),
    )
end


function project!(bufmask::MaskBinary, P, mask::MaskBinary, indices)
    bounds_ = P.(getbounds(mask))
    a = OffsetArray(parent(itemdata(bufmask)), indices)
    res = warp!(
        a,
        mask_extrapolation(itemdata(mask)),
        inv(P),
    )
    return MaskBinary(res, P.(getbounds(mask)))
end

function showitem!(img, mask::MaskBinary)
    showimage!(img, colorview(Gray, itemdata(mask)))
end
# ## Helpers


function mask_extrapolation(
        mask::AbstractArray{T};
        t = T,
        degree = Constant(),
        boundary = Flat()) where T
    itp = interpolate(t, t, mask, BSpline(degree))
    etp = extrapolate(itp, Flat())
    return etp
end
