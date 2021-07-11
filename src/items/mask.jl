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
struct MaskMulti{N, T<:Integer, U} <: AbstractArrayItem{N, T}
    data::AbstractArray{T, N}
    classes::AbstractVector{U}
    bounds::Bounds{N}
end


function MaskMulti(a::AbstractArray, classes = unique(a))
    bounds = Bounds(size(a))
    minimum(a) >= 1 || error("Class values must start at 1")
    return MaskMulti(a, classes, bounds)
end

MaskMulti(a::AbstractArray{<:Gray{T}}, args...) where T = MaskMulti(reinterpret(T, a), args...)
MaskMulti(a::AbstractArray{<:Normed{T}}, args...) where T = MaskMulti(reinterpret(T, a), args...)
MaskMulti(a::IndirectArray, classes = a.values, bounds = Bounds(size(a))) =
    MaskMulti(a.index, classes, bounds)

Base.show(io::IO, mask::MaskMulti{N, T}) where {N, T} =
    print(io, "MaskMulti{$N, $T}() with size $(size(itemdata(mask))) and $(length(mask.classes)) classes")


getbounds(mask::MaskMulti) = mask.bounds


function project(P, mask::MaskMulti, bounds::Bounds)
    a = itemdata(mask)
    etp = mask_extrapolation(a)
    res = warp(etp, inv(P), bounds.rs)
    return MaskMulti(
        res,
        mask.classes,
        bounds
    )
end


function project!(bufmask::MaskMulti, P, mask::MaskMulti, bounds)
    a = OffsetArray(parent(itemdata(bufmask)), bounds.rs)
    warp!(
        a,
        mask_extrapolation(itemdata(mask)),
        inv(P),
    )
    return MaskMulti(
        a,
        mask.classes,
        bounds
    )
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
struct MaskBinary{N} <: AbstractArrayItem{N, Bool}
    data::AbstractArray{Bool, N}
    bounds::Bounds{N}
end

function MaskBinary(a::AbstractArray{Bool, N}, bounds = Bounds(size(a))) where N
    return MaskBinary(a, bounds)
end

Base.show(io::IO, mask::MaskBinary{N}) where {N} =
    print(io, "MaskBinary{$N}() with size $(size(itemdata(mask)))")

getbounds(mask::MaskBinary) = mask.bounds

function project(P, mask::MaskBinary, bounds::Bounds)
    etp = mask_extrapolation(itemdata(mask))
    return MaskBinary(
        warp(etp, inv(P), bounds.rs),
        bounds,
    )
end


function project!(bufmask::MaskBinary, P, mask::MaskBinary, bounds)
    a = OffsetArray(parent(itemdata(bufmask)), bounds.rs)
    res = warp!(
        a,
        mask_extrapolation(itemdata(mask)),
        inv(P),
    )
    return MaskBinary(
        a,
        bounds
    )
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
