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
showitem(mask)
```
"""
struct MaskMulti{N, T, B} <: AbstractArrayItem{N, T}
    data::AbstractArray{T, N}
    classes::AbstractVector{T}
    bounds::AbstractArray{<:SVector{N, B}, N}
end


function MaskMulti(a::AbstractArray, classes = unique(a), bounds = makebounds(size(a)))
    return MaskMulti(a, classes = bounds)
end

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
    bounds_ = P.(getbounds(mask))
    warp!(
        itemdata(bufmask),
        mask_extrapolation(itemdata(mask)),
        inv(P),
    )
    return MaskMulti(itemdata(bufmask), mask.classes, P.(getbounds(mask)))
end


function showitem(mask::MaskMulti)
    colors = distinguishable_colors(length(mask.classes))
    map(itemdata(mask)) do val
        colors[findfirst(==(val), mask.classes)]
    end
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
showitem(mask)
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
    warp!(
        itemdata(bufmask),
        mask_extrapolation(itemdata(mask)),
        inv(P),
    )
    return MaskBinary(itemdata(bufmask), P.(getbounds(mask)))
end

function showitem(mask::MaskBinary)
    return colorview(Gray, itemdata(mask))
end
# ## Helpers


function mask_extrapolation(
        mask::AbstractArray{T};
        degree = Constant(),
        boundary = Flat()) where T
    itp = interpolate(T, T, mask, BSpline(degree))
    etp = extrapolate(itp, Flat())
    return etp
end
