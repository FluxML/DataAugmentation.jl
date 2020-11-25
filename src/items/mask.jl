"""
    MaskMulti(a, [classes])

An `N`-dimensional multilabel mask with labels `classes`.
"""
struct MaskMulti{N, T, B} <: AbstractArrayItem{N, T}
    data::AbstractArray{T, N}
    classes::AbstractVector{T}
    bounds::AbstractArray{<:SVector{N, B}, N}
end


function MaskMulti(a::AbstractArray, classes = unique(a), bounds = makebounds(size(a)))
    return MaskMulti(a, classes = bounds)
end


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


# ## Binary masks

"""
    MaskBinary(a)

An `N`-dimensional binary mask with labels `classes`.
"""
struct MaskBinary{N, B} <: AbstractArrayItem{N, Bool}
    data::AbstractArray{Bool, N}
    bounds::AbstractArray{<:SVector{N, B}, N}
end

function MaskBinary(a::AbstractArray{Bool, N}, bounds = makebounds(size(a))) where N
    return MaskBinary(a, bounds)
end


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


# ## Helpers


function mask_extrapolation(
        mask::AbstractArray{T};
        degree = Constant(),
        boundary = Flat()) where T
    itp = interpolate(T, T, mask, BSpline(degree))
    etp = extrapolate(itp, Flat())
    return etp
end
