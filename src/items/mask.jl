"""
    MaskMulti(a, [classes]; interpolate=BSpline(Constant()), extrapolate=Flat())

An `N`-dimensional multilabel mask with labels `classes`. Optionally, the
interpolation and extrapolation method can be provided. Interpolation here
refers to how the values of projected pixels that fall into the transformed
content bounds are calculated. Extrapolation refers to how to assign values
that fall outside the projected content bounds. The default is nearest neighbor
interpolation and flat extrapolation of the edges into new regions.

!!! info
    The `Interpolations` package provides numerous methods for use with
    the `interpolate` and `extrapolate` keyword arguments.  For instance,
    `BSpline(Linear())` and `BSpline(Constant())` provide linear and nearest
    neighbor interpolation, respectively. In addition `Flat()`, `Reflect()` and
    `Periodic()` boundary conditions are available for extrapolation.

## Examples

```julia
using DataAugmentation

mask = MaskMulti(rand(1:3, 100, 100))
```
```julia
showitems(mask)
```
"""
struct MaskMulti{N, T<:Integer, U} <: AbstractArrayItem{N, T}
    data::AbstractArray{T, N}
    classes::AbstractVector{U}
    bounds::Bounds{N}
    interpolate::Interpolations.InterpolationType
    extrapolate::ImageTransformations.FillType
end

function MaskMulti(
    data::AbstractArray{T,N},
    classes::AbstractVector{U},
    bounds::Bounds{N};
    interpolate::Interpolations.InterpolationType=BSpline(Constant()),
    extrapolate::ImageTransformations.FillType=Flat(),
) where {N, T<:Integer, U}
    return MaskMulti(data, classes, bounds, interpolate, extrapolate)
end

function MaskMulti(a::AbstractArray, classes = unique(a); kwargs...)
    bounds = Bounds(axes(a))
    minimum(a) >= 1 || error("Class values must start at 1")
    return MaskMulti(a, classes, bounds; kwargs...)
end

MaskMulti(a::AbstractArray{<:Gray{T}}, args...; kwargs...) where T = MaskMulti(reinterpret(T, a), args...; kwargs...)
MaskMulti(a::AbstractArray{<:Normed{T}}, args...; kwargs...) where T = MaskMulti(reinterpret(T, a), args...; kwargs...)
MaskMulti(a::IndirectArray, classes = a.values, bounds = Bounds(axes(a)); kwargs...) =
    MaskMulti(a.index, classes, bounds; kwargs...)
MaskMulti(a::IndirectArray{<:Integer, N}, classes::AbstractVector, bounds::Bounds{N}; kwargs...) where N =
    MaskMulti(a.index, classes, bounds; kwargs...)

Base.show(io::IO, mask::MaskMulti{N, T}) where {N, T} =
    print(io, "MaskMulti{$N, $T}() with size $(size(itemdata(mask))) and $(length(mask.classes)) classes")


getbounds(mask::MaskMulti) = mask.bounds


function project(P, mask::MaskMulti{N, T, U}, bounds::Bounds{N}) where {N, T, U}
    source = _project_extrapolation(itemdata(mask), mask.interpolate, mask.extrapolate)
    res = warp(source, inv(P), bounds.rs)
    return MaskMulti(
        convert.(T, res),
        mask.classes,
        bounds
    )
end


function project!(bufmask::MaskMulti, P, mask::MaskMulti, bounds)
    size(itemdata(bufmask)) == length.(bounds.rs) || return project(P, mask, bounds)
    a = OffsetArray(parent(itemdata(bufmask)), bounds.rs)
    warp!(
        a,
        _project_extrapolation(itemdata(mask), mask.interpolate, mask.extrapolate),
        inv(P),
    )
    return MaskMulti(
        a,
        mask.classes,
        bounds
    )
end


function showitem!(img, mask::MaskMulti; opacity = 1, background = nothing)
    colors = distinguishable_colors(length(mask.classes))
    showmask!(img, itemdata(mask), opacity, background) do label
        colors[findfirst(==(label), mask.classes)]
    end
end


# ## Binary masks

"""
    MaskBinary(a; interpolate=BSpline(Constant()), extrapolate=Flat())

An `N`-dimensional binary mask. Optionally, the interpolation and extrapolation
method can be provided. Interpolation here refers to how the values of
projected pixels that fall into the transformed content bounds are calculated.
Extrapolation refers to how to assign values that fall outside the projected
content bounds. The default is nearest neighbor interpolation and flat
extrapolation of the edges into new regions.

!!! info
    The `Interpolations` package provides numerous methods for use with
    the `interpolate` and `extrapolate` keyword arguments.  For instance,
    `BSpline(Linear())` and `BSpline(Constant())` provide linear and nearest
    neighbor interpolation, respectively. In addition `Flat()`, `Reflect()` and
    `Periodic()` boundary conditions are available for extrapolation.

## Examples

```julia
using DataAugmentation

mask = MaskBinary(rand(Bool, 100, 100))
```
```julia
showitems(mask)
```
"""
struct MaskBinary{N} <: AbstractArrayItem{N, Bool}
    data::AbstractArray{Bool, N}
    bounds::Bounds{N}
    interpolate::Interpolations.InterpolationType
    extrapolate::ImageTransformations.FillType
end

function MaskBinary(
    a::AbstractArray,
    bounds = Bounds(axes(a));
    interpolate::Interpolations.InterpolationType=BSpline(Constant()),
    extrapolate::ImageTransformations.FillType=Flat(),
)
    return MaskBinary(a, bounds, interpolate, extrapolate)
end

Base.show(io::IO, mask::MaskBinary{N}) where {N} =
    print(io, "MaskBinary{$N}() with size $(size(itemdata(mask)))")

getbounds(mask::MaskBinary) = mask.bounds

function project(P, mask::MaskBinary, bounds::Bounds)
    source = _project_extrapolation(itemdata(mask), mask.interpolate, mask.extrapolate)
    res = warp(source, inv(P), bounds.rs)
    return MaskBinary(convert.(Bool, res), bounds)
end


function project!(bufmask::MaskBinary, P, mask::MaskBinary, bounds)
    size(itemdata(bufmask)) == length.(bounds.rs) || return project(P, mask, bounds)
    a = OffsetArray(parent(itemdata(bufmask)), bounds.rs)
    warp!(
        a,
        _project_extrapolation(itemdata(mask), mask.interpolate, mask.extrapolate),
        inv(P),
    )
    return MaskBinary(
        a,
        bounds
    )
end

function showitem!(img, mask::MaskBinary; opacity = 1, background = nothing)
    showmask!(Gray, img, itemdata(mask), opacity, background)
end
