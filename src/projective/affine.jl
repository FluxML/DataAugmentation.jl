

# ## Scaling

function scaleprojection(ratios::NTuple{N}, T = Float32) where N
    a = zeros(Float32, N, N)
    a[I(N)] = SVector{N}(ratios)
    return LinearMap(SArray{Tuple{N, N}}(a))
end


"""
    ScaleRatio(minlengths) <: ProjectiveTransform

Scales the aspect ratio
"""
struct ScaleRatio{N} <: ProjectiveTransform
    ratios::NTuple{N}
end


function getprojection(scale::ScaleRatio, bounds; randstate = nothing)
    return scaleprojection(scale.ratios)
end

"""
    ScaleKeepAspect(minlengths) <: ProjectiveTransform

Scales the shortest side of `item` to `minlengths`, keeping the
original aspect ratio.

## Examples

```@example
using DataAugmentation, TestImages
image = testimage("lighthouse")
tfm = ScaleKeepAspect((200, 200))
apply(tfm, Image(image))
```
"""
struct ScaleKeepAspect{N} <: ProjectiveTransform
    minlengths::NTuple{N, Int}
end


function getprojection(scale::ScaleKeepAspect{N}, bounds::Bounds{N}; randstate = nothing) where N
    # If no scaling needs to be done, return a noop transform
    scale.minlengths == length.(bounds.rs) && return IdentityTransformation()

    # Offset `minlengths` by 1 to avoid black border on one side
    ratio = maximum((scale.minlengths .+ 1) ./ length.(bounds.rs))
    upperleft = SVector{N, Float32}(minimum.(bounds.rs)) .- 0.5
    P = scaleprojection(Tuple(ratio for _ in 1:N))
    if any(upperleft .!= 0)
        P = P ∘ Translation((Float32.(P(upperleft)) .+ 0.5f0))
    end
    return P
end

function projectionbounds(tfm::ScaleKeepAspect{N}, P, bounds::Bounds{N}; randstate = nothing) where N
    origsz = length.(bounds.rs)
    ratio = maximum((tfm.minlengths) ./ origsz)
    sz = floor.(Int,ratio .* origsz)
    bounds_ = transformbounds(bounds, P)
    bs_ = offsetcropbounds(sz, bounds_, ntuple(_ -> 0.5, N))
    return bs_
end

"""
    ScaleFixed(sizes)

Projective transformation that scales sides to `sizes`, disregarding
aspect ratio.

See also [`ScaleKeepAspect`](@ref).
"""
struct ScaleFixed{N} <: ProjectiveTransform
    sizes::NTuple{N, Int}
end


function getprojection(scale::ScaleFixed, bounds::Bounds{N}; randstate = nothing) where N
    ratios = (scale.sizes .+ 1) ./ length.(bounds.rs)
    upperleft = SVector{N, Float32}(minimum.(bounds.rs)) .- 1
    P = scaleprojection(ratios)
    if any(upperleft .!= 0)
        P = P  ∘ Translation(-upperleft)
    end
    return P
end


function projectionbounds(tfm::ScaleFixed{N}, P, bounds::Bounds{N}; randstate = nothing) where N
    bounds_ = transformbounds(bounds, P)
    return offsetcropbounds(tfm.sizes, bounds_, ntuple(_ -> 1., N))
end

"""
    Zoom(scales = (1, 1.2)) <: ProjectiveTransform
    Zoom(distribution)

Zoom into an item by a factor chosen from the interval `scales`
or `distribution`.
"""
struct Zoom{D<:Sampleable} <: ProjectiveTransform
    dist::D
end

Zoom(scales::NTuple{2, T} = (1., 1.2)) where T = Zoom(Uniform(scales[1], scales[2]))

"""
Return random state of the transform
"""
getrandstate(tfm::Zoom) = rand(tfm.dist)

function getprojection(tfm::Zoom, bounds::Bounds{N}; randstate = getrandstate(tfm)) where N
    ratio = randstate
    return scaleprojection(ntuple(_ -> ratio, N))
end

"""
    Rotate(γ)
    Rotate(distribution)
    Rotate(α, β, γ)
    Rotate(α_distribution, β_distribution, γ_distribution)

Rotate spatial data around its center. Rotate(γ) is a 2D rotation
by an angle chosen uniformly from [-γ, γ], an angle given in degrees.
Rotate(α, β, γ) is a 3D rotation by angles chosen uniformly from
[-α, α], [-β, β], and [-γ, γ], for X, Y, and Z rotations.

You can also pass any `Distributions.Sampleable` from which the
angle is selected.

## Examples

```julia
tfm2d = Rotate(10)
apply(tfm2d, Image(rand(Float32, 16, 16)))

tfm3d = Rotate(10, 20, 30)
apply(tfm3d, Image(rand(Float32, 16, 16, 16)))
```
"""
struct Rotate{N, R, S<:Sampleable} <: ProjectiveTransform
    dist::S
end

Rotate{N, R}(s::S) where {N, R, S} = Rotate{N, R, S}(s)

function Rotate{N, R}(s::S) where {N, R, S<:Number}
    if s == 0
        return Identity()
    end
    Rotate{N, R, Uniform}(Uniform(-abs(s), abs(s)))
end

Rotate(s) = Rotate{2, Type{RotMatrix{2, Float64}}}(s)
Rotate(sx, sy, sz) = RotateX(sx) |> RotateY(sy) |> RotateZ(sz)
Rotate{3}(s) = Rotate(s, s, s)
Rotate{2}(s) = Rotate(s)

"""
    RotateX(γ)
    RotateX(distribution)

X-Axis rotation of 3D spatial data around the center by an angle chosen
uniformly from [-γ, γ], an angle given in degrees.

You can also pass any `Distributions.Sampleable` from which the
angle is selected.
"""
RotateX(s) = Rotate{3, Type{RotX{Float64}}}(s)

"""
    RotateY(γ)
    RotateY(distribution)

Y-Axis rotation of 3D spatial data around the center by an angle chosen
uniformly from [-γ, γ], an angle given in degrees.

You can also pass any `Distributions.Sampleable` from which the
angle is selected.
"""
RotateY(s) = Rotate{3, Type{RotY{Float64}}}(s)

"""
    RotateZ(γ)
    RotateZ(distribution)

Z-Axis rotation of 3D spatial data around the center by an angle chosen
uniformly from [-γ, γ], an angle given in degrees.

You can also pass any `Distributions.Sampleable` from which the
angle is selected.
"""
RotateZ(s) = Rotate{3, Type{RotZ{Float64}}}(s)

getrandstate(tfm::Rotate) = rand(tfm.dist)

function getprojection(
        tfm::Rotate{N, Type{R}, S},
        bounds::Bounds{N};
        randstate = getrandstate(tfm)) where {N, R, S}
    γ = randstate
    middlepoint = SVector{N, Float32}(mean.(bounds.rs))
    r = γ / 360 * 2pi
    return recenter(RotMatrix{N, Float32}(R(convert(Float64, r))), middlepoint)
end


"""
    Reflect(γ)
    Reflect(distribution)

Reflect 2D spatial data around the center by an angle chosen at
uniformly from [-γ, γ], an angle given in degrees.

You can also pass any `Distributions.Sampleable` from which the
angle is selected.

## Examples

```julia
tfm = Reflect(10)
```
"""
struct Reflect <: ProjectiveTransform
    γ
end


function getprojection(tfm::Reflect, bounds::Bounds{2}; randstate = getrandstate(tfm))
    r = tfm.γ / 360 * 2pi
    return centered(LinearMap(reflectionmatrix(r)), bounds)
end

"""
    centered(P, bounds)

Transform `P` so that is applied around the center of `bounds`
instead of the origin
"""
function centered(P, bounds::Bounds{N}) where N
    upperleft = minimum.(bounds.rs)
    bottomright = maximum.(bounds.rs)

    midpoint = SVector{N, Float32}((bottomright .- upperleft) ./ 2) .+ .5f0
    return recenter(P, midpoint)
end


function reflectionmatrix(r)
    A = SMatrix{2, 2, Float32}(cos(2r), sin(2r), sin(2r), -cos(2r))
    return round.(A; digits = 12)
end


"""
    FlipDim{N}(dim)
Reflect `N` dimensional data along the axis of dimension `dim`. Must satisfy 1 <= `dim` <= `N`.
## Examples
```julia
tfm = FlipDim{2}(1)
```
"""
struct FlipDim{N} <: ProjectiveTransform
    dim::Int
    FlipDim{N}(dim) where N = 1 <= dim <= N ? new{N}(dim) : error("invalid dimension")
end

# 2D images use (r, c) = (y, x) convention
struct FlipX{N}
    FlipX{N}() where N = FlipDim{N}(N==2 ? 2 : 1)
end

struct FlipY{N}
    FlipY{N}() where N = FlipDim{N}(N==2 ? 1 : 2)
end

struct FlipZ{N}
    FlipZ{N}() where N = FlipDim{N}(3)
end

function getprojection(tfm::FlipDim{N}, bounds::Bounds{N}; randstate = nothing) where N
    arr = 1I(N)
    arr[tfm.dim, tfm.dim] = -1
    M = SMatrix{N, N, Float32}(arr)
    return DataAugmentation.centered(LinearMap(M), bounds)
end


"""
    PinOrigin()

Projective transformation that translates the data so that
the upper left bounding corner is at the origin `(0, 0)` (or
the multidimensional equivalent).

Projective transformations on images return `OffsetArray`s,
but not on keypoints. Hardware like GPUs do not support OffsetArrays,
so they will be unwrapped and no longer match up with the keypoints.

Pinning the data to the origin makes sure that the resulting
`OffsetArray` has the same indices as a regular array, starting
at one.
"""
struct PinOrigin <: ProjectiveTransform end

function getprojection(::PinOrigin, bounds::Bounds{N}; randstate = nothing) where N
    p = (-SVector{N, Float32}(minimum.(bounds.rs))) .+ 1
    P = Translation(p)
    return P
end

function apply(::PinOrigin, item::Union{<:Image, <:MaskMulti, <:MaskBinary}; randstate = nothing)
    item = @set item.data = parent(itemdata(item))
    item = @set item.bounds = Bounds(size(itemdata(item)))
    return item
end

function apply!(buf::AbstractItem, ::PinOrigin, item::Union{Image, MaskMulti, MaskBinary}; randstate = nothing)
    item = @set item.data = parent(itemdata(item))
    copyitemdata!(buf, item)
    return buf
end

# `PinOrigin` should not compose with a cropped transform otherwise the pinning won't work.
# This overwrites the default composition.

compose(cropped::CroppedProjectiveTransform, pin::PinOrigin) = Sequence(cropped, pin)
compose(cropped::ComposedProjectiveTransform, pin::PinOrigin) = Sequence(cropped, pin)
compose(cropped::ProjectiveTransform, pin::PinOrigin) = Sequence(cropped, pin)

# ## Resize crops
"""
ScaleKeepAspect(sz) |> RandomCrop(sz) |> PinOrigin()
"""
RandomResizeCrop(sz) = ScaleKeepAspect(sz) |> RandomCrop(sz) |> PinOrigin()
"""
ScaleKeepAspect(sz) |> CenterCrop(sz) |> PinOrigin()
"""
CenterResizeCrop(sz) = ScaleKeepAspect(sz) |> CenterCrop(sz) |> PinOrigin()
"""
ScaleKeepAspect(sz) |> PadDivisible(by) |> PinOrigin()
"""
ResizePadDivisible(sz, by) = ScaleKeepAspect(sz) |> PadDivisible(by) |> PinOrigin()
