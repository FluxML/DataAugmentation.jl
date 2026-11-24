

# ## Scaling

function scaleprojection(ratios::NTuple{N}, T = Float32) where N
    a = zeros(Float32, N, N)
    a[I(N)] = SVector{N}(ratios)
    return LinearMap(SArray{Tuple{N, N}}(a))
end


"""
    ScaleRatio(ratios) <: ProjectiveTransform

Scales the aspect ratio
"""
struct ScaleRatio{N} <: ProjectiveTransform
    ratios::NTuple{N}
end

# This allows for roundtrip through ScaleFixed and avoids code duplication
fixed_sizes(scale::ScaleRatio, bounds::Bounds) = round.(Int,  scale.ratios .* length.(bounds.rs))

function getprojection(scale::ScaleRatio{N}, bounds::Bounds{N}; randstate = nothing) where N
    return getprojection(ScaleFixed{N}(fixed_sizes(scale, bounds)), bounds; randstate)
end

function projectionbounds(tfm::ScaleRatio{N}, P, bounds::Bounds{N}; randstate = nothing) where N
    projectionbounds(ScaleFixed{N}(fixed_sizes(tfm, bounds)), P, bounds; randstate)
end

"""
    ScaleKeepAspect(minlengths) <: ProjectiveTransform

Scales the sides of `item` to be as least as large as `minlengths`,
keeping the original aspect ratio.

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

# This allows for roundtrip through ScaleFixed and avoids code duplication
fixed_sizes(scale::ScaleKeepAspect, bounds::Bounds) = round.(Int, maximum(scale.minlengths ./ length.(bounds.rs)) .* length.(bounds.rs))

function getprojection(scale::ScaleKeepAspect{N}, bounds::Bounds{N}; randstate = nothing) where N
    getprojection(ScaleFixed{N}(fixed_sizes(scale, bounds)), bounds; randstate)
end

function projectionbounds(tfm::ScaleKeepAspect{N}, P, bounds::Bounds{N}; randstate = nothing) where N
    projectionbounds(ScaleFixed{N}(fixed_sizes(tfm, bounds)), P, bounds; randstate)
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


# The span between the first and last pixel center of each axis is split into `size`
# equal cells, and the output pixels sample the cell centers. All samples then lie
# half a cell inside the input, so there are no black borders however large the scaling
# (an exact alignment of the corners would be at the mercy of rounding errors).
# The output bounds are exactly `1:size`.
function getprojection(scale::ScaleFixed, bounds::Bounds{N}; randstate = nothing) where N
    lengths = length.(bounds.rs)
    # If no scaling needs to be done, return a noop transform
    (scale.sizes == lengths) && return IdentityTransformation()

    # Axes of unchanged size are left alone; `max` keeps axes of length one invertible
    ratios = map((s, n) -> s == n ? 1.0 : s / max(n - 1, 1), scale.sizes, lengths)
    center = SVector{N, Float32}(mean.(bounds.rs))
    newcenter = SVector{N, Float32}((scale.sizes .+ 1) ./ 2)
    P = Translation(newcenter) ∘ scaleprojection(ratios) ∘ Translation(-center)
    return _singletonprojection(P, bounds, scale.sizes)
end


function projectionbounds(tfm::ScaleFixed{N}, P, bounds::Bounds{N}; randstate = nothing) where N
    P isa IdentityTransformation && return bounds
    return Bounds(map(s -> 1:s, tfm.sizes))
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

zoomscale(ratio, ::Bounds{N}) where N = ScaleRatio{N}(ntuple(_ -> ratio, N))

function getprojection(tfm::Zoom, bounds::Bounds; randstate = getrandstate(tfm))
    return getprojection(zoomscale(randstate, bounds), bounds)
end

function projectionbounds(tfm::Zoom, P, bounds::Bounds; randstate = getrandstate(tfm))
    return projectionbounds(zoomscale(randstate, bounds), P, bounds)
end

"""
    Rotate(γ; expand = true, center = nothing, fill = nothing)
    Rotate(distribution; expand = true, center = nothing, fill = nothing)
    Rotate(α, β, γ)
    Rotate(α_distribution, β_distribution, γ_distribution)

Rotate spatial data around its center. Rotate(γ) is a 2D rotation
by an angle chosen uniformly from [-γ, γ], an angle given in degrees.
Rotate(α, β, γ) is a 3D rotation by angles chosen uniformly from
[-α, α], [-β, β], and [-γ, γ], for X, Y, and Z rotations.

You can also pass any `Distributions.Sampleable` from which the
angle is selected.

For 2D rotations, `expand = false` keeps the input bounds. Pass an absolute
`center = (row, column)` in pixel coordinates to rotate around another point.
By default, out-of-bounds samples use the item's extrapolation setting. An
explicit `fill` value applies to this rotation only.

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
    expand::Bool
    center::Union{Nothing, SVector{N, Float32}}
end

Rotate{N, R, S}(s::S) where {N, R, S} = Rotate{N, R, S}(s, true, nothing)

function Rotate{N, R}(s::S; expand::Bool = true, center = nothing) where {N, R, S<:Sampleable}
    if center !== nothing
        center isa Union{Tuple, AbstractVector} && length(center) == N &&
            all(x -> x isa Real && isfinite(Float32(x)), center) ||
            throw(ArgumentError("center must contain $N finite coordinates"))
        center = SVector{N, Float32}(center)
    end
    return Rotate{N, R, S}(s, expand, center)
end

function Rotate{N, R}(s::Number; kwargs...) where {N, R}
    iszero(s) && return Identity()
    return Rotate{N, R}(Uniform(-abs(s), abs(s)); kwargs...)
end

struct FilledRotation{R<:Rotate, F} <: Transform
    rotation::R
    fill::F
end

function Rotate(s; expand::Bool = true, center = nothing, fill = nothing)
    rotation = Rotate{2, Type{RotMatrix{2, Float64}}}(s; expand, center)
    (isnothing(fill) || rotation isa Identity) && return rotation
    fill isa Union{Number, Colorant} || throw(ArgumentError("fill must be a number or a color"))
    return FilledRotation(rotation, fill)
end

Rotate(sx, sy, sz) = RotateX(sx) |> RotateY(sy) |> RotateZ(sz)
Rotate{3}(s) = Rotate(s, s, s)
Rotate{2}(s; kwargs...) = Rotate(s; kwargs...)

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
getrandstate(tfm::FilledRotation) = getrandstate(tfm.rotation)

function getprojection(
        tfm::Rotate{N, Type{R}, S},
        bounds::Bounds{N};
        randstate = getrandstate(tfm)) where {N, R, S}
    γ = randstate
    middlepoint = isnothing(tfm.center) ? SVector{N, Float32}(mean.(bounds.rs)) : tfm.center
    r = γ / 360 * 2pi
    return recenter(RotMatrix{N, Float32}(R(convert(Float64, r))), middlepoint)
end

function projectionbounds(tfm::Rotate, P, bounds; randstate = nothing)
    return tfm.expand ? transformbounds(bounds, P) : bounds
end

function apply(tfm::FilledRotation, item::Item; randstate = getrandstate(tfm))
    return apply(tfm.rotation, item; randstate)
end

function apply(tfm::FilledRotation, item::Union{Image, MaskMulti, MaskBinary}; randstate = getrandstate(tfm))
    source = @set item.extrapolate = tfm.fill
    return apply(tfm.rotation, source; randstate)
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

struct FlipAxis <: ProjectiveTransform
    axis::Symbol
end

function getprojection(tfm::FlipAxis, bounds::Bounds{N}; randstate = nothing) where N
    dims = N == 2 ? (x = 2, y = 1, z = 3) : (x = 1, y = 2, z = 3)
    dim = dims[tfm.axis]
    dim <= N || throw(ArgumentError("cannot flip the $(tfm.axis)-axis of $N-dimensional data"))
    return getprojection(FlipDim{N}(dim), bounds; randstate)
end

"""
    FlipX()

Flip data along the x-axis. In 2D, the layout is `(row, col) = (y, x)`,
so this flips dimension 2. For other dimensionalities, it flips dimension 1.

The `FlipX{N}()` form is kept for compatibility and also uses the item's dimensionality.
"""
struct FlipX{N}
    FlipX{N}() where N = FlipX()
end

FlipX() = FlipAxis(:x)

"""
    FlipY()

Flip data along the y-axis. In 2D, the layout is `(row, col) = (y, x)`,
so this flips dimension 1. For three or more dimensions, it flips dimension 2.

The `FlipY{N}()` form is kept for compatibility and also uses the item's dimensionality.
"""
struct FlipY{N}
    FlipY{N}() where N = FlipY()
end

FlipY() = FlipAxis(:y)

"""
    FlipZ()

Flip data along the z-axis (dimension 3). The item must have at least three dimensions.

The `FlipZ{N}()` form is kept for compatibility and also uses the item's dimensionality.
"""
struct FlipZ{N}
    FlipZ{N}() where N = FlipZ()
end

FlipZ() = FlipAxis(:z)

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
