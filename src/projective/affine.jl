

# ## Scaling

function scaleprojection(ratios::NTuple{N}, T = Float32) where N
    a = zeros(Float32, N, N)
    a[I(N)] = SVector{N}(ratios)
    return LinearMap(SArray{Tuple{N, N}}(a))
end


# A single number for a scaling applies to every dimension. It is expanded to one
# value per dimension once the item, and with it the dimensionality, is known.
struct IsotropicScale{F, T<:Real} <: ProjectiveTransform
    value::T
end

IsotropicScale{F}(value::T) where {F, T<:Real} = IsotropicScale{F, T}(value)

_expand(tfm::IsotropicScale{F}, ::Bounds{N}) where {F, N} = F(ntuple(_ -> tfm.value, Val(N)))

function getprojection(tfm::IsotropicScale, bounds::Bounds; randstate = nothing)
    return getprojection(_expand(tfm, bounds), bounds; randstate)
end

function projectionbounds(tfm::IsotropicScale, P, bounds::Bounds; randstate = nothing)
    return projectionbounds(_expand(tfm, bounds), P, bounds; randstate)
end

"""
    ScaleRatio(ratios) <: ProjectiveTransform
    ScaleRatio(ratio)

Scales the sides by `ratios`, one per dimension. A number `ratio` scales every
dimension by the same factor, whatever the dimensionality of the item.
"""
struct ScaleRatio{N} <: ProjectiveTransform
    ratios::NTuple{N}
end

function ScaleRatio(ratio::Real)
    isfinite(ratio) && ratio > 0 || throw(ArgumentError("ratio must be positive and finite"))
    return IsotropicScale{ScaleRatio}(ratio)
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
    ScaleKeepAspect(minlength)

Scales the sides of `item` to be as least as large as `minlengths`,
keeping the original aspect ratio. A number `minlength` is the same minimum
for every dimension, whatever the dimensionality of the item: the shortest
side becomes `minlength`.

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

function ScaleKeepAspect(minlength::Integer)
    minlength > 0 || throw(ArgumentError("size must be positive"))
    return IsotropicScale{ScaleKeepAspect}(Int(minlength))
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
    ScaleFixed(size)

Projective transformation that scales sides to `sizes`, disregarding
aspect ratio. A number `size` is the same size for every dimension, whatever
the dimensionality of the item.

See also [`ScaleKeepAspect`](@ref).
"""
struct ScaleFixed{N} <: ProjectiveTransform
    sizes::NTuple{N, Int}
end

function ScaleFixed(size::Integer)
    size > 0 || throw(ArgumentError("size must be positive"))
    return IsotropicScale{ScaleFixed}(Int(size))
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

Rotate spatial data around its center, by angles given in degrees. `Rotate(γ)`
takes the dimensionality from the item. In 2D, it rotates by an angle chosen
uniformly from [-γ, γ]. In 3D, it is the same as `Rotate(γ, γ, γ)`: a rotation
around each of the three axes, by angles chosen independently from [-γ, γ].
Use `Rotate{2}(γ)` or `Rotate{3}(γ)` to accept only 2D or 3D data.

`Rotate(α, β, γ)` is a 3D rotation by angles chosen uniformly from
[-α, α], [-β, β], and [-γ, γ], for X, Y, and Z rotations.

You can also pass any `Distributions.Sampleable` from which the
angle is selected.

`expand = false` keeps the input bounds. Pass an absolute `center = (row, column)`
(or one coordinate per dimension in 3D) in pixel coordinates to rotate around another
point. By default, out-of-bounds samples use the item's extrapolation setting. An
explicit `fill` value applies to this rotation only and to [`Image`](@ref) items
only; masks keep their own extrapolation. Rotations with a `fill` are not
projective transforms and are not fused with neighboring ones.

## Examples

```julia
tfm = Rotate(10)
apply(tfm, Image(rand(Float32, 16, 16)))
apply(tfm, Image(rand(Float32, 16, 16, 16)))

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

struct FilledRotation{R<:ProjectiveTransform, F} <: Transform
    rotation::R
    fill::F
end

# `Rotate(γ)` draws three angles before the item is seen, as a 3D rotation needs them.
# A 2D rotation uses the first one.
struct RotateND{S<:Sampleable} <: ProjectiveTransform
    dist::S
    expand::Bool
    center::Union{Nothing, Tuple{Vararg{Float32}}}
end

function RotateND(s::Union{Real, Sampleable}; expand::Bool = true, center = nothing)
    if center !== nothing
        center isa Union{Tuple, AbstractVector} && length(center) in (2, 3) &&
            all(x -> x isa Real && isfinite(Float32(x)), center) ||
            throw(ArgumentError("center must contain 2 or 3 finite coordinates"))
        center = Tuple(Float32.(center))
    end
    s isa Real && iszero(s) && return Identity()
    dist = s isa Real ? Uniform(-abs(s), abs(s)) : s
    return RotateND(dist, expand, center)
end

function _rotate2d(s; expand::Bool = true, center = nothing, fill = nothing)
    rotation = Rotate{2, Type{RotMatrix{2, Float64}}}(s; expand, center)
    return _filled(rotation, fill)
end

function _filled(rotation, fill)
    (isnothing(fill) || rotation isa Identity) && return rotation
    fill isa Union{Number, Colorant} || throw(ArgumentError("fill must be a number or a color"))
    return FilledRotation(rotation, fill)
end

function Rotate(s; expand::Bool = true, center = nothing, fill = nothing)
    return _filled(RotateND(s; expand, center), fill)
end

Rotate(sx, sy, sz) = RotateX(sx) |> RotateY(sy) |> RotateZ(sz)
Rotate{3}(s) = Rotate(s, s, s)
Rotate{2}(s; kwargs...) = _rotate2d(s; kwargs...)

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
getrandstate(tfm::RotateND) = (rand(tfm.dist), rand(tfm.dist), rand(tfm.dist))
getrandstate(tfm::FilledRotation) = getrandstate(tfm.rotation)

function _rotation(tfm::RotateND, ::Bounds{2})
    return Rotate{2, Type{RotMatrix{2, Float64}}}(tfm.dist; tfm.expand, tfm.center)
end

function _rotation(tfm::RotateND, ::Bounds{3})
    rotations = (RotX{Float64}, RotY{Float64}, RotZ{Float64})
    return ComposedProjectiveTransform(
        map(R -> Rotate{3, Type{R}}(tfm.dist; tfm.expand, tfm.center), rotations)...)
end

function _rotation(::RotateND, ::Bounds{N}) where N
    throw(ArgumentError("cannot rotate $N-dimensional data, use 2D or 3D data"))
end

# A single angle, as for a 2D rotation, is the angle around the first axis.
_rotation_state(angle::Real, ::Bounds{2}) = angle
_rotation_state(angles, ::Bounds{2}) = first(angles)
_rotation_state(angle::Real, ::Bounds{3}) = (angle, zero(angle), zero(angle))
_rotation_state(angles, ::Bounds{3}) = Tuple(angles)

function _projection_geometry(tfm::RotateND, bounds::Bounds, randstate)
    rotation = _rotation(tfm, bounds)
    return _projection_geometry(rotation, bounds, _rotation_state(randstate, bounds))
end

function getprojection(tfm::RotateND, bounds::Bounds; randstate = getrandstate(tfm))
    return first(_projection_geometry(tfm, bounds, randstate))
end

function projectionbounds(tfm::RotateND, P, bounds::Bounds; randstate = getrandstate(tfm))
    return last(_projection_geometry(tfm, bounds, randstate))
end

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

function apply(tfm::FilledRotation, item::Image; randstate = getrandstate(tfm))
    source = @set item.extrapolate = tfm.fill
    result = apply(tfm.rotation, source; randstate)
    return @set result.extrapolate = item.extrapolate
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
    midpoint = SVector{N, Float32}(mean.(bounds.rs))
    return recenter(P, midpoint)
end


function reflectionmatrix(r)
    A = SMatrix{2, 2, Float32}(cos(2r), sin(2r), sin(2r), -cos(2r))
    return round.(A; digits = 12)
end


"""
    FlipDim(dim)
    FlipDim{N}(dim)

Reflect spatial data along the axis of dimension `dim`. `FlipDim(dim)` takes the
dimensionality from the item and requires it to be at least `dim`. `FlipDim{N}(dim)`
also checks that `1 <= dim <= N` right away.

## Examples

```julia
tfm = FlipDim(1)
```
"""
struct FlipDim{N}
    function FlipDim{N}(dim) where N
        1 <= dim <= N || error("invalid dimension")
        return FlipDim(dim)
    end
end

# `axis` is the name of an axis (`:x`, `:y` or `:z`) or a dimension.
struct FlipAxis <: ProjectiveTransform
    axis::Union{Symbol, Int}
end

function FlipDim(dim::Integer)
    dim >= 1 || throw(ArgumentError("dimension must be positive"))
    return FlipAxis(Int(dim))
end

_flipdim(dim::Int, N) = dim
_flipdim(axis::Symbol, N) = (N == 2 ? (x = 2, y = 1, z = 3) : (x = 1, y = 2, z = 3))[axis]
_flipname(dim::Int) = "dimension $dim"
_flipname(axis::Symbol) = "the $axis-axis"

function getprojection(tfm::FlipAxis, bounds::Bounds{N}; randstate = nothing) where N
    dim = _flipdim(tfm.axis, N)
    dim <= N || throw(ArgumentError("cannot flip $(_flipname(tfm.axis)) of $N-dimensional data"))
    arr = 1I(N)
    arr[dim, dim] = -1
    M = SMatrix{N, N, Float32}(arr)
    return DataAugmentation.centered(LinearMap(M), bounds)
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
