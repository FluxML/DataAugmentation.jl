

# ## Scaling

function scaleprojection(ratios::NTuple{N}, T = Float32) where N
    a = zeros(Float32, N, N)
    a[I(N)] = SVector{N}(ratios)
    return LinearMap(SArray{Tuple{N, N}}(a))
end


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

{cell=ScaleKeepAspect}
```julia
using DataAugmentation, TestImages
image = testimage("lighthouse")
tfm = ScaleKeepAspect((200, 200))
apply(tfm, Image(image)) |> showitems
```
"""
struct ScaleKeepAspect{N} <: ProjectiveTransform
    minlengths::NTuple{N, Int}
end


function getprojection(scale::ScaleKeepAspect{N}, bounds; randstate = nothing) where N
    ratio = maximum(scale.minlengths ./ boundssize(bounds))
    return scaleprojection(Tuple(ratio for _ in 1:N))
end

"""
    ScaleFixed(sizes)

Projective transformation that scales sides to `sizes`, disregarding
aspect ratio.

See also [`ScaleKeepAspect`](#).
"""
struct ScaleFixed{N} <: ProjectiveTransform
    sizes::NTuple{N, Int}
end


function getprojection(scale::ScaleFixed{N}, bounds; randstate = nothing) where N
    ratios = scale.sizes ./ boundssize(bounds)
    return scaleprojection(ratios)
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

getrandstate(tfm::Zoom) = rand(tfm.dist)

function getprojection(tfm::Zoom, bounds::AbstractArray{<:SVector{N}}; randstate = getrandstate(tfm)) where N
    ratio = randstate
    return scaleprojection(ntuple(_ -> ratio, N))
end

"""
    Rotate(γ)
    Rotate(γs)

Rotate 2D spatial data around the center by an angle chosen at
uniformly from [-γ, γ], an angle given in degrees.

You can also pass any `Distributions.Sampleable` from which the
angle is selected.

## Examples

```julia
tfm = Rotate(10)
```

"""
struct Rotate{S<:Sampleable} <: ProjectiveTransform
    dist::S
end
Rotate(γ) = Rotate(Uniform(-abs(γ), abs(γ)))

getrandstate(tfm::Rotate) = rand(tfm.dist)

function getprojection(tfm::Rotate, bounds; randstate = getrandstate(tfm))
    γ = randstate
    middlepoint = sum(bounds) ./ length(bounds)
    r = γ / 360 * 2pi
    return recenter(RotMatrix(r), middlepoint)
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


function getprojection(tfm::Reflect, bounds; randstate = getrandstate(tfm))
    midpoint = sum(bounds) ./ length(bounds)
    r = tfm.γ / 360 * 2pi
    return recenter(reflectionmatrix(r), midpoint)
end


FlipX() = Reflect(180)
FlipY() = Reflect(90)

reflectionmatrix(r) = SMatrix{2, 2, Float32}(cos(2r), sin(2r), sin(2r), -cos(2r))


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

function getprojection(::PinOrigin, bounds; randstate = nothing)
    # TODO: translate by actual minimum x and y coordinates
    return Translation(-bounds[1])
end

function apply(::PinOrigin, item::Union{Image, MaskMulti, MaskBinary}; randstate = nothing)
    item = @set item.data = parent(itemdata(item))
    item = @set item.bounds = makebounds(size(itemdata(item)))
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

# ## Resize crops

RandomResizeCrop(sz) = ScaleKeepAspect(sz) |> RandomCrop(sz) |> PinOrigin()
CenterResizeCrop(sz) = ScaleKeepAspect(sz) |> CenterCrop(sz) |> PinOrigin()

ResizePadDivisible(sz, by) = ScaleKeepAspect(sz) |> PadDivisible(by) |> PinOrigin()
