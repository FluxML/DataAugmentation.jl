

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
    ScaleKeepAspect(minlengths)

Projective transformation that scales the shortest side of `item`
to `minlengths`, keeping the original aspect ratio.
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
    Rotate(γ)
    Rotate(γs)

Rotate 2D spatial data counter-clockwise by angle γ around the
center. If you pass in a vector of angles, one will be
randomly selected.

## Examples

```julia
tfm = Rotate(10)
```

```julia
tfm = Rotate(-10:10)
```
"""
struct Rotate{T} <: ProjectiveTransform
    γ::T
end


getrandstate(r::Rotate{<:Real}) = r.γ
getrandstate(r::Rotate{<:AbstractVector}) = rand(r.γ)

function getprojection(rotate::Rotate, bounds; randstate = getrandstate(rotate))
    γ = randstate
    middlepoint = sum(bounds) ./ length(bounds)
    r = γ / 360 * 2pi
    return recenter(RotMatrix(r), middlepoint)
end


"""
    Reflect(γ)
    Reflect(γs)

Reflect 2D spatial data by angle γ around the center.
If you pass in a vector of angles, one will be
randomly selected.

## Examples

```julia
tfm = Reflect(10)
```

```julia
tfm = Reflect(-10:10)
```
"""
struct Reflect{T} <: ProjectiveTransform
    γ::T
end

getrandstate(r::Reflect{<:Real}) = r.γ
getrandstate(r::Reflect{<:AbstractVector}) = rand(r.γ)

function getprojection(reflect::Reflect, bounds; randstate = nothing)
    γ = randstate
    middlepoint = sum(bounds) ./ length(bounds)
    r = γ / 360 * 2pi
    return recenter(reflectionmatrix(r), middlepoint)
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
    return Translation(-bounds[1])
end

function apply(::PinOrigin, item::Union{Image, MaskMulti, MaskBinary}; randstate = nothing)
    item = @set item.data = parent(itemdata(item))
    item = @set item.bounds = makebounds(size(itemdata(item)))
    return item
end

# `PinOrigin` should not compose with a cropped transform otherwise the pinning won't work.
# This overwrites the default composition.

compose(cropped::CroppedProjectiveTransform, pin::PinOrigin) = Sequence(cropped, pin)

# ## Resize crops

RandomResizeCrop(sz) = ScaleKeepAspect(sz) |> RandomCrop(sz) |> PinOrigin()
CenterResizeCrop(sz) = ScaleKeepAspect(sz) |> CenterCrop(sz) |> PinOrigin()

ResizePadDivisible(sz, by) = ScaleKeepAspect(sz) |> PadDivisible(by) |> PinOrigin()
