# AbstractAffineTransform interface

"""
    AbstractAffineTransform

Abstract supertype for affine transformations

# Supporting custom `AbstractAffineTransform`s and `Item`s

## `AbstractAffineTransform` interface

- `gettfmmatrix(t::MyAffineTransform, getbounds(item), getparam(t))`
  should return a transformation matrix (s. `CoordinateTransformations.jl`)

## Item interface

- `getbounds(item::MyItem)::Tuple` returns the spatial bounds of an item,
  e.g. `size(img)` for an image array
- `applytfm(item::MyItem, tfm)::MyItem` applies transformation matrix `tfm`
  (constructed with `gettfmmatrix`) to `item` and returns an item of the same
  type
"""
abstract type AbstractAffineTransform <: AbstractTransform end

function (t::AbstractAffineTransform)(item::Item, param)
    tfm = gettfmmatrix(t, getbounds(item), param)
    return applytfm(item, tfm)
end

# Item interface implementation
# TODO: rename to applyaffine
"""
    applytfm(item, tfm, crop = nothing)

Applies affine transformation matrix `tfm` to `item`, optionally cropping to window
of size `crop`
"""
function applytfm(item::Image{C}, tfm, crop::Union{Nothing,Tuple} = nothing) where {C}
    if crop isa Tuple
        indices = (1:crop[1], 1:crop[2])
        return Image(warp(parent(item.data), inv(tfm), indices, zero(C)))
    else
        return Image(warp(parent(item.data), tfm, zero(C)))
    end
end

function applytfm(item::Keypoints, tfm, crop::Union{Nothing,Tuple} = nothing)::Keypoints
    return Keypoints(
        map(k -> fmap(tfm, k), itemdata(item)),
        isnothing(crop) ? item.bounds : crop,
    )
end


# AbstractAffineTransform composition

"""
    ComposedAffineTransform(transforms)

Composes several affine transformations.

Due to associativity of affine transformations, the transforms can be
combined before applying, leading to large performance improvements.

Chaining (`|>`) multiple `AbstractAffineTransformation`s automatically
creates a `ComposedAffineTransform`.
"""
struct ComposedAffineTransform <: AbstractAffineTransform
    transforms::NTuple{N,AbstractAffineTransform} where N
end

getparam(cat::ComposedAffineTransform) = Tuple(getparam(t) for t in cat.transforms)

# TODO: maybe refactor to use a fold
function gettfmmatrix(cat::ComposedAffineTransform, bounds, params::Tuple)
    tfm = IdentityTransformation()

    for (t, param) in zip(cat.transforms, params)
        tfm = gettfmmatrix(t, bounds, param) ∘ tfm
    end

    return tfm
end

Base.:(|>)(tfm1::AbstractAffineTransform, tfm2::AbstractAffineTransform) =
    ComposedAffineTransform((tfm1, tfm2))
Base.:(|>)(cat::ComposedAffineTransform, tfm::AbstractAffineTransform) =
    ComposedAffineTransform((cat.transforms..., tfm))
Base.:(|>)(tfm::AbstractAffineTransform, cat::ComposedAffineTransform) =
    ComposedAffineTransform((tfm, cat.transforms))


# Cropping transforms

"""
    CroppedAffineTransform(transform, croptransform)

Applies an affine `transform` and crops with `croptransform`, such
that `getbounds(t(item)) == getcrop(croptransform)`

This wrapper leads to performance improvements when warping an
image, since only the indices within the bounds of `crop` need
to be evaluated.

Composing any `AbstractAffineTransform` with a `CropTransform`
constructs a `CroppedAffineTransform`.
"""
struct CroppedAffineTransform <: AbstractAffineTransform
    transform::AbstractAffineTransform
    croptransform::AbstractCropTransform
end

getparam(t::CroppedAffineTransform) = getparam(t.transform)

function (t::CroppedAffineTransform)(item::Item, param)
    tfm = gettfmmatrix(t.transform, getbounds(item), param)
    return applytfm(item, tfm, getcrop(t.croptransform, item))
end

Base.:(|>)(at::AbstractAffineTransform, ct::AbstractCropTransform) =
    CroppedAffineTransform(at, ct)
Base.:(|>)(cat::CroppedAffineTransform, ct::AbstractCropTransform) =
    CroppedAffineTransform(cat.transform, ct)
Base.:(|>)(cat::CroppedAffineTransform, at::AbstractAffineTransform) =
    CroppedAffineTransform(cat.transform |> at, cat.crop)
Base.:(|>)(at::AbstractAffineTransform, cat::CroppedAffineTransform) =
    CroppedAffineTransform(at |> cat.transform, cat.crop)
Base.:(|>)(cat1::CroppedAffineTransform, cat2::CroppedAffineTransform) =
    CroppedAffineTransform(cat1.transform |> cat2.transform, cat2.croptransform)


# AbstractAffineTransformations implementations

"""
    AffineTransform

Applies static transformation matrix `tfm` to an item
"""
struct AffineTransform <: AbstractAffineTransform
    tfm
end
gettfmmatrix(t::AffineTransform, bounds, param) = t.tfm

"""

"""
abstract type AbstractResizedTransform <: AbstractAffineTransform end

struct RandomResizedTransform <: AbstractResizedTransform
    size
end
struct CenterResizedTransform <: AbstractResizedTransform
    size
end

getparam(tfm::RandomResizedTransform) = (rand(), rand())
getparam(tfm::CenterResizedTransform) = (1/2, 1/2)

# FIXME: black borders
# TODO: clean up naming
function gettfmmatrix(t::AbstractResizedTransform, bounds, param)
    h, w = t.size
    k, l = bounds
    factor = min(k / h, l / w)
    scaletfm = getscale(1/factor, 1/factor)

    h_, w_ = Int.(floor.(bounds ./ factor))

    ry, rx = param
    ty = pickfromrange(0:max(0, h_ - h), ry)
    tx = pickfromrange(0:max(0, w_ - w), rx)

    translatetfm = Translation(-ty, -tx)

    return translatetfm ∘ scaletfm
end

RandomResizedCrop(crop) = RandomResizedTransform(crop) |> CropTransform(crop)
CenterResizedCrop(crop) = CenterResizedTransform(crop) |> CropTransform(crop)


"""
    Scale
"""
Scale(factors::Tuple) = AffineTransform(getscale(factors...)) |> CropTransform(factors = factors)
Scale(factor) = Scale((factor, factor))


# AffineRotate

"""
    RotateTransform(angle)
    RotateTransform(angles)

Rotates an `Item` by `angle` degrees counter-clockwise using an affine
transformation.
If a vector or range `angles` of degrees is given, picks a random element.

See also [`Rotate90`](@ref), [`Rotate180`](@ref), [`Rotate270`](@ref)
"""
struct Rotate <: AbstractAffineTransform
    angles
end

# Pick a random angle
getparam(t::Rotate) = t.angles isa Number ? t.angles : rand(t.angles)

# Transformation matrix centered in the item's bounds
function gettfmmatrix(t::Rotate, bounds, param)
    return recenter(RotMatrix(param * (pi / 180)), center(bounds))
end

Base.:(|>)(::Rotate90, at::AbstractAffineTransform) = Rotate(90) |> at
Base.:(|>)(at::AbstractAffineTransform, ::Rotate90) = at |> Rotate(90)

Base.:(|>)(::Rotate180, at::AbstractAffineTransform) = Rotate(180) |> at
Base.:(|>)(at::AbstractAffineTransform, ::Rotate180) = at |> Rotate(180)

Base.:(|>)(::Rotate270, at::AbstractAffineTransform) = Rotate(270) |> at
Base.:(|>)(at::AbstractAffineTransform, ::Rotate270) = at |> Rotate(270)


# Utilities

"""
    pickfromrange(rng::AbstractRange, rnd::AbstractFloat)

Given a number 0 <= `rnd` <= 1 that's generated by `rand()`,
pick the element in range `rng` that is closest to it relative
to `rng`'s length
"""
pickfromrange(rng::AbstractRange, rnd::AbstractFloat) = rng[Int(ceil(length(rng) * rnd))]

getscale(scaley, scalex) = LinearMap([scaley 0; 0 scalex])

import ImageTransformations: center, _center
using StaticArrays: SVector

center(bounds::NTuple{N}) where N = SVector{N}(map(b -> ImageTransformations._center(1:b), bounds))
