# AbstractAffineTransform interface

"""
    AbstractAffineTransform

Abstract supertype for affine transformations

# Supporting custom `AbstractAffineTransform`s and `Item`s

## `AbstractAffineTransform` interface

- `getaffine(t::MyAffineTransform, getbounds(item), getparam(t))`
  should return a transformation matrix (s. `CoordinateTransformations.jl`)

## Item interface

- `getbounds(item::MyItem)::Tuple` returns the spatial bounds of an item,
  e.g. `size(img)` for an image array
- `applyaffine(item::MyItem, A)::MyItem` applies transformation matrix `A`
  (constructed with `getaffine`) to `item` and returns an item of the same
  type
"""
abstract type AbstractAffine <: Transform end

function apply(tfm::AbstractAffine, item::Item, param)
    A = getaffine(tfm, getbounds(item), param)
    return applyaffine(item, A)
end

"""
    applyaffine(item::Item, A, crop = nothing)

Applies an affine transformation `A` to `item`, optionally cropping
to tuple `crop`.
"""
function applyaffine(item::Item, A, crop = nothing) end

# Item interface implementation

"""
    applyaffine(item, tfm, crop = nothing)

Applies affine transformation matrix `tfm` to `item`, optionally cropping to window
of size `crop`
"""
function applyaffine(item::Image{C}, A, crop::Union{Nothing,Tuple} = nothing) where {C}
    if crop isa Tuple
        indices = (1:crop[1], 1:crop[2])
        return Image(warp(parent(item.data), inv(A), indices, zero(C)))
    else
        return Image(warp(parent(item.data), inv(A), zero(C)))
    end
end

function applyaffine(keypoints::Keypoints, A, crop::Union{Nothing,Tuple} = nothing)::Keypoints
    keypoints_ = mapmaybe(A, keypoints.data)
    if isnothing(crop)
        # TODO: calculate actual bounds
        bounds_ = keypoints.bounds
    else
        bounds_ = crop
    end
    return Keypoints(
        keypoints_,
        bounds_
    )
end


# AbstractAffineTransform composition

"""
    ComposedAffine(transforms)

Composes several affine transformations.

Due to associativity of affine transformations, the transforms can be
combined before applying, leading to large performance improvements.

`compose`ing multiple `AbstractAffineTransformation`s automatically
creates a `ComposedAffine`.
"""
struct ComposedAffine <: AbstractAffine
    transforms::NTuple{N,AbstractAffine} where N
end

getparam(cat::ComposedAffine) = Tuple(getparam(t) for t in cat.transforms)


function getaffine(cat::ComposedAffine, bounds, params::Tuple, T = Float32)
    A = IdentityTransformation()
    for (t, param) in zip(cat.transforms, params)
        A = getaffine(t, bounds, param, T) ∘ A
    end
    return A
end

compose(tfm1::AbstractAffine, tfm2::AbstractAffine) =
    ComposedAffine((tfm1, tfm2))
compose(cat::ComposedAffine, tfm::AbstractAffine) =
    ComposedAffine((cat.transforms..., tfm))
compose(tfm::AbstractAffine, cat::ComposedAffine) =
    ComposedAffine((tfm, cat.transforms))


# Cropping transforms

"""
    CroppedAffine(transform, croptransform)

Applies an affine `transform` and crops with `croptransform`, such
that `getbounds(t(item)) == getcrop(croptransform)`

This wrapper leads to performance improvements when warping an
image, since only the indices within the bounds of `crop` need
to be evaluated.

`compose`ing any `AbstractAffine` with a `CropTransform`
constructs a `CroppedAffine`.
"""
struct CroppedAffine <: AbstractAffine
    transform::AbstractAffine
    croptransform::Crop
end

getparam(t::CroppedAffine) = getparam(t.transform)

function apply(t::CroppedAffine, item::Item, param)
    tfm = getaffine(t.transform, getbounds(item), param)
    return applyaffine(item, tfm, getcrop(t.croptransform, item))
end

compose(at::AbstractAffine, ct::Crop) = CroppedAffine(at, ct)

compose(cat::CroppedAffine, ct::Crop) = CroppedAffine(cat.transform, ct)

compose(cat::CroppedAffine, at::AbstractAffine) = CroppedAffine(cat.transform |> at, cat.croptransform)

compose(at::AbstractAffine, cat::CroppedAffine) = CroppedAffine(at |> cat.transform, cat.croptransform)

compose(cat1::CroppedAffine, cat2::CroppedAffine) =
    CroppedAffine(cat1.transform |> cat2.transform, cat2.croptransform)


# AffineTransformations implementations

"""
    Affine

Applies static transformation matrix `tfm` to an item
"""
struct Affine <: AbstractAffine
    tfm
end
getaffine(t::Affine, bounds, param) = t.tfm

abstract type AbstractResize <: AbstractAffine end

struct RandomResize <: AbstractResize
    size
end
struct CenterResize <: AbstractResize
    size
end


getparam(tfm::RandomResize) = (rand(), rand())
getparam(tfm::CenterResize) = (1/2, 1/2)


# TODO: clean up naming
function getaffine(t::AbstractResize, bounds, param, T = Float32)
    h, w = t.size
    k, l = bounds
    factor = min(k / h, l / w)
    scaletfm = getscale(1/factor, 1/factor)

    h_, w_ = Int.(floor.(bounds ./ factor))

    ry, rx = param
    ty = convert(T, pickfromrange(0:max(0, h_ - h), ry))
    tx = convert(T, pickfromrange(0:max(0, w_ - w), rx))

    translatetfm = Translation(-ty, -tx)

    return translatetfm ∘ scaletfm
end

RandomResizeCrop(crop) = RandomResize(crop) |> Crop(crop)
CenterResizeCrop(crop) = CenterResize(crop) |> Crop(crop)

# TODO: add Translate

"""
    Scale
"""
Scale(factors::Tuple) = Affine(getscale(factors...))
Scale(factor) = Scale((factor, factor))

ScaleCrop(factors::Tuple) = Scale(factors) |> Crop(factors = factors)
ScaleCrop(factor) = ScaleCrop((factor, factor))


# AffineRotate

"""
    RotateTransform(angle)
    RotateTransform(angles)

Rotates an `Item` by `angle` degrees counter-clockwise using an affine
transformation.
If a vector or range `angles` of degrees is given, picks one at random.

See also [`Rotate90`](@ref), [`Rotate180`](@ref), [`Rotate270`](@ref) for
more efficient non-affine versions for those specific angles.
"""
struct Rotate <: AbstractAffine
    angles
end

# Pick a random angle
getparam(t::Rotate) = t.angles isa Number ? t.angles : rand(t.angles)

# Transformation matrix centered in the item's bounds
function getaffine(t::Rotate, bounds, param, T = Float32)
    angle = convert(T, param * (pi / 180))
    return recenter(RotMatrix(angle), center(bounds))
end

compose(::Rotate90, at::AbstractAffine) = Rotate(90) |> at
compose(at::AbstractAffine, ::Rotate90) = at |> Rotate(90)

compose(::Rotate180, at::AbstractAffine) = Rotate(180) |> at
compose(at::AbstractAffine, ::Rotate180) = at |> Rotate(180)

compose(::Rotate270, at::AbstractAffine) = Rotate(270) |> at
compose(at::AbstractAffine, ::Rotate270) = at |> Rotate(270)


# Utilities

"""
    pickfromrange(rng::AbstractRange, rnd::AbstractFloat)

Given a number 0 <= `rnd` <= 1 that's generated by `rand()`,
pick the element in range `rng` that is closest to it relative
to `rng`'s length
"""
pickfromrange(rng::AbstractRange, rnd::AbstractFloat) = rng[Int(ceil(length(rng) * rnd))]

getscale(scaley, scalex) = LinearMap(SMatrix{2, 2}([scaley 0; 0 scalex]))


ImageTransformations.center(bounds::NTuple{N}) where N = SVector{N}(map(b -> ImageTransformations._center(1:b), bounds))
