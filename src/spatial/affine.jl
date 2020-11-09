



"""
    abstract type AbstractAffine <: Transform

Abstract supertype for affine transformations.

## Interface

An `AbstractAffine` transform "`T`" has to implement:

- [`getaffine`](#)`(tfm::T, getbounds(item), getrandstate(tfm))`

To be able to apply affine transformations an `Item` type `I` must
implement:

- [`getbounds`](#)`(item::MyItem)` returns the spatial bounds of an item,
  e.g. `size(img)` for an image array
- `applyaffine(item::MyItem, A)::MyItem` applies transformation matrix `A`
  (constructed with `getaffine`) to `item` and returns an item of the same
  type
"""
abstract type AbstractAffine <: Transform end


"""
    getaffine(tfm, bounds, randstate)

Return an affine transformation matrix, see
[CoordinateTransformations.jl](https://github.com/JuliaGeometry/CoordinateTransformations.jl).

Takes into account the `bounds` of the item it is applied to as well
as the `tfm`'s `randstate`.
"""
function getaffine end


"""

"""
function getbounds() end

getbounds(item::Image) = item.bounds
getbounds(item::Keypoints) = item.bounds
getbounds(wrapper::ItemWrapper) = getbounds(getwrapped(wrapper))
getbounds(a::AbstractMatrix) = makebounds(a)

affinetype(item) = Float32
affinetype(keypoints::Keypoints{N, T}) where {N, T} = T

function apply(tfm::AbstractAffine, item::Item; randstate=getrandstate(tfm))
    A = getaffine(tfm, getbounds(item), randstate, affinetype(item))
    return applyaffine(item, A)
end


# Image implementation

function applyaffine(item::Image{N, T}, A, crop=nothing) where {N, T}
    if crop isa Tuple
        newdata = warp(itemdata(item), inv(A), crop, zero(T))
        # TODO: add correct bounds for offset image
        return Image(newdata)
    else
        newdata = warp(itemdata(item), inv(A), zero(T))
        newbounds = A.(getbounds(item))
        return Image(newdata, newbounds)
    end
end


# Keypoints implementation

function applyaffine(keypoints::Keypoints{N, T}, A, crop = nothing) where {N, T}
    if isnothing(crop)
        newbounds = A.(getbounds(keypoints))
    else
        newbounds = makebounds(length.(crop), T)
    end
    return Keypoints(
        mapmaybe(A, keypoints.data),
        newbounds
    )
end

struct Affine <: AbstractAffine
    A
end

getaffine(tfm::Affine, bounds, randstate, T = Float32) = tfm.A




"""
    ComposedAffine(transforms)

Composes several affine transformations.

Due to associativity of affine transformations, the transforms can be
combined before applying, leading to large performance improvements.

`compose`ing multiple `AbstractAffine`s automatically
creates a `ComposedAffine`.
"""
struct ComposedAffine <: AbstractAffine
    transforms::NTuple{N,AbstractAffine} where N
end

getrandstate(composed::ComposedAffine) = getrandstate.(composed.transforms)


function getaffine(composed::ComposedAffine, bounds, randstate, T = Float32)
    A_all = IdentityTransformation()
    for (tfm, r) in zip(composed.transforms, randstate)
        A = getaffine(tfm, bounds, r, T)
        bounds = A.(bounds)
        A_all = A ∘ A_all
    end
    return A_all
end

compose(tfm1::AbstractAffine, tfm2::AbstractAffine) =
    ComposedAffine((tfm1, tfm2))
compose(cat::ComposedAffine, tfm::AbstractAffine) =
    ComposedAffine((cat.transforms..., tfm))
compose(tfm::AbstractAffine, cat::ComposedAffine) =
    ComposedAffine((tfm, cat.transforms))



mapmaybe(f, a) = map(x -> isnothing(x) ? nothing : f(x), a)
mapmaybe!(f, dest, a) = map!(x -> isnothing(x) ? nothing : f(x), dest, a)
