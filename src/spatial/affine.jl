

getbounds(item::Image) = item.bounds
getbounds(item::Keypoints) = item.bounds
getbounds(wrapper::ItemWrapper) = getbounds(getwrapped(wrapper))
getbounds(a::AbstractMatrix) = makebounds(a)




abstract type AbstractAffine <: Transform end

function apply(tfm::AbstractAffine, item::Item; randstate=getrandstate(tfm))
    A = getaffine(tfm, getbounds(item), randstate)
    return applyaffine(item, A)
end


# Image implementation

function applyaffine(item::Image{C}, A, crop=nothing) where {C}
    if crop isa Tuple
        # FIXME: why `parent`?
        newdata = warp(itemdata(item), inv(A), crop, zero(C))
        return Image(
            newdata
        )
    else
        newdata = warp(itemdata(item), inv(A), zero(C))
        newbounds = A.(getbounds(item))

        return Image(newdata, newbounds)
    end
end


# Keypoints implementation

function applyaffine(keypoints::Keypoints, A, crop = nothing)::Keypoints
    if isnothing(crop)
        newbounds = A.(getbounds(keypoints))
    else
        newbounds = makebounds(crop...)
    end
    return Keypoints(
        mapmaybe(A, keypoints.data),
        newbounds
    )
end

struct Affine <: AbstractAffine
    A
end

getaffine(tfm::Affine, _, _) = tfm.A




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

getrandstate(cat::ComposedAffine) = getrandstate.(cat.transforms)


function getaffine(cat::ComposedAffine, bounds, randstate)
    A_all = IdentityTransformation()
    for (tfm, r) in zip(cat.transforms, randstate)
        A = getaffine(tfm, bounds, r)
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
