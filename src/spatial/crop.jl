abstract type CropFrom end

struct FromOrigin <: CropFrom end
struct FromCenter <: CropFrom end
struct FromRandom <: CropFrom end


abstract type Crop <: Transform end

@with_kw struct CropFixed <: Crop
    size::Tuple{Int,Int}
    from::CropFrom = FromOrigin()
end

CropFixed(sz; from=FromOrigin()) = CropFixed(sz, from)
CropFixed(h::Int, w::Int; from=FromOrigin()) = CropFixed((h, w), from)



struct CropRatio <: Crop
    ratios
    from::CropFrom
    CropRatio(ratios, from=FromOrigin()) = new(ratios, from)
end


struct CropIndices <: Crop
    indices
end

"""
    CropDivisible(factor, [from])

Does not remove any pixels, but pads width and height so they're
divisible by `factor`
"""
struct CropDivisible <: Crop
    factor::Int
    from::CropFrom
    CropDivisible(factor, from=FromOrigin()) = new(factor, from)
end


getrandstate(::Crop) = (rand(), rand())


"""
    cropindices(croptfm::Crop, bounds, randstate)
    cropindices(size, from::CropFrom, bounds, randstate)

# Example:

```julia
cropindices((25, 25), FromOrigin(), [SVector(1, 1), SVector(50, 50)], nothing) == (1:25, 1:25)
```
"""
cropindices(crop::CropFixed, bounds, randstate) = cropindices(crop.size, crop.from, bounds, randstate)

cropindices(crop::CropIndices, _, _) = crop.indices

function cropindices(crop::CropDivisible, bounds, randstate)
    sz = roundtodivisible.(length.(boundsranges(bounds)), crop.factor)
    cropindices(sz, crop.from, bounds, randstate)
end

roundtodivisible(x::Int, factor::Int) = (x ÷ factor + Bool(x % factor > 0)) * factor

function cropindices(crop::CropRatio, bounds, randstate)
    h, w = length.(boundsranges(bounds))
    sz = ceil.(Int, (h, w) .* crop.ratios)
    cropindices(sz, crop.from, bounds, randstate)
end


function cropindices(sz, ::FromRandom, bounds, r::Tuple{Float64,Float64})
    ranges = boundsranges(bounds)
    mins = minimum.(ranges)
    maxs = maximum.(ranges)
    lengths = length.(ranges)
    ds = lengths .- sz
    starts = floor.(Int, r .* ds) .+ mins
    ends = starts .+ sz .- 1
    return Tuple(start:end_ for (start, end_) in zip(starts, ends))
end

function cropindices(sz, ::FromCenter, bounds, _)
    return cropindices(sz, FromRandom(), bounds, (.5, .5))
end

function cropindices(sz, ::FromOrigin, bounds, _)
    return cropindices(sz, FromRandom(), bounds, (0., 0.))
end


Crop(args...) = Affine(CoordinateTransformations.IdentityTransformation()) |> CropFixed(args...)


"""
    CroppedAffine(transform, croptransform)

Applies an affine `transform` and crops with `croptransform`, such
that `getbounds(apply(transform, item)) == getcrop(croptransform)`

This wrapper leads to performance improvements when warping an
image, since only the indices within the bounds of `crop` need
to be evaluated.

`compose`ing any `AbstractAffine` with a `Crop`
constructs a `CroppedAffine`.
"""
struct CroppedAffine <: AbstractAffine
    transform::AbstractAffine
    croptransform::Crop
    fixcorner::Bool
    CroppedAffine(transform, croptransform, fixcorner = true) = new(
        transform, croptransform, fixcorner
    )
end


getrandstate(tfm::CroppedAffine) = getrandstate(tfm.transform), getrandstate(tfm.croptransform)


# TODO: take into account if not cropping from origin
function getaffine(tfm::CroppedAffine, bounds, randstate, T = Float32)
    randtfm, randcrop = randstate
    A = getaffine(tfm.transform, bounds, randstate, T)
    if tfm.fixcorner
        newbounds = A.(bounds)
        indices = cropindices(tfm.croptransform, newbounds, randcrop)
        A = Translation(-indices[1][begin]+1, -indices[2][begin]+1) ∘ A
    end
    return A
end

function apply(tfm::CroppedAffine, item::Item; randstate=getrandstate(tfm))
    tfmr, cropr = randstate
    A = getaffine(tfm, getbounds(item), randstate, affinetype(item))
    newbounds = A.(getbounds(item))
    indices = cropindices(tfm.croptransform, newbounds, cropr)
    return applyaffine(item, A, indices)
end


function apply!(buffer, tfm::CroppedAffine, item::Item; randstate=getrandstate(tfm))
    tfmr, cropr = randstate
    A = getaffine(tfm, getbounds(item), randstate)
    return applyaffine!(buffer, item, A)
end

function applyaffine!(buffer::Image, item::Image, A)
    warp!(buffer.data, box_extrapolation(item.data), inv(A))
    return buffer
end

compose(at::AbstractAffine, ct::Crop) = CroppedAffine(at, ct)

compose(cat::CroppedAffine, ct::Crop) = CroppedAffine(cat.transform, ct)

compose(cat::CroppedAffine, at::AbstractAffine) = CroppedAffine(cat.transform |> at, cat.croptransform)

compose(at::AbstractAffine, cat::CroppedAffine) = CroppedAffine(at |> cat.transform, cat.croptransform)

compose(cat1::CroppedAffine, cat2::CroppedAffine) =
    CroppedAffine(cat1.transform |> cat2.transform, cat2.croptransform)
