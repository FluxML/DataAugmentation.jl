abstract type CropFrom end

struct CropFromOrigin <: CropFrom end
struct CropFromCenter <: CropFrom end
struct CropFromRandom <: CropFrom end


abstract type Crop <: Transform end

@with_kw struct CropFixed <: Crop
    size::Tuple{Int,Int}
    from::CropFrom = CropFromOrigin()
end

CropFixed(h::Int, w::Int; from=CropFromOrigin()) = CropFixed((h, w), from)



struct CropRatio <: Crop
    ratios
    from::CropFrom
    CropRatio(ratios, from=CropFromOrigin()) = new(ratios, from)
end

getrandstate(::Crop) = (rand(), rand())


"""
    cropindices(croptfm::Crop, bounds, randstate)
    cropindices(size, from::CropFrom, bounds, randstate)

# Example:

```julia
cropindices((25, 25), CropFromOrigin(), [SVector(1, 1), SVector(50, 50)], nothing) == (1:25, 1:25)
```
"""
cropindices(crop::CropFixed, bounds, randstate) = cropindices(crop.size, crop.from, bounds, randstate)


function cropindices(crop::CropRatio, bounds, randstate)
    h, w = length.(index_ranges(bounds))
    sz = ceil.(Int, (h, w) .* crop.ratios)
    cropindices(sz, crop.from, bounds, randstate)
end


function cropindices(sz, ::CropFromRandom, bounds, r::Tuple{Float64,Float64})
    ranges = index_ranges(bounds)
    mins = minimum.(ranges)
    maxs = maximum.(ranges)
    lengths = length.(ranges)
    ds = lengths .- sz
    starts = floor.(Int, r .* ds) .+ mins
    ends = starts .+ sz .- 1
    return Tuple(start:end_ for (start, end_) in zip(starts, ends))
end

function cropindices(sz, ::CropFromCenter, bounds, _)
    return cropindices(sz, CropFromRandom(), bounds, (.5, .5))
end

function cropindices(sz, ::CropFromOrigin, bounds, _)
    return cropindices(sz, CropFromRandom(), bounds, (0., 0.))
end


"""
    getcropsizes(t::Crop, ::Item)

Compute the indices to crop to, e.g. `(1:100, 1:100)`.
"""
getcropsizes(t::CropFixed, ::Item) = t.sizes
function getcropsizes(t::CropRatio, item::Item)
    r1, r2 = index_ranges(getbounds(item))

    return (floor(Int, r1[begin]):floor(Int, r1[begin] + t.ratios[1] * (r1[end] - r1[begin])),
        floor(Int, r2[begin]):floor(Int, r2[begin] + t.ratios[2] * (r2[end] - r2[begin])),)
end

Crop(args...) = CropFixed(args...)


"""
    CroppedAffine(transform, croptransform)

Applies an affine `transform` and crops with `croptransform`, such
that `getbounds(t(item)) == getcrop(croptransform)`

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
function getaffine(tfm::CroppedAffine, bounds, randstate)
    randtfm, randcrop = randstate
    A = getaffine(tfm.transform, bounds, randstate)
    if tfm.fixcorner
        newbounds = A.(bounds)
        indices = cropindices(tfm.croptransform, newbounds, randcrop)
        A = Translation(-indices[1][begin]+1, -indices[2][begin]+1) ∘ A
    end
    return A
end

function apply(tfm::CroppedAffine, item::Item; randstate=getrandstate(tfm))
    tfmr, cropr = randstate
    A = getaffine(tfm, getbounds(item), randstate)
    newbounds = A.(getbounds(item))
    indices = cropindices(tfm.croptransform, newbounds, cropr)
    return applyaffine(item, A, indices)
end


function apply!(buffer, tfm::AbstractAffine, item::Item; randstate=getrandstate(tfm))
    tfmr, cropr = randstate
    A = getaffine(tfm, getbounds(item), tfmr)
    indices = cropindices(tfm.croptransform, getbounds(item), cropr)
    return applyaffine!(buffer, item, A, indices)
end

function applyaffine!(buffer::Image, item::Image, A, cropsizes)
    warp!(buffer.data, box_extrapolation(item.data), A, cropsizes)
end

compose(at::AbstractAffine, ct::Crop) = CroppedAffine(at, ct)

compose(cat::CroppedAffine, ct::Crop) = CroppedAffine(cat.transform, ct)

compose(cat::CroppedAffine, at::AbstractAffine) = CroppedAffine(cat.transform |> at, cat.croptransform)

compose(at::AbstractAffine, cat::CroppedAffine) = CroppedAffine(at |> cat.transform, cat.croptransform)

compose(cat1::CroppedAffine, cat2::CroppedAffine) =
    CroppedAffine(cat1.transform |> cat2.transform, cat2.croptransform)
