
abstract type Scale <: AbstractAffine end

struct ScaleFixed{T<:Number} <: Scale
    size::Tuple{T, T}
end
struct ScaleRatio <: Scale
    ratios
end
ScaleRatio(fy::Number, fx::Number) = ScaleRatio((fy, fx))
ScaleRatio(f::Number) = ScaleRatio((f, f))

ScaleFixed(h::Int, w::Int) = ScaleFixed((h, w))

function getaffine(tfm::ScaleFixed, bounds, randstate)
    ratios = boundssize(bounds) ./ tfm.size
    return getaffine(ScaleRatio(ratios), bounds, randstate)
end

function getaffine(tfm::ScaleRatio, _, _)
    fy, fx = tfm.ratios
    return LinearMap(SMatrix{2, 2}([fy 0; 0 fx]))
end


"""
    ScaleKeepAspect(minglengths) <: Scale <: AbstractAffine
    ScaleKeepAspect(minglength)

Affine transformation that scales the shortest side of `item`
to `minlengths`, keeping the original aspect ratio.
"""
struct ScaleKeepAspect <: Scale
    minlengths::Tuple{Int, Int}
end

ScaleKeepAspect(minlength::Int) = ScaleKeepAspect((minlength, minlength))

function getaffine(tfm::ScaleKeepAspect, bounds, randstate)
    l1, l2 = tfm.minlengths
    ratio = maximum((l1, l2) ./ boundssize(bounds))
    getaffine(ScaleRatio((ratio, ratio)), bounds, randstate)
end
