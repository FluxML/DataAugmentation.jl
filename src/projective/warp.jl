"""
    WarpAffine(σ = 0.1) <: ProjectiveTransform

A three-point affine warp calculated by randomly moving 3 corners
of an item. Similar to a random translation, shear and rotation.
"""
struct WarpAffine <: ProjectiveTransform
    σ
end


getrandstate(::WarpAffine) = abs(rand(Int))


function getprojection(
        tfm::WarpAffine,
        bounds::Bounds{2};
        randstate = getrandstate(tfm))
    T = Float32
    rng = Random.seed!(Random.MersenneTwister(), randstate)
    scale = sqrt(prod(length.(bounds.rs)))

    srcps = shuffle(rng, SVector{2, T}.(corners(bounds)))[1:3]
    offsets = rand(rng, SVector{2, T}, 3)
	offsets = map(v -> (v .* (2one(T)) .- one(T)) .* convert(T, scale * tfm.σ), offsets)

	return threepointwarpaffine(srcps, srcps .+ offsets)
end


# Adapted from
"""
    threepointwarpaffine(srcps, dstps)

Calculate an affine `CoordinateTransformations.LinearMap`
from 3 source points to 3 destination points.

Adapted from  [CoordinateTransformations.jl#30](https://github.com/JuliaGeometry/CoordinateTransformations.jl/issues/30#issuecomment-610337378).
"""
function threepointwarpaffine(
        srcps::T, dstps::T) where {V, T<:AbstractArray{<:SVector{2,V}}}
	X = vcat(hcat(dstps...), ones(1,3))'
    Y = hcat(srcps...)'
    c = (X \ Y)'
    A = SMatrix{2, 2, V}(c[:, 1:2])
    b = SVector{2, V}(c[:, 3])
    return AffineMap(A, b)
end


function corners(bounds::Bounds)
    (Tuple(x) for x in ImageTransformations.CornerIterator(CartesianIndices(bounds.rs)))
end
