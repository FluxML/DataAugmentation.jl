"""
    WarpAffine(σ = 0.1) <: ProjectiveTransform

An affine warp calculated by randomly moving `N + 1` corners of an `N`-dimensional
item, which is three corners of an image. Similar to a random translation, shear and
rotation. The corners move by up to `σ` times the geometric mean of the side lengths.
"""
struct WarpAffine <: ProjectiveTransform
    σ
end


getrandstate(::WarpAffine) = abs(rand(Int))


function getprojection(
        tfm::WarpAffine,
        bounds::Bounds{N};
        randstate = getrandstate(tfm)) where N
    T = Float32
    rng = Random.seed!(Random.MersenneTwister(), randstate)
    scale = _warpscale(bounds)

    srcps = _warpcorners(rng, bounds)
    offsets = rand(rng, SVector{N, T}, N + 1)
    offsets = map(v -> (v .* (2one(T)) .- one(T)) .* convert(T, scale * tfm.σ), offsets)

    return threepointwarpaffine(srcps, srcps .+ offsets)
end

_warpscale(bounds::Bounds{2}) = sqrt(prod(length.(bounds.rs)))
_warpscale(bounds::Bounds{N}) where N = prod(float, length.(bounds.rs))^(1 / N)

# Any 3 of the 4 corners of a rectangle are affinely independent.
function _warpcorners(rng, bounds::Bounds{2})
    return shuffle(rng, SVector{2, Float32}.(corners(bounds)))[1:3]
end

# The corners of a box are not: 4 corners of a cube can lie on one face. A corner
# and its neighbors along each axis are.
function _warpcorners(rng, bounds::Bounds{N}) where N
    low = SVector{N, Float32}(first.(bounds.rs))
    high = SVector{N, Float32}(last.(bounds.rs))
    flip = SVector{N, Bool}(ntuple(_ -> rand(rng, Bool), Val(N)))
    corner = SVector{N, Float32}(ifelse.(flip, high, low))
    neighbors = ntuple(d -> setindex(corner, flip[d] ? low[d] : high[d], d), Val(N))
    return [corner, neighbors...]
end

# Adapted from
"""
    threepointwarpaffine(srcps, dstps)

Calculate an affine `CoordinateTransformations.AffineMap`
from `N + 1` source points to `N + 1` destination points in `N` dimensions,
which are three points for 2D data. The points must be affinely independent.

Adapted from  [CoordinateTransformations.jl#30](https://github.com/JuliaGeometry/CoordinateTransformations.jl/issues/30#issuecomment-610337378).
"""
function threepointwarpaffine(
        srcps::T, dstps::T) where {N, V, T<:AbstractArray{<:SVector{N,V}}}
    X = vcat(hcat(dstps...), ones(1, N + 1))'
    Y = hcat(srcps...)'
    c = (X \ Y)'
    A = SMatrix{N, N, V}(c[:, 1:N])
    b = SVector{N, V}(c[:, N + 1])
    return AffineMap(A, b)
end


function corners(bounds::Bounds)
    (Tuple(x) for x in ImageTransformations.CornerIterator(CartesianIndices(bounds.rs)))
end
