"""
    ElasticTransform(alpha, sigma = 5)
    ElasticTransform(; alpha = 25, sigma = 5)

Apply a smooth random elastic deformation to 2D spatial data. `alpha` controls
the maximum displacement in pixels and `sigma` controls the Gaussian smoothing
scale; either may be a scalar or a `(row, column)` pair. The same deformation
is used for images, masks, and keypoints. The sampled field is reduced when
necessary to keep its displacement map contractive, which makes the backward
map used for image sampling well-defined and numerically invertible.
"""
struct ElasticTransform <: ProjectiveTransform
    alpha::NTuple{2, Float64}
    sigma::NTuple{2, Float64}

    function ElasticTransform(alpha, sigma = 5)
        a = _elastic_pair("alpha", alpha; positive=false)
        s = _elastic_pair("sigma", sigma; positive=true)
        return new(a, s)
    end
end

ElasticTransform(; alpha = 25, sigma = 5) = ElasticTransform(alpha, sigma)

_elastic_pair(name, x::Real; positive) = _elastic_pair(name, (x, x); positive)
function _elastic_pair(name, x::Tuple{<:Real,<:Real}; positive)
    values = (Float64(x[1]), Float64(x[2]))
    all(isfinite, values) || throw(ArgumentError("$name values must be finite"))
    all(v -> positive ? v > 0 : v >= 0, values) ||
        throw(ArgumentError(positive ? "$name values must be positive" : "$name values must be nonnegative"))
    return values
end
_elastic_pair(name, x; positive) = throw(ArgumentError("$name must be a scalar or a (row, column) pair"))

getrandstate(::ElasticTransform) = rand(UInt64)

struct _ElasticMap <: CoordinateTransformations.Transformation
    displacement::Array{Float64,3}
    origin::SVector{2,Float64}
end

struct _ElasticInverse <: CoordinateTransformations.Transformation
    map::_ElasticMap
end

function _elastic_displacement(map::_ElasticMap, point::StaticVector{2})
    field = map.displacement
    h, w = size(field, 1), size(field, 2)
    y = clamp(Float64(point[1]) - map.origin[1] + 1, 1.0, Float64(h))
    x = clamp(Float64(point[2]) - map.origin[2] + 1, 1.0, Float64(w))
    y0, x0 = floor(Int, y), floor(Int, x)
    y1, x1 = min(y0 + 1, h), min(x0 + 1, w)
    fy, fx = y - y0, x - x0
    return SVector{2,Float64}(
        (1-fy) * ((1-fx) * field[y0,x0,1] + fx * field[y0,x1,1]) +
            fy * ((1-fx) * field[y1,x0,1] + fx * field[y1,x1,1]),
        (1-fy) * ((1-fx) * field[y0,x0,2] + fx * field[y0,x1,2]) +
            fy * ((1-fx) * field[y1,x0,2] + fx * field[y1,x1,2]),
    )
end

(map::_ElasticMap)(point::StaticVector{2}) = SVector{2,Float64}(point) - _elastic_displacement(map, point)

function (inverse::_ElasticInverse)(point::StaticVector{2})
    target = SVector{2,Float64}(point)
    source = target
    # The field is scaled so this fixed-point inversion is a contraction.
    for _ in 1:100
        next = target + _elastic_displacement(inverse.map, source)
        maximum(abs, next - source) <= 1e-10 && return next
        source = next
    end
    return source
end

Base.inv(map::_ElasticMap) = _ElasticInverse(map)
Base.inv(inverse::_ElasticInverse) = inverse.map

function _elastic_lipschitz(field)
    h, w = size(field, 1), size(field, 2)
    bound = 0.0
    for component in 1:2
        dy = h > 1 ? maximum(abs, diff(view(field, :, :, component); dims=1)) : 0.0
        dx = w > 1 ? maximum(abs, diff(view(field, :, :, component); dims=2)) : 0.0
        bound = max(bound, dy + dx)
    end
    return bound
end

function getprojection(tfm::ElasticTransform, bounds::Bounds{2}; randstate=getrandstate(tfm))
    h, w = length.(bounds.rs)
    (iszero(tfm.alpha[1]) && iszero(tfm.alpha[2]) || h == 0 || w == 0) &&
        return IdentityTransformation()
    rng = Xoshiro(randstate)
    field = Array{Float64}(undef, h, w, 2)
    for component in 1:2
        noise = rand(rng, h, w) .* 2 .- 1
        row_smoothed = similar(noise)
        smoothed = similar(noise)
        row_kernel = _elastic_kernel_size(tfm.sigma[1], h)
        column_kernel = _elastic_kernel_size(tfm.sigma[2], w)
        _gaussian_pass!(row_smoothed, noise, 1, tfm.sigma[1], row_kernel)
        _gaussian_pass!(smoothed, row_smoothed, 2, tfm.sigma[2], column_kernel)
        field[:, :, component] .= tfm.alpha[component] .* smoothed
    end
    # Bound the bilinear field's componentwise Lipschitz constant below one.
    # This guarantees convergence of x ↦ y - d(x) for the forward-map inverse.
    lipschitz = _elastic_lipschitz(field)
    lipschitz > 0.45 && (field .*= 0.45 / lipschitz)
    # The projective interface expects its inverse for image sampling. Returning
    # the inverse map here keeps that per-pixel path to one bilinear lookup; the
    # iterative contraction is reserved for forward annotation coordinates.
    return _ElasticInverse(_ElasticMap(field, SVector{2,Float64}(first.(bounds.rs))))
end

_elastic_kernel_size(sigma, n) = 2ceil(Int, min(3sigma, max(n - 1, 1))) + 1

projectionbounds(::ElasticTransform, P, bounds::Bounds{2}; randstate=nothing) = bounds
