"""
    RandomAffine(degrees; translate = (0, 0), scale = (1, 1), shear = 0)

Apply a random 2D affine warp on the original canvas. A scalar `degrees` samples
a rotation symmetrically around zero; a pair gives the angle limits directly.
`translate` gives maximum fractions of the height and width. Sample a uniform
scale from the two limits in `scale`. A scalar `shear` samples horizontal shear
symmetrically, a pair sets its limits, and four values give horizontal then
vertical shear limits. Angles are in degrees.
"""
struct RandomAffine <: ProjectiveTransform
    degrees::Tuple{Float64, Float64}
    translate::Tuple{Float64, Float64}
    scale::Tuple{Float64, Float64}
    shear::NTuple{4, Float64}
end

_angle_limits(angle::Real) = (-abs(Float64(angle)), abs(Float64(angle)))
_angle_limits(angles::Tuple{Real, Real}) = Float64.(angles)

function RandomAffine(degrees; translate::Tuple{Real, Real} = (0, 0),
                      scale::Tuple{Real, Real} = (1, 1), shear = 0)
    d = _angle_limits(degrees)
    -Inf < d[1] <= d[2] < Inf || throw(ArgumentError("degrees must be finite and ordered"))
    all(x -> 0 <= x <= 1, translate) || throw(ArgumentError("translate fractions must be in [0, 1]"))
    sc = Float64.(scale)
    0 < sc[1] <= sc[2] < Inf || throw(ArgumentError("scale limits must be positive, finite, and ordered"))
    sh = shear isa NTuple{4, Real} ? Float64.(shear) : (_angle_limits(shear)..., 0.0, 0.0)
    -90 < sh[1] <= sh[2] < 90 && -90 < sh[3] <= sh[4] < 90 ||
        throw(ArgumentError("shear limits must be ordered and between -90 and 90 degrees"))
    return RandomAffine(d, Float64.(translate), sc, sh)
end

getrandstate(::RandomAffine) = rand(UInt64)

function getprojection(tfm::RandomAffine, bounds::Bounds{2}; randstate = getrandstate(tfm))
    rng = Xoshiro(randstate)
    h, w = length.(bounds.rs)
    s, c = sincosd(_sample_between(rng, tfm.degrees...))
    dy, dx = (h, w) .* tfm.translate
    shift = SVector(round(Int, _sample_between(rng, -dy, dy)), round(Int, _sample_between(rng, -dx, dx)))
    scale = _sample_between(rng, tfm.scale...)
    hx = tand(_sample_between(rng, tfm.shear[1], tfm.shear[2]))
    hy = tand(_sample_between(rng, tfm.shear[3], tfm.shear[4]))
    rotation = @SMatrix [c s; -s c]
    shear = @SMatrix [1+hx*hy -hy; -hx 1]
    midpoint = SVector(mean.(bounds.rs))
    return Translation(shift) ∘ recenter(LinearMap(scale * (rotation * shear)), midpoint)
end

projectionbounds(::RandomAffine, P, bounds::Bounds{2}; randstate = nothing) = bounds

"""
    RandomPerspective(distortion_scale = 0.5; p = 0.5)

With probability `p`, warp a 2D canvas by moving its four corners inward.
`distortion_scale` runs from zero to one and controls how far each corner can
move along each axis. The output keeps the original canvas bounds.
"""
struct RandomPerspective <: ProjectiveTransform
    distortion_scale::Float64
    p::Float64

    function RandomPerspective(distortion_scale::Real = 0.5; p::Real = 0.5)
        0 <= distortion_scale <= 1 || throw(ArgumentError("distortion_scale must be in [0, 1]"))
        0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1]"))
        return new(distortion_scale, p)
    end
end

getrandstate(::RandomPerspective) = rand(UInt64)

struct _Homography <: CoordinateTransformations.Transformation
    matrix::SMatrix{3, 3, Float64, 9}
    origin::SVector{2, Float64}
    span::SVector{2, Float64}
end

function (H::_Homography)(point::StaticVector{2})
    y, x = (point - H.origin) ./ H.span
    a, b, w = H.matrix * SVector(y, x, 1.0)
    denom = abs(w) < eps(Float64) ? copysign(eps(Float64), w) : w
    return H.origin + H.span .* (SVector(a, b) / denom)
end

Base.inv(H::_Homography) = _Homography(inv(H.matrix), H.origin, H.span)

function _homography(src, dst)
    M = Matrix{Float64}(undef, 8, 8)
    v = Vector{Float64}(undef, 8)
    for i in 1:4
        r, c = src[i]
        rr, cc = dst[i]
        j = 2i - 1
        M[j, :] .= (r, c, 1, 0, 0, 0, -rr * r, -rr * c)
        M[j+1, :] .= (0, 0, 0, r, c, 1, -cc * r, -cc * c)
        v[j:j+1] .= (rr, cc)
    end
    x = M \ v
    return @SMatrix [x[1] x[2] x[3]; x[4] x[5] x[6]; x[7] x[8] 1.0]
end

function _clockwise_turn(a, b, c)
    u, v = b - a, c - b
    return widemul(u[2], v[1]) > widemul(u[1], v[2])
end

function getprojection(tfm::RandomPerspective, bounds::Bounds{2}; randstate = getrandstate(tfm))
    h, w = length.(bounds.rs)
    (h < 2 || w < 2) && return IdentityTransformation()
    rng = Xoshiro(randstate)
    rand(rng) < tfm.p || return IdentityTransformation()
    dy = floor(Int, tfm.distortion_scale * (h ÷ 2))
    dx = floor(Int, tfm.distortion_scale * (w ÷ 2))
    dy == dx == 0 && return IdentityTransformation()
    source = (SVector(0, 0), SVector(0, 1), SVector(1, 1), SVector(1, 0))
    span = SVector{2, Float64}(h - 1, w - 1)
    for _ in 1:8
        pixels = map(source) do corner
            y = rand(rng, 0:dy)
            x = rand(rng, 0:dx)
            return SVector(iszero(corner[1]) ? y : h - 1 - y,
                           iszero(corner[2]) ? x : w - 1 - x)
        end
        all(i -> _clockwise_turn(pixels[i], pixels[mod1(i + 1, 4)], pixels[mod1(i + 2, 4)]), 1:4) ||
            continue
        destination = map(p -> p ./ span, pixels)
        H = _homography(source, destination)
        abs(det(H)) > eps(Float64) * maximum(abs, H)^3 || continue
        return _Homography(H, SVector{2, Float64}(first.(bounds.rs)), span)
    end
    return IdentityTransformation()
end

projectionbounds(::RandomPerspective, P, bounds::Bounds{2}; randstate = nothing) = bounds
