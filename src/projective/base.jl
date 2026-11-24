"""
    abstract type ProjectiveTransform <: Transform

Abstract supertype for projective transformations. See
"""
abstract type ProjectiveTransform <: Transform end


"""
    getprojection(tfm, bounds; randstate)

Create a projection for an item with spatial bounds `bounds`.
The projection should be a `CoordinateTransformations.Transformation`.
See [CoordinateTransformations.jl](https://github.com/JuliaGeometry/CoordinateTransformations.jl)
"""
function getprojection end


"""
    Bounds(sz)
    Bounds(rs)

Spatial bounds of an `N`-dimensional item, stored as one index range per dimension.
`Bounds(sz)` for a size tuple `sz` gives the ranges `1:sz[i]`; `rs` is a tuple of
ranges.
"""
struct Bounds{N}
    rs::NTuple{N, UnitRange{Int}}
end

function Base.show(io::IO, bounds::Bounds{N}) where N
    print(io, "Bounds(")
    for (i, r) in enumerate(bounds.rs)
        print(io, first(r), ':', last(r))
        i == N || print(io, '×')
    end
    print(io, ")")
end

Bounds(sz::NTuple{N, <:Integer}) where N = Bounds(Tuple(1:n for n in sz))
Bounds(axes::NTuple{N, AbstractUnitRange}) where N = Bounds(map(r -> Int(first(r)):Int(last(r)), axes))
Bounds(::Tuple{}) = Bounds{0}(())


Base.:(==)(bs1::Bounds, bs2::Bounds) = false
function Base.:(==)(bs1::Bounds{N}, bs2::Bounds{N}) where N
    c1 = all((first(bs1.rs[i]) == first(bs2.rs[i])) for i in 1:N)
    c2 = all((last(bs1.rs[i]) == last(bs2.rs[i])) for i in 1:N)
    return c1 && c2
end



"""
    transformbounds(bounds, P)

Apply `CoordinateTransformations.Transformation` to `bounds`.
"""
function transformbounds(bounds::Bounds, P::CoordinateTransformations.Transformation)
    return Bounds(_autorange(CartesianIndices(bounds.rs), P))
end

"""
    getbounds(item)

Return the spatial bounds of `item`. For a 2D-image (`Image{2}`)
the bounds are the 4 corners of the bounding rectangle. In general,
for an N-dimensional item, the bounds are a vector of the N^2 corners
of the N-dimensional hypercube bounding the data.
"""
function getbounds end

getbounds(wrapper::ItemWrapper) = getbounds(getwrapped(wrapper))

"""
    project(P, item, indices)

Project `item` using projection `P` and crop to `indices` if given.
"""
function project end


"""
    project!(bufitem, P, item, indices)

Project `item` using projection `P` and crop to `indices` if given.
Store result in `bufitem`. Inplace version of [`project`](@ref).

Default implementation falls back to `project`.
"""
function project!(bufitem, P, item, indices)
    titem = project(P, item, indices)
    copyitemdata!(bufitem, titem)

    return bufitem
end

project(P, itemw::ItemWrapper, indices) =
    setwrapped(itemw, project(P, getwrapped(itemw), indices))

project!(bufw::ItemWrapper, P, itemw::ItemWrapper, indices) =
    setwrapped(bufw, project!(getwrapped(bufw), P, getwrapped(itemw), indices))


"""
    projectionbounds(tfm, P, bounds, randstate)
"""
function projectionbounds(tfm, P, bounds; randstate = getrandstate(tfm))
    return transformbounds(bounds, P)
end

function _projection_geometry(tfm::ProjectiveTransform, bounds, randstate)
    P = getprojection(tfm, bounds; randstate)
    return P, projectionbounds(tfm, P, bounds; randstate)
end

_sample_between(rng, lo, hi) = lo + rand(rng) * (hi - lo)

function _project_extrapolation(data, method, fillvalue)
    if method isa BSpline && any(isone, size(data))
        # A linear stencil needs at least two samples on every interpolated axis.
        schemes = map(n -> n == 1 ? NoInterp() : method, size(data))
        itp = interpolate(data, schemes)
        if fillvalue isa Union{Periodic, Interpolations.Reflect}
            boundaries = map(n -> n == 1 ? Flat() : fillvalue, size(data))
            return extrapolate(itp, boundaries)
        end
        return box_extrapolation(itp; fillvalue)
    end
    return box_extrapolation(data; method, fillvalue)
end

function apply(tfm::ProjectiveTransform, item::Item; randstate = getrandstate(tfm))
    P, bounds = _projection_geometry(tfm, getbounds(item), randstate)
    return project(P, item, bounds)
end

# For the buffered version, `project!` is used. Image and mask projections
# reuse the buffer when the projected size matches.

function apply!(
        bufitem::AbstractItem,
        tfm::ProjectiveTransform,
        item::AbstractItem;
        randstate = getrandstate(tfm))
    P, bounds = _projection_geometry(tfm, getbounds(item), randstate)
    return project!(bufitem, P, item, bounds)
end


# The simplest `ProjectiveTransform` is the static `Project`.

"""
    Project(P)

Projective transformation that applies the fixed projection `P`, a
`CoordinateTransformations.Transformation`, regardless of the item's bounds.
"""
struct Project{T<:CoordinateTransformations.Transformation} <: ProjectiveTransform
    P::T
end

getprojection(tfm::Project, bounds; randstate = nothing) = tfm.P

# ## ImageTransformations.jl 0.8 internal functionality port
#
# Ported from ImageTransformations 0.8, since 0.9 introduced changes that broke
# some assumptions.

function _autorange(img, tform)
    R = CartesianIndices(axes(img))
    autorange(R, tform)
end

function _autorange(R::CartesianIndices, tform)
    tform = _round(tform)
    mn = mx = tform(SVector(first(R).I))
    for I in ImageTransformations.CornerIterator(R)
        x = tform(SVector(I.I))
        # we map min and max to prevent type-inference issues
        # (because min(::SVector,::SVector) -> Vector)
        mn = map(min, x, mn)
        mx = map(max, x, mx)
    end
    _autorange(Tuple(mn), Tuple(mx))
end

@noinline _autorange(mn::Tuple, mx::Tuple) = map((a,b)->floor(Int,a):ceil(Int,b), mn, mx)


# Slightly round/discretize the transformation so that the warpped image size isn't affected by
# numerical stability
# https://github.com/JuliaImages/ImageTransformations.jl/issues/104
_default_digits(::Type{T}) where T<:Number = _default_digits(floattype(T))
# these constants come from eps() digits
_default_digits(::Type{<:AbstractFloat}) = 15
_default_digits(::Type{Float64}) = 15
_default_digits(::Type{Float32}) = 7

function _round(tform::T; kwargs...) where T<:CoordinateTransformations.Transformation
    rounded_fields = map(Base.OneTo(fieldcount(T))) do i
        __round(getfield(tform, i); kwargs...)
    end
    T(rounded_fields...)
end
if isdefined(Base, :ComposedFunction)
    _round(tform::ComposedFunction; kwargs...) = _round(tform.outer; kwargs...) ∘ _round(tform.inner; kwargs...)
end
_round(tform; kwargs...) = tform

__round(x; kwargs...) = x
__round(x::AbstractArray; digits=_default_digits(eltype(x)), kwargs...) = round.(x; digits=digits, kwargs...)
__round(x::T; digits=_default_digits(T), kwargs...) where T<:Number = round(x; digits=digits, kwargs...)
