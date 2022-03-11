"""
    abstract type ProjectiveTransform <: Transform

Abstract supertype for projective transformations. See
[Projective transformations](../docs/projective/interface.md).
"""
abstract type ProjectiveTransform <: Transform end


"""
    getprojection(tfm, bounds; randstate)

Create a projection for an item with spatial bounds `bounds`.
The projection should be a `CoordinateTransformations.Transformation`.
See [CoordinateTransformations.jl](https://github.com/JuliaGeometry/CoordinateTransformations.jl)
"""
function getprojection end


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
Bounds(axes::NTuple{N, <:Base.OneTo}) where N = Bounds(convert.(UnitRange{Int}, axes))


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
    return Bounds(ImageTransformations.autorange(CartesianIndices(bounds.rs), P))
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
Store result in `bufitem`. Inplace version of [`project`](#).

Default implementation falls back to `project`.
"""
function project!(bufitem, P, item, indices)
    titem = project(P, item, indices)
    copyitemdata!(bufitem, titem)

    return bufitem
end


"""
    projectionbounds(tfm, P, bounds, randstate)
"""
function projectionbounds(tfm, P, bounds; randstate = getrandstate(tfm))
    return transformbounds(bounds, P)
end

# With the interface defined, any `ProjectiveTransform` can be `apply`ed
# like this:

function apply(tfm::ProjectiveTransform, item::Item; randstate = getrandstate(tfm))
    bounds = getbounds(item)
    P = getprojection(tfm, bounds; randstate = randstate)
    bounds_ = projectionbounds(tfm, P, bounds; randstate = randstate)
    return project(P, item, bounds_)
end

# For the buffered version, `project!` is used. Of course the size
# of the data produced by `tfm` needs to be the same every time it
# is applied for this to work.

function apply!(
        bufitem::AbstractItem,
        tfm::ProjectiveTransform,
        item::AbstractItem;
        randstate = getrandstate(tfm))
    bounds = getbounds(item)
    P = getprojection(tfm, bounds; randstate = randstate)
    bounds_ = projectionbounds(tfm, P, bounds; randstate = randstate)
    res = project!(bufitem, P, item, bounds_)
    return res
end


# The simplest `ProjectiveTransform` is the static `Project`.

struct Project{T<:CoordinateTransformations.Transformation} <: ProjectiveTransform
    P::T
end

getprojection(tfm::Project, bounds; randstate = nothing) = tfm.P
