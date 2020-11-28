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


"""
    getbounds(item)

Return the spatial bounds of `item`. For a 2D-image (`Image{2}`)
the bounds are the 4 corners of the bounding rectangle. In general,
for an N-dimensional item, the bounds are a vector of the N^2 corners
of the N-dimensional hypercube bounding the data. In practive, use
[`makebounds`](#) to construct the bounds from a tuple of side lengths.
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
project!(bufitem, P, item, indices) = project(P, item, indices)


"""
    cropindices(tfm, P, bounds, randstate)
"""
function cropindices(tfm, P, bounds; randstate = getrandstate(tfm))
    return ImageTransformations.autorange(CartesianIndices(boundsranges(bounds)), P)
end

# With the interface defined, any `ProjectiveTransform` can be `apply`ed
# like this:

function apply(tfm::ProjectiveTransform, item::Item; randstate = getrandstate(tfm))
    bounds = getbounds(item)
    P = getprojection(tfm, bounds; randstate = randstate)
    indices = cropindices(tfm, P, bounds; randstate = randstate)
    return project(P, item, indices)
end

# For the buffered version, `project!` is used. Of course the size
# of the data produced by `tfm` needs to be the same every time it
# is applied for this to work.

function apply!(
        bufitem,
        tfm::ProjectiveTransform,
        item::Item;
        randstate = getrandstate(tfm))
    bounds = getbounds(item)
    P = getprojection(tfm, bounds; randstate = randstate)
    indices = cropindices(tfm, P, bounds; randstate = randstate)
    return project!(bufitem, P, item, bounds)
end


# The simplest `ProjectiveTransform` is the static `Project`.

struct Project{T<:CoordinateTransformations.Transformation} <: ProjectiveTransform
    P::T
end

getprojection(tfm::Project, bounds; randstate = nothing) = tfm.P
