"""
    ClampKeypoints()

Clamp each coordinate of a [`Keypoints`](@ref) item to the first and last
indices of its dimension. Use after a geometric transformation to move
out-of-bounds coordinates onto the boundary. Each dimension with points to
clamp must have a nonempty index range.

Accepts a tuple of aligned items, such as `(image, keypoints)`, and targets
its `Keypoints` entries.
"""
struct ClampKeypoints <: Transform end

"""
    SanitizeKeypoints()

Replace each out-of-bounds point in a [`Keypoints`](@ref) item with a point
whose coordinates are all `NaN`. For a dimension with index range `lo:hi`,
the coordinate is inside if `lo <= coordinate < hi + 1`.

The result preserves the shape and ordering of the keypoint array and uses
floating-point coordinates. Accepts a tuple of aligned items, such as
`(image, keypoints)`, and targets its `Keypoints` entries.
"""
struct SanitizeKeypoints <: Transform end

const KeypointBoundsTransform = Union{ClampKeypoints, SanitizeKeypoints}

apply(::KeypointBoundsTransform, item::Item; randstate = nothing) = item
apply(::KeypointBoundsTransform, item::ItemWrapper; randstate = nothing) = item

function apply(::ClampKeypoints, item::Keypoints{N, T}; randstate = nothing) where {N, T}
    ranges = getbounds(item).rs
    if any(isempty, ranges) && any(!isnothing, itemdata(item))
        throw(ArgumentError("cannot clamp keypoints to an empty index range"))
    end
    U = T <: Integer ? Int : promote_type(T, Int)
    return _map_keypoints(item, U) do point
        coords = map(Tuple(point), ranges) do x, r
            x < first(r) && return first(r)
            x > last(r) && return last(r)
            return x
        end
        return SVector{N, U}(coords)
    end
end

function apply(::SanitizeKeypoints, item::Keypoints{N, T}; randstate = nothing) where {N, T}
    ranges = getbounds(item).rs
    F = float(T)
    P = SVector{N, F}
    missing_point = P(ntuple(_ -> F(NaN), Val(N)))
    return _map_keypoints(item, F) do point
        inside = all(first(r) <= x < widen(last(r)) + 1 for (x, r) in zip(point, ranges))
        return inside ? convert(P, point) : missing_point
    end
end

function _map_keypoints(f, item::Keypoints{N, T, S}, ::Type{U}) where {N, T, S, U}
    P = SVector{N, U}
    R = Nothing <: S ? Union{Nothing, P} : P
    points = map(fmap(f), itemdata(item))
    # Mapping legacy `nothing` entries can erase the point type.
    if eltype(points) !== R
        points = copyto!(similar(Array{R}, axes(points)), points)
    end
    return Keypoints(points, getbounds(item))
end
