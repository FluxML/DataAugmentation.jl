"""
    Pad(padding; fill = nothing, padding_mode = :constant)

Pad every side by an integer, or pass one integer per axis in array order.
In 2D, a four-tuple specifies `(left, top, right, bottom)`. For arbitrary
dimensions, pairs `(before, after)` per axis also work.

Modes are `:constant`, `:edge`, `:reflect` (excluding the edge pixel), and
`:symmetric` (including the edge pixel). Use `fill = nothing` with
`:constant` to use the item's extrapolation setting. The input content keeps
its coordinates while the bounds grow; use [`PinOrigin`](@ref) for one-based
output coordinates.
"""
struct Pad{P, F} <: Transform
    padding::P
    fill::F
    padding_mode::Symbol
end

function Pad(padding; fill = nothing, padding_mode::Symbol = :constant)
    _validate_padding_mode(padding_mode)
    return Pad(padding, fill, padding_mode)
end

function _validate_padding_mode(mode)
    mode in (:constant, :edge, :reflect, :symmetric) ||
        throw(ArgumentError("unsupported padding mode: $mode"))
end

function _padparams(padding, n)
    sides = if padding isa Integer
        ntuple(_ -> (Int(padding), Int(padding)), n)
    elseif n == 2 && padding isa NTuple{4, Integer}
        left, top, right, bottom = Int.(padding)
        ((top, bottom), (left, right))
    elseif padding isa NTuple{n, Integer}
        map(p -> (Int(p), Int(p)), padding)
    elseif padding isa NTuple{n, Tuple{Integer, Integer}}
        map(p -> Int.(p), padding)
    else
        throw(ArgumentError("padding must be an integer, one integer or pair per axis, or a 2D four-tuple"))
    end
    all(p -> all(>=(0), p), sides) || throw(ArgumentError("padding must be nonnegative"))
    return sides
end

_padded_bounds(bounds::Bounds, padding) =
    Bounds(map((r, p) -> (first(r) - p[1]):(last(r) + p[2]), bounds.rs, padding))

_padindex(i, n, ::Val{:constant}) = 1 <= i <= n ? i : 0
_padindex(i, n, ::Val{:edge}) = clamp(i, 1, n)
_padindex(i, n, ::Val{:periodic}) = mod1(i, n)

function _padindex(i, n, ::Val{:symmetric})
    j = mod(i - 1, 2n)
    return j < n ? j + 1 : 2n - j
end

function _padindex(i, n, ::Val{:reflect})
    n == 1 && return 1
    period = 2n - 2
    j = mod(i - 1, period)
    return (j < n ? j : period - j) + 1
end

function _padarray(data::AbstractArray{T,N}, bounds, padding, fill, mode) where {T,N}
    isempty(data) && mode !== Val(:constant) &&
        throw(ArgumentError("this padding mode requires nonempty spatial axes"))
    output = similar(Array{T}, bounds.rs)
    input_size = size(data)
    input_start = first.(axes(data))
    output_start = first.(bounds.rs)
    for i in CartesianIndices(output)
        src = ntuple(d -> _padindex(i[d] - output_start[d] - padding[d][1] + 1, input_size[d], mode), N)
        output[i] = any(iszero, src) ? fill : data[(src .+ input_start .- 1)...]
    end
    return output
end

function _pad_boundary(tfm::Pad, item)
    tfm.padding_mode !== :constant && return nothing, Val(tfm.padding_mode)
    fill = isnothing(tfm.fill) ? item.extrapolate : tfm.fill
    fill isa Flat && return nothing, Val(:edge)
    fill isa Interpolations.Reflect && return nothing, Val(:reflect)
    fill isa Periodic && return nothing, Val(:periodic)
    return convert(eltype(itemdata(item)), fill), Val(:constant)
end

function apply(tfm::Pad, item::Union{Image, MaskMulti, MaskBinary}; randstate = nothing)
    bounds = getbounds(item)
    padding = _padparams(tfm.padding, length(bounds.rs))
    all(p -> all(iszero, p), padding) && return item
    newbounds = _padded_bounds(bounds, padding)
    fill, mode = _pad_boundary(tfm, item)
    data = _padarray(itemdata(item), newbounds, padding, fill, mode)
    return @set setdata(item, data).bounds = newbounds
end

function apply(tfm::Pad, item::Keypoints{N}; randstate = nothing) where N
    return Keypoints(itemdata(item), _padded_bounds(getbounds(item), _padparams(tfm.padding, N)))
end

"""
    RandomZoomOut(; fill = nothing, side_range = (1.0, 4.0), p = 0.5)

With probability `p`, place 2D spatial data at a random position within a larger
canvas. Sample the canvas scale factor from `side_range` and fill the new pixels
as in [`Pad`](@ref).
"""
struct RandomZoomOut{F} <: Transform
    fill::F
    side_range::Tuple{Float64, Float64}
    p::Float64
end

function RandomZoomOut(; fill = nothing, side_range::Tuple{Real, Real} = (1.0, 4.0), p::Real = 0.5)
    lo, hi = Float64.(side_range)
    1 <= lo <= hi < Inf || throw(ArgumentError("side_range must contain ordered, finite factors ≥ 1"))
    0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1]"))
    return RandomZoomOut(fill, (lo, hi), Float64(p))
end

function getrandstate(tfm::RandomZoomOut)
    lo, hi = tfm.side_range
    return (rand() < tfm.p, lo + rand() * (hi - lo), (rand(), rand()))
end

function apply(tfm::RandomZoomOut, item::Item; randstate = getrandstate(tfm))
    enabled, factor, placement = randstate
    enabled || return item
    bounds = getbounds(item)
    bounds isa Bounds{2} || throw(ArgumentError("RandomZoomOut requires 2D spatial data"))
    padding = map(length.(bounds.rs), placement) do n, position
        extra = floor(Int, n * factor) - n
        start = min(extra, floor(Int, (extra + 1) * position))
        return (start, extra - start)
    end
    return apply(Pad(padding; tfm.fill), item)
end
