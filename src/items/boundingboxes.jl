"""
    BoundingBoxes(data, bounds; format = :XYXY, labels = nothing)

A collection of axis-aligned boxes on a two-dimensional canvas. Pass a one-based
`4 × N` matrix of real coordinates with one box per column. Give the canvas as
`(height, width)` or as [`Bounds`](@ref) with row and column ranges. Optional
`labels` is a one-based vector with one entry per box.

The coordinate order is `(x, y)`, with `x` along columns and `y` along rows.
Pixel `i` is centered at coordinate `i`, so a canvas axis `a:b` spans the
continuous interval `[a - 0.5, b + 0.5]`. Supported formats are:

- `:XYXY` for `(xmin, ymin, xmax, ymax)`
- `:XYWH` for `(xmin, ymin, width, height)`
- `:CXCYWH` for `(xcenter, ycenter, width, height)`

Use [`itemdata`](@ref) for the coordinate matrix, `boxes.format` for its format,
and `boxes.labels` for the labels. Geometric transforms enclose the four
projected corners of each valid box and update the canvas. Use
[`ClampBoundingBoxes`](@ref) and [`SanitizeBoundingBoxes`](@ref) to clip or
remove boxes after a crop or warp.
"""
struct BoundingBoxes{A<:AbstractMatrix{<:Real}, L<:Union{Nothing, AbstractVector}} <: Item
    data::A
    bounds::Bounds{2}
    format::Symbol
    labels::L

    function BoundingBoxes(data::A, bounds::Bounds{2}, format::Symbol, labels::L) where
            {A<:AbstractMatrix{<:Real}, L<:Union{Nothing, AbstractVector}}
        Base.require_one_based_indexing(data)
        size(data, 1) == 4 || throw(DimensionMismatch("box coordinates must have four rows"))
        _check_box_format(format)
        if !isnothing(labels)
            Base.require_one_based_indexing(labels)
            length(labels) == size(data, 2) ||
                throw(DimensionMismatch("there must be one label per box"))
        end
        return new{A, L}(data, bounds, format, labels)
    end
end

BoundingBoxes(data, bounds::Bounds{2}; format = :XYXY, labels = nothing) =
    BoundingBoxes(data, bounds, format, labels)
BoundingBoxes(data, size::Tuple{Integer, Integer}; kwargs...) =
    BoundingBoxes(data, Bounds(size); kwargs...)

Base.length(boxes::BoundingBoxes) = size(itemdata(boxes), 2)
getbounds(boxes::BoundingBoxes) = boxes.bounds

Base.show(io::IO, boxes::BoundingBoxes) =
    print(io, "BoundingBoxes with $(length(boxes)) boxes in $(boxes.format) format")

function _check_box_format(format::Symbol)
    format in (:XYXY, :XYWH, :CXCYWH) ||
        throw(ArgumentError("box format must be :XYXY, :XYWH, or :CXCYWH"))
    return format
end

function _box_corners(boxes::BoundingBoxes, i)
    data = itemdata(boxes)
    origin = SVector(float(data[2, i]), float(data[1, i]))
    endpoint = SVector(float(data[4, i]), float(data[3, i]))
    boxes.format === :XYXY && return origin, endpoint
    boxes.format === :XYWH && return origin, origin + endpoint
    half = endpoint / 2
    return origin - half, origin + half
end

function _format_box(lo, hi, format)
    y, x = lo
    format === :XYXY && return SVector(x, y, hi[2], hi[1])
    h, w = hi - lo
    format === :XYWH && return SVector(x, y, w, h)
    return SVector(x + w / 2, y + h / 2, w, h)
end

_copy_box_labels(labels::AbstractVector) = collect(labels)
_copy_box_labels(::Nothing) = nothing

function _map_boxes(f, boxes::BoundingBoxes; bounds = boxes.bounds, format = boxes.format,
                    encode = _format_box)
    data = if iszero(length(boxes))
        copy(itemdata(boxes))
    else
        stack(1:length(boxes)) do i
            lo, hi = f(_box_corners(boxes, i)...)
            encode(lo, hi, format)
        end
    end
    return BoundingBoxes(data, bounds; format, labels = _copy_box_labels(boxes.labels))
end

"""
    box_convert(boxes::BoundingBoxes, format)

Convert the coordinates to `:XYXY`, `:XYWH`, or `:CXCYWH`, preserving
the canvas, ordering, and labels. Coordinate arithmetic uses floating point.
"""
function box_convert(boxes::BoundingBoxes, format::Symbol)
    _check_box_format(format)
    return boxes.format === format ? boxes : _map_boxes(tuple, boxes; format)
end

_valid_box(lo, hi) = all(isfinite, lo) && all(isfinite, hi) && all(lo .< hi)

_box_edges(bounds::Bounds{2}) =
    SVector(first.(bounds.rs)) .- 0.5, SVector(last.(bounds.rs)) .+ 0.5

function _clamp_box(lo, hi, lower, upper)
    all(isfinite, lo) && all(isfinite, hi) || return lo, hi
    return clamp.(lo, lower, upper), clamp.(hi, lower, upper)
end

function _project_box(P, lo, hi)
    _valid_box(lo, hi) || return lo, hi
    a, b = P(lo), P(SVector(lo[1], hi[2]))
    c, d = P(SVector(hi[1], lo[2])), P(hi)
    return min.(a, b, c, d), max.(a, b, c, d)
end

project(P, boxes::BoundingBoxes, bounds::Bounds{2}) =
    _map_boxes((lo, hi) -> _project_box(P, lo, hi), boxes; bounds)

project(::IdentityTransformation, boxes::BoundingBoxes, bounds::Bounds{2}) =
    BoundingBoxes(copy(itemdata(boxes)), bounds; boxes.format, labels = _copy_box_labels(boxes.labels))

project!(buf, P, boxes::BoundingBoxes, bounds::Bounds{2}) = project(P, boxes, bounds)

copyitem(boxes::BoundingBoxes) = project(IdentityTransformation(), boxes, boxes.bounds)

function copyitemdata!(buf::B, boxes::B) where {B<:BoundingBoxes}
    length(buf) == length(boxes) || throw(DimensionMismatch("box counts must match the buffer"))
    buf.bounds == boxes.bounds && buf.format === boxes.format ||
        throw(ArgumentError("box canvas and format must match the buffer"))
    if !isnothing(boxes.labels) && buf.labels !== boxes.labels
        copyto!(buf.labels, boxes.labels)
    end
    copyto!(itemdata(buf), itemdata(boxes))
    return buf
end

function showitem!(img, boxes::BoundingBoxes)
    isempty(img) && return img
    lower, upper = _box_edges(Bounds(axes(img)))
    first_pixel, last_pixel = SVector(first.(axes(img))), SVector(last.(axes(img)))
    for i in 1:length(boxes)
        lo, hi = _clamp_box(_box_corners(boxes, i)..., lower, upper)
        _valid_box(lo, hi) || continue
        showbox!(img, clamp.(lo, first_pixel, last_pixel), clamp.(hi, first_pixel, last_pixel))
    end
    return img
end
