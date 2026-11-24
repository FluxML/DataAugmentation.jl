"""
    ClampBoundingBoxes()

Clip finite [`BoundingBoxes`](@ref) coordinates to the pixel edges of the canvas.
Preserve the format, box count, and labels. Boxes fully outside the canvas
collapse along at least one axis. Follow with [`SanitizeBoundingBoxes`](@ref)
to remove boxes with zero area. Accepts a box collection or a tuple of items
such as `(image, boxes)`.
"""
struct ClampBoundingBoxes <: Transform end

"""
    SanitizeBoundingBoxes()

Remove any [`BoundingBoxes`](@ref) entry with nonfinite coordinates, nonpositive
width or height, or any coordinate outside the canvas's pixel edges. Remove
the corresponding labels in the same operation. Apply
[`ClampBoundingBoxes`](@ref) first to keep the visible parts of partially
out-of-bounds boxes. Accepts a box collection or a tuple of items such as
`(image, boxes)`.
"""
struct SanitizeBoundingBoxes <: Transform end

const BoxBoundsTransform = Union{ClampBoundingBoxes, SanitizeBoundingBoxes}

apply(::BoxBoundsTransform, item::Item; randstate = nothing) = item
apply(::BoxBoundsTransform, item::ItemWrapper; randstate = nothing) = item

function _format_clipped_box(lo, hi, format)
    (format === :XYXY || !_valid_box(lo, hi)) && return _format_box(lo, hi, format)
    extent = hi - lo
    # Encoding an extent can move a rounded corner back outside the crop.
    if format === :XYWH
        h, w = map((l, u, d) -> l + d > u ? prevfloat(d) : d, lo, hi, extent)
        return SVector(lo[2], lo[1], w, h)
    end
    center = lo + extent / 2
    half_h, half_w = map(lo, hi, center) do l, u, c
        r = min(c - l, u - c)
        return c - r < l || c + r > u ? prevfloat(r) : r
    end
    return SVector(center[2], center[1], 2 * half_w, 2 * half_h)
end

function apply(::ClampBoundingBoxes, boxes::BoundingBoxes; randstate = nothing)
    lower, upper = _box_edges(getbounds(boxes))
    return _map_boxes((lo, hi) -> _clamp_box(lo, hi, lower, upper), boxes; encode = _format_clipped_box)
end

function apply(::SanitizeBoundingBoxes, boxes::BoundingBoxes; randstate = nothing)
    lower, upper = _box_edges(getbounds(boxes))
    keep = findall(1:length(boxes)) do i
        lo, hi = _box_corners(boxes, i)
        _valid_box(lo, hi) && all(lower .<= lo) && all(hi .<= upper)
    end
    length(keep) == length(boxes) && return boxes
    labels = isnothing(boxes.labels) ? nothing : boxes.labels[keep]
    return BoundingBoxes(itemdata(boxes)[:, keep], boxes.bounds; boxes.format, labels)
end
