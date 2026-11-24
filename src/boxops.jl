"""
    masks_to_boxes(masks::AbstractArray{Bool, 3}; labels = nothing)
    masks_to_boxes(mask::MaskBinary{2}; labels = nothing)

Build [`BoundingBoxes`](@ref) in `:XYXY` format from a `(height, width, count)`
stack of instance masks or a single binary mask item. Each box encloses the
full pixels marked `true`, so a one-pixel object has width and height `1`.
An empty mask yields a zero-area box at the canvas's upper-left edge so the
box and label ordering is preserved. Spatial axes can be offset.
"""
function masks_to_boxes(masks::AbstractArray{Bool, 3}; labels = nothing)
    bounds = Bounds((axes(masks, 1), axes(masks, 2)))
    data = Matrix{Float64}(undef, 4, size(masks, 3))
    for (i, k) in enumerate(axes(masks, 3))
        data[:, i] = _mask_box(selectdim(masks, 3, k))
    end
    return BoundingBoxes(data, bounds; labels)
end

function masks_to_boxes(mask::MaskBinary{2}; labels = nothing)
    bounds = getbounds(mask)
    data = OffsetArray(itemdata(mask), bounds.rs)
    return BoundingBoxes(reshape(collect(_mask_box(data)), 4, 1), bounds; labels)
end

function _mask_box(mask)
    ymin = xmin = typemax(Int)
    ymax = xmax = typemin(Int)
    for i in CartesianIndices(mask)
        mask[i] || continue
        y, x = Tuple(i)
        ymin, xmin = min(ymin, y), min(xmin, x)
        ymax, xmax = max(ymax, y), max(xmax, x)
    end
    if xmax < xmin
        y, x = first.(axes(mask)) .- 0.5
        return SVector(x, y, x, y)
    end
    return SVector(xmin - 0.5, ymin - 0.5, xmax + 0.5, ymax + 0.5)
end

"""
    box_iou(a::BoundingBoxes, b::BoundingBoxes)

Return the pairwise intersection over union of two box collections in the
same coordinate system. Entry `(i, j)` compares box `i` in `a` with box `j`
in `b`. Pairs containing an invalid or degenerate box have IoU `0`.
"""
function box_iou(a::BoundingBoxes, b::BoundingBoxes)
    first_boxes = [_box_corners(a, i) for i in 1:length(a)]
    second_boxes = [_box_corners(b, j) for j in 1:length(b)]
    return [_box_iou(first..., second...) for first in first_boxes, second in second_boxes]
end

function _box_iou(lo1, hi1, lo2, hi2)
    T = promote_type(Float64, eltype(lo1), eltype(hi1), eltype(lo2), eltype(hi2))
    _valid_box(lo1, hi1) && _valid_box(lo2, hi2) || return zero(T)
    lo1, hi1, lo2, hi2 = map(SVector{2, T}, (lo1, hi1, lo2, hi2))
    intersection = prod(max.(min.(hi1, hi2) - max.(lo1, lo2), zero(T)))
    return intersection / (prod(hi1 - lo1) + prod(hi2 - lo2) - intersection)
end
