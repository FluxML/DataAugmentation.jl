"""
    RandomIoUCrop(; min_scale = 0.3, max_scale = 1, min_aspect_ratio = 0.5,
                    max_aspect_ratio = 2, sampler_options = (0, 0.1, 0.3, 0.5, 0.7, 0.9, 1),
                    trials = 40)

Crop a detection sample to a randomly selected 2D window. Apply to a tuple
containing the image, its [`BoundingBoxes`](@ref) collection or individual
[`BoundingBox`](@ref) items, and any aligned masks or keypoints. Individual boxes
can also be grouped in a vector. A window is accepted if it contains the center
of a box whose intersection over union with the window meets the sampled threshold.

Box corners are continuous coordinates in the pixel-center system, so a window
of pixels `a:b` spans `a - 0.5` to `b + 0.5`. Boxes with centers inside the crop
are clipped to these edges. Other boxes are collapsed to zero area so that
their positions and labels are preserved. Use [`SanitizeBoundingBoxes`](@ref)
to remove collapsed boxes and their labels from a `BoundingBoxes` collection.
A threshold of `1` keeps the full sample, as does running out of trials or
applying the transform to a sample without boxes.
"""
struct RandomIoUCrop{T<:Tuple} <: Transform
    scale::Tuple{Float64, Float64}
    ratio::Tuple{Float64, Float64}
    sampler_options::T
    trials::Int
end

function RandomIoUCrop(;
    min_scale::Real = 0.3, max_scale::Real = 1,
    min_aspect_ratio::Real = 0.5, max_aspect_ratio::Real = 2,
    sampler_options = (0, 0.1, 0.3, 0.5, 0.7, 0.9, 1), trials::Integer = 40,
)
    0 < min_scale <= max_scale <= 1 || throw(ArgumentError("expected 0 < min_scale <= max_scale <= 1"))
    0 < min_aspect_ratio <= max_aspect_ratio < Inf ||
        throw(ArgumentError("aspect ratio limits must be positive, finite, and ordered"))
    options = Tuple(Float64(x) for x in sampler_options)
    !isempty(options) && all(x -> 0 <= x <= 1, options) ||
        throw(ArgumentError("sampler_options must contain at least one threshold in [0, 1]"))
    trials > 0 || throw(ArgumentError("trials must be positive"))
    return RandomIoUCrop(Float64.((min_scale, max_scale)), Float64.((min_aspect_ratio, max_aspect_ratio)),
                         options, Int(trials))
end

getrandstate(::RandomIoUCrop) = rand(UInt64)

_detection_boxes(item) = ()
_detection_boxes(box::BoundingBox{2}) = (box,)
_detection_boxes(boxes::BoundingBoxes) = (boxes,)
_detection_boxes(items::Union{Tuple, AbstractVector}) = Iterators.flatten(map(_detection_boxes, items))

_detection_corners(box::BoundingBox{2}) = (itemdata(box),)
_detection_corners(boxes::BoundingBoxes) = (_box_corners(boxes, i) for i in 1:length(boxes))

function _center_in_crop(lo, hi, lower, upper)
    center = lo / 2 + hi / 2
    return all(lower .<= center .< upper)
end

function _meets_iou_threshold(box, crop::Bounds{2}, threshold)
    lo, hi = box
    lower, upper = _box_edges(crop)
    _center_in_crop(lo, hi, lower, upper) || return false
    iou = _box_iou(lo, hi, lower, upper)
    return iou > 0 && iou >= threshold
end

function _iou_crop_bounds(tfm::RandomIoUCrop, bounds::Bounds{2}, boxes, rng)
    h, w = length.(bounds.rs)
    (iszero(h) || iszero(w)) && return nothing
    lo, hi = tfm.scale
    for threshold in shuffle(rng, collect(tfm.sampler_options))
        threshold == 1 && return nothing
        for _ in 1:tfm.trials
            ch = max(1, floor(Int, h * (lo + rand(rng) * (hi - lo))))
            cw = max(1, floor(Int, w * (lo + rand(rng) * (hi - lo))))
            tfm.ratio[1] <= cw / ch <= tfm.ratio[2] || continue
            top = first(bounds.rs[1]) + rand(rng, 0:(h - ch))
            left = first(bounds.rs[2]) + rand(rng, 0:(w - cw))
            crop = Bounds((top:(top + ch - 1), left:(left + cw - 1)))
            any(box -> _meets_iou_threshold(box, crop, threshold), boxes) && return crop
        end
    end
    return nothing
end

function _crop_detection(item::AbstractItem, source::Bounds, crop::Bounds)
    getbounds(item) == source || throw(DimensionMismatch("detection items must share the same spatial bounds"))
    return project(IdentityTransformation(), item, crop)
end

function _clip_detection_box(lo, hi, lower, upper)
    first_corner = clamp.(lo, lower, upper)
    last_corner = _center_in_crop(lo, hi, lower, upper) ? clamp.(hi, lower, upper) : first_corner
    return first_corner, last_corner
end

function _crop_detection(box::BoundingBox{2}, source::Bounds, crop::Bounds{2})
    lo, hi = _clip_detection_box(itemdata(box)..., _box_edges(crop)...)
    return BoundingBox([lo, hi], crop)
end

function _crop_detection(boxes::BoundingBoxes, source::Bounds, crop::Bounds{2})
    getbounds(boxes) == source || throw(DimensionMismatch("detection items must share the same spatial bounds"))
    lower, upper = _box_edges(crop)
    return _map_boxes((lo, hi) -> _clip_detection_box(lo, hi, lower, upper), boxes;
                      bounds = crop, encode = _format_clipped_box)
end

_crop_detection(items::Union{Tuple, AbstractVector}, source, crop) =
    map(item -> _crop_detection(item, source, crop), items)

function apply(tfm::RandomIoUCrop, items::Union{Tuple, AbstractVector}; randstate = getrandstate(tfm))
    boxes = collect(_detection_boxes(items))
    isempty(boxes) && return items
    bounds = getbounds(first(boxes))
    all(box -> getbounds(box) == bounds, boxes) ||
        throw(DimensionMismatch("detection boxes must share the same spatial bounds"))
    corners = collect(Iterators.flatten(map(_detection_corners, boxes)))
    isempty(corners) && return items
    crop = _iou_crop_bounds(tfm, bounds, corners, Xoshiro(randstate))
    (isnothing(crop) || crop == bounds) && return items
    return _crop_detection(items, bounds, crop)
end

apply(tfm::RandomIoUCrop, box::BoundingBox{2}; randstate = getrandstate(tfm)) =
    only(apply(tfm, (box,); randstate))
apply(tfm::RandomIoUCrop, boxes::BoundingBoxes; randstate = getrandstate(tfm)) =
    only(apply(tfm, (boxes,); randstate))
apply(::RandomIoUCrop, item::Item; randstate = nothing) = item

function apply!(bufs::Union{Tuple, AbstractVector}, tfm::RandomIoUCrop,
                items::Union{Tuple, AbstractVector}; randstate = getrandstate(tfm))
    return apply(tfm, items; randstate)
end
