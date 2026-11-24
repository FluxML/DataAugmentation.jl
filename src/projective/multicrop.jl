abstract type MultiCrop <: Transform end

"""
    FiveCrop(size)

Return five views of 2D spatial data, in order from the top left, top right,
bottom left, bottom right, and center. Pass an integer for a square crop or
a `(height, width)` tuple. Each crop keeps its coordinates on the input canvas.
For a tuple of items, each item produces its own tuple of views.
"""
struct FiveCrop <: MultiCrop
    size::NTuple{2, Int}

    function FiveCrop(size::Tuple{Integer, Integer})
        all(>(0), size) || throw(ArgumentError("crop sizes must be positive"))
        return new(size)
    end
end

FiveCrop(size::Integer) = FiveCrop((size, size))

"""
    TenCrop(size; vertical_flip = false)

Return the five views from [`FiveCrop`](@ref), followed by the same five crops
of the horizontally flipped input. Use `vertical_flip = true` to flip vertically.
"""
struct TenCrop <: MultiCrop
    crop::FiveCrop
    vertical_flip::Bool
end

TenCrop(size; vertical_flip::Bool = false) = TenCrop(FiveCrop(size), vertical_flip)

function _fivecropbounds(crop::FiveCrop, bounds::Bounds{2})
    h, w = crop.size
    rows, cols = bounds.rs
    h <= length(rows) && w <= length(cols) ||
        throw(ArgumentError("crop size must fit within the input bounds"))
    top, left = first(rows), first(cols)
    bottom, right = last(rows) - h + 1, last(cols) - w + 1
    corners = ((top, left), (top, right), (bottom, left), (bottom, right))
    cornerbounds = map(((r, c),) -> Bounds((r:(r + h - 1), c:(c + w - 1))), corners)
    center = offsetcropbounds(crop.size, bounds, (0.5, 0.5))
    return (cornerbounds..., center)
end

function _multicrop(tfm::FiveCrop, item, bounds::Bounds{2})
    return map(crop -> project(IdentityTransformation(), item, crop), _fivecropbounds(tfm, bounds))
end

function _multicrop(tfm::TenCrop, item, bounds::Bounds{2})
    crops = _fivecropbounds(tfm.crop, bounds)
    signs = tfm.vertical_flip ? (-1, 1) : (1, -1)
    mirror = recenter(scaleprojection(signs), SVector(mean.(bounds.rs)))
    original = map(crop -> project(IdentityTransformation(), item, crop), crops)
    flipped = map(crop -> project(mirror, item, crop), crops)
    return (original..., flipped...)
end

function apply(tfm::MultiCrop, item::Item; randstate = nothing)
    return _multicrop(tfm, item, getbounds(item))
end

function apply(tfm::MultiCrop, item::ItemWrapper; randstate = nothing)
    return _multicrop(tfm, item, getbounds(item))
end
