"""
    ImageBatch(images, labels)
    ImageBatch(images, class_indices; num_classes)

An item for collated images and their classification labels. Use one-based real
arrays of size `(width, height, batch)` for grayscale images or
`(width, height, channels, batch)` for tensors collated after [`ImageToTensor`](@ref).
Volumes collated after `ImageToTensor` have size `(width, height, depth, channels, batch)`:
from four dimensions on, the last one is the batch, the one before it the channels, and
all the dimensions before those are spatial.
The label matrix has size `(classes, batch)` and holds one-hot or soft labels.
The second constructor encodes class indices in `1:num_classes` as `Float32`
one-hot columns. [`itemdata`](@ref) returns `(images, labels)`.
"""
struct ImageBatch{A<:AbstractArray, L<:AbstractMatrix} <: Item
    data::Tuple{A, L}
    function ImageBatch(data::Tuple{A, L}) where {A<:AbstractArray, L<:AbstractMatrix}
        images, labels = data
        ndims(images) >= 3 || throw(ArgumentError("images must have at least 3 dimensions"))
        Base.require_one_based_indexing(images, labels)
        eltype(images) <: Real || throw(ArgumentError("images must contain real numbers"))
        eltype(labels) <: Real || throw(ArgumentError("labels must contain real numbers"))
        size(images, ndims(images)) == size(labels, 2) ||
            throw(DimensionMismatch("image and label batch sizes must match"))
        all(>(0), Base.front(size(images))) ||
            throw(ArgumentError("image spatial and channel dimensions must be nonempty"))
        size(labels, 1) > 0 || throw(ArgumentError("labels must contain at least one class"))
        return new{A, L}(data)
    end
end

ImageBatch(images::AbstractArray, labels::AbstractMatrix) = ImageBatch((images, labels))

function ImageBatch(images::AbstractArray, class_indices::AbstractVector{<:Integer}; num_classes::Integer)
    num_classes > 0 || throw(ArgumentError("num_classes must be positive"))
    labels = zeros(Float32, num_classes, length(class_indices))
    for (j, i) in enumerate(class_indices)
        1 <= i <= num_classes || throw(ArgumentError("class indices must be in 1:num_classes"))
        labels[i, j] = 1
    end
    return ImageBatch(images, labels)
end

Base.length(batch::ImageBatch) = size(itemdata(batch)[2], 2)

copyitem(item::ImageBatch) = ImageBatch(map(copy, itemdata(item)))

function copyitemdata!(buf::B, item::B) where {B<:ImageBatch}
    dst = itemdata(buf)
    src = itemdata(item)
    axes.(dst) == axes.(src) || throw(DimensionMismatch("batch buffer dimensions must match the result"))
    map(copyto!, dst, src)
    return buf
end
