"""
    $(TYPEDEF)

Designate that the inplace-version of `Transform::T` should be used
for all items that are a subtype of `itemtype`.

`apply!(buffer, tfm::T, item)` needs to be implemented for all `Item`
types that should be buffered.
"""
@with_kw struct Buffered{T<:Transform} <: Transform
    transform::T
    itemtype::Type{<:Item}  = Item
end
Buffered(tfm::T) where {T<:Transform} = Buffered{T}(transform = tfm)


getrandstate(buffered::Buffered) = getrandstate(buffered.transform)
apply(buffered::Buffered, item::Item, randstate = getrandstate(buffered)) =
    apply(buffered.transform, item)


"""
    makebuffer(tfm::Buffered, item) = apply(buffered.transform, item)
    makebuffer(tfm::Transform, item) = nothing

Allocate a buffer. If `tfm` is not `Buffered`, default to `nothing`.
If `tfm` is a `Buffered{T}`, default to `buffer = apply(tfm.transform, item)`.
"""
makebuffer(::Transform, _) = nothing
makebuffer(buffered::Buffered, item::Item) = apply(buffered.transform, item)


"""
    apply!(buffer, buffered::Buffered{T}, item::I)

Applies `buffered.transform` to `item`, mutating the preallocated
`buffer`.

`buffer` can be obtained with `buffer = makebuffer(buffered, item)`

    apply!(buffer, tfm::Transform, item::I; randstate) = apply(tfm, item; randstate)

If `tfm` is not a `Buffered`, simply return `apply(tfm, item)` (non-mutating
version).
"""
apply!(_, tfm::Transform, items) = apply(tfm, items)


function makebuffer(pipeline::Sequential, items)
    buffers = []
    for tfm in pipeline.transforms
        push!(buffers, makebuffer(tfm, items))
        items = apply(tfm, items)
    end
    return buffers
end


function apply!(buffers, pipeline::Sequential, items)
    for (tfm, buffer) in zip(pipeline.transforms, buffers)
        items = apply!(buffer, tfm, items)
    end
    return items
end
