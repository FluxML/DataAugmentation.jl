

# To make composition possible, we implement [`compose`](#), which
# defaults to returning a [`Sequence`](#).

"""
    Sequence(transforms...)

`Transform` that applies multiple `transformations`
after each other.

You should not use this explicitly. Instead use [`compose`](#).
"""
struct Sequence{T<:Tuple where N} <: Transform
    transforms::T
end

Sequence(tfms...) = Sequence{typeof(tfms)}(tfms)

getrandstate(seq::Sequence) = getrandstate.(seq.transforms)


compose(tfm1::Transform, tfm2::Transform) = Sequence(tfm1, tfm2)
compose(seq::Sequence, tfm::Transform) = Sequence(seq.transforms..., tfm)
compose(seq::Sequence, ::Identity) = seq


function apply(seq::Sequence, items::Tuple; randstate = getrandstate(seq))
    for (tfm, r) in zip(seq.transforms, randstate)
        items = apply(tfm, items; randstate = r)
    end
    return items
end


apply(seq::Sequence, item::Item; randstate = getrandstate(seq)) =
    apply(seq, (item,); randstate = randstate) |> only


makebuffer(pipeline::Sequence, item::Item) = only.(makebuffer(pipeline, (item,)))
function makebuffer(pipeline::Sequence, items::Tuple)
    buffers = []
    for tfm in pipeline.transforms
        push!(buffers, makebuffer(tfm, items))
        items = apply(tfm, items)
    end
    return Tuple(buffers)
end


function apply!(buffers::Tuple, pipeline::Sequence, items::Tuple; randstate = getrandstate(pipeline))
    for (tfm, buffer, r) in zip(pipeline.transforms, buffers, randstate)
        items = apply!(buffer, tfm, items; randstate = r)
    end
    return items
end

function apply!(buffer::I, pipeline::Sequence, item::I; randstate = getrandstate(pipeline)) where {I<:Item}
    return apply!((buffer,), pipeline, (item,); randstate = randstate) |> only
end
