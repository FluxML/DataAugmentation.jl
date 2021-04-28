

# To make composition possible, we implement [`compose`](#), which
# defaults to returning a [`Sequence`](#).

"""
    Sequence(transforms...)

`Transform` that applies multiple `transformations`
after each other.

You should not use this explicitly. Instead use [`compose`](#).
"""
struct Sequence{T<:Tuple} <: Transform
    transforms::T
end

Sequence(tfms...) = Sequence{typeof(tfms)}(tfms)
Sequence(tfm::Transform) = tfm

getrandstate(seq::Sequence) = getrandstate.(seq.transforms)


compose(tfm1::Transform, tfm2::Transform) = Sequence(tfm1, tfm2)
compose(seq::Sequence, tfm::Transform) = Sequence(seq.transforms..., tfm)
compose(tfm::Transform, seq::Sequence) = compose(tfm, seq.transforms...)
compose(::Identity, seq::Sequence) = seq
compose(seq::Sequence, ::Identity) = seq


function apply(seq::Sequence, items::Tuple; randstate = getrandstate(seq))
    for (tfm, r) in zip(seq.transforms, randstate)
        items = apply(tfm, items; randstate = r)
    end
    return items
end


apply(seq::Sequence, item::Item; randstate = getrandstate(seq)) =
    apply(seq, (item,); randstate = randstate) |> only


function makebuffer(pipeline::Sequence, items)
    buffers = []
    for tfm in pipeline.transforms
        push!(buffers, makebuffer(tfm, items))
        items = apply(tfm, items)
    end
    return buffers
end


function apply!(buffers::Item, pipeline::Sequence, items::Item; randstate = getrandstate(pipeline))
    @assert length(buffers) == length(pipeline.transforms)
    for (tfm, buffer, r) in zip(pipeline.transforms, buffers, randstate)
        items = apply!(buffer, tfm, items; randstate = r)
    end
    return items
end

function apply!(buffers::Vector, pipeline::Sequence, items::Vector; randstate = getrandstate(pipeline))
    @assert length(buffers) == length(pipeline.transforms)
    for (tfm, buffer, r) in zip(pipeline.transforms, buffers, randstate)
        items = apply!(buffer, tfm, items; randstate = r)
    end
    return items
end
