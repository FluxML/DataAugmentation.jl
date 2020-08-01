abstract type Transform end

"""
    getrandstate(transform)

Random state to pass as keyword argument to `apply`. Useful for
stochastic transforms.
"""
getrandstate(::Transform) = nothing


apply(tfm::Transform, item::Item) = apply(tfm, item; randstate = getrandstate(tfm))


function apply(tfm::Transform, itemw::ItemWrapper; randstate = getrandstate(tfm))
    item = apply(tfm, getwrapped(itemw); randstate = getrandstate(tfm))
    itemw = setwrapped(itemw, item)
    return itemw
end

function apply(tfm::Transform, many::Many; randstate = getrandstate(tfm))
    return Many(map(item -> apply(tfm, item, randstate = randstate), many.items))
end

# FIXME: Stack overflow when passing an `Int` to `apply`
apply(tfm::Transform, items; randstate = getrandstate(tfm)) =
    map(item -> apply(tfm, item; randstate = randstate), items)


struct Sequential <: Transform
    transforms
end

getrandstate(seq::Sequential) = getrandstate.(seq.transforms)

function apply(seq::Sequential, items; randstate = getrandstate(seq))
    for (tfm, r) in zip(seq.transforms, randstate)
        items = apply(tfm, items; randstate = r)
    end
    return items
end

apply(seq::Sequential, item::Item; randstate = getrandstate(seq)) =
    apply(seq, [item]; randstate = randstate)[1]

compose(tfm) = tfm
compose(tfm1::Transform, tfm2::Transform) = Sequential([tfm1, tfm2])
function compose(seq::Sequential, tfm::Transform)
    push!(seq.transforms, tfm)
    return seq
end
compose(tfms...) = compose(compose(tfms[1], tfms[2]), tfms[3:end]...)
Base.:(|>)(tfm1::Transform, tfm2::Transform) = compose(tfm1, tfm2)

# Simple Transforms

"""
    Identity()

Does nothing.
"""
struct Identity <: Transform end
apply(::Identity, item::Item; randstate = nothing) = item
compose(::Identity, ::Identity) = Identity()
compose(tfm::Transform, ::Identity) = tfm
compose(::Identity, tfm::Transform) = tfm