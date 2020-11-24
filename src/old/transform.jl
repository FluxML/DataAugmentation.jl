"""
    abstract type Transform

Abstract supertype for all transformations.
"""
abstract type Transform end

"""
    getrandstate(transform)

Generates random state for stochastic transformations.
Calling `apply(tfm, item)` is equivalent to
`apply(tfm, item; randstate = getrandstate(tfm))`. It
defaults to `nothing`, so you it only needs to be implemented
for stochastic `Transform`s.
"""
getrandstate(::Transform) = nothing


"""
    apply(tfm, item[; randstate])
    apply(tfm, items[; randstate])

Apply `tfm` to an `item` or a tuple `items`.

"""
apply(tfm::Transform, items) = apply(tfm, items; randstate = getrandstate(tfm))



function apply(tfm::Transform, items::Tuple; randstate = getrandstate(tfm))
    map(item -> apply(tfm, item; randstate = randstate), items)
end


# ## Composition

"""
    Sequence(transforms...)

`Transform` that applies multiple `transformations`
after each other.

You should not use this explicitly. Instead use [`compose`](#).
"""
struct Sequence{T<:NTuple{N, Transform} where N} <: Transform
    transforms::T
end

getrandstate(seq::Sequence) = getrandstate.(seq.transforms)

function apply(seq::Sequence, items::Tuple; randstate = getrandstate(seq))
    for (tfm, r) in zip(seq.transforms, randstate)
        items = apply(tfm, items; randstate = r)
    end
    return items
end


apply(seq::Sequence, item::Item; randstate = getrandstate(seq)) =
    apply(seq, (item,); randstate = randstate) |> only

"""
    compose(transforms...)

Compose tranformations. Use `|>` as an alias.

Defaults to creating a [`Sequence`](#) of transformations,
but smarter behavior can be implemented.
For example, `MapElem(f) |> MapElem(g) == MapElem(g ∘ f)`.
"""
compose(tfm) = tfm
compose(tfm1::Transform, tfm2::Transform) = Sequence(tfm1, tfm2)
compose(tfms...) = compose(compose(tfms[1], tfms[2]), tfms[3:end]...)

compose(seq::Sequence, tfm::Transform) = Sequence(seq.transforms..., tfm)

Base.:(|>)(tfm1::Transform, tfm2::Transform) = compose(tfm1, tfm2)


"""
    Identity()

The identity transformation.
"""
struct Identity <: Transform end
apply(::Identity, item::Item; randstate = nothing) = item
compose(::Identity, ::Identity) = Identity()
compose(tfm::Transform, ::Identity) = tfm
compose(::Identity, tfm::Transform) = tfm


"""
    MapElem(f)

Applies `f` to every element in an [`AbstractArrayItem`].
"""
struct MapElem <: Transform
    fn
end

function apply(tfm::MapElem, item::AbstractArrayItem; randstate = nothing)
    return setdata(item, map(tfm.fn, itemdata(item)))
end

function apply!(
        bufitem::I,
        tfm::MapElem,
        item::I;
        randstate = nothing) where I <: AbstractArrayItem
    map!(tfm.f, itemdata(bufitem), itemdata(item))
    return bufitem
end
