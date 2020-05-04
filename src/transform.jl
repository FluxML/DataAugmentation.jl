# Transform

abstract type Transform end

getparam(::Transform) = nothing

"""
    apply(tfm, item)
    apply(tfm, items)

Apply transform `tfm` to a single `item` or a tuple of `items`.
If given a tuple, the same random state is used to transform
the individual items.
"""
apply(tfm::Transform, item::Item) = apply(tfm, item, getparam(tfm))

apply(tfm::Transform, item::ItemWrapper) = apply(tfm, item, getparam(tfm))

function apply(tfm::Transform, items::Tuple)
    param = getparam(tfm)
    return Tuple(apply(tfm, item, param) for item in items)
end


# Transform composition

"""
    Pipeline

A `Pipeline` combines several transformations and applies them sequentially
"""
struct Pipeline <: Transform
    transforms
end

apply(pipeline::Pipeline, item::Item, param) = foldl(
    (item, tfm) -> apply(tfm, item),
    pipeline.transforms;
    init = item)
apply(pipeline::Pipeline, items::Tuple) = foldl(
    (items, tfm) -> apply(tfm, items),
    pipeline.transforms;
    init = items)

Base.:(|>)(t1::Transform, t2::Transform) = compose(t1, t2)

# FIXME: custom composition on pipelines will not work easily
"""
    compose(tfm1, tfm2)
    tfm1 |> tfm2

Combines two `Transform`s. The default behavior is to group them
in a `Pipeline` that applies them sequentially.

Custom methods can be used to implement performance improvements,
for example merging two affine transformations.
"""
compose(tfm1::Transform, tfm2::Transform) = Pipeline([tfm1, tfm2])
function compose(pipeline::Pipeline, tfm::Transform)
    push!(pipeline.transforms, tfm)
    return pipeline
end

# Basic Transforms

"""
    Lambda(f)

Applies function `f` to an `Item`
"""
struct Lambda <: Transform
    f
end

apply(tfm::Lambda, item::Item, param) = tfm.f(item)

# None

"""
    Identity()

Does not transform an `item`
"""
struct Identity <: Transform end
apply(tfm::Identity, item::Item, param) = item

# Transformer

"""
    Either(transformations, probabilities)

Chooses one of `transformations` with probability in
`probabilities`
"""
struct Either{N} <: Transform
    transforms::NTuple{N, Transform}
    probabilities::NTuple{N, Float32}
end

"""
    Either(transform, p)

Applies `tfm` with probability `p`
"""
Either(t::Transform, p::Number) = Either{2}((t, Identity()), (p, 1 - p))


getparam(t::Either)::Int = pick(t.probabilities, rand())
pick(ps, r) = r < ps[1] ? 1 : pick(ps[2:end], r - ps[1]) + 1

# FIXME: revisit this bit
#(t::Either)(item::Item, param::Int) = (@show param; t.transforms[param](item))

function apply(tfm::Either, items::Tuple)
    t = tfm.transforms[getparam(tfm)]
    return t(items)
end

function apply(tfm::Either, items::Tuple, param)
    t = tfm.transforms[param]
    return t(items)
end
