# AbstractTransform

"""
    AbstractTransform

Abstract type for all transforms

# `AbstractTransform` interface
TODO: Document
"""
abstract type AbstractTransform end

getparam(::AbstractTransform) = nothing

function (tfm::AbstractTransform)(items::Tuple)
    param = getparam(tfm)
    return Tuple(tfm(item, param) for item in items)
end

(tfm::AbstractTransform)(item::Item) = tfm(item, getparam(tfm))
(tfm::AbstractTransform)(item::ItemWrapper) = tfm(item, getparam(tfm))

# Pipeline

"""
    Pipeline

A `Pipeline` combines several transformations and applies them sequentially
"""
struct Pipeline <: AbstractTransform
    transforms
end

(pipeline::Pipeline)(item::Item, param) = foldl((item, f) -> f(item), pipeline.transforms; init = item)
(pipeline::Pipeline)(items::Tuple) = foldl((items, f) -> f(items), pipeline.transforms; init = items)

# FIXME: custom composition on pipelines will not work easily
Base.:(|>)(tfm1::AbstractTransform, tfm2::AbstractTransform) = Pipeline([tfm1, tfm2])
function Base.:(|>)(pipeline::Pipeline, tfm::AbstractTransform)
    push!(pipeline.transforms, tfm)
    return pipeline
end

# Basic Transforms

struct LambdaTransform <: AbstractTransform
    f
end

(t::LambdaTransform)(item::Item) = t.f(item)

# None

struct None <: AbstractTransform end
(t::None)(item::Item, param) = item

# Transformer

struct Either{N} <: AbstractTransform
    transforms::NTuple{N, AbstractTransform}
    probabilities::NTuple{N, Float32}
end
Either(t::AbstractTransform, p::AbstractFloat) = Either{2}((t, None()), (p, 1 - p))


getparam(t::Either)::Int = pick(t.probabilities, rand())
pick(ps, r) = r < ps[1] ? 1 : pick(ps[2:end], r - ps[1]) + 1

#(t::Either)(item::Item, param::Int) = (@show param; t.transforms[param](item))

function (either::Either)(items::Tuple)
    t = either.transforms[getparam(either)]
    return t(items)
end

function (either::Either)(items::Tuple, param)
    t = either.transforms[param]
    return t(items)
end
