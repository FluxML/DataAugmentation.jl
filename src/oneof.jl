"""
    OneOf(tfms)
    OneOf(tfms, ps)

Apply one of `tfms` selected randomly with probability `ps` each
or uniformly chosen if no `ps` is given.
"""
struct OneOf{T<:Transform, D<:Sampleable} <: Transform
    tfms::Vector{T}
    dist::D
end

function OneOf(tfms::Vector{<:Transform}, ps = ones(length(tfms)) ./ length(tfms))
    @assert length(tfms) == length(ps)
    return OneOf(tfms, Categorical(ps))
end

function getrandstate(oneof::OneOf)
    i = rand(oneof.dist)
    return i, getrandstate(oneof.tfms[i])
end


function apply(oneof::OneOf, item::Item; randstate = getrandstate(oneof))
    i, tfmrandstate = randstate
    return apply(oneof.tfms[i], item; randstate = tfmrandstate)
end


function makebuffer(oneof::OneOf, items)
    return Tuple([makebuffer(tfm, items) for tfm in oneof.tfms])
end

function apply!(bufs, oneof::OneOf, item::Item; randstate = getrandstate(oneof))
    i, tfmrandstate = randstate
    buf = bufs[i]
    return apply!(buf, oneof.tfms[i], item; randstate = tfmrandstate)
end

"""
    Maybe(tfm, p = 0.5) <: Transform

With probability `p`, apply transformation `tfm`.
"""
Maybe(tfm, p = 0.5) = OneOf([tfm, Identity()], [p, 1-p])


struct OneOfProjective{T<:Transform, D<:Sampleable} <: ProjectiveTransform
    tfms::Vector{T}
    dist::D
end


function OneOf(tfms::Vector{<:ProjectiveTransform}, ps = ones(length(tfms)) ./ length(tfms))
    @assert length(tfms) == length(ps)
    return OneOfProjective(tfms, Categorical(ps))
end

OneOf(tfms::Vector{<:ProjectiveTransform}, dist::Sampleable) = OneOfProjective(tfms, dist)


function getrandstate(oneof::OneOfProjective)
    i = rand(oneof.dist)
    return i, getrandstate(oneof.tfms[i])
end


function Maybe(tfm::ProjectiveTransform, p = 1/2)
    return OneOf([tfm, Project(IdentityTransformation())], [p, 1-p])
end


function getprojection(oneof::OneOfProjective, bounds; randstate = getrandstate(oneof))
    i, tfmrandstate = randstate
    return getprojection(oneof.tfms[i], bounds; randstate = tfmrandstate)
end

function projectionbounds(oneof::OneOfProjective, P, bounds; randstate = getrandstate(oneof))
    i, tfmrandstate = randstate
    return projectionbounds(oneof.tfms[i], P, bounds; randstate = tfmrandstate)
end


function makebuffer(oneof::OneOfProjective, items)
    return [makebuffer(tfm, items) for tfm in oneof.tfms]
end


"""
    RandomOrder(tfms)
    RandomOrder(tfms...)

Apply all of the transformations `tfms`, a vector, tuple or list of arguments, in a
random order. Each call draws a new permutation, and applies the same one to all items
of a tuple. The order changes the shape of the buffers in between, so [`Buffered`](@ref)
does not reuse any for it.

See also [`OneOf`](@ref) to apply a single randomly selected transformation.

## Examples

```julia
tfm = RandomOrder(AdjustBrightness(0.2), GaussianBlur(1), FlipX())
```
"""
struct RandomOrder{T<:Tuple{Vararg{Transform}}} <: Transform
    transforms::T

    function RandomOrder(transforms::T) where T<:Tuple{Vararg{Transform}}
        isempty(transforms) && throw(ArgumentError("pass at least one transform"))
        return new{T}(transforms)
    end
end

RandomOrder(transforms::Transform...) = RandomOrder(transforms)
RandomOrder(transforms::AbstractVector{<:Transform}) = RandomOrder(Tuple(transforms))

function getrandstate(tfm::RandomOrder)
    return Tuple(randperm(length(tfm.transforms))), map(getrandstate, tfm.transforms)
end

function apply(tfm::RandomOrder, items::Tuple; randstate = getrandstate(tfm))
    order, states = randstate
    for i in order
        items = apply(tfm.transforms[i], items; randstate = states[i])
    end
    return items
end

apply(tfm::RandomOrder, item::Item; randstate = getrandstate(tfm)) =
    apply(tfm, (item,); randstate) |> only

apply(tfm::RandomOrder, item::ItemWrapper; randstate = getrandstate(tfm)) =
    apply(tfm, (item,); randstate) |> only
