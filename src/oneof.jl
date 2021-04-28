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


function makebuffer(oneof::OneOfProjective, items)
    return [makebuffer(tfm, items) for tfm in oneof.tfms]
end
