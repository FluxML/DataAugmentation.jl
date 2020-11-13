

"""
    makebuffer(tfm::Transform, item) = apply(tfm, item)

Allocate a buffer. Default to `buffer = apply(tfm, item)`.
"""
makebuffer(tfm::Transform, items) = apply(tfm, items)


"""
    apply!(buffer, tfm, item::I)

Applies `tfm` to `item`, mutating the preallocated `buffer`.

`buffer` can be obtained with `buffer = makebuffer(tfm, item)`

    apply!(buffer, tfm::Transform, item::I; randstate) = apply(tfm, item; randstate)

Default to `apply(tfm, item)` (non-mutating version).
"""
apply!(buf, tfm::Transform, items; randstate = getrandstate(tfm)) = apply(tfm, items, randstate = randstate)


function makebuffer(pipeline::Sequential, items)
    buffers = []
    for tfm in pipeline.transforms
        push!(buffers, makebuffer(tfm, items))
        items = apply(tfm, items)
    end
    return buffers
end


function apply!(buffers, pipeline::Sequential, items; randstate = getrandstate(pipeline))
    for (tfm, buffer, r) in zip(pipeline.transforms, buffers, randstate)
        items = apply!(buffer, tfm, items; randstate = r)
    end
    return items
end


# Inplace wrappers

mutable struct Inplace{T<:Transform} <: Transform
    tfm::T
    buffer
    Inplace(tfm::T, buffer = nothing) where T = new{T}(tfm, buffer)
end

getrandstate(inplace::Inplace) = getrandstate(inplace.tfm)

function apply(inplace::Inplace, items; randstate = getrandstate(inplace))
    if isnothing(inplace.buffer)
        inplace.buffer = makebuffer(inplace.tfm, items)
    end
    return apply!(inplace.buffer, inplace.tfm, items, randstate = randstate)
end

function apply!(buf, inplace::Inplace, items; randstate = getrandstate(inplace))
    titems = apply(inplace, items, randstate = randstate)
    copyitemdata!(buf, titems)
    return titems
end


struct InplaceThreadsafe
    inplaces::Vector{Inplace}
    function InplaceThreadsafe(tfm; n = Threads.nthreads())
        @assert n >= 1
        return new([Inplace(tfm) for _ in 1:n])
    end
end


function apply(inplacet::InplaceThreadsafe, items; kwargs...)
    inplacethread = inplacet.inplaces[Threads.threadid()]
    return apply(inplacethread, items; kwargs...)
end


function apply!(buf, inplacet::InplaceThreadsafe, items; kwargs...)
    inplacethread = inplacet.inplaces[Threads.threadid()]
    return apply!(buf, inplacethread, items; kwargs...)
end


# Utils

copyitemdata!(buf::I, item::I) where I<:Item = copy!(itemdata(buf), itemdata(item))
copyitemdata!(bufs::T, items::T) where T<:Tuple = (copyitemdata!.(bufs, items); bufs)
copyitemdata!(bufs::T, items::T) where T<:AbstractVector = (copyitemdata!.(bufs, items); bufs)
