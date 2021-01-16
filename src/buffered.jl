
"""
    makebuffer(tfm, item)

Create a buffer `buf` that can be used in a call to `apply!(buf, tfm, item)`.
Default to `buffer = apply(tfm, item)`.

You only need to implement this if the default `apply(tfm, item)` isn't
enough. See `apply(tfm::Sequence, item)` for an example of this.
"""
makebuffer(tfm::Transform, items) = apply(tfm, items)


"""
    apply!(buffer::I, tfm, item::I)

Applies `tfm` to `item`, mutating the preallocated `buffer`.

`buffer` can be obtained with `buffer = makebuffer(tfm, item)`

    apply!(buffer, tfm::Transform, item::I; randstate) = apply(tfm, item; randstate)

Default to `apply(tfm, item)` (non-mutating version).
"""
apply!(buf, tfm::Transform, items; randstate = getrandstate(tfm)) = apply(tfm, items, randstate = randstate)
apply!(buf, tfm::Transform, item::Item; randstate = getrandstate(tfm)) = apply(tfm, item, randstate = randstate)


# Applying transforms inplace to tuples of items works fine when they always have the same
# length. When you have a varying number of items however, (e.g. different number of
# bounding boxes per sample) the number of items doesn't match up with the buffer and
# we fall back to regular `apply`.

function apply!(bufs::Tuple, tfm::Transform, items::Tuple; randstate = getrandstate(tfm))
    if length(bufs) == length(items)
        return map((item, buf) -> apply!(buf, tfm, item; randstate = randstate), items, bufs)
    else
        return apply(tfm, items; randstate = randstate)
    end
end


# ## Buffered transforms

mutable struct Buffered{T<:Transform} <: Transform
    tfm::T
    buffer
    Buffered(tfm::T, buffer = nothing) where T = new{T}(tfm, buffer)
end

getrandstate(buffered::Buffered) = getrandstate(buffered.tfm)

function apply(buffered::Buffered, items::Tuple; randstate = getrandstate(buffered))
    if isnothing(buffered.buffer)
        buffered.buffer = makebuffer(buffered.tfm, items)
    end
    titems = apply!(buffered.buffer, buffered.tfm, items, randstate = randstate)
    return titems
end

apply(buffered::Buffered, item::Item; randstate = getrandstate(buffered)) =
    apply(buffered, (item,); randstate = randstate) |> only


function apply!(buf::Tuple, buffered::Buffered, items::Tuple; randstate = getrandstate(buffered))
    if isnothing(buffered.buffer)
        buffered.buffer = makebuffer(buffered.tfm, items)
    end
    buffered.buffer = apply!(buffered.buffer, buffered.tfm, items; randstate = randstate)
    copyitemdata!(buf, buffered.buffer)
    return buf
end

function apply!(buf::I, buffered::Buffered, item::I; randstate = getrandstate(buffered)) where {I<:Item}
    bufs, items = (buf,), (item,)
    titems = apply!(bufs, buffered, items; randstate = randstate)
    return only(titems)
end


struct BufferedThreadsafe
    buffereds::Vector{Buffered}
    function BufferedThreadsafe(tfm; n = Threads.nthreads())
        @assert n >= 1
        return new([Buffered(tfm) for _ in 1:n])
    end
end

Base.show(io::IO, bt::BufferedThreadsafe) = print(io, "BufferedThreadsafe($(bt.buffereds[1].tfm))")


function apply(bufferedt::BufferedThreadsafe, items; kwargs...)
    bufferedthread = bufferedt.buffereds[Threads.threadid()]
    return apply(bufferedthread, items; kwargs...)
end


function apply!(buf, bufferedt::BufferedThreadsafe, items; kwargs...)
    bufferedthread = bufferedt.buffereds[Threads.threadid()]
    return apply!(buf, bufferedthread, items; kwargs...)
end


# Utils

copyitemdata!(buf::I, item::I) where I<:Item = copy!(itemdata(buf), itemdata(item))
copyitemdata!(bufs::T, items::T) where T<:Tuple = (copyitemdata!.(bufs, items); bufs)
copyitemdata!(bufs::T, items::T) where T<:AbstractVector = (copyitemdata!.(bufs, items); bufs)
