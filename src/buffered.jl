
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


# Applying transforms inplace to tuples of items works fine when they always have the same
# length. When you have a varying number of items however, (e.g. different number of
# bounding boxes per sample) the number of items doesn't match up with the buffer and
# we fall back to regular `apply`.

function apply!(bufs::Union{Tuple, AbstractVector}, tfm::Transform, items::Union{Tuple, AbstractVector}; randstate = getrandstate(tfm))
    if length(bufs) == length(items)
        return map((item, buf) -> apply!(buf, tfm, item; randstate = randstate), items, bufs)
    else
        return apply(tfm, items; randstate = randstate)
    end
end


# ## Buffered transforms

"""
    Buffered(tfm; copy = false)

Wrapper transformation that applies `tfm` with `apply!`, reusing buffers that are
created the first time the wrapper is applied (one set of buffers per item type).

By default the returned item is the final buffer itself, so the next `apply`
overwrites it. This is a problem for code that keeps several results alive at the
same time, like a data loader that collates a batch after loading all its samples
(e.g. `DataLoader(...; collate = true)` from MLUtils.jl). With `copy = true` the
buffers are still reused internally, but the returned item is a fresh copy that
the caller owns.

Only one set of buffers is kept, so don't share a `Buffered` between threads; use
[`BufferedThreadsafe`](@ref) instead.
"""
mutable struct Buffered{T<:Transform} <: Transform
    tfm::T
    buffers::Dict
    copy::Bool
    Buffered(tfm::T; copy::Bool = false) where T = new{T}(tfm, Dict(), copy)
end

getrandstate(buffered::Buffered) = getrandstate(buffered.tfm)

function getbuffer!(buffered::Buffered, items::T) where T
    return get!(() -> makebuffer(buffered.tfm, items), buffered.buffers, T)
end

function apply(buffered::Buffered, items; randstate = getrandstate(buffered))
    buf = getbuffer!(buffered, items)
    res = apply!(buf, buffered.tfm, items; randstate = randstate)
    return buffered.copy ? copyitem(res) : res
end


function apply!(buf, buffered::Buffered, items; randstate = getrandstate(buffered))
    res = apply!(getbuffer!(buffered, items), buffered.tfm, items; randstate = randstate)
    copyitemdata!(buf, res)
    return buf
end


"""
    BufferedThreadsafe(tfm; n = Threads.nthreads())

Like [`Buffered`](@ref), but can be applied from multiple threads and tasks at the
same time. Each concurrent call checks out its own buffers from a pool that grows on
demand (`n` is the number of buffer sets created upfront) and returns them when done.

Since the buffers go back to the pool, the item returned by `apply` is always a copy
that the caller owns, and it is safe to keep several results alive at once.
`apply!(buf, btfm, item)` writes the result into `buf` instead.
"""
struct BufferedThreadsafe{T<:Transform} <: Transform
    tfm::T
    pool::Vector{Buffered{T}}
    lock::ReentrantLock
    function BufferedThreadsafe(tfm::T; n = Threads.nthreads()) where T<:Transform
        @assert n >= 0
        return new{T}(tfm, [Buffered(tfm) for _ in 1:n], ReentrantLock())
    end
end

getrandstate(btfm::BufferedThreadsafe) = getrandstate(btfm.tfm)
Base.show(io::IO, bt::BufferedThreadsafe) = print(io, "BufferedThreadsafe($(bt.tfm))")

function withbuffered(f, bt::BufferedThreadsafe)
    buffered = @lock bt.lock isempty(bt.pool) ? Buffered(bt.tfm) : pop!(bt.pool)
    try
        return f(buffered)
    finally
        @lock bt.lock push!(bt.pool, buffered)
    end
end

function apply(bt::BufferedThreadsafe, items; randstate = getrandstate(bt))
    return withbuffered(bt) do buffered
        copyitem(apply(buffered, items; randstate = randstate))
    end
end

function apply!(buf, bt::BufferedThreadsafe, items; randstate = getrandstate(bt))
    return withbuffered(bt) do buffered
        apply!(buf, buffered, items; randstate = randstate)
    end
end

# Resolve ambiguities with the generic methods for wrapped items and collections of items.

apply(buffered::Buffered, item::ItemWrapper; kwargs...) =
    invoke(apply, Tuple{Buffered, Any}, buffered, item; kwargs...)

apply(bufferedt::BufferedThreadsafe, item::ItemWrapper; kwargs...) =
    invoke(apply, Tuple{BufferedThreadsafe, Any}, bufferedt, item; kwargs...)

apply!(bufs::Union{Tuple, AbstractVector}, buffered::Buffered, items::Union{Tuple, AbstractVector}; kwargs...) =
    invoke(apply!, Tuple{Any, Buffered, Any}, bufs, buffered, items; kwargs...)

apply!(bufs::Union{Tuple, AbstractVector}, bufferedt::BufferedThreadsafe, items::Union{Tuple, AbstractVector}; kwargs...) =
    invoke(apply!, Tuple{Any, BufferedThreadsafe, Any}, bufs, bufferedt, items; kwargs...)


# Utils

copyitemdata!(buf::I, item::I) where I<:Item = (copy!(itemdata(buf), itemdata(item)); return buf)
copyitemdata!(bufw::I, itemw::I) where I<:ItemWrapper = (copyitemdata!(getwrapped(bufw), getwrapped(itemw)); return bufw)
copyitemdata!(bufs::T, items::T) where T<:Tuple = ((copyitemdata!.(bufs, items); bufs); return bufs)
copyitemdata!(bufs::T, items::T) where T<:AbstractVector = ((copyitemdata!.(bufs, items); bufs); return bufs)

copyitem(item::Item) = setdata(item, copydata(itemdata(item)))
copyitem(itemw::ItemWrapper) = setwrapped(itemw, copyitem(getwrapped(itemw)))
copyitem(items::Union{Tuple, AbstractVector}) = map(copyitem, items)

copydata(data::AbstractArray) = copy(data)
copydata(data) = data
