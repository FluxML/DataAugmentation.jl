# Buffering

Data augmentation pipelines often run on large amounts of data, and preallocating the output of the transformations saves memory allocations and garbage collection. Transformations that support [`apply!`](@ref) can write their result into an existing item.

```julia
buffer = tfm(item)
apply!(buffer, tfm, item)
```

A [`Buffered`](@ref) wrapper creates a buffer on its first call and reuses it through `apply!` on later calls.

```julia
btfm = Buffered(tfm)
titem = btfm(item)  # creates the buffer
titem = btfm(item)  # reuses the buffer
```

To support `apply!` in your own transformations, see [Defining transformations](tfminterface.md#Supporting-buffering).

## Who owns the result

For transformations that support `apply!`, `btfm(item)` returns the buffer itself, so the next call can overwrite it. Copy the result or create the wrapper with `copy = true` to keep it.

```julia
btfm = Buffered(tfm; copy = true)
titem = btfm(item)  # returns a copy
```

With `copy = true` the intermediate steps of a [`Sequence`](@ref) still reuse their buffers; only the final result is copied. This matters whenever several results are alive at the same time. A data loader that stacks the samples of a batch after loading all of them, such as `DataLoader(...; collate = true)` from MLUtils.jl, would otherwise receive the same array for every sample and silently build a batch that contains the last sample repeated.

## Multi-threading

Since `Buffered` only stores one set of buffers, don't share it between threads or tasks. Use [`BufferedThreadsafe`](@ref) instead: every concurrent call checks out its own buffers from a pool that grows on demand, and the result is always a copy that the caller owns. It is therefore safe in parallel loaders and with `collate = true`.

Buffering mainly saves allocations (and so garbage collection), not necessarily time: for a single transformation on one thread, it can cost as much in copies as it saves in allocations. Measure with your pipeline before relying on it.
