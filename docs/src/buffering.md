# Buffering

Data augmentation pipelines often run on large amounts of data, and preallocating the output of the transformations saves memory allocations and garbage collection. Transformations that support it can write their result into an existing item with [`apply!`](@ref) instead of allocating a new one:

```julia
buffer = apply(tfm, item)
apply!(buffer, tfm, item)
```

Managing the buffers manually quickly becomes tedious. For convenience, this library implements [`Buffered`](@ref), a transformation wrapper that will use a buffer internally. `btfm = Buffered(tfm)` will create a buffer the first time it is `apply`ed and then use it by internally calling `apply!`.

```julia
btfm = Buffered(tfm)
titem = apply(btfm, item)  # creates the buffer
titem = apply(btfm, item)  # reuses it, using apply! internally
```

Note that for transformations that support `apply!` the result is the buffer itself, so it is overwritten by the next call to `apply`. Copy it if you need to keep it.

Since `Buffered` only stores one buffer, you may run into problems when using it in a multi-threading context where different threads invalidate the buffer before it can be used. In that case, you can use [`BufferedThreadsafe`](@ref), a version of `Buffered` that keeps a separate buffer for every thread.

To support `apply!` in your own transformations, see [Defining transformations](tfminterface.md#Supporting-buffering).
