# Buffering

As mentioned in the section on [transformations](./tfminterface.md), you can implement [`apply!`](@ref) as an inplace version of [`apply`](@ref) to support buffered transformations. Usually, the result of a regular `apply` can be used as a buffer. You may write

```julia
buffer = apply(tfm, item)
apply!(buffer, tfm, item)
```

However, for some transformations, a different buffer is needed. [`Sequence`](@ref), for example, needs to reuse all intermediate results. That is why the buffer creation can be customized:

- [`makebuffer`](@ref)`(tfm, item)` creates a buffer `buf` that can be used in an `apply!` call: `apply!(buf, tfm, item)`.

---

Managing the buffers manually quickly becomes tedious. For convenience, this library implements [`Buffered`](@ref), a transformation wrapper that will use a buffer internally. `btfm = Buffered(tfm)` will create a buffer the first time it is `apply`ed and then use it by internally calling `apply!`.

```julia
buffered = Buffered(tfm)
buffer = apply(tfm, item)  # uses apply! internally
```

Since `Buffered` only stores one buffer, you may run into problems when using it in a multi-threading context where different threads invalidate the buffer before it can be used. In that case, you can use [`BufferedThreadsafe`](@ref), a version of `Buffered` that keeps a separate buffer for every thread. 