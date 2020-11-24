fmap(::Any, ::Nothing) = nothing
fmap(f, a::AbstractArray) = map(x -> fmap(f, x), a)
fmap(f, x::SVector) = f(x)
