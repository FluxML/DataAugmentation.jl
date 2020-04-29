fmap(f, x::Nothing) = nothing
fmap(f, x) = f(x)
