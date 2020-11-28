# ## `projective/bounds.jl`
#
# To make it easier to handle spatial bounds, we define
# some helpers here.
#
# [`makebounds`](#) creates bounds from side lengths or ranges.


"""
    makebounds(sz[, T])
    makebounds(ranges[, T])

Helper for creating spatial bounds.

## Examples

{cell=makebounds}
```julia
using DataAugmentation: makebounds, showbounds
makebounds((100, 100), Float32)
```
{cell=makebounds}
```julia
makebounds((100, 100)) == makebounds((1:100, 1:100))
```

{cell=makebounds}
```julia
bounds = makebounds((100, 100))
showbounds(bounds)
```
"""
function makebounds(sz::NTuple{N, Int}, T = Float32) where N
    return makebounds(Tuple(1:a for a in sz), T)
end

function makebounds(ranges::NTuple{N, R}, T = Float32) where {N, R<:AbstractUnitRange}
    return collect(SVector{N, T}, Iterators.product(((r[begin]-1, r[end]) for r in ranges)...))
end


# [`boundsextrema`](#), [`boundsranges`](#) and [`boundssize`](#) represent
# the inverse of `makebounds`.

function boundsextrema(bounds::AbstractArray{<:SVector{N}}) where N
    mins = Tuple(floor(Int, minimum(getindex.(bounds, i))) for i = 1:N)
    maxs = Tuple(ceil(Int, maximum(getindex.(bounds, i))) for i = 1:N)
    return mins, maxs
end


function boundsranges(bounds)
    mins, maxs = boundsextrema(bounds)
    return UnitRange.(mins .+ 1, maxs)
end


"""
    boundssize(bounds)

`(100, 100) |> makebounds |> boundssize == (100, 100)`
"""
boundssize(bounds) = length.(boundsranges(bounds))
