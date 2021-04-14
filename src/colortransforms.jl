
"""
    AdjustBrightness(δ = 0.2)
    AdjustBrightness(distribution)

Adjust the brightness of an image by a factor chosen uniformly
from `f ∈ [1-δ, 1+δ]` by multiplying each color channel by `f`.

You can also pass any `Distributions.Sampleable` from which the
factor is selected.

{cell=AdjustBrightness}
```julia
using DataAugmentation, TestImages

img = testimage("lighthouse")
tfm = AdjustBrightness(0.2)
apply(tfm, Image(img)) |> showitem
```
"""
struct AdjustBrightness{S<:Sampleable} <: Transform
    dist::S
end

AdjustBrightness(f::Real) = AdjustBrightness(Uniform(max(0, 1 - f), 1 + f))

getrandstate(tfm::AdjustBrightness) = rand(tfm.dist)


function apply(tfm::AdjustBrightness, item::Image; randstate = getrandstate(tfm))
    factor = randstate
    return setdata(item, adjustbrightness(itemdata(item), factor))
end

function apply!(buf, tfm::AdjustBrightness, item::Image; randstate = getrandstate(tfm))
    factor = randstate
    adjustbrightness!(itemdata(buf), itemdata(item), factor)
    return buf
end


function adjustbrightness(img, factor)
    return adjustbrightness!(copy(img), factor)
end


adjustbrightness!(img, factor) = adjustbrightness!(img, img, factor)

# TODO: add methods for non-RGB/Gray images
function adjustbrightness!(dst::AbstractArray{U}, img::AbstractArray{T}, factor) where {T, U}
    map!(dst, img) do x
        convert(U, clamp01(x * factor))
    end
end

##
"""
    AdjustContrast(factor = 0.2)
    AdjustContrast(distribution)

Adjust the contrast of an image by a factor chosen uniformly
from `f ∈ [1-δ, 1+δ]`.

Pixels `c` are transformed `c + μ*(1-f)` where `μ` is the mean color
of the image.

You can also pass any `Distributions.Sampleable` from which the
factor is selected.

{cell=AdjustBrightness}
```julia
using DataAugmentation, TestImages

img = testimage("lighthouse")
tfm = AdjustContrast(0.2)
apply(tfm, Image(img)) |> showitem
```
"""
struct AdjustContrast{S<:Sampleable} <: Transform
    dist::S
end

AdjustContrast(f::Real) = AdjustContrast(Uniform(max(0, 1 - f), 1 + f))

getrandstate(tfm::AdjustContrast) = rand(tfm.dist)


function apply(tfm::AdjustContrast, item::Image; randstate = getrandstate(tfm))
    factor = randstate
    return setdata(item, adjustcontrast(itemdata(item), factor))
end

function apply!(buf, tfm::AdjustContrast, item::Image; randstate = getrandstate(tfm))
    factor = randstate
    adjustcontrast!(itemdata(buf), itemdata(item), factor)
    return buf
end


function adjustcontrast(img, factor)
    return adjustcontrast!(copy(img), factor)
end


# TODO: add methods for non-RGB/Gray images
function adjustcontrast!(dst::AbstractArray{U}, img::AbstractArray{T}, factor) where {T, U}
    μ = mean(img)
    map!(dst, img) do x
        convert(U, clamp01(x + μ * (1 - factor)))
    end
end

adjustcontrast!(img, factor) = adjustcontrast!(img, img, factor)
