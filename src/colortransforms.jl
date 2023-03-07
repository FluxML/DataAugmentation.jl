
"""
    AdjustBrightness(δ = 0.2)
    AdjustBrightness(distribution)

Adjust the brightness of an image by a factor chosen uniformly
from `f ∈ [1-δ, 1+δ]` by multiplying each color channel by `f`.

You can also pass any `Distributions.Sampleable` from which the
factor is selected.

Pixels are clamped to [0,1] unless `clamp=false` is passed.

## Example

{cell=AdjustBrightness}
```julia
using DataAugmentation, TestImages

item = Image(testimage("lighthouse"))
tfm = AdjustBrightness(0.2)
titems = [apply(tfm, item) for _ in 1:8]
showgrid(titems; ncol = 4, npad = 16)
```
"""
struct AdjustBrightness{S<:Sampleable} <: Transform
    dist::S
    clamp::Bool
end

AdjustBrightness(f::Real; clamp::Bool=true) = AdjustBrightness(Uniform(max(0, 1 - f), 1 + f), clamp)

getrandstate(tfm::AdjustBrightness) = rand(tfm.dist)


function apply(tfm::AdjustBrightness, item::Image; randstate = getrandstate(tfm))
    factor = randstate
    return setdata(item, adjustbrightness(itemdata(item), factor, tfm.clamp))
end

function apply!(buf, tfm::AdjustBrightness, item::Image; randstate = getrandstate(tfm))
    factor = randstate
    adjustbrightness!(itemdata(buf), itemdata(item), factor, tfm.clamp)
    return buf
end


function adjustbrightness(img, factor, clamp)
    return adjustbrightness!(copy(img), factor, clamp)
end


adjustbrightness!(img, factor, clamp) = adjustbrightness!(img, img, factor, clamp)

# TODO: add methods for non-RGB/Gray images
function adjustbrightness!(dst::AbstractArray{U}, img::AbstractArray{T}, factor, clamp) where {T, U}
    map!(dst, img) do x
        convert(U, clamp ? clamp01(x * factor) : x * factor)
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

Pixels are clamped to [0,1] unless `clamp=false` is passed.

## Example

{cell=AdjustBrightness}
```julia
using DataAugmentation, TestImages

item = Image(testimage("lighthouse"))
tfm = AdjustContrast(0.2)
titems = [apply(tfm, item) for _ in 1:8]
showgrid(titems; ncol = 4, npad = 16)
```
"""
struct AdjustContrast{S<:Sampleable} <: Transform
    dist::S
    clamp::Bool
end

AdjustContrast(f::Real; clamp::Bool=true) = AdjustContrast(Uniform(max(0, 1 - f), 1 + f), clamp)

getrandstate(tfm::AdjustContrast) = rand(tfm.dist)


function apply(tfm::AdjustContrast, item::Image; randstate = getrandstate(tfm))
    factor = randstate
    return setdata(item, adjustcontrast(itemdata(item), factor, tfm.clamp))
end

function apply!(buf, tfm::AdjustContrast, item::Image; randstate = getrandstate(tfm))
    factor = randstate
    adjustcontrast!(itemdata(buf), itemdata(item), factor, tfm.clamp)
    return buf
end


function adjustcontrast(img, factor, clamp)
    return adjustcontrast!(copy(img), factor, clamp)
end


# TODO: add methods for non-RGB/Gray images
function adjustcontrast!(dst::AbstractArray{U}, img::AbstractArray{T}, factor, clamp) where {T, U}
    μ = mean(img)
    map!(dst, img) do x
        convert(U, clamp ? clamp01(x + μ * (1 - factor)) : x + μ * (1 - factor))
    end
end

adjustcontrast!(img, factor, clamp) = adjustcontrast!(img, img, factor, clamp)
