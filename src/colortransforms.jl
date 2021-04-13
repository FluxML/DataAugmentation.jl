
struct AdjustBrightness{S<:Sampleable} <: Transform
    dist::S
end

AdjustBrightness(f::Real) = AdjustBrightness(Uniform(max(0, 1 - f), 1 + f))

getrandstate(tfm::AdjustBrightness) = rand(tfm.dist)


function apply(tfm::AdjustBrightness, item::AbstractArrayItem; randstate = getrandstate(tfm))
    factor = randstate
    return setdata(item, adjustbrightness(itemdata(item), factor))
end

function apply!(buf, tfm::AdjustBrightness, item::AbstractArrayItem; randstate = getrandstate(tfm))
    factor = randstate
    adjustbrightness!(itemdata(buf), itemdata(item), factor)
    return buf
end


function adjustbrightness(img, factor)
    return adjustbrightness!(copy(img), factor)
end


adjustbrightness!(img, factor) = adjustbrightness!(img, img, factor)

function adjustbrightness!(dst::AbstractArray{U}, img::AbstractArray{T}, factor) where {T, U}
    map!(dst, img) do x
        convert(U, clamp01(x * factor))
    end
end

##

struct AdjustContrast{S<:Sampleable} <: Transform
    dist::S
end

AdjustContrast(f::Real) = AdjustContrast(Uniform(max(0, 1 - f), 1 + f))

getrandstate(tfm::AdjustContrast) = rand(tfm.dist)


function apply(tfm::AdjustContrast, item::AbstractArrayItem; randstate = getrandstate(tfm))
    factor = randstate
    return setdata(item, adjustcontrast(itemdata(item), factor))
end

function apply!(buf, tfm::AdjustContrast, item::AbstractArrayItem; randstate = getrandstate(tfm))
    factor = randstate
    adjustcontrast!(itemdata(buf), itemdata(item), factor)
    return buf
end


function adjustcontrast(img, factor)
    return adjustcontrast!(copy(img), factor)
end

function adjustcontrast!(dst::AbstractArray{U}, img::AbstractArray{T}, factor) where {T, U}
    μ = mean(img)
    map!(dst, img) do x
        convert(U, clamp01(x + μ * (1 - factor)))
    end
end

adjustcontrast!(img, factor) = adjustcontrast!(img, img, factor)
