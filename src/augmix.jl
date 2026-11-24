"""
    AugMix(; severity = 3, mixture_width = 3, chain_depth = -1, alpha = 1,
             all_ops = true, fill = nothing)

Mix an image with augmented versions of itself, as in
[AugMix](https://arxiv.org/abs/1912.02781). It draws `mixture_width` chains of random
operations, each of `chain_depth` operations or, for a negative `chain_depth`, of one to
three. The result is a convex combination of the image and the chains, with weights from
a Dirichlet distribution and a Beta distribution with parameter `alpha`.

Operations draw their magnitude from the `severity` weakest entries of their tables, with
`severity` in `1:10`. The operations are shearing and translating along each axis,
rotation, posterization, solarization, [`Autocontrast`](@ref) and [`Equalize`](@ref),
plus adjustments of brightness, color (saturation), contrast and sharpness with
`all_ops = true`. The tables are those of `torchvision.transforms.v2.AugMix`:
translations up to a third of the side length. The exception is posterization at the
two strongest magnitudes, which would keep no bits at all, and keeps one bit here.

AugMix takes an [`Image`](@ref) with two spatial dimensions and only that. Masks and keypoints
are rejected since the chains warp the image differently, so they can't be mixed. The
interpolation and extrapolation of the image are used as given. A `fill` value, a number
or color, replaces its extrapolation for the geometric operations. Pixels are returned in
the element type of the image.

## Example

```julia
using DataAugmentation, TestImages

item = Image(testimage("lighthouse"))
tfm = AugMix(severity = 5)
titems = [apply(tfm, item) for _ in 1:8]
showgrid(titems; ncol = 4, npad = 16)
```
"""
struct AugMix <: Transform
    ops::Vector{AugmentationOp}
    severity::Int
    mixture_width::Int
    chain_depth::Int
    alpha::Float64
    fill::Union{Nothing, Number, Colorant}
end

function AugMix(; severity::Integer = 3, mixture_width::Integer = 3, chain_depth::Integer = -1,
                alpha::Real = 1, all_ops::Bool = true, fill = nothing)
    1 <= severity <= 10 || throw(ArgumentError("severity must be in 1:10, got $severity"))
    mixture_width >= 1 || throw(ArgumentError("mixture_width must be positive"))
    chain_depth != 0 || throw(ArgumentError("chain_depth must be positive, or negative for a random depth"))
    isfinite(alpha) && alpha > 0 || throw(ArgumentError("alpha must be positive and finite"))
    _check_fill(fill)
    ops = [_geometric_ops(10; shear = 0.3, translate = 1 / 3, rotate = 30, relative = true);
           _posterize_op(10, 4, 4); _solarize_op(10);
           _unsigned_op(:AutoContrast); _unsigned_op(:Equalize)]
    all_ops && append!(ops, _enhancement_ops(10, 0.9))
    return AugMix(ops, severity, mixture_width, chain_depth, alpha, fill)
end

function getrandstate(tfm::AugMix)
    original = rand(Beta(tfm.alpha, tfm.alpha))
    weights = rand(Dirichlet(tfm.mixture_width, tfm.alpha)) .* (1 - original)
    chains = map(1:tfm.mixture_width) do _
        depth = tfm.chain_depth > 0 ? tfm.chain_depth : rand(1:3)
        Step[_random_bin_step(rand(tfm.ops), tfm.severity) for _ in 1:depth]
    end
    return (; original, weights, chains)
end

function apply(tfm::AugMix, item::Image; randstate = getrandstate(tfm))
    src = itemdata(item)
    T = promote_type(Float32, floattype(eltype(eltype(src))))
    mix = T(randstate.original) .* _photometric_color.(src)
    for (weight, steps) in zip(randstate.weights, randstate.chains)
        chain = itemdata(apply_steps(steps, item, tfm.fill))
        mix .+= T(weight) .* _photometric_color.(chain)
    end
    return setdata(item, _mixed_pixel.(src, mix))
end

function apply(::AugMix, item::Item; randstate = nothing)
    throw(ArgumentError("AugMix only applies to images, got $(nameof(typeof(item)))"))
end

# The weights sum to one up to rounding, which fixed-point pixels do not tolerate.
_mixed_pixel(pixel::T, value) where T =
    _restore_photometric_color(pixel, eltype(T) <: FixedPoint ? _map_channels(clamp01, value) : value)
