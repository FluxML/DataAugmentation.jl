# The learned policies of the AutoAugment paper, as in `torchvision.transforms.v2`. A
# policy is a list of sub-policies, one of which is drawn for each sample. Its
# operations are `(name, probability, magnitude)`, where the magnitude is an index in
# `0:9` into the table of the operation, or `nothing` for operations without one.
const AUTOAUGMENT_POLICIES = (
    imagenet = [
        ((:Posterize, 0.4, 8), (:Rotate, 0.6, 9)),
        ((:Solarize, 0.6, 5), (:AutoContrast, 0.6, nothing)),
        ((:Equalize, 0.8, nothing), (:Equalize, 0.6, nothing)),
        ((:Posterize, 0.6, 7), (:Posterize, 0.6, 6)),
        ((:Equalize, 0.4, nothing), (:Solarize, 0.2, 4)),
        ((:Equalize, 0.4, nothing), (:Rotate, 0.8, 8)),
        ((:Solarize, 0.6, 3), (:Equalize, 0.6, nothing)),
        ((:Posterize, 0.8, 5), (:Equalize, 1.0, nothing)),
        ((:Rotate, 0.2, 3), (:Solarize, 0.6, 8)),
        ((:Equalize, 0.6, nothing), (:Posterize, 0.4, 6)),
        ((:Rotate, 0.8, 8), (:Color, 0.4, 0)),
        ((:Rotate, 0.4, 9), (:Equalize, 0.6, nothing)),
        ((:Equalize, 0.0, nothing), (:Equalize, 0.8, nothing)),
        ((:Invert, 0.6, nothing), (:Equalize, 1.0, nothing)),
        ((:Color, 0.6, 4), (:Contrast, 1.0, 8)),
        ((:Rotate, 0.8, 8), (:Color, 1.0, 2)),
        ((:Color, 0.8, 8), (:Solarize, 0.8, 7)),
        ((:Sharpness, 0.4, 7), (:Invert, 0.6, nothing)),
        ((:ShearX, 0.6, 5), (:Equalize, 1.0, nothing)),
        ((:Color, 0.4, 0), (:Equalize, 0.6, nothing)),
        ((:Equalize, 0.4, nothing), (:Solarize, 0.2, 4)),
        ((:Solarize, 0.6, 5), (:AutoContrast, 0.6, nothing)),
        ((:Invert, 0.6, nothing), (:Equalize, 1.0, nothing)),
        ((:Color, 0.6, 4), (:Contrast, 1.0, 8)),
        ((:Equalize, 0.8, nothing), (:Equalize, 0.6, nothing)),
    ],
    cifar10 = [
        ((:Invert, 0.1, nothing), (:Contrast, 0.2, 6)),
        ((:Rotate, 0.7, 2), (:TranslateX, 0.3, 9)),
        ((:Sharpness, 0.8, 1), (:Sharpness, 0.9, 3)),
        ((:ShearY, 0.5, 8), (:TranslateY, 0.7, 9)),
        ((:AutoContrast, 0.5, nothing), (:Equalize, 0.9, nothing)),
        ((:ShearY, 0.2, 7), (:Posterize, 0.3, 7)),
        ((:Color, 0.4, 3), (:Brightness, 0.6, 7)),
        ((:Sharpness, 0.3, 9), (:Brightness, 0.7, 9)),
        ((:Equalize, 0.6, nothing), (:Equalize, 0.5, nothing)),
        ((:Contrast, 0.6, 7), (:Sharpness, 0.6, 5)),
        ((:Color, 0.7, 7), (:TranslateX, 0.5, 8)),
        ((:Equalize, 0.3, nothing), (:AutoContrast, 0.4, nothing)),
        ((:TranslateY, 0.4, 3), (:Sharpness, 0.2, 6)),
        ((:Brightness, 0.9, 6), (:Color, 0.2, 8)),
        ((:Solarize, 0.5, 2), (:Invert, 0.0, nothing)),
        ((:Equalize, 0.2, nothing), (:AutoContrast, 0.6, nothing)),
        ((:Equalize, 0.2, nothing), (:Equalize, 0.6, nothing)),
        ((:Color, 0.9, 9), (:Equalize, 0.6, nothing)),
        ((:AutoContrast, 0.8, nothing), (:Solarize, 0.2, 8)),
        ((:Brightness, 0.1, 3), (:Color, 0.7, 0)),
        ((:Solarize, 0.4, 5), (:AutoContrast, 0.9, nothing)),
        ((:TranslateY, 0.9, 9), (:TranslateY, 0.7, 9)),
        ((:AutoContrast, 0.9, nothing), (:Solarize, 0.8, 3)),
        ((:Equalize, 0.8, nothing), (:Invert, 0.1, nothing)),
        ((:TranslateY, 0.7, 9), (:AutoContrast, 0.9, nothing)),
    ],
    svhn = [
        ((:ShearX, 0.9, 4), (:Invert, 0.2, nothing)),
        ((:ShearY, 0.9, 8), (:Invert, 0.7, nothing)),
        ((:Equalize, 0.6, nothing), (:Solarize, 0.6, 6)),
        ((:Invert, 0.9, nothing), (:Equalize, 0.6, nothing)),
        ((:Equalize, 0.6, nothing), (:Rotate, 0.9, 3)),
        ((:ShearX, 0.9, 4), (:AutoContrast, 0.8, nothing)),
        ((:ShearY, 0.9, 8), (:Invert, 0.4, nothing)),
        ((:ShearY, 0.9, 5), (:Solarize, 0.2, 6)),
        ((:Invert, 0.9, nothing), (:AutoContrast, 0.8, nothing)),
        ((:Equalize, 0.6, nothing), (:Rotate, 0.9, 3)),
        ((:ShearX, 0.9, 4), (:Solarize, 0.3, 3)),
        ((:ShearY, 0.8, 8), (:Invert, 0.7, nothing)),
        ((:Equalize, 0.9, nothing), (:TranslateY, 0.6, 6)),
        ((:Invert, 0.9, nothing), (:Equalize, 0.6, nothing)),
        ((:Contrast, 0.3, 3), (:Rotate, 0.8, 4)),
        ((:Invert, 0.8, nothing), (:TranslateY, 0.0, 2)),
        ((:ShearY, 0.7, 6), (:Solarize, 0.4, 8)),
        ((:Invert, 0.6, nothing), (:Rotate, 0.8, 4)),
        ((:ShearY, 0.3, 7), (:TranslateX, 0.9, 3)),
        ((:ShearX, 0.1, 6), (:Invert, 0.6, nothing)),
        ((:Solarize, 0.7, 2), (:TranslateY, 0.6, 7)),
        ((:ShearY, 0.8, 4), (:Invert, 0.8, nothing)),
        ((:ShearX, 0.7, 9), (:TranslateY, 0.8, 3)),
        ((:ShearY, 0.8, 5), (:AutoContrast, 0.7, nothing)),
        ((:ShearX, 0.7, 2), (:Invert, 0.1, nothing)),
    ],
)

function autoaugment_ops()
    return [_geometric_ops(10; shear = 0.3, translate = 150 / 331, rotate = 30, relative = true);
            _enhancement_ops(10, 0.9);
            _posterize_op(10, 8, 4); _solarize_op(10);
            _unsigned_op(:AutoContrast); _unsigned_op(:Equalize); _unsigned_op(:Invert)]
end

"""
    AutoAugment(policy = :imagenet; fill = nothing)
    AutoAugment(subpolicies; fill = nothing)

Apply a sub-policy drawn uniformly at random from a policy learned by
[AutoAugment](https://arxiv.org/abs/1805.09501). A sub-policy is a pair of operations,
each applied with its own probability and with a fixed magnitude and a random sign.

`policy` is `:imagenet`, `:cifar10` or `:svhn`, the policies of
`torchvision.transforms.v2.AutoAugment`. You can also pass your own `subpolicies`, a
collection of sub-policies, each a collection of operations `(name, probability, magnitude)`.
The names are `:ShearX`, `:ShearY`, `:TranslateX`, `:TranslateY`, `:Rotate`, `:Brightness`,
`:Color`, `:Contrast`, `:Sharpness`, `:Posterize`, `:Solarize`, `:AutoContrast`, `:Equalize`
and `:Invert`. The magnitude is an index in `0:9` into the table of the operation, and
`nothing` for the last three. The tables are those of [`RandAugment`](@ref), with ten bins.

Items, interpolation and `fill` work as for `RandAugment`.

## Example

```julia
using DataAugmentation, TestImages

item = Image(testimage("lighthouse"))
tfm = AutoAugment(:cifar10)
titems = [apply(tfm, item) for _ in 1:8]
showgrid(titems; ncol = 4, npad = 16)

custom = AutoAugment([((:Rotate, 0.8, 5), (:Equalize, 0.5, nothing))])
```
"""
struct AutoAugment <: PolicyAugmentation
    subpolicies::Vector{Vector{Tuple{AugmentationOp, Float64, Float64}}}
    fill::Union{Nothing, Number, Colorant}
end

function AutoAugment(policy::Symbol = :imagenet; fill = nothing)
    policy in keys(AUTOAUGMENT_POLICIES) ||
        throw(ArgumentError("policy must be one of $(join(repr.(keys(AUTOAUGMENT_POLICIES)), ", ")), got $(repr(policy))"))
    return AutoAugment(AUTOAUGMENT_POLICIES[policy]; fill)
end

function AutoAugment(subpolicies; fill = nothing)
    isempty(subpolicies) && throw(ArgumentError("pass at least one sub-policy"))
    _check_fill(fill)
    ops = Dict(op.name => op for op in autoaugment_ops())
    return AutoAugment(map(sub -> _autoaugment_subpolicy(ops, sub), collect(subpolicies)), fill)
end

function _autoaugment_subpolicy(ops, subpolicy)
    return map(collect(subpolicy)) do (name, probability, bin)
        haskey(ops, name) || throw(ArgumentError("unknown operation $(repr(name))"))
        0 <= probability <= 1 || throw(ArgumentError("probability must be in [0, 1], got $probability"))
        op = ops[name]
        if isempty(op.magnitudes)
            isnothing(bin) || throw(ArgumentError("$name has no magnitude, use nothing"))
        else
            bin isa Integer && 0 <= bin < length(op.magnitudes) ||
                throw(ArgumentError("the magnitude of $name must be an integer in 0:$(length(op.magnitudes) - 1)"))
        end
        return (op, Float64(probability), isnothing(bin) ? 0.0 : op.magnitudes[bin + 1])
    end
end

function getrandstate(tfm::AutoAugment)
    steps = Step[]
    for (op, probability, magnitude) in rand(tfm.subpolicies)
        rand() < probability && push!(steps, _sample_step(op, magnitude))
    end
    return steps
end
