# Policy-based augmentations apply randomly chosen operations from a fixed set, each
# with a magnitude taken from a table. This file defines the operations and what
# `RandAugment`, `TrivialAugmentWide`, `AutoAugment` and `AugMix` share. The tables
# follow `torchvision.transforms.v2`.

# ## Geometric operations
#
# The geometric operations keep the canvas, so that they can be applied one after the
# other to images, masks and keypoints. They take 2D data, in the layout (row, column).
# Policy augmentations check the dimensionality before drawing anything, so that they do
# not fail only when a geometric operation happens to be drawn.

# Shear about the first pixel. The coordinate along `dim` moves by `factor` times the
# other one, so `dim = 2` moves pixels sideways.
struct Shear <: ProjectiveTransform
    dim::Int
    factor::Float64
end

# Translation by a whole number of pixels along `dim`.
struct Translate <: ProjectiveTransform
    dim::Int
    offset::Int
end

const CanvasOperation = Union{Shear, Translate}

function getprojection(tfm::Shear, bounds::Bounds{2}; randstate = nothing)
    f = Float32(tfm.factor)
    M = tfm.dim == 2 ? SMatrix{2, 2, Float32}(1, f, 0, 1) : SMatrix{2, 2, Float32}(1, 0, f, 1)
    return recenter(LinearMap(M), SVector{2, Float32}(first.(bounds.rs)))
end

function getprojection(tfm::Translate, bounds::Bounds{2}; randstate = nothing)
    return Translation(SVector{2, Float32}(ntuple(d -> d == tfm.dim ? tfm.offset : 0, 2)))
end

projectionbounds(::CanvasOperation, P, bounds::Bounds; randstate = nothing) = bounds

# ## Operations
#
# An operation has a name and a table of magnitudes, from the weakest to the strongest
# one. Signed operations are applied with either sign. Translations are given in
# pixels, or as a fraction of the side length if `relative`.
struct AugmentationOp
    name::Symbol
    magnitudes::Vector{Float64}
    signed::Bool
    relative::Bool
end

_linspace(lo, hi, n) = n == 1 ? [float(lo)] : collect(range(float(lo), float(hi); length = n))

_signed_op(name, hi, bins; relative = false) =
    AugmentationOp(name, _linspace(0, hi, bins), true, relative)

_unsigned_op(name, magnitudes = Float64[]) = AugmentationOp(name, magnitudes, false, false)

# The number of bits goes from `bits` down by `range` over the table.
_posterize_op(bins, bits, range) =
    _unsigned_op(:Posterize, [bits - round(Int, i / ((bins - 1) / range)) for i in 0:(bins - 1)])

_solarize_op(bins) = _unsigned_op(:Solarize, _linspace(1, 0, bins))

function _geometric_ops(bins; shear, translate, rotate, relative)
    return [_signed_op(:ShearX, shear, bins), _signed_op(:ShearY, shear, bins),
            _signed_op(:TranslateX, translate, bins; relative),
            _signed_op(:TranslateY, translate, bins; relative),
            _signed_op(:Rotate, rotate, bins)]
end

function _enhancement_ops(bins, hi)
    return [_signed_op(name, hi, bins) for name in (:Brightness, :Color, :Contrast, :Sharpness)]
end

_identity_op() = _unsigned_op(:Identity)

# Ops are built for the item at hand, since translations can depend on its size.
# A magnitude of zero leaves the item as it is.

_require_2d(::Bounds{2}) = nothing
_require_2d(::Bounds{N}) where N =
    throw(ArgumentError("policy augmentations need 2D data, got $N dimensions"))

function _translation(dim, magnitude, bounds, relative)
    offset = trunc(Int, relative ? magnitude * length(bounds.rs[dim]) : magnitude)
    return iszero(offset) ? Identity() : Translate(dim, offset)
end

function _shear(dim, factor)
    return iszero(factor) ? Identity() : Shear(dim, factor)
end

function _rotation(angle)
    return iszero(angle) ? Identity() : Rotate{2}(Dirac(float(angle)); expand = false)
end

_enhancement(T, magnitude) = iszero(magnitude) ? Identity() : T((1 + magnitude, 1 + magnitude))

const OPERATIONS = Dict{Symbol, Function}(
    :Identity => (m, bounds, relative) -> Identity(),
    :ShearX => (m, bounds, relative) -> _shear(2, m),
    :ShearY => (m, bounds, relative) -> _shear(1, m),
    :TranslateX => (m, bounds, relative) -> _translation(2, m, bounds, relative),
    :TranslateY => (m, bounds, relative) -> _translation(1, m, bounds, relative),
    :Rotate => (m, bounds, relative) -> _rotation(m),
    :Brightness => (m, bounds, relative) -> _enhancement(AdjustBrightness, m),
    :Color => (m, bounds, relative) -> _enhancement(AdjustSaturation, m),
    :Contrast => (m, bounds, relative) -> _enhancement(AdjustContrast, m),
    :Sharpness => (m, bounds, relative) -> iszero(m) ? Identity() : AdjustSharpness(1 + m),
    # AugMix reaches zero bits at its strongest magnitudes, but Posterize needs one.
    :Posterize => (m, bounds, relative) -> Posterize(clamp(round(Int, m), 1, 8)),
    :Solarize => (m, bounds, relative) -> Solarize(m),
    :AutoContrast => (m, bounds, relative) -> Autocontrast(),
    :Equalize => (m, bounds, relative) -> Equalize(),
    :Invert => (m, bounds, relative) -> Invert(),
)

_transform(op::AugmentationOp, magnitude, bounds) = OPERATIONS[op.name](magnitude, bounds, op.relative)

# ## Sampling and applying operations
#
# The random state of a policy augmentation is a list of steps `(op, magnitude)` with
# the sign included, so that all items passed together are transformed in the same way.

_sample_step(op::AugmentationOp, magnitude) =
    (op, op.signed && rand(Bool) ? -float(magnitude) : float(magnitude))

# The magnitude of bin `bin`, counted from zero, or none for an operation without.
_bin_step(op::AugmentationOp, bin) = _sample_step(op, isempty(op.magnitudes) ? 0.0 : op.magnitudes[bin + 1])

_random_bin_step(op::AugmentationOp, bins) = _bin_step(op, rand(0:(bins - 1)))

const Step = Tuple{AugmentationOp, Float64}

function apply_steps(steps, item, fill)
    _require_2d(getbounds(item))
    return _with_fill(item, fill) do item
        for (op, magnitude) in steps
            item = apply(_transform(op, magnitude, getbounds(item)), item)
        end
        return item
    end
end

# Masks keep their own extrapolation.
_with_fill(f, item, fill) = f(item)

function _with_fill(f, item::Image, fill::Union{Number, Colorant})
    result = f(@set item.extrapolate = fill)
    return @set result.extrapolate = item.extrapolate
end

_check_fill(::Nothing) = nothing
_check_fill(::Union{Number, Colorant}) = nothing
_check_fill(fill) = throw(ArgumentError("fill must be nothing, a number or a color"))

function _check_bins(bins)
    bins >= 2 || throw(ArgumentError("num_magnitude_bins must be at least 2"))
    return Int(bins)
end

abstract type PolicyAugmentation <: Transform end

function apply(tfm::PolicyAugmentation, item::Item; randstate = getrandstate(tfm))
    return apply_steps(randstate, item, tfm.fill)
end

# ## RandAugment

"""
    RandAugment(num_ops = 2, magnitude = 9; num_magnitude_bins = 31, fill = nothing)

Apply `num_ops` operations drawn uniformly at random from a fixed set, each with the
same strength, as in
[RandAugment](https://arxiv.org/abs/1909.13719). `magnitude` selects the strength as
an index in `0:num_magnitude_bins-1` into the table of each operation, from the
identity to the strongest value.

The operations are `Identity`, shearing and translating along each axis, rotation,
brightness, color (saturation), contrast and sharpness adjustments, posterization,
solarization, `AutoContrast` ([`Autocontrast`](@ref)) and [`Equalize`](@ref). Signed
operations are applied in either direction with equal probability. Shears are given
as the shear factor, translations as up to 150/331 of the side length, and rotations
up to 30 degrees. The tables are those of `torchvision.transforms.v2.RandAugment`.

The input is an [`Image`](@ref) with two spatial dimensions, with its masks and keypoints
passed alongside. Like the color transformations it uses, it expects pixel values in `[0, 1]`. The same operations apply to all items: the geometric ones to every item
and the others to the image alone. The interpolation and extrapolation of each item are
used as given: images interpolate linearly by default, where torchvision uses nearest
neighbor, which `Image(data; interpolate = BSpline(Constant()))` gives. A `fill` value,
a number or color, replaces the extrapolation of the image for the geometric operations.

See also [`TrivialAugmentWide`](@ref), [`AutoAugment`](@ref).

## Example

```julia
using DataAugmentation, TestImages

item = Image(testimage("lighthouse"))
tfm = RandAugment(2, 9)
titems = [apply(tfm, item) for _ in 1:8]
showgrid(titems; ncol = 4, npad = 16)
```
"""
struct RandAugment <: PolicyAugmentation
    ops::Vector{AugmentationOp}
    num_ops::Int
    magnitude::Int
    fill::Union{Nothing, Number, Colorant}
end

function RandAugment(num_ops::Integer = 2, magnitude::Integer = 9;
                     num_magnitude_bins::Integer = 31, fill = nothing)
    num_ops >= 0 || throw(ArgumentError("num_ops must be nonnegative"))
    bins = _check_bins(num_magnitude_bins)
    0 <= magnitude < bins || throw(ArgumentError("magnitude must be in 0:$(bins - 1)"))
    _check_fill(fill)
    ops = [_identity_op();
           _geometric_ops(bins; shear = 0.3, translate = 150 / 331, rotate = 30, relative = true);
           _enhancement_ops(bins, 0.9);
           _posterize_op(bins, 8, 4); _solarize_op(bins);
           _unsigned_op(:AutoContrast); _unsigned_op(:Equalize)]
    return RandAugment(ops, num_ops, magnitude, fill)
end

function getrandstate(tfm::RandAugment)
    return Step[_bin_step(op, tfm.magnitude) for op in rand(tfm.ops, tfm.num_ops)]
end

# ## TrivialAugmentWide

"""
    TrivialAugmentWide(; num_magnitude_bins = 31, fill = nothing)

Apply one operation drawn uniformly at random, with a magnitude drawn uniformly at
random from its table, as in
[TrivialAugment](https://arxiv.org/abs/2103.10158) (the wide variant).

The operations are those of [`RandAugment`](@ref), with wider ranges: shear factors up
to 0.99, translations up to 32 pixels, rotations up to 135 degrees, and enhancement
factors down to 0.01 and up to 1.99. The tables are those of
`torchvision.transforms.v2.TrivialAugmentWide`. Items, interpolation and `fill`
work as for `RandAugment`.

## Example

```julia
using DataAugmentation, TestImages

item = Image(testimage("lighthouse"))
tfm = TrivialAugmentWide()
titems = [apply(tfm, item) for _ in 1:8]
showgrid(titems; ncol = 4, npad = 16)
```
"""
struct TrivialAugmentWide <: PolicyAugmentation
    ops::Vector{AugmentationOp}
    fill::Union{Nothing, Number, Colorant}
end

function TrivialAugmentWide(; num_magnitude_bins::Integer = 31, fill = nothing)
    bins = _check_bins(num_magnitude_bins)
    _check_fill(fill)
    ops = [_identity_op();
           _geometric_ops(bins; shear = 0.99, translate = 32, rotate = 135, relative = false);
           _enhancement_ops(bins, 0.99);
           _posterize_op(bins, 8, 6); _solarize_op(bins);
           _unsigned_op(:AutoContrast); _unsigned_op(:Equalize)]
    return TrivialAugmentWide(ops, fill)
end

function getrandstate(tfm::TrivialAugmentWide)
    op = rand(tfm.ops)
    return Step[_random_bin_step(op, max(length(op.magnitudes), 1))]
end
