```@setup tsm
using DataAugmentation, TestImages, ImageShow
```
# Transformations

Transformations are applied to [items](items.md) with [`apply`](@ref)`(tfm, item)`. This page shows how to combine and randomize them and gives an overview of the available ones; the [Gallery](gallery.md) shows what the spatial transformations look like, and the [API reference](ref.md) documents each in detail.

## Composing transformations

Using transformations is easy. Simply `compose` them with `|>`:

```@example tsm
tfm = Rotate(10) |> ScaleRatio((0.7,0.1,1.2)) |> FlipX() |> Crop((128, 128))
```

[Projective transformations](projective.md) are fused into a single, more efficient transformation; the rest are applied one after the other.

## Affine transformations

Affine transformations are a kind of [projective transformation](projective.md) that can be composed very efficiently. They apply to [`Image`](@ref)s and the keypoint-based items alike, in 2 and 3 dimensions.

- [`FlipX`](@ref)`()`, [`FlipY`](@ref)`()` and [`FlipZ`](@ref)`()` flip along a named axis according to the item's dimensionality. Use [`FlipDim`](@ref)`{N}(dim)` to specify the dimension directly.
- [`Reflect`](@ref)`(γ)` reflects 2D data around its center by an angle chosen uniformly from `[-γ, γ]` degrees.
- [`Rotate`](@ref)`(γ)` rotates 2D data around its center by an angle chosen uniformly from `[-γ, γ]` degrees; `Rotate(α, β, γ)` rotates 3D data, and [`RotateX`](@ref), [`RotateY`](@ref) and [`RotateZ`](@ref) rotate around a single axis.
- [`ScaleRatio`](@ref)`(ratios)` scales the sides by the given ratios, [`ScaleFixed`](@ref)`(sizes)` scales them to the given sizes disregarding the aspect ratio, and [`ScaleKeepAspect`](@ref)`(minlengths)` scales so that the sides are at least `minlengths`, keeping the aspect ratio. [`Zoom`](@ref)`(scales)` zooms in by a factor chosen from the interval `scales`.
- [`WarpAffine`](@ref)`(σ)` randomly moves three corners of the item, which is similar to a random translation, shear and rotation.

## Crops

To get a cropped result, simply `compose` any projective transformation with a crop:

- [`CenterCrop`](@ref)`(sz)` and [`RandomCrop`](@ref)`(sz)` crop a region of size `sz` from the center or from a random location. They are special cases of [`Crop`](@ref)`(sz, from)`. With the `padding` keyword the item is padded on every side before cropping, as in torchvision: `RandomCrop((32, 32); padding = 4)` takes a random 32×32 window from a 32×32 image padded by 4. The padded region is filled according to the `extrapolate` option of the item.
- [`CenterResizeCrop`](@ref)`(sz)` and [`RandomResizeCrop`](@ref)`(sz)` first scale the item with [`ScaleKeepAspect`](@ref) and then crop it from the center or from a random location, so that images of any size end up with the same size `sz`. The result is moved to the origin with [`PinOrigin`](@ref)`()`.
- [`ResizePadDivisible`](@ref)`(sz, by)` scales the item with [`ScaleKeepAspect`](@ref) and pads it so that its side lengths are divisible by `by`.

## Color transformations

Color transformations operate on [`Image`](@ref) items. You can apply them to an image together with its masks or keypoints in the same pipeline.

- [`AdjustBrightness`](@ref), [`AdjustContrast`](@ref), [`AdjustSaturation`](@ref), and [`AdjustHue`](@ref) sample an adjustment for a single property. [`ColorJitter`](@ref) applies those adjustments in random order.
- [`Grayscale`](@ref) and [`ToRGB`](@ref) convert between grayscale and RGB. [`RandomGrayscale`](@ref) converts to grayscale with a configurable probability.
- [`GaussianBlur`](@ref), [`GaussianNoise`](@ref), [`AdjustSharpness`](@ref), and [`JPEG`](@ref) change the detail or quality of an image. Load `JpegTurbo` with `using JpegTurbo` before applying `JPEG`.
- [`AdjustGamma`](@ref), [`Solarize`](@ref), [`Posterize`](@ref), [`Invert`](@ref), [`Equalize`](@ref), and [`Autocontrast`](@ref) adjust channel intensities.
- [`RandomChannelPermutation`](@ref) shuffles the RGB channels. [`RandomPhotometricDistort`](@ref) combines channel permutation with randomly selected brightness, contrast, saturation, and hue adjustments.

## Erasing and batch mixing

[`RandomErasing`](@ref) fills a random rectangle in a 2D image with a constant or with noise. Masks and keypoints passed alongside the image are preserved.

[`MixUp`](@ref) blends image pairs and their labels. [`CutMix`](@ref) copies a rectangle from one image into another and weights the labels by the area each image contributes. Both work on an [`ImageBatch`](@ref) after collation. The last image axis indexes samples, and each label column represents one sample.

```@example mixing
using DataAugmentation, Images

samples = [Image(rand(RGB{Float32}, 24, 32)) for _ in 1:4]
sample_tfm = RandomErasing(value = :random) |> ImageToTensor()
images = stack(itemdata(apply(sample_tfm, image)) for image in samples)

batch = ImageBatch(images, [1, 2, 3, 2]; num_classes = 3)
mixed_images, mixed_labels = itemdata(apply(CutMix(), batch))
size(mixed_images), size(mixed_labels)
```

## Stochastic transformations

When augmenting data, it is often useful to apply a transformation only with some probability or choose from a set of transformations. Unlike in other data augmentation libraries like *albumentations*, in DataAugmentation.jl you can use wrapper transformations for this functionality.

- [`Maybe`](@ref)`(tfm, p = 0.5)` applies a transformation with probability `p`; and
- [`OneOf`](@ref)`([tfm1, tfm2])` randomly selects a transformation to apply.

Let's say we have an image classification dataset. For most datasets, horizontally flipping the image does not change the label: a flipped image of a cat still shows a cat. So let's flip every image horizontally half of the time to improve the generalization of the model we might be training.

```@example tsm
item = Image(testimage("lighthouse"))
tfm = Maybe(FlipX())
titems = [apply(tfm, item) for _ in 1:8]
showgrid(titems; ncol = 4, npad = 16)
```

## Other transformations

[Preprocessing](preprocessing.md) covers the transformations that get data ready to be put into a model, such as normalization and one-hot encoding. [`Buffered`](@ref) wraps any transformation to reuse its output buffers; see [Buffering](buffering.md).
