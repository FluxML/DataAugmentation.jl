```@setup pipe
using DataAugmentation, Images, MosaicViews, StaticArrays, TestImages, ImageShow
```
# Pipelines

This page walks through complete augmentation pipelines for common computer vision tasks. They all follow the same recipe:

1. wrap the data that belongs together in [items](items.md);
2. put the items in a tuple and `apply` a transformation to it, so that every item undergoes the *same* random transformation (see [Design](design.md));
3. prepare each item for the model with the [preprocessing](preprocessing.md) transformations.

The examples use the lighthouse picture from the [Gallery](gallery.md).

```@example pipe
image = testimage("lighthouse")
image = imresize(image, ratio = 196 / size(image, 1))
```

## 2D images

For image classification only the image is transformed, since the label does not change. Training uses random transformations; validation uses deterministic ones, so that results are comparable between epochs.

```@example pipe
train_tfm = Maybe(FlipX{2}()) |> RandomResizeCrop((128, 128)) |> AdjustBrightness(0.2)
val_tfm = CenterResizeCrop((128, 128))

showgrid([apply(train_tfm, Image(image)) for _ in 1:6]; ncol = 6, npad = 8)
```

The spatial transformations come first and are fused into a single, efficient one, followed by the color transformations. Then [`ImageToTensor`](@ref) and [`Normalize`](@ref) turn the image into a normalized array of the element type `Float32`:

```@example pipe
prep = ImageToTensor() |> Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

x = itemdata(apply(train_tfm |> prep, Image(image)))
size(x), eltype(x)
```

The array has the dimensions `(width, height, channels)`, the layout expected by Flux's convolutional layers; see [Using with Flux](flux.md).

## Images and segmentation masks

For segmentation, the image and its mask have to be transformed in exactly the same way. Wrap the mask in a [`MaskMulti`](@ref) with the list of its classes. Here is a mask with the background, the lighthouse and the house, drawn by hand:

```@example pipe
labels = ones(Int, size(image))
labels[28:118, 122:146] .= 2
labels[72:112, 14:95] .= 3

mask = MaskMulti(labels, 1:3)
mosaicview([showitems(Image(image)), showitems(mask)]; nrow = 1, npad = 8)
```

[`showitems`](@ref) draws a mask as a map of its classes, so we show the image and the mask next to each other. Applying a transformation to the tuple `(image, mask)` uses one random state for both. The mask is projected with nearest-neighbor interpolation, so the class labels are never blended. Color transformations like [`AdjustBrightness`](@ref) leave masks unchanged, so they can be part of the same pipeline:

```@example pipe
tfm = Maybe(FlipX{2}()) |> Rotate(10) |> RandomResizeCrop((128, 128)) |> AdjustBrightness(0.2)

pairs = [apply(tfm, (Image(image), mask)) for _ in 1:4]
mosaicview(vcat([showitems(i) for (i, _) in pairs], [showitems(m) for (_, m) in pairs]);
           ncol = 4, rowmajor = true, npad = 8)
```

The top row shows the transformed images, and the bottom row their masks, which were transformed in the same way.

Preprocessing differs between the two items: [`ImageToTensor`](@ref) only works on images, and [`OneHot`](@ref) only on masks. So apply the tuple pipeline first, and prepare each item on its own:

```@example pipe
image_prep = ImageToTensor() |> Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
mask_prep = OneHot() |> PermuteDims(2, 1, 3)

timage, tmask = apply(tfm, (Image(image), mask))
x = itemdata(apply(image_prep, timage))
y = itemdata(apply(mask_prep, tmask))
size(x), size(y)
```

[`OneHot`](@ref) keeps the dimensions of the mask, `(height, width, classes)`, whereas [`ImageToTensor`](@ref) swaps height and width. The [`PermuteDims`](@ref) step makes the one-hot mask match the image.

## Images and keypoints

[`Keypoints`](@ref) represent points on an image, for example landmarks. They are given as a vector of `SVector`s whose entries are the coordinates along the dimensions of the image, so `SVector(i, j)` is the position of `image[i, j]`. The second argument is the size of the image the points belong to. For objects given by a bounding box or a polygon, use [`BoundingBox`](@ref) and [`Polygon`](@ref) instead, which are used just the same.

```@example pipe
points = SVector{2, Float32}[(30, 134), (116, 134), (72, 55)]
keypoints = Keypoints(points, size(image))
showitems((Image(image), keypoints))
```

Again, pass the image and the keypoints to `apply` together:

```@example pipe
tfm = Maybe(FlipX{2}()) |> Rotate(10) |> RandomResizeCrop((128, 128))

showgrid([apply(tfm, (Image(image), keypoints)) for _ in 1:4]; ncol = 4, npad = 8)
```

Keypoints that end up outside of the cropped region are *not* removed. Their coordinates just lie outside of the image, so filter them if your task needs that:

```@example pipe
timage, tkeypoints = apply(tfm, (Image(image), keypoints))
sz = size(itemdata(timage))
inside = [all(1 .<= p .<= sz) for p in itemdata(tkeypoints)]
```

## 3D images

Transformations that are defined for any number of dimensions work on volumes, too. The transformation has to match the dimensionality: use `FlipX{3}()` and a 3-dimensional crop size instead of `FlipX{2}()` and `(128, 128)`, and `Rotate(α, β, γ)` to rotate around all three axes. A volume is an array of colors like `Gray`, the same as for 2D images, and its segmentation is a [`MaskMulti`](@ref) of the same size.

Let's make a volume with a blob in it, and a mask that marks the blob:

```@example pipe
dist2(i, j, k) = (i - 18)^2 + (j - 24)^2 + (k - 24)^2
volume = [Gray{Float32}(exp(-dist2(i, j, k) / 150)) for i in 1:48, j in 1:48, k in 1:48]
segmentation = [dist2(i, j, k) < 100 ? 2 : 1 for i in 1:48, j in 1:48, k in 1:48]
mask3d = MaskMulti(segmentation, 1:2)
```

The volume is interpolated linearly and the mask with nearest-neighbor interpolation, as in two dimensions. We can look at the middle slice of the results (with the offset axes of the cropped arrays removed by [`parent`](https://docs.julialang.org/en/v1/base/arrays/#Base.parent)):

```@example pipe
tfm = Maybe(FlipX{3}()) |> Rotate(30, 30, 30) |> CenterCrop((32, 32, 32))

middleslice(item) = parent(itemdata(item))[:, :, 16]
samples = [apply(tfm, (Image(volume), mask3d)) for _ in 1:4]
imageslices = [middleslice(i) for (i, _) in samples]
maskslices = [Gray.(Float32.(middleslice(m) .- 1)) for (_, m) in samples]
mosaicview(vcat(imageslices, maskslices); ncol = 4, rowmajor = true, npad = 4, fillvalue = 1)
```

The top row shows the slices of the volume and the bottom row those of the mask. As in 2D, [`ImageToTensor`](@ref) and [`OneHot`](@ref) add the dimension of the channels or of the classes at the end:

```@example pipe
timage, tmask = apply(tfm, (Image(volume), mask3d))
size(itemdata(apply(ImageToTensor(), timage))), size(itemdata(apply(OneHot(), tmask)))
```
