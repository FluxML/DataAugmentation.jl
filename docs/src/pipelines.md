```@setup pipe
using DataAugmentation, Images, MosaicViews, StaticArrays, TestImages, ImageShow
```
# Pipelines

This page walks through complete augmentation pipelines for common computer vision tasks.

1. Wrap the data in [items](items.md).
2. Call the transform on the item or on a tuple of aligned items, such as `tfm((image, mask))`. The items in the tuple share the transform's random state. See [Design](design.md).
3. Prepare each item for the model with [preprocessing](preprocessing.md) transforms and unwrap the result with [`itemdata`](@ref).

The examples use the lighthouse picture from the [Gallery](gallery.md).

```@example pipe
image = testimage("lighthouse")
image = imresize(image, ratio = 196 / size(image, 1))
```

## 2D images

For image classification only the image is transformed, since the label does not change. Training uses random transformations; validation uses deterministic ones, so that results are comparable between epochs.

```@example pipe
train_tfm = Maybe(FlipX()) |> RandomResizedCrop(128; scale = (0.5, 1.0)) |> AdjustBrightness(0.2)
val_tfm = CenterResizeCrop(128)
item = Image(image; antialias = true)

showgrid([train_tfm(item) for _ in 1:6]; ncol = 6, npad = 8)
```

[`RandomResizedCrop`](@ref) samples a window by area and aspect ratio, then resizes it to the target size. [`FlipX`](@ref) flips around the input bounds and shares a single projection with the crop. `antialias = true` prefilters the image when downsampling. [`ImageToTensor`](@ref) and [`Normalize`](@ref) turn each result into a normalized `Float32` array.

```@example pipe
prep = ImageToTensor() |> Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

train = train_tfm |> prep
validate = val_tfm |> prep

x = itemdata(train(item))
size(x), size(itemdata(validate(item))), eltype(x)
```

Each sample has shape `(width, height, channels)`. See [Using with Flux](flux.md) for batching and [policy-based augmentations](transformations.md#Policy-based-augmentations) for preset training policies.

## Images and segmentation masks

For segmentation, the image and its mask have to be transformed in exactly the same way. Wrap the mask in a [`MaskMulti`](@ref) with the list of its classes. Here is a mask with the background, the lighthouse and the house, drawn by hand:

```@example pipe
labels = ones(Int, size(image))
labels[28:118, 122:146] .= 2
labels[72:112, 14:95] .= 3

mask = MaskMulti(labels, 1:3)
showitems((Image(image), mask); maskbackground = 1)
```

[`showitems`](@ref) blends the classes of the mask over the image and skips background class `1` here. Pass the image and mask together so they share the flip, rotation, and crop. Masks use nearest-neighbor interpolation to preserve class labels. [`AdjustBrightness`](@ref) targets the image.

```@example pipe
tfm = Maybe(FlipX()) |> Rotate(10) |> RandomResizeCrop((128, 160)) |> AdjustBrightness(0.2)

showgrid([tfm((Image(image), mask)) for _ in 1:4]; ncol = 4, npad = 8, maskbackground = 1)
```

After the shared transform, use [`ImageToTensor`](@ref) on the image and [`OneHot`](@ref) on the mask.

```@example pipe
mask_prep = OneHot() |> PermuteDims(2, 1, 3)

timage, tmask = tfm((Image(image), mask))
x = itemdata(prep(timage))
y = itemdata(mask_prep(tmask))
size(x), size(y)
```

[`OneHot`](@ref) returns `(height, width, classes)`. The [`PermuteDims`](@ref) step puts width first to match the image tensor.

## Images and keypoints

[`Keypoints`](@ref) represent landmarks as `SVector`s in array dimension order. `SVector(i, j)` is the position of `image[i, j]`. The second argument gives the size of the image. Use [`Polygon`](@ref) for vertices that outline an object.

```@example pipe
points = SVector{2, Float32}[(30, 134), (116, 134), (72, 55)]
keypoints = Keypoints(points, size(image))
showitems((Image(image), keypoints))
```

Pass the image and keypoints to the same transform. [`SanitizeKeypoints`](@ref) marks landmarks outside the crop with `NaN` while preserving their positions in the landmark array. [`showitems`](@ref) skips those points. Use [`ClampKeypoints`](@ref) to move coordinates into the image's index ranges.

```@example pipe
tfm = Maybe(FlipX()) |> Rotate(10) |> RandomResizeCrop(128) |> SanitizeKeypoints()

showgrid([tfm((Image(image), keypoints)) for _ in 1:4]; ncol = 4, npad = 8)
```

```@example pipe
_, tkeypoints = tfm((Image(image), keypoints))
itemdata(tkeypoints)
```

## Images and bounding boxes

For object detection, wrap all boxes for an image in a [`BoundingBoxes`](@ref) collection. In the default `:XYXY` format each column holds `(xmin, ymin, xmax, ymax)`. Coordinates use `x` along columns and `y` along rows, with integer coordinates at pixel centers. The canvas size is `(height, width)`.

```@example pipe
coords = [122  14
           28  72
          146  95
          118 112]
boxes = BoundingBoxes(coords, size(image); labels = [:lighthouse, :house])

detect_tfm = Maybe(FlipX()) |> RandomIoUCrop() |> PinOrigin() |> SanitizeBoundingBoxes()
timage, tboxes = detect_tfm((Image(image), boxes))
showitems((timage, tboxes))
```

[`RandomIoUCrop`](@ref) selects a window based on overlap with the boxes or keeps the full image. When it crops, it clips boxes whose centers stay inside the window and collapses the others. [`PinOrigin`](@ref) makes the image axes one-based and shifts the boxes by the same amount. [`SanitizeBoundingBoxes`](@ref) drops collapsed boxes together with their labels. After a plain crop or warp, use [`ClampBoundingBoxes`](@ref) before sanitizing to keep the visible part of partially cropped boxes.

```@example pipe
x = itemdata(prep(timage))
target = (boxes = itemdata(tboxes), labels = tboxes.labels)
size(x), target
```

Use [`box_convert`](@ref) if the detector needs a different box format. For instance masks, [`masks_to_boxes`](@ref) builds the collection from a stack of binary masks. See [Using with Flux](flux.md#Object-detection) for batching detection samples.

## 3D images

Flips such as [`FlipX`](@ref) adapt to the dimensionality of the item. [`CenterCrop`](@ref)`(32)` crops 32 voxels along every axis. Use a size tuple for different extents along each axis. [`Rotate`](@ref)`(30)` samples an angle for each of the three axes of a volume, and `Rotate(α, β, γ)` sets a separate range for each. Wrap a volume of colors such as `Gray` in [`Image`](@ref) and its segmentation in a [`MaskMulti`](@ref) of the same size.

Here is a synthetic volume and a mask for its central region.

```@example pipe
dist2(i, j, k) = (i - 18)^2 + (j - 24)^2 + (k - 24)^2
volume = [Gray{Float32}(exp(-dist2(i, j, k) / 150)) for i in 1:48, j in 1:48, k in 1:48]
segmentation = [dist2(i, j, k) < 100 ? 2 : 1 for i in 1:48, j in 1:48, k in 1:48]
mask3d = MaskMulti(segmentation, 1:2)
```

The volume uses linear interpolation and the mask uses nearest-neighbor interpolation. [`PinOrigin`](@ref) makes the cropped axes one-based so we can display the same slice index in each sample.

```@example pipe
tfm = Maybe(FlipX()) |> Rotate(30) |> CenterCrop(32) |> PinOrigin()

middleslice(item) = itemdata(item)[:, :, 16]
samples = [tfm((Image(volume), mask3d)) for _ in 1:4]
imageslices = [middleslice(i) for (i, _) in samples]
maskslices = [Gray.(Float32.(middleslice(m) .- 1)) for (_, m) in samples]
mosaicview(vcat(imageslices, maskslices); ncol = 4, rowmajor = true, npad = 4, fillvalue = 1)
```

The top row shows volume slices and the bottom row shows mask slices. [`ImageToTensor`](@ref) returns `(width, height, depth, channels)`. Use `PermuteDims(2, 1, 3, 4)` after [`OneHot`](@ref) to put width first in the mask as well.

```@example pipe
timage, tmask = tfm((Image(volume), mask3d))
to_tensor = ImageToTensor()
volume_mask_prep = OneHot() |> PermuteDims(2, 1, 3, 4)
size(itemdata(to_tensor(timage))), size(itemdata(volume_mask_prep(tmask)))
```

For a video with time on the third axis, [`UniformTemporalSubsample`](@ref) selects evenly spaced frames from the video and its aligned masks. See [Video](transformations.md#Video).
