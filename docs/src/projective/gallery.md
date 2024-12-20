# Gallery

Let's visualize what these projective transformations look like.

You can apply them to [`Image`](@ref)s and
the keypoint-based items [`Keypoints`](@ref), [`Polygon`](@ref), and [`BoundingBox`](@ref).

Let's take this picture of a light house:

```@example deps
using DataAugmentation
using MosaicViews
using Images
using TestImages
using StaticArrays

imagedata = testimage("lighthouse")
imagedata = imresize(imagedata, ratio = 196 / size(imagedata, 1))
```

To apply a transformation `tfm` to it, wrap it in
`Image`, apply the transformation and unwrap it using [`itemdata`](@ref):

```@example deps
tfm = CenterCrop((196, 196))
image = Image(imagedata)
apply(tfm, image) |> itemdata
```
Customization of how pixel values are interpolated and extrapolated during
transformations is done with the Item types ([`Image`](@ref),
[`MaskBinary`](@ref), [`MaskMulti`](@ref)). For example, if we scale the image
we can see how the interpolation affects how values of projected pixels are
calculated.
```@example deps
using Interpolations: BSpline, Constant, Linear
tfm = ScaleFixed((2000, 2000)) |> CenterCrop((200, 200))
showgrid(
    [
        # Default is linear interpolation for Image
        apply(tfm, Image(imagedata)),
        # Nearest neighbor interpolation
        apply(tfm, Image(imagedata; interpolate=BSpline(Constant()))),
        # Linear interpolation
        apply(tfm, Image(imagedata; interpolate=BSpline(Linear()))),
    ];
    ncol=3,
    npad=8,
)
```
Similarly, if we crop to a larger region than the image, we can see how
extrapolation affects how pixel values are calculated in the regions outside
the original image bounds.
```@example deps
import Interpolations
tfm = CenterCrop((400, 400))
showgrid(
    [
        apply(tfm, Image(imagedata)),
        apply(tfm, Image(imagedata; extrapolate=1)),
        apply(tfm, Image(imagedata; extrapolate=Interpolations.Flat())),
        apply(tfm, Image(imagedata; extrapolate=Interpolations.Periodic())),
        apply(tfm, Image(imagedata; extrapolate=Interpolations.Reflect())),
    ];
    ncol=5,
    npad=8,
)
```

Now let's say we want to train a light house detector and have a bounding box for the light house. We can use the [`BoundingBox`](@ref) item to represent it. It takes the two corners of the bounding rectangle as the first argument. As the second argument we have to pass the size of the corresponding image.

```@example deps
points = SVector{2, Float32}[SVector(23., 120.), SVector(120., 150.)]
bbox = BoundingBox(points, size(imagedata))
```

[`showitems`](@ref) visualizes the two items:
```@example deps
showitems((image, bbox))
```
If we apply transformations like translation and cropping to the image, then the same transformations have to be applied to the bounding box. Otherwise, the bounding box will no longer match up with the light house.

Another problem can occur with stochastic transformations like [`RandomResizeCrop`](@ref) If we apply it separately to the image and the bounding box, they will be cropped from slightly different locations:

```@example deps
tfm = RandomResizeCrop((128, 128))
showitems((
    apply(tfm, image),
    apply(tfm, bbox)
))
```
Instead, pass a tuple of the items to a single `apply` call so the same random state will be used for both image and bounding box:

```@example deps
apply(tfm, (image, bbox)) |> showitems
```

!!! info "3D Projective dimensions"

    We'll use a 2-dimensional [`Image`](@ref) and [`BoundingBox`](@ref) here, but you can apply most projective transformations to any spatial item (including [`Keypoints`](@ref), [`MaskBinary`](@ref) and [`MaskMulti`](@ref)) in 3 dimensions.
    
    Of course, you have to create a 3-dimensional transformation, i.e. `CenterCrop((128, 128, 128))` instead of `CenterCrop((128, 128))`.

## [`RandomResizeCrop`](@ref)`(sz)`

Resizes the sides so that one of them is no longer than `sz` and crops a region of size `sz` *from a random location*.

```@example deps
tfm = RandomResizeCrop((128, 128))
showgrid([apply(tfm, (image, bbox)) for _ in 1:6]; ncol=6, npad=8)
```

## [`CenterResizeCrop`](@ref)

Resizes the sides so that one of them is no longer than `sz` and crops a region of size `sz` *from the center*.

```@example deps
tfm = CenterResizeCrop((128, 128))
showgrid([apply(tfm, (image, bbox))]; ncol=6, npad=8)
```

## [`Crop`](@ref)`(sz[, from])`

Crops a region of size `sz` from the image, *without resizing* the image first.

```@example deps
using DataAugmentation: FromOrigin, FromCenter, FromRandom
tfms = [
    Crop((128, 128), FromOrigin()),
    Crop((128, 128), FromCenter()),
    Crop((128, 128), FromRandom()),
    Crop((128, 128), FromRandom()),
    Crop((128, 128), FromRandom()),
    Crop((128, 128), FromRandom()),
]
showgrid([apply(tfm, (image, bbox)) for tfm in tfms]; ncol=6, npad=8)
```

## [`FlipX`](@ref), [`FlipY`](@ref), [`FlipZ`](@ref), [`Reflect`](@ref)

Flip the data on the horizontally and vertically, respectively. More generally, reflect around an angle from the x-axis.

```@example deps
tfms = [
    FlipX{2}(),
    FlipY{2}(),
    Reflect(30),
]
showgrid([apply(tfm, (image, bbox)) for tfm in tfms]; ncol=6, npad=8)
```

## [`Rotate`](@ref), [`RotateX`](@ref), [`RotateY`](@ref), [`RotateZ`](@ref)

Rotate a 2D image counter-clockwise by an angle.

```@example deps
tfm = Rotate(20) |> CenterCrop((256, 256))
showgrid([apply(tfm, (image, bbox)) for _ in 1:6]; ncol=6, npad=8)
```

Rotate also works with 3D images in addition to 3D specific transforms RotateX, RotateY, and RotateZ.

```@example deps
image3D = Image([RGB(i, j, k) for i=0:0.01:1, j=0:0.01:1, k=0:0.01:1])
tfms = [
    Rotate(20, 30, 40),
    Rotate{3}(45),
    RotateX(45),
    RotateY(45),
    RotateZ(45),
]
transformed = [apply(tfm, image3D) |> itemdata for tfm in tfms]
slices = [Image(parent(t[:, :, 50])) for t in transformed]
showgrid(slices; ncol=6, npad=8)
```
