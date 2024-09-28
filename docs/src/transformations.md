```@setup tsm
using DataAugmentation
```
# Usage

Using  transformations is easy. Simply `compose` them:

```@example tsm
tfm = Rotate(10) |> ScaleRatio((0.7,0.1,1.2)) |> FlipX{2}() |> Crop((128, 128))
```

# Projective transformations
DataAugmentation.jl has great support for transforming spatial data like images and keypoints. Most of these transformations are projective transformations. For our purposes, a projection means a mapping between two coordinate spaces. In computer vision, these are frequently used for preprocessing and augmenting image data: images are randomly scaled, maybe flipped horizontally and finally cropped to the same size.

This library generalizes projective transformations for different kinds of image and keypoint data in an N-dimensional Euclidean space. It also uses composition for performance improvements like fusing affine transformations.

Unlike mathematical objects, the spatial data we want to transform has spatial bounds. For an image, these bounds are akin to the array size. But keypoint data aligned with an image has the same bounds even if they are not explicitly encoded in the representation of the data. These spatial bounds can be used to dynamically create useful transformations. For example, a rotation around the center or a horizontal flip of keypoint annotations can be calculated from the bounds.

Often, we also want to crop an area from the projected results. By evaluating only the parts of a projection that fall inside the cropped area, a lot of unnecessary computation can be avoided.

Projective transformations include:
1. [Affine transformations](@ref)
2. [Crops](@ref)
## Affine transformations

Affine transformations are a subgroup of projective transformations that can be composed very efficiently: composing two affine transformations results in another affine transformation. Affine transformations can represent translation, scaling, reflection and rotation. Available `Transform`s are:

```@docs; canonical=false
FlipX
FlipY
FlipZ
Reflect
Rotate
RotateX
RotateY
RotateZ
ScaleKeepAspect
ScaleFixed
ScaleRatio
WarpAffine
Zoom
```
    
## Crops

To get a cropped result, simply `compose` any `ProjectiveTransform` with

```@docs; canonical=false
CenterCrop
RandomCrop
```

# Color transformations

DataAugmentation.jl currently supports the following color transformations for augmentation:

```@docs; canonical=false
AdjustBrightness
AdjustContrast
```

# Stochastic transformations
When augmenting data, it is often useful to apply a transformation only with some probability or choose from a set of transformations. Unlike in other data augmentation libraries like *albumentations*, in DataAugmentation.jl you can use wrapper transformations for this functionality.

- [`Maybe`](@ref)`(tfm, p = 0.5)` applies a transformation with probability `p`; and
- [`OneOf`](@ref)`([tfm1, tfm2])` randomly selects a transformation to apply.
```@docs; canonical=false
Maybe
OneOf
```

Let's say we have an image classification dataset. For most datasets, horizontally flipping the image does not change the label: a flipped image of a cat still shows a cat. So let's flip every image horizontally half of the time to improve the generalization of the model we might be training.

```@example
using DataAugmentation, TestImages
item = Image(testimage("lighthouse"))
tfm = Maybe(FlipX{2}())
titems = [apply(tfm, item) for _ in 1:8]
showgrid(titems; ncol = 4, npad = 16)
```

```@docs; canonical=false
DataAugmentation.ImageToTensor
```
