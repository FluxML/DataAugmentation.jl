```@meta
CollapsedDocStrings = true
```

# API reference

Names that are not exported are listed with the `DataAugmentation.` prefix and need to be imported or qualified. The sections [Extending](@ref reference-extending) and [Testing](@ref) cover the names used for defining your own items and transformations.

[Jump to the symbol index](@ref reference-index).

## Items

```@docs
Image
ImageBatch
MaskBinary
MaskMulti
Keypoints
BoundingBox
BoundingBoxes
Polygon
ArrayItem
TabularItem
itemdata
```

## Applying and composing transformations

```@docs
apply
apply!
DataAugmentation.compose
Sequence
Identity
```

## Geometric warps

```@docs
FlipDim
FlipX
FlipY
FlipZ
Reflect
Rotate
RotateX
RotateY
RotateZ
WarpAffine
RandomAffine
RandomPerspective
ElasticTransform
Project
PinOrigin
```

## Resizing

```@docs
ScaleRatio
ScaleFixed
ScaleKeepAspect
Zoom
Resize
ScaleJitter
RandomShortestSize
RandomResize
```

## Crops and padding

```@docs
Crop
CenterCrop
RandomCrop
RandomResizedCrop
CenterResizeCrop
RandomResizeCrop
RandomIoUCrop
FiveCrop
TenCrop
Pad
RandomZoomOut
PadDivisible
ResizePadDivisible
```

## Keypoint bounds

```@docs
ClampKeypoints
SanitizeKeypoints
```

## Bounding boxes

```@docs
box_convert
ClampBoundingBoxes
SanitizeBoundingBoxes
masks_to_boxes
box_iou
```

## Color transformations

```@docs
AdjustBrightness
AdjustContrast
AdjustSaturation
AdjustHue
ColorJitter
Grayscale
RandomGrayscale
ToRGB
GaussianBlur
GaussianNoise
AdjustSharpness
AdjustGamma
Solarize
Posterize
Invert
Equalize
Autocontrast
RandomPhotometricDistort
RandomChannelPermutation
JPEG
```

## Policy-based augmentations

```@docs
RandAugment
TrivialAugmentWide
AutoAugment
AugMix
```

## Erasing and batch mixing

```@docs
RandomErasing
MixUp
CutMix
```

## Video

```@docs
UniformTemporalSubsample
```

## Stochastic transformations

```@docs
Maybe
OneOf
RandomOrder
```

## Buffering

```@docs
Buffered
BufferedThreadsafe
```

## Preprocessing

```@docs
ToEltype
ImageToTensor
PermuteDims
Normalize
NormalizeIntensity
LinearTransformation
OneHot
onehot
MapElem
```

## Tabular data

```@docs
DataAugmentation.FillMissing
DataAugmentation.Categorify
DataAugmentation.NormalizeRow
```

## Visualization

```@docs
showitems
showgrid
```

## [Extending](@id reference-extending)

### Items

```@docs
Item
DataAugmentation.AbstractItem
DataAugmentation.AbstractArrayItem
DataAugmentation.ItemWrapper
DataAugmentation.setdata
DataAugmentation.showitem!
```

### Transformations

```@docs
Transform
DataAugmentation.getrandstate
DataAugmentation.makebuffer
```

### Projective transformations

```@docs
DataAugmentation.ProjectiveTransform
DataAugmentation.ComposedProjectiveTransform
DataAugmentation.getprojection
DataAugmentation.getbounds
DataAugmentation.project
DataAugmentation.project!
Bounds
DataAugmentation.boundsof
DataAugmentation.centered
DataAugmentation.offsetcropbounds
DataAugmentation.projectionbounds
DataAugmentation.threepointwarpaffine
DataAugmentation.transformbounds
```

## Testing

```@docs
DataAugmentation.testapply
DataAugmentation.testapply!
DataAugmentation.testitem
DataAugmentation.testprojective
```

## [Index](@id reference-index)

```@index
Pages = ["ref.md"]
```
