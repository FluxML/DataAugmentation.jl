# API reference

Names that are not exported are listed with the `DataAugmentation.` prefix and need to be imported or qualified. The sections [Extending](@ref reference-extending) and [Testing](@ref) cover the names used for defining your own items and transformations.

## Items

```@docs
Image
MaskBinary
MaskMulti
Keypoints
BoundingBox
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

## Affine transformations

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
ScaleRatio
ScaleFixed
ScaleKeepAspect
Zoom
WarpAffine
Project
PinOrigin
```

## Crops

```@docs
Crop
CenterCrop
RandomCrop
CenterResizeCrop
RandomResizeCrop
PadDivisible
ResizePadDivisible
```

## Color transformations

```@docs
AdjustBrightness
AdjustContrast
```

## Stochastic transformations

```@docs
Maybe
OneOf
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
