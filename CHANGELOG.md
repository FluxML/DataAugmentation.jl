# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.7]

### Added

- Add the geometric transformations of `torchvision.transforms.v2`: `Resize`, `ScaleJitter`, `RandomShortestSize`,
  `RandomResize`, `RandomResizedCrop`, `RandomIoUCrop`, `FiveCrop`, `TenCrop`, `Pad`, `RandomZoomOut`, `RandomAffine`,
  `RandomPerspective` and `ElasticTransform`
- Add `padding`, `pad_if_needed`, `fill` and `padding_mode` keywords to `Crop`, `CenterCrop` and `RandomCrop`,
  and `CenterCrop(n)` for a square crop
- Add `expand`, `center` and `fill` keywords to 2D `Rotate`
- Add `antialias` keyword to `Image` to prefilter before downsampling with an affine projection
- Add photometric transformations: `AdjustSaturation`, `AdjustHue`, `ColorJitter`, `Grayscale`, `RandomGrayscale`,
  `ToRGB`, `GaussianBlur`, `GaussianNoise`, `AdjustSharpness`, `AdjustGamma`, `Solarize`, `Posterize`, `Invert`,
  `Equalize`, `Autocontrast`, `RandomPhotometricDistort`, `RandomChannelPermutation` and `JPEG`
  (requires JpegTurbo.jl, loaded as a package extension). They leave masks and keypoints unchanged
- `AdjustBrightness` and `AdjustContrast` accept a `(min, max)` interval and work on more color spaces
- Add `RandomErasing`, and `ImageBatch` with `MixUp` and `CutMix` for collated classification batches
- Add `RandAugment`, `TrivialAugmentWide`, `AutoAugment` (ImageNet, CIFAR10, SVHN and custom policies) and
  `AugMix`, with the magnitude tables of `torchvision.transforms.v2`, and `RandomOrder` to apply transformations
  in a random order
- Add `ClampKeypoints` and `SanitizeKeypoints` to clamp keypoints to the bounds of the item or mark those
  outside as `NaN`
- Add `LinearTransformation` (e.g. for ZCA whitening) and `UniformTemporalSubsample`
- `showitems` and `showgrid` blend masks over the image, with the new `maskalpha` and `maskbackground` keywords

### Changed

- `Zoom` is now implemented through `ScaleRatio`: its output has bounds `1:size` like the other scaling transformations
- Make transformations dimension-free: `FlipX()`, `FlipY()`, `FlipZ()` and `FlipDim(dim)` take the dimensionality from the item
  (the `{N}` forms still work), `Rotate(γ)` rotates in 2D or, as `Rotate(γ, γ, γ)`, in 3D, and a number for
  `Crop`, `CenterCrop`, `RandomCrop`, `ScaleFixed`, `ScaleKeepAspect`, `ScaleRatio` and the resize crops applies
  to every dimension. Results in 2D are unchanged for a given random state
- `WarpAffine`, `RandomErasing`, `GaussianBlur`, `AdjustSharpness`, `ImageBatch` and `CutMix` support N-dimensional items
- `Buffered` accepts `copy = true` to reuse its buffers but return a copy of the result
- `BufferedThreadsafe` now keeps a pool of buffers and always returns a copy
- Reorganize the documentation into Guide, Concepts, Extending and API reference sections (the pages under `projective/` moved), and document `Project`, `PadDivisible`, `TabularItem`, `onehot`, `showgrid`, `Bounds` and `NormalizeIntensity`

### Fixed

- Fix `apply!` and `Buffered`/`BufferedThreadsafe` projective transforms (e.g. `ScaleRatio`)
  throwing a `MethodError` for `BoundingBox` and `Polygon` items
- `ScaleFixed`, `ScaleRatio`, `ScaleKeepAspect` and `Zoom` no longer produce black borders when scaling images, for any combination of input and output sizes
- Flips and `Reflect` are centered on the input bounds: they no longer shift the item by one pixel along the flipped
  dimension, so flipped and unflipped samples have the same axes and can be stacked
- `BufferedThreadsafe` no longer throws a `BoundsError` with interactive threads, and returns copies
  so that collating a batch after loading it no longer repeats the last sample (use `Buffered(tfm; copy = true)`
  for the same with `Buffered`)
- `AdjustBrightness` and `AdjustContrast` return masks and keypoints unchanged instead of throwing a `MethodError`

## [0.3.6] - 2026-10-26

### Changed

- Require Julia 1.10
- Replace the Setfield.jl dependency with Accessors.jl
- Remove unused dependencies (ColorBlendModes.jl, Parameters.jl)

### Fixed

- Remove the undefined `getcategorypools` from the exports
- Fix method ambiguity errors when composing two cropped transforms, calling `OneOf` with
  projective transforms and a distribution, applying `Buffered`/`BufferedThreadsafe` to
  `BoundingBox`/`Polygon` items or with `apply!` to tuples of items, and creating a
  `MaskMulti` from an integer-valued `IndirectArray`
- `Keypoints` with only `nothing` elements now throws an informative `ArgumentError`

## [0.3.0]

### Added

- Add `PermuteDims` transformation

### Changed

- `ImageToTensor` now returns arrays in `WHC` format instead of `HWC` 

## [0.2.12]

### Added

- Added optional `clamp` flag to `AdjustBrightness` and `AdjustContrast`

### Changed

- Transfered repository to FluxML

## [0.2.11]

### Fixed

- Bump Setfield compat

## [0.2.10]

### Fixed 

- Fix `RandomCrop` transform

## [0.2.9]

### Added 

- Set up Pollen.jl documentation 

### Fixed 

- Fix deprecated call to `warp`

## [0.2.8]

### Fixed

- Fix compatibility with ImageTransformations 0.9

## [0.2.7]

### Changed

- Replace Images.jl dependency with ImageCore.jl

### Fixed

- Fix crop-projective composition

## [0.2.6]

### Fixed

- `CenterCrop` and co. now work properly on their own. (fixes [#52](https://github.com/lorenzoh/DataAugmentation.jl/issues/52))

## [0.2.5]

### Added

- `TableRow` item
- transforms for `TableRow`: `FillMissing`, `Categorify` and `NormalizeRow`

## [0.2.4]

### Changed

- `Buffered` now stores a `Dict` of item type -> buffer, to alleviate the need to recreate
  buffers if the buffered transform is applied to multiple items

## [0.2.3]

### Changed

- `BufferedThreadsafe` now properly passes through explicit random state
- `ScaleKeepAspect` no longer sometimes produces a black border
- *fix* `Sequence |> Sequence` now has a method
- `ToTensor` now works on different color types and N-dimensional arrays
- `MaskMulti` now has a constructor for `IndirectArray`s

## [0.1.5] - 2021-04-17

### Added
- [Color augmentation transforms](https://lorenzoh.github.io/DataAugmentation.jl/dev/docs/literate/colortransforms.md.html)
- [Stochastic transformation wrappers](https://lorenzoh.github.io/DataAugmentation.jl/dev/docs/literate/stochastic.md.html)
- `WarpAffine` projective transformation

### Changed
- Moved documentation generation to [Pollen.jl](https://github.com/lorenzoh/Pollen.jl)

[0.3.7]: https://github.com/FluxML/DataAugmentation.jl/compare/v0.3.6...HEAD
[0.3.6]: https://github.com/FluxML/DataAugmentation.jl/compare/v0.3.5...v0.3.6
