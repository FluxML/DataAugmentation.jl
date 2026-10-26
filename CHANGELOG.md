# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.6]

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

[0.3.6]: https://github.com/FluxML/DataAugmentation.jl/compare/v0.3.5...HEAD
