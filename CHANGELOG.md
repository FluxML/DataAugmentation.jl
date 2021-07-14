

# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
