# DataAugmentation.jl

Julia package (FluxML) for composable data augmentation of images, masks, keypoints and tabular rows. Supports Julia ≥ 1.10.

## Layout
- `src/DataAugmentation.jl`: module, `include` order (matters) and `export` list. New public names must be exported here.
- `src/base.jl`: core interface: `Item`, `Transform`, `apply`, `itemdata`, `getrandstate`.
- `src/items/`: item types (`Image`, `Keypoints`, `MaskMulti`/`MaskBinary`, `TabularItem`, …).
- `src/projective/`: geometric transforms (crop, affine, warp) built on `Project` / `ProjectiveTransform`.
- Other transforms: `sequence.jl`, `oneof.jl`, `colortransforms.jl`, `rowtransforms.jl`, `preprocessing.jl`, `buffered.jl`, `wrapper.jl`.
- `src/testing.jl`, `src/visualization.jl`: test helpers and `showitem!`.
- `test/` mirrors `src/`; `docs/` is Documenter (`docs/src/*.md`, API in `ref.md`).

## Conventions
- Stochastic transforms implement `getrandstate`; `apply(tfm, item)` ≡ `apply(tfm, item; randstate = getrandstate(tfm))`.
- `apply!(buf, tfm, item)` is the buffer-reusing variant of `apply`.
- Exported symbols need a docstring; list them in `docs/src/ref.md` for the docs.
- Do not edit `CHANGELOG.md` in PRs; it is updated at release time.

## Commands
- Test: `julia --project -e 'using Pkg; Pkg.test()'` (includes Aqua checks: no method ambiguities, no unbound type parameters, compat bounds for all deps).
- Docs: `julia --project=docs docs/make.jl` (workspace project; `Pkg.instantiate` first).
- Add new test files to the list in `test/runtests.jl`.

## Notes
- `Manifest*.toml` and `docs/build/` are gitignored.
- Default branch is `master`.
