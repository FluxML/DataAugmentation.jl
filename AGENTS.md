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

## Release (only when asked)
1. Update `CHANGELOG.md` from the PRs merged since the last tag (`gh pr list --state merged`, `git log vX.Y.Z..HEAD`), under the top undated `## [X.Y.W]` heading (the next patch version): add its release date (`## [X.Y.W] - YYYY-MM-DD`), point its compare link at `vX.Y.W` instead of `HEAD`, and add an empty `## [X.Y.(W+1)]` above it with its link to `HEAD`.
2. Bump `version` in `Project.toml` to `X.Y.W` and commit with the changelog update as `Bump version from X.Y.Z to X.Y.W`.
3. Push the commit directly to `master`, no PR.
4. Register: `gh api repos/FluxML/DataAugmentation.jl/commits/SHA/comments -f body='@JuliaRegistrator register'`. The reply links the General PR; TagBot tags the release after it merges.

## Notes
- `Manifest*.toml` and `docs/build/` are gitignored.
- Default branch is `master`.
