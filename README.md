# DataAugmentation.jl

[![Documentation Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://fluxml.ai/DataAugmentation.jl/stable/)
[![Documentation Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fluxml.ai/DataAugmentation.jl/dev/)
[![CI](https://github.com/FluxML/DataAugmentation.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/FluxML/DataAugmentation.jl/actions/workflows/ci.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/FluxML/DataAugmentation.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FluxML/DataAugmentation.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Efficient, composable data augmentation for machine and deep learning with support for n-dimensional images, keypoints and categorical masks.

## Installation

```julia
using Pkg
Pkg.add("DataAugmentation")
```

## Example

```julia
using DataAugmentation, TestImages

item = Image(testimage("lighthouse"))
tfm = Maybe(FlipX{2}()) |> RandomResizeCrop((128, 128))
timage = itemdata(apply(tfm, item))
```

See the [documentation](https://fluxml.ai/DataAugmentation.jl/stable/) for the available transformations and how to define your own.
