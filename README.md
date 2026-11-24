# DataAugmentation.jl

[![Documentation Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://fluxml.ai/DataAugmentation.jl/stable/)
[![Documentation Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://fluxml.ai/DataAugmentation.jl/dev/)
[![CI](https://github.com/FluxML/DataAugmentation.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/FluxML/DataAugmentation.jl/actions/workflows/ci.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/FluxML/DataAugmentation.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FluxML/DataAugmentation.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Compose data augmentation pipelines for image classification, segmentation, object detection, and 3D volumes in Julia.

![A lighthouse photo followed by a flipped, rotated crop and a color-jittered version](assets/readme/transforms.png)

## Quickstart

Install the registered package and the sample images used below.

```julia
using Pkg
Pkg.add(["DataAugmentation", "TestImages"])
```

Compose a random rotation and crop with `|>`. Use `itemdata` to get the resulting pixel array.

```julia
using DataAugmentation, TestImages

image = Image(testimage("lighthouse"))
tfm = Rotate(15) |> RandomResizeCrop((192, 256))
pixels = itemdata(tfm(image))
```

## Feature showcase

### Keep annotations aligned

Pass aligned items as a tuple with `tfm((image, mask, boxes))` to share the random geometry. Here `FlipX` and `RandomIoUCrop` move the segmentation mask and labeled boxes with the image.

![Before and after a shared flip and crop, with masks and bounding boxes aligned to the lighthouse and the house](assets/readme/aligned-annotations.png)

### Mix images and labels

`MixUp` blends paired images. `CutMix` copies a rectangular patch from one image into another. Both return an `ImageBatch` with class weights that reflect the mix.

![A cat and a cup of coffee as source images, a MixUp blend with 55% cat and 45% coffee, and a CutMix sample with 65% cat and 35% coffee](assets/readme/batch-mixing.png)

## Documentation

- Browse crops, warps, color transforms, and policies such as `RandAugment` and `AutoAugment`. [Transformations](https://fluxml.ai/DataAugmentation.jl/dev/transformations/)
- Run complete examples for masks, keypoints, boxes, and volumes. [Pipelines](docs/src/pipelines.md)
- Prepare image tensors, load batches, and train a model. [Using with Flux](docs/src/flux.md)
