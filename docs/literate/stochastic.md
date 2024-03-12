# Stochastic transformations


When augmenting data, it is often useful to apply a transformation only with some probability or choose from a set of transformations. Unlike in other data augmentation libraries like *albumentations*, in DataAugmentation.jl you can use wrapper transformations for this functionality.

- [`Maybe`](#)`(tfm, p = 0.5)` applies a transformation with probability `p`; and
- [`OneOf`](#)`([tfm1, tfm2])` randomly selects a transformation to apply.

## Example
Let's say we have an image classification dataset. For most datasets, horizontally flipping the image does not change the label: a flipped image of a cat still shows a cat. So let's flip every image horizontally half of the time to improve the generalization of the model we might be training.

{cell=main}
```julia
using DataAugmentation, TestImages
item = Image(testimage("lighthouse"))
tfm = Maybe(FlipX{2}())
titems = [apply(tfm, item) for _ in 1:8]
showgrid(titems; ncol = 4, npad = 16)
```