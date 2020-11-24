# Transforms

Now we can get to the projection transformations themselves. We can break down most augmentation used in practive into a single (possibly stochastic) projection and a crop.

As an example, consider an image augmentation pipeline: A random horizontal flip, followed by a random resized crop. The latter resizes and crops (irregularly sized) images to a common size without distorting the aspect ratio.

```julia
Maybe(FlipX()) |> RandomResizeCrop((h, w))
```

Let's pull apart the steps involved. 

1. Half of the time, flip the image horizontally.

2. Scale the image down without distortion so that the shorter side length is 128. With an input of size `(512, 256)` this result in scaling both dimensions by `1/2`, resulting in an image with side lengths `(256, 128)`.

3. Crop a random `(128, 128)` portion from that image. There is only "wiggle room" on the y-axis (which, by convention, is the first).

All of these steps can be efficiently computed in one step with two tricks:

- Some projections like reflection, translation, scaling and rotation can be composed into a single projection matrix. This means in the above example we only need to apply one projection which represents both the flipping (a reflection) and the scaling. Especially in pipelines with many augmentation steps this avoids a lot of unnecessary computation.

- In cases, where the result of the projection is cropped, we can save additional computing by only evaluating the parts that we want to keep. 

By default, the bounds of a projected item will be chosen so they still encase all the data. So after applying a `Scale((2, 2))` to an `Image`, its bounds will also be scaled by 2. Sometimes, however, we want to crop a part of the projected output, for example so a number of images can later be batched into a single array. While the crop usually has a fixed size, the region to crop still needs to be chosen. For validation data (which should be transformed deterministically), a center crop is usually used. For training data, on the other hand, a random region is selected to add additional augmentation. 

## In practice

For practical usage, `compose` makes sure to create a performant transform. Composing projective transformations results in a `ComposedProjectiveTransform` that simplifies the projections where possible. These transformations, for example, are reduced to a single projection:

```julia
Rotate(-10:10) |> ScaleRatio(0.7:0.1:1.2) |> FlipX() |> Crop((128, 128))
```

The following projective transformations are currently implemented:

- [`ScaleRatio`](#), [`ScaleKeepAspect`](#)
- [`Rotate`](#)
- [`FlipX`](#)
- [`FlipY`](#)

To crop the result, you can `compose` those `ProjectiveTransform`s with

- [`CenterCrop`](#) to crop a fixed-size region from the center; or
- [`RandomCrop`](#) to crop a fixed-size region from a random position
