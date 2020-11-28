

# Projective transformations

DataAugmentation.jl has great support for transforming spatial data like images and keypoints. Most of these transformations are projective transformations. For our purposes, a projection means a mapping between two coordinate spaces. In computer vision, these are frequently used for preprocessing and augmenting image data: images are randomly scaled, maybe flipped horizontally and finally cropped to the same size.

This library generalizes projective transformations for different kinds of image and keypoint data in an N-dimensional Euclidean space. It also uses composition for performance improvements like fusing affine transformations.

Unlike mathematical objects, the spatial data we want to transform has *spatial bounds*. For an image, these bounds are akin to the array size. But keypoint data aligned with an image has the same bounds even if they are not explicitly encoded in the representation of the data.
These spatial bounds can be used to dynamically create useful transformations. For example, a rotation around the center or a horizontal flip of keypoint annotations can be calculated from the bounds.

Often, we also want to *crop* an area from the projected results. By evaluating only the parts of a projection that fall inside the cropped area, a lot of unnecessary computation can be avoided.

## An example pipeline


We can break down most augmentation used in practive into a single (possibly stochastic) projection and a crop.

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

## Cropping

By default, the bounds of a projected item will be chosen so they still encase all the data. So after applying a `Scale((2, 2))` to an `Image`, its bounds will also be scaled by 2. Sometimes, however, we want to crop a part of the projected output, for example so a number of images can later be batched into a single array. While the crop usually has a fixed size, the region to crop still needs to be chosen. For validation data (which should be transformed deterministically), a center crop is usually used. For training data, on the other hand, a random region is selected to add additional augmentation. 

---

Read on to find out [how projective transformations are implemented](./interface.md) or jump straight to the [usage section](./usage.md).