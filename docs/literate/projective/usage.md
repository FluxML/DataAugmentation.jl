# Usage

Using projective transformations is as simple as any other transformations. Simply `compose` them:

```julia
Rotate(-10:10) |> ScaleRatio(0.7:0.1:1.2) |> FlipX() |> Crop((128, 128))
```

The composition will automatically create a single projective transformation and evaluate only the cropped area.

## Affine transformations

Affine transformations are a subgroup of projective transformations that can be composed very efficiently: composing two affine transformations results in another affine transformation. Affine transformations can represent translation, scaling, reflection and rotation. Available `Transform`s are:

- [`ScaleRatio`](#), [`ScaleKeepAspect`](#)
- [`Rotate`](#)
- [`FlipX`](#)
- [`FlipY`](#)

## Crops

To get a cropped result, simply `compose` any `ProjectiveTransform` with

- [`CenterCrop`](#) to crop a fixed-size region from the center; or
- [`RandomCrop`](#) to crop a fixed-size region from a random position
