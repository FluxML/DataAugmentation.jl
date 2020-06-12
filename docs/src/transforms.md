# Image Transforms

```@contents
pages = ["transforms.md"]
```

```@setup imt
using Images
using TestImages
function showimagetransform(image, tfm)
    imaget = apply(tfm, Image(image)).data

    return mosaicview(image, imaget; nrow = 1, npad = 10, fillvalue = zero(RGBA))
end
```

```@example imt
using DataAugmentation
using Images: imresize
using TestImages

image = imresize(testimage("lighthouse"); ratio = 1/2)
```

## [`FlipX`](@ref)

```@example imt
tfm = FlipX()
showimagetransform(image, tfm) # hide
```

## [`FlipY`](@ref)

```@example imt
tfm = FlipY()
showimagetransform(image, tfm) # hide
```