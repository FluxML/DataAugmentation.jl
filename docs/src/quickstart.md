# Quickstart

```@setup deps
using TestImages
using ImageShow
using Images
using DataAugmentation
```

Import the library:
```
using DataAugmentation  # Image, CenterResizeCrop, apply, showitem
using TestImages: testimage
```
Load your data:
```@example deps
image = testimage("lighthouse")
```
Create an item that contains the data you want to augment:
```@example deps
item = Image(image)
```
Create a transform:
```@example deps
tfm = CenterResizeCrop((128, 128))
```
Apply the transformation and unwrap the data:
```@example deps
titem = apply(tfm, item)
timage = itemdata(titem)
```

## Next steps

- [Items](items.md) explains what can be wrapped in an item.
- [Transformations](transformations.md) shows how to compose transformations and randomize them, and lists the available ones.
- The [Gallery](gallery.md) shows what the spatial transformations do.
- [Pipelines](pipelines.md) puts them together for images, segmentation masks, keypoints and 3D images.
- [Using with Flux](flux.md) feeds augmented batches to a model.
