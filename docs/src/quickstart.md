# Quickstart

```@setup deps
using TestImages
using ImageShow
using Images
using DataAugmentation
```

Import the library:
```
using DataAugmentation
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
Call the transform on the item, then unwrap its data.

```@example deps
titem = tfm(item)
timage = itemdata(titem)
```

Pass a tuple such as `tfm((item, mask))` to share the transform's random state
across aligned items.

## Next steps

- [Items](items.md) explains what can be wrapped in an item.
- [Transformations](transformations.md) shows how to compose transformations and randomize them, and lists the available ones.
- The [Gallery](gallery.md) shows what the spatial transformations do.
- [Pipelines](pipelines.md) puts them together for images, segmentation masks, keypoints and 3D images.
- [Using with Flux](flux.md) feeds augmented batches to a model.
