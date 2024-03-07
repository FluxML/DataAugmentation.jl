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
