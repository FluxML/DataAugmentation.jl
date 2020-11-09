
# Quickstart

{cell=main style="display:none;"}
```julia
using TestImages
using ImageShow
using Images
using DataAugmentation
```

1. Import the library:
    ```julia
    using DataAugmentation  # Image, CenterResizeCrop, apply, showitem
    ```

2. Load your data

    ```julia
    image = testimage("lighthouse")   # from TestImages.jl
    ```

    {cell=main style="display:none;"}
    ```julia
    image = imresize(testimage("lighthouse"), ratio = 1/3)
    ```

3. Create an item that contains the data you want to augment:
    {cell=main result=false}
    ```julia
    item = Image(image)
    ```

4. Create a transform:
    {cell=main result=false}
    ```julia
    tfm = CenterResizeCrop((128, 128))
    ```

5. Apply the transformation and unwrap the data:
    {cell=main}
    ```julia
    titem = apply(tfm, item)
    timage = itemdata(titem)
    ```


Now read the [overview](./overview.md).



