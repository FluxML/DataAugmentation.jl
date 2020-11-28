# Item interface

{style="opacity:60%;"}
*[`base.jl`](../../src/base.jl)*

As described previously, items are simply containers for data: an [`Image`](#) represents an image, and [`Keypoints`](#) some keypoints.

### Why do I need to wrap my data in an item?

For one, the item `struct`s may contain metadata that is useful for some transformations. More importantly, though, by wrapping data in an item type, the *meaning* of the data is separated from the *representation*, that is, the concrete type.

An `Array{Integer, 2}` could represent an image, but also a multi-class segmentation mask. Is `Array{Float32, 3}` a 3-dimensional image or a 2-dimensional image with the color channels expanded?

Separating the representation from the data's meaning resolves those ambiguities.

## Creating items

To create a new item, you can simply subtype [`Item`](#):

```julia
struct MyItem <: Item
    data
end
```

The only function that is expected to be implemented is [`itemdata`](#), which simply returns the wrapped data. If, as above, you simply call the field holding the data `data`, you do not need to implement it. The same goes for the [`setdata`](#) helper.

For some items, it also makes sense to implement the following:

- [`showitem`](#)`(item::I)` creates a visual representation of an item. Should return something that can be shown as an image.