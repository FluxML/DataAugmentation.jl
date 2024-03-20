# Item interface

As described previously, items are simply containers for data: an [`Image`](@ref) represents an image, and [`Keypoints`](@ref) some keypoints.

### Why do I need to wrap my data in an item?

For one, the item `struct`s may contain metadata that is useful for some transformations. More importantly, though, by wrapping data in an item type, the *meaning* of the data is separated from the *representation*, that is, the concrete type.

An `Array{Integer, 2}` could represent an image, but also a multi-class segmentation mask. Is `Array{Float32, 3}` a 3-dimensional image or a 2-dimensional image with the color channels expanded?

Separating the representation from the data's meaning resolves those ambiguities.

## Creating items

To create a new item, you can simply subtype [`Item`](@ref):

```julia
struct MyItem <: Item
    data
end
```

The only function that is expected to be implemented is [`itemdata`](@ref), which simply returns the wrapped data. If, as above, you simply call the field holding the data `data`, you do not need to implement it. The same goes for the [`DataAugmentation.setdata`](@ref) helper.

For some items, it also makes sense to implement the following:

- [`DataAugmentation.showitem!`](@ref)`(img, item::I)` creates a visual representation of an item on top of `img`.
