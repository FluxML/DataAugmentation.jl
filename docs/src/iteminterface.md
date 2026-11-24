# Defining items

Items are containers for data; see [Items](items.md) for the items that are provided and why data is wrapped in them.

To create a new item, you can simply subtype [`Item`](@ref):

```julia
struct MyItem <: Item
    data
end
```

The only function that is expected to be implemented is [`itemdata`](@ref), which simply returns the wrapped data. If, as above, you simply call the field holding the data `data`, you do not need to implement it. The same goes for the [`DataAugmentation.setdata`](@ref) helper.

For some items, it also makes sense to implement the following:

- [`DataAugmentation.showitem!`](@ref)`(img, item::I)` creates a visual representation of an item on top of `img`.

To make a transformation work with your item, see [Defining transformations](tfminterface.md). For spatial items that should support [projective transformations](projective.md), see [Defining projective transformations](projinterface.md).
