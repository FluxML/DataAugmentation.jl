# Preprocessing

This library also implements some general transformations useful for getting data ready to be put into a model.

- [`ToEltype`](@ref)`(T)` converts the element type of any [`DataAugmentation.AbstractArrayItem`](@ref) to `T`.
- [`ImageToTensor`](@ref) converts an image to an `ArrayItem` with another dimension for the color channels 
- [`Normalize`](@ref) normalizes image tensors
- [`OneHot`](@ref) to one-hot encode multi-class masks ([`MaskMulti`](@ref)s)

