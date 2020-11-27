# Preprocessing

This library also implements some general transformations useful for getting data ready to be put into a model.

- [`ToEltype`](#)`(T)` converts the element type of any [`AbstractArrayItem`](#) to `T`.
- [`ImageToTensor`](#) converts an image to an `ArrayItem` with another dimension for the color channels 
- [`Normalize`](#) normalizes image tensors

