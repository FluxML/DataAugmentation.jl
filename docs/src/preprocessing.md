# Preprocessing

This library also implements some general transformations useful for getting data ready to be put into a model. They work on [`ArrayItem`](@ref)s and, for [`ImageToTensor`](@ref) and [`OneHot`](@ref), on the corresponding spatial items.

## Arrays

- [`ToEltype`](@ref)`(T)` converts the element type of any [`DataAugmentation.AbstractArrayItem`](@ref) to `T`.
- [`ImageToTensor`](@ref) converts an image to an `ArrayItem` with another dimension for the color channels, in `WHC` format.
- [`PermuteDims`](@ref)`(perm)` permutes the dimensions of an `ArrayItem`, for example to turn `WHC` into `HWC`.
- [`Normalize`](@ref)`(means, stds)` normalizes image tensors along the channel dimension.
- [`NormalizeIntensity`](@ref)`()` normalizes an array with its own mean and standard deviation.
- [`OneHot`](@ref) to one-hot encode multi-class masks ([`MaskMulti`](@ref)s); [`onehot`](@ref)`(x, n)` one-hot encodes a single class index.
- [`MapElem`](@ref)`(f)` applies a function to every element of an array item.

Preprocessing steps are transformations, so they compose with `|>` like any other:

```@example prep
using DataAugmentation, Images

image = Image(rand(RGB{Float32}, 40, 50))
tfm = ImageToTensor() |> Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
size(itemdata(apply(tfm, image)))
```

One-hot encoding a mask with 4 classes adds a dimension for the classes:

```@example prep
mask = MaskMulti(rand(1:4, 8, 8), 1:4)
size(itemdata(apply(OneHot(), mask)))
```

## Tabular data

A row of a table is wrapped in a [`TabularItem`](@ref)`(data, columns)`, where `columns` are the names of the columns of `data`. Three transformations prepare rows for a model; they are not exported, so import them from `DataAugmentation`:

- [`DataAugmentation.FillMissing`](@ref)`(dict, cols)` fills missing values in the columns `cols` with the values given in `dict`.
- [`DataAugmentation.Categorify`](@ref)`(dict, cols)` label encodes the values in the columns `cols` using the unique values given in `dict`. A value is replaced by its position in that list plus one, and a missing value by `1`.
- [`DataAugmentation.NormalizeRow`](@ref)`(dict, cols)` normalizes the values in the columns `cols` with the mean and standard deviation given in `dict`.

```@example prep
using DataAugmentation: FillMissing, Categorify, NormalizeRow

cols = [:age, :city, :income]
item = TabularItem((; zip(cols, [missing, "rome", 52_000.0])...), cols)

tfm = FillMissing(Dict(:age => 40), [:age]) |>
    Categorify(Dict(:city => ["milan", "rome"]), [:city]) |>
    NormalizeRow(Dict(:income => (50_000.0, 10_000.0)), [:income])
itemdata(apply(tfm, item))
```
