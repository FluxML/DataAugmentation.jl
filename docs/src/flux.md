# Using with Flux

Use DataAugmentation.jl to transform each sample and the `DataLoader` shared by Flux and [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl) to load, shuffle, and batch samples.

This example builds a per-sample transform and loads augmented batches in the background for a Flux training loop.

## Array layout

Flux's 2D convolutions expect arrays in `(width, height, channels, batch)` order. [`ImageToTensor`](@ref) converts a `(height, width)` array of colors to a `Float32` tensor with shape `(width, height, channels)`. The data loader adds the batch axis when it stacks samples.

## A transformation for each sample

For CIFAR-10, a common pipeline randomly flips the image, pads it by four pixels, and crops a window of the original size. The `padding` keyword of [`RandomCrop`](@ref) extends the canvas, using black pixels by default for images.

```@example flux
using DataAugmentation, Images, MLUtils

means, stds = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)

train_tfm = Maybe(FlipX()) |> RandomCrop(32; padding = 4) |> PinOrigin() |>
    ImageToTensor() |> Normalize(means, stds)
test_tfm = ImageToTensor() |> Normalize(means, stds)

augment(tfm, image) = itemdata(tfm(Image(image)))
```

For a preset CIFAR-10 augmentation policy, compose [`AutoAugment`](@ref)`(:cifar10)` with the preprocessing in `test_tfm`. For evaluation, call `test_tfm` on its own.

The example uses random images and labels. The images stay in memory as 8-bit colors, a quarter of the size of `Float32` colors. [`ImageToTensor`](@ref) converts one sample at a time.

```@example flux
images = [rand(RGB{N0f8}, 32, 32) for _ in 1:64]
labels = rand(1:10, 64)

x = augment(train_tfm, images[1])
size(x), eltype(x)
```

!!! warning "End crops with `PinOrigin`"

    A plain crop can leave offset axes tied to the crop location. Batching requires matching axes across samples. Add [`PinOrigin`](@ref)`()` after [`Crop`](@ref), [`CenterCrop`](@ref), or [`RandomCrop`](@ref) to make the axes one-based before preprocessing. [`RandomResizedCrop`](@ref), [`RandomResizeCrop`](@ref), and [`CenterResizeCrop`](@ref) already return one-based axes.

## A data loader

[`mapobs`](https://juliaml.github.io/MLUtils.jl/stable/api/#MLUtils.mapobs) applies a function lazily to each `(image, label)` pair in `(images, labels)`. The function runs each time the loader requests an observation, so every epoch draws fresh transformations.

- `collate = true` loads each observation individually and stacks the image tensors along a new last dimension.
- `parallel = true` loads batches with worker tasks while the model works on the current batch. Start Julia with multiple threads, for example with `julia -t auto`. Batches can arrive out of order.
- `partial = false` drops a short final batch to keep the batch size fixed.

```@example flux
train_data = mapobs(((image, label),) -> (augment(train_tfm, image), label), (images, labels))
train_loader = DataLoader(train_data; batchsize = 16, shuffle = true, collate = true,
                          partial = false, parallel = true)

x, y = first(train_loader)
size(x), size(y)
```

The test loader iterates over all samples with shuffling disabled.

```@example flux
test_data = mapobs(((image, label),) -> (augment(test_tfm, image), label), (images, labels))
test_loader = DataLoader(test_data; batchsize = 16, collate = true, parallel = true)
```

Without MLUtils, a comprehension and `stack` do the same for a single batch: `stack([augment(train_tfm, image) for image in images[1:16]])`.

!!! warning "Buffered transformations in a data loader"

    [`Buffered`](@ref) returns a reusable output buffer by default. Later calls can overwrite samples before the loader collates them. [`BufferedThreadsafe`](@ref) uses a separate set of buffers for each concurrent call and returns a copy. See [Buffering](buffering.md).

## Mixing samples of a batch

[`MixUp`](@ref) and [`CutMix`](@ref) combine pairs of samples and their labels within a batch. Wrap a batch from the loader in an [`ImageBatch`](@ref) with the number of classes to encode labels as one-hot columns. [`OneOf`](@ref) selects a mixing transform for each batch.

```@example flux
mix = OneOf([MixUp(), CutMix()])

x, y = first(train_loader)
xmix, ymix = itemdata(mix(ImageBatch(x, y; num_classes = 10)))
size(xmix), size(ymix)
```

Each column of the mixed label matrix contains class weights for one sample.

## Training

This example uses the `Float32` batches in a Flux training loop. Select the device once and move the model to it. Mix each batch on the CPU before transferring it to the device. `gpu_device()` selects the CPU when a GPU backend is unavailable.

```julia
using Flux

device = gpu_device()

model = device(Chain(
    Conv((3, 3), 3 => 16, relu; pad = 1), MaxPool((2, 2)),
    Conv((3, 3), 16 => 32, relu; pad = 1), GlobalMeanPool(), Flux.flatten,
    Dense(32 => 10),
))
opt_state = Flux.setup(Adam(1e-3), model)

for epoch in 1:10
    for (x, y) in train_loader
        x, y = itemdata(mix(ImageBatch(x, y; num_classes = 10)))
        x, y = device(x), device(y)
        loss, grads = Flux.withgradient(model) do m
            Flux.logitcrossentropy(m(x), y)
        end
        Flux.update!(opt_state, model, grads[1])
    end
end

accuracy = sum(sum(Flux.onecold(cpu(model(device(x)))) .== y) for (x, y) in test_loader) / numobs(images)
```

## Segmentation

Call the spatial transform on the image and mask together, then prepare each result separately as in [Pipelines](pipelines.md#Images-and-segmentation-masks). By default, padding a mask repeats its edge labels.

```@example flux
masks = [rand(1:3, 32, 32) for _ in 1:64]

seg_tfm = Maybe(FlipX()) |> RandomCrop(32; padding = 4) |> PinOrigin()
mask_prep = OneHot() |> PermuteDims(2, 1, 3)

function augment_pair(image, labels)
    timage, tmask = seg_tfm((Image(image), MaskMulti(labels, 1:3)))
    return itemdata(test_tfm(timage)), itemdata(mask_prep(tmask))
end

seg_data = mapobs(((image, labels),) -> augment_pair(image, labels), (images, masks))
seg_loader = DataLoader(seg_data; batchsize = 16, shuffle = true, collate = true,
                        partial = false, parallel = true)

x, y = first(seg_loader)
size(x), size(y)
```

## Object detection

Use the [image and bounding box pipeline](pipelines.md#Images-and-bounding-boxes) in `mapobs` to return an image tensor with the transformed boxes and their labels. Crop sizes and box counts can vary across samples. Use `collate = false` to receive a vector of `(image, target)` pairs, or supply a custom collation function for the detector's batch format.
