# Using with Flux

DataAugmentation.jl does not come with a data loader. It provides the part that is specific to augmentation: a function that turns one sample into one array ready for the model. Loading, shuffling and batching of samples is left to the tools made for it, such as `DataLoader` from [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl), which is also the `DataLoader` of Flux.

This page builds a training pipeline for an image classifier in three steps: a transformation for each sample, a data loader that applies it in the background while the model trains, and a Flux training loop.

## Array layout

Flux's convolutional layers, as well as the pretrained models of Metalhead.jl, expect images in `WHCN` order: width, height, channels, and a batch dimension last. Images, as arrays of colors, are stored as `(height, width)`. The preprocessing transformations take care of the conversion of a single image: [`ImageToTensor`](@ref) permutes the dimensions into `(width, height, channels)` and converts the colors to `Float32`. The batch dimension is then added by the data loader when it stacks samples.

## A transformation for each sample

The usual augmentation of CIFAR10 images is to flip them horizontally, pad them by four pixels and crop a random region of the original size. The `padding` keyword of [`RandomCrop`](@ref) does the last two steps, and pads images with black pixels:

```@example flux
using DataAugmentation, Images, MLUtils

means, stds = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)

train_tfm = Maybe(FlipX()) |> RandomCrop((32, 32); padding = 4) |> PinOrigin() |>
    ImageToTensor() |> Normalize(means, stds)
test_tfm = ImageToTensor() |> Normalize(means, stds)

augment(tfm, image) = itemdata(apply(tfm, Image(image)))
```

At test time there is nothing to augment, so only the preprocessing is applied. As stand-in for a dataset like CIFAR10, we use random images and labels. Note that the images stay in memory as 8-bit colors, a quarter of the size of `Float32` ones. [`ImageToTensor`](@ref) converts them one sample at a time:

```@example flux
images = [rand(RGB{N0f8}, 32, 32) for _ in 1:64]
labels = rand(1:10, 64)

x = augment(train_tfm, images[1])
size(x), eltype(x)
```

!!! warning "End crops with `PinOrigin`"

    A crop leaves its result with axes that do not start at 1: where they start depends on the random location of the crop. Arrays with different axes cannot be batched. [`PinOrigin`](@ref)`()` moves the result back to the origin. [`RandomResizeCrop`](@ref) and [`CenterResizeCrop`](@ref) do that already, but a [`Crop`](@ref), [`CenterCrop`](@ref) or [`RandomCrop`](@ref) on its own does not.

## A data loader

[`mapobs`](https://juliaml.github.io/MLUtils.jl/stable/api/#MLUtils.mapobs) applies a function lazily to every observation of a data container, here the tuple `(images, labels)`, whose observations are pairs of an image and a label. The function runs each time an observation is loaded, so every epoch sees new random transformations. The options of the `DataLoader` that matter for speed and for the shape of the batches are:

- `collate = true` applies the function to each observation first, and then stacks the results along a new last dimension. Without it, the function would be called with the whole batch at once.
- `parallel = true` loads batches with worker tasks spread over the available threads, while the training loop works on the current batch. Augmentation is work for the CPU, and this keeps the model from waiting for data. It needs Julia to be started with multiple threads, for example with `julia -t auto`, and does not preserve the order of the batches.
- `partial = false` drops the last batch if it is smaller than the others, so that all batches have the same size. This avoids recompiling or retracing the model for a differently sized batch.

```@example flux
train_data = mapobs(((image, label),) -> (augment(train_tfm, image), label), (images, labels))
train_loader = DataLoader(train_data; batchsize = 16, shuffle = true, collate = true,
                          partial = false, parallel = true)

x, y = first(train_loader)
size(x), size(y)
```

The loader for the test data is built in the same way, without shuffling and with all the samples:

```@example flux
test_data = mapobs(((image, label),) -> (augment(test_tfm, image), label), (images, labels))
test_loader = DataLoader(test_data; batchsize = 16, collate = true, parallel = true)
```

Without MLUtils, a comprehension and `stack` do the same for a single batch: `stack([augment(train_tfm, image) for image in images[1:16]])`.

!!! warning "Buffered transformations in a data loader"

    The result of a [`Buffered`](@ref) transformation is its buffer, which the next call overwrites. A loader stacks the observations of a batch only after all of them have been loaded, so every sample of the batch would be the last one, and worker tasks would share the buffer. To reuse buffers in a loader, wrap the transformation in [`BufferedThreadsafe`](@ref) instead, which gives every task its own buffers and returns a copy. See [Buffering](buffering.md), also on when buffering pays off.

## Mixing samples of a batch

[`MixUp`](@ref) and [`CutMix`](@ref) combine pairs of samples and their labels, so they work on whole batches instead of single samples. Wrap a batch from the loader in an [`ImageBatch`](@ref) with the number of classes, which encodes the labels as one-hot columns. Here we choose one of the two for every batch, as is common:

```@example flux
mix = OneOf([MixUp(), CutMix()])

x, y = first(train_loader)
xmix, ymix = itemdata(apply(mix, ImageBatch(x, y; num_classes = 10)))
size(xmix), size(ymix)
```

The labels are now soft: each column holds the weights of the two classes mixed in the sample.

## Training

The batches are ordinary arrays of `Float32`, so the training loop is the same as without augmentation. This is a small example with Flux (not run here). Define the device once, move the model to it, and move every batch to it inside the loop, after mixing it on the CPU. Without a GPU, `gpu_device()` is the CPU:

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
        x, y = itemdata(apply(mix, ImageBatch(x, y; num_classes = 10)))
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

For tasks with a target that needs to be transformed together with the input, the function of a sample returns both. As described in [Pipelines](pipelines.md), the image and the mask go through the same transformation, and are then prepared separately. The padding of a mask repeats the labels at its edges:

```@example flux
masks = [rand(1:3, 32, 32) for _ in 1:64]

seg_tfm = Maybe(FlipX()) |> RandomCrop((32, 32); padding = 4) |> PinOrigin()
mask_prep = OneHot() |> PermuteDims(2, 1, 3)

function augment_pair(image, labels)
    timage, tmask = apply(seg_tfm, (Image(image), MaskMulti(labels, 1:3)))
    return itemdata(apply(test_tfm, timage)), itemdata(apply(mask_prep, tmask))
end

seg_data = mapobs(((image, labels),) -> augment_pair(image, labels), (images, masks))
seg_loader = DataLoader(seg_data; batchsize = 16, shuffle = true, collate = true,
                        partial = false, parallel = true)

x, y = first(seg_loader)
size(x), size(y)
```
