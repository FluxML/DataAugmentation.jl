using DataAugmentation: Random

@testset "ImageBatch" begin
    images = reshape(Float32.(1:12), 2, 2, 3)
    batch = ImageBatch(images, [1, 2, 1]; num_classes = 2)
    @test itemdata(batch)[1] === images
    @test itemdata(batch)[2] == Float32[1 0 1; 0 1 0]
    @test length(batch) == 3
    @test_throws ArgumentError ImageBatch(images, [0, 1, 1]; num_classes = 2)
    @test_throws DimensionMismatch ImageBatch(images, zeros(2, 2))
    @test_throws ArgumentError ImageBatch(ones(2, 2), zeros(2, 1))
    @test_throws ArgumentError ImageBatch(ones(0, 2, 3), zeros(2, 3))

    empty = ImageBatch(zeros(Float32, 2, 2, 0), Int[]; num_classes = 2)
    @test itemdata(apply(MixUp(), empty)) == itemdata(empty)
    @test itemdata(apply(CutMix(), empty)) == itemdata(empty)
end

@testset "MixUp" begin
    images = [i for _ in 1:2, _ in 1:3, i in 1:4]
    labels = Matrix{Bool}(I, 4, 4)
    batch = ImageBatch(images, labels)
    tfm = MixUp(alpha = 0.5)
    state = (weight = 0.25, seed = UInt64(0x1234))
    mixed_images, mixed_labels = itemdata(apply(tfm, batch; randstate = state))
    partners = Random.randperm(Random.Xoshiro(state.seed), length(batch))
    for (i, j) in enumerate(partners)
        @test mixed_images[:, :, i] == fill(0.25 * i + 0.75 * j, 2, 3)
        @test mixed_labels[:, i] == 0.25 .* labels[:, i] .+ 0.75 .* labels[:, j]
    end
    @test eltype(mixed_images) == Float64
    @test eltype(mixed_labels) == Float64
    @test images == [i for _ in 1:2, _ in 1:3, i in 1:4]
    @test labels == Matrix{Bool}(I, 4, 4)
    testapply(tfm, batch)
    @test_throws ArgumentError MixUp(0)
    @test_throws ArgumentError CutMix(Inf)
end

@testset "CutMix" begin
    images = [UInt8(10 * i + c) for _ in 1:8, _ in 1:7, c in 1:2, i in 1:4]
    labels = Matrix{Float32}(I, 4, 4)
    batch = ImageBatch(images, labels)
    original = copy(images)
    state = (weight = 0.5, seed = UInt64(0x5678))
    mixed_images, mixed_labels = itemdata(apply(CutMix(), batch; randstate = state))
    partners = Random.randperm(Random.Xoshiro(state.seed), length(batch))
    @test eltype(mixed_images) == UInt8
    @test mixed_images != images
    for (i, j) in enumerate(partners)
        changed = mixed_images[:, :, 1, i] .!= images[:, :, 1, i]
        @test changed == (mixed_images[:, :, 2, i] .!= images[:, :, 2, i])
        for c in axes(images, 3)
            @test mixed_images[:, :, c, i] == ifelse.(changed, images[:, :, c, j], images[:, :, c, i])
        end
        borrowed = count(changed) / length(changed)
        @test mixed_labels[:, i] ≈ (1 - borrowed) .* labels[:, i] .+ borrowed .* labels[:, j]
    end
    @test images == original

    tiny = ImageBatch(reshape(UInt8[1, 2, 3, 4], 1, 1, 4), labels)
    @test itemdata(apply(CutMix(), tiny; randstate = (weight = 0.9, seed = state.seed))) == itemdata(tiny)
end

@testset "Batch mixing composition and buffers" begin
    images = rand(Float32, 4, 4, 3)
    labels = Matrix{Float32}(I, 3, 3)
    batch = ImageBatch(images, labels)
    mixup, cutmix = MixUp(), CutMix()
    states = ((weight = 0.4, seed = UInt64(1)), (weight = 0.6, seed = UInt64(2)))
    mixed = apply(mixup, batch; randstate = states[1])
    expected = itemdata(apply(cutmix, mixed; randstate = states[2]))
    @test itemdata(apply(mixup |> cutmix, batch; randstate = states)) == expected
    @test itemdata(apply(Maybe(mixup), batch; randstate = (1, states[1]))) == itemdata(mixed)
    @test apply(Maybe(mixup), batch; randstate = (2, nothing)) === batch

    buf = makebuffer(mixup, batch)
    @test apply!(buf, mixup, batch; randstate = states[1]) === buf
    @test itemdata(buf) == itemdata(mixed)
    @test itemdata(apply!(batch, mixup, batch; randstate = states[1])) == itemdata(mixed)

    cached = Buffered(mixup; copy = true)
    result = apply(cached, batch; randstate = states[1])
    @test itemdata(result) == itemdata(mixed)
    apply(cached, batch; randstate = states[2])
    @test itemdata(result) == itemdata(mixed)
    for (before, after) in zip(itemdata(result), itemdata(cached.buffers[typeof(batch)]))
        @test before !== after
    end

    shorter = ImageBatch(images[:, :, 1:2], labels[:, 1:2])
    @test itemdata(apply(cached, shorter; randstate = states[1])) ==
        itemdata(apply(mixup, shorter; randstate = states[1]))
    @test apply!(buf, BufferedThreadsafe(mixup), batch; randstate = states[1]) === buf
    @test itemdata(buf) == itemdata(mixed)
end
