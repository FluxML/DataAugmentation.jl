include("./imports.jl")

#=
Test

- Normalize
    - normalize
    - denormalize
- ToEltype
- SplitChannels
    - imagetotensor
    - tensortoimage
- OneHotEncode
    - onehot

- ImagePipeline
=#


@testset ExtendedTestSet "Normalize" begin
    @testset ExtendedTestSet "normalize" begin
        image = rand(Float32, 10, 10, 3)
        means = rand(Float32, 3)
        stds = rand(Float32, 3)
        @test denormalize(normalize(copy(image), means, stds), means, stds) ≈ image
    end

    @testset ExtendedTestSet "Normalize" begin
        item = ArrayItem(zeros(10, 10, 3))
        means = ones(3)
        stds = ones(3)
        tfm = Normalize(means, stds)
        @test itemdata(apply(tfm, item)) ≈ -1 .* ones(10, 10, 3)
    end
end

@testset ExtendedTestSet "ToEltype" begin
    tfm = ToEltype(RGB)
    data = rand(RGB, 10, 10)
    imagergb = Image(data)
    imagegray = Image((Gray.(data)))
    @test itemdata(apply(ToEltype(Gray), imagegray)) === itemdata(imagegray)
    @test itemdata(apply(ToEltype(Gray), imagergb)) ≈ itemdata(imagegray)
end


@testset ExtendedTestSet "SplitChannels" begin
    @testset ExtendedTestSet "imagetotensor,tensortoimage" begin
        data = rand(RGB, 10, 10)
        @test size(imagetotensor(data)) == (10, 10, 3)
        @test data |> imagetotensor |> tensortoimage ≈ data

    end

    @testset ExtendedTestSet "SplitChannels" begin
        image = Image(rand(RGB, 10, 10))
        tfm = SplitChannels()
        @test_nowarn apply(tfm, image)
        a = itemdata(apply(tfm, image))
        @test size(a) == (10, 10, 3)
    end
end

@testset ExtendedTestSet "Image pipeline" begin
    image = Image(testimage("lena"))

    tfm = compose(
        ToEltype(RGB),
        RandomResizeCrop((128, 64)),
        SplitChannels(),
        Normalize([0, 0, 0], [1, 1, 1])
    )

    @test_nowarn apply(tfm, image)

    res = apply(tfm, image)
    a = itemdata(res)
    @test size(a) == (128, 64, 3)
    @test eltype(a) == Float32
end
