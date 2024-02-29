include("./imports.jl")



@testset ExtendedTestSet "Normalize" begin
    @testset ExtendedTestSet "normalize" begin
        image = rand(Float32, 10, 10, 3)
        means = SVector{3}(rand(Float32, 1, 1, 3))
        stds = SVector{3}(rand(Float32, 1, 1, 3))
        @test denormalize(normalize(copy(image), means, stds), means, stds) ≈ image
    end

    @testset ExtendedTestSet "Normalize" begin
        item = ArrayItem(zeros(10, 10, 3))
        means =  SVector{3}(ones(1, 1, 3))
        stds =  SVector{3}(ones(1, 1, 3))
        tfm = Normalize(means, stds)
        @test itemdata(apply(tfm, item)) ≈ -1 .* ones(10, 10, 3)
        testapply(tfm, item)
        item1 = ArrayItem(rand(10, 10, 3))
        item2 = ArrayItem(rand(10, 10, 3))
        testapply!(tfm, item1, item2)
    end
end

@testset ExtendedTestSet "NormalizeIntensity" begin
    item = ArrayItem(Float32.(collect(1:5)))
    ground_truth = [-1.2649, -0.6325, 0, 0.6325, 1.2649]
    tfm = NormalizeIntensity()
    @test itemdata(apply(tfm, item)) ≈ ground_truth
end

@testset ExtendedTestSet "ToEltype" begin
    data = rand(RGB, 10, 10)
    imagergb = Image(data)
    imagegray = Image((Gray.(data)))
    @test itemdata(apply(ToEltype(Gray), imagegray)) === itemdata(imagegray)
    @test itemdata(apply(ToEltype(Gray), imagergb)) ≈ itemdata(imagegray)

    tfm = ToEltype(Gray)
    testapply(tfm, Image)
    testapply!(tfm, Image)
end


@testset ExtendedTestSet "ImageToTensor" begin
    @testset ExtendedTestSet "imagetotensor,tensortoimage" begin
        data = rand(RGB, 10, 10)
        @test size(imagetotensor(data)) == (10, 10, 3)
        @test data |> imagetotensor |> tensortoimage ≈ data

    end

    @testset ExtendedTestSet "ImageToTensor" begin
        image = Image(rand(RGB, 10, 10))
        tfm = ImageToTensor()
        @test_nowarn apply(tfm, image)
        a = itemdata(apply(tfm, image))
        @test size(a) == (10, 10, 3)
        testapply(tfm, Image)
        testapply!(tfm, Image)
    end
end

@testset ExtendedTestSet "PermuteDims" begin
    tfm = PermuteDims(2, 1, 4, 3)
    A = rand(3, 4, 5, 6)
    item = ArrayItem(A)
    @test_nowarn apply(tfm, item)
    B = itemdata(apply(tfm, item))
    @test size(B) == (4, 3, 6, 5)
    testapply(tfm, item)
end

@testset ExtendedTestSet "OneHot" begin
    tfm = OneHot()
    mask = rand(1:4, 10, 10)
    item1 = MaskMulti(mask, 1:4)
    item2 = MaskMulti(rand(1:3, 10, 10), 1:4)
    @test_nowarn apply(tfm, item1)
    aitem = apply(tfm, item1)
    @test size(itemdata(aitem)) == (10, 10, 4)

    testapply(tfm, item1)
    testapply!(tfm, item1, item2)

end

@testset ExtendedTestSet "Image pipeline" begin
    item1 = Image(rand(RGB, 64, 64))
    item2 = Image(rand(RGB, 64, 64))

    tfm = compose(
        ToEltype(RGB),
        RandomResizeCrop((32, 48)),
        ImageToTensor(),
        Normalize([0, 0, 0], [1, 1, 1])
    )


    res = apply(tfm, item1)
    a = itemdata(res)
    @test size(a) == (32, 48, 3)
    @test eltype(a) == Float32

    testapply(tfm, item1)
    testapply!(tfm, item1, item2)
end
