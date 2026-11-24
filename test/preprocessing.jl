include("./imports.jl")



@testset "Normalize" begin
    @testset "normalize" begin
        image = rand(Float32, 10, 10, 3)
        means = SVector{3}(rand(Float32, 1, 1, 3))
        stds = SVector{3}(rand(Float32, 1, 1, 3))
        @test denormalize(normalize(copy(image), means, stds), means, stds) ≈ image
    end

    @testset "Normalize" begin
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

@testset "NormalizeIntensity" begin
    item = ArrayItem(Float32.(collect(1:5)))
    ground_truth = [-1.2649, -0.6325, 0, 0.6325, 1.2649]
    tfm = NormalizeIntensity()
    @test itemdata(apply(tfm, item)) ≈ ground_truth
end

@testset "LinearTransformation" begin
    matrix = Float32[1 2 0 0; 0 1 0 1; 3 0 1 0; 0 0 0 2]
    mean = Float32[1, 0, -1, 2]
    data = rand(Float32, 2, 2)
    item = ArrayItem(data)

    @testset "values" begin
        res = itemdata(apply(LinearTransformation(matrix; mean), item))
        @test size(res) == (2, 2)
        @test eltype(res) == Float32
        @test vec(res) ≈ matrix * (vec(data) - mean)
        @test vec(itemdata(apply(LinearTransformation(matrix), item))) ≈ matrix * vec(data)
    end

    @testset "element type" begin
        @test eltype(itemdata(apply(LinearTransformation(Float64.(matrix)), item))) == Float64
        @test eltype(itemdata(apply(LinearTransformation(matrix), ArrayItem(rand(UInt8, 2, 2))))) == Float32
    end

    @testset "whitening" begin
        samples = randn(Float64, 4, 2000)
        samples = [1 0.5 0 0; 0 2 0 0; 0 0 3 0; 0.2 0 0 1] * samples .+ [1, 2, 3, 4]
        μ = vec(Statistics.mean(samples; dims = 2))
        eig = eigen(Symmetric(cov(samples; dims = 2)))
        W = eig.vectors * Diagonal(1 ./ sqrt.(eig.values)) * eig.vectors'
        tfm = LinearTransformation(W; mean = μ)
        white = reduce(hcat, vec(itemdata(apply(tfm, ArrayItem(reshape(c, 2, 2))))) for c in eachcol(samples))
        @test cov(white; dims = 2) ≈ I atol = 1e-8
    end

    @testset "apply!" begin
        tfm = LinearTransformation(matrix; mean)
        item2 = ArrayItem(data .+ 1)
        testapply(tfm, item)
        testapply!(tfm, item, item2)
        buf = makebuffer(tfm, item)
        res = apply!(buf, tfm, item2)
        @test res === buf
        @test itemdata(res) ≈ itemdata(apply(tfm, item2))
        # an aliased buffer must not be overwritten while it is read
        aliased = ArrayItem(copy(data))
        res = apply!(aliased, tfm, aliased)
        @test vec(itemdata(res)) ≈ matrix * (vec(data) - mean)
        @test itemdata(aliased) == data
    end

    @testset "pipeline" begin
        tfm = ImageToTensor() |> LinearTransformation(Matrix{Float32}(I, 12, 12) * 2)
        res = apply(tfm, Image(rand(RGB{Float32}, 2, 2)))
        @test size(itemdata(res)) == (2, 2, 3)
    end

    @testset "errors" begin
        @test_throws DimensionMismatch LinearTransformation(rand(3, 4))
        @test_throws DimensionMismatch LinearTransformation(rand(3, 3); mean = zeros(2))
        @test_throws DimensionMismatch apply(LinearTransformation(rand(3, 3)), item)
    end
end

@testset "ToEltype" begin
    data = rand(RGB, 10, 10)
    imagergb = Image(data)
    imagegray = Image((Gray.(data)))
    @test itemdata(apply(ToEltype(Gray), imagegray)) === itemdata(imagegray)
    @test itemdata(apply(ToEltype(Gray), imagergb)) ≈ itemdata(imagegray)

    tfm = ToEltype(Gray)
    testapply(tfm, Image)
    testapply!(tfm, Image)
end


@testset "ImageToTensor" begin
    @testset "imagetotensor,tensortoimage" begin
        data = rand(RGB, 10, 10)
        @test size(imagetotensor(data)) == (10, 10, 3)
        @test data |> imagetotensor |> tensortoimage ≈ data

    end

    @testset "ImageToTensor" begin
        image = Image(rand(RGB, 10, 10))
        tfm = ImageToTensor()
        @test_nowarn apply(tfm, image)
        a = itemdata(apply(tfm, image))
        @test size(a) == (10, 10, 3)
        testapply(tfm, Image)
        testapply!(tfm, Image)
    end
end

@testset "PermuteDims" begin
    tfm = PermuteDims(2, 1, 4, 3)
    item1 = ArrayItem(rand(3, 4, 5, 6))
    item2 = ArrayItem(rand(3, 4, 5, 6))
    @test_nowarn apply(tfm, item1)
    a = itemdata(apply(tfm, item1))
    @test size(a) == (4, 3, 6, 5)

    testapply(tfm, item1)
    testapply!(tfm, item1, item2)
end

@testset "OneHot" begin
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

@testset "Image pipeline" begin
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
    @test size(a) == (48, 32, 3)
    @test eltype(a) == Float32

    testapply(tfm, item1)
    testapply!(tfm, item1, item2)
end
