include("imports.jl")


@testset ExtendedTestSet "`AdjustContrast`" begin
    item = Image(rand(RGB{N0f8}, 64, 64))
    tfm = AdjustContrast(0.2)
    @test_nowarn apply(tfm, item)
    titem = apply(tfm, item)
    @test eltype(itemdata(titem)) == RGB{N0f8}
    testapply(tfm, Image)
    testapply!(tfm, Image)
end

@testset ExtendedTestSet "`AdjustBrightness`" begin
    item = Image(rand(RGB{N0f8}, 64, 64))
    tfm = AdjustBrightness(0.2)
    @test_nowarn apply(tfm, item)
    titem = apply(tfm, item)
    @test eltype(itemdata(titem)) == RGB{N0f8}

    testapply(tfm, Image)
    testapply!(tfm, Image)
end
