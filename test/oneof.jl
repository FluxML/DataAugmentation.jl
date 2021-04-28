include("imports.jl")


@testset ExtendedTestSet "OneOf" begin
    item = ArrayItem(rand(10, 10))
    tfm = OneOf([MapElem(x -> x + 1), MapElem(x -> x - 1)])
    @test apply(tfm, item; randstate = (1, nothing)).data ≈ item.data .+ 1
    @test apply(tfm, item; randstate = (2, nothing)).data ≈ item.data .- 1
    testapply(tfm, ArrayItem)
    testapply!(tfm, ArrayItem)
end

@testset ExtendedTestSet "Maybe" begin
    item = ArrayItem(rand(10, 10))
    tfm = Maybe(MapElem(x -> x + 1))
    @test apply(tfm, item; randstate = (1, nothing)).data ≈ item.data .+ 1
    @test apply(tfm, item; randstate = (2, nothing)).data ≈ item.data
    testapply(tfm, ArrayItem)
end


@testset ExtendedTestSet "OneOfProjective" begin
    item = Image(rand(10, 10))
    tfm = Rotate(10) |> CenterCrop((5, 5))
    oneof = Maybe(tfm)
    @test oneof isa DataAugmentation.OneOfProjective
    @test_nowarn DataAugmentation.getprojection(oneof, getbounds(item); randstate = getrandstate(oneof))
    @test apply(oneof, item; randstate = (1, getrandstate(tfm))) isa Image
    @test apply(oneof, item; randstate = (2, nothing)) isa Image
end


@testset ExtendedTestSet "Sequence(oneofs...)" begin
    item = ArrayItem(rand(10, 10))
    tfm = Maybe(MapElem(x -> x + 1)) |> Maybe(MapElem(x -> x + 1))
    testapply(tfm, ArrayItem)
    testapply!(tfm, ArrayItem)
end


@testset ExtendedTestSet "Sequence" begin
    tfm = Maybe(AdjustBrightness(0.1), .75) |> Maybe(AdjustContrast(0.1), .75)
    item = DataAugmentation.Image(rand(RGB{Float32}, 10, 10))
    testapply(tfm, Image)
    testapply!(tfm, Image)
end
