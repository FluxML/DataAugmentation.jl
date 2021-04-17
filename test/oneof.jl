include("imports.jl")


@testset ExtendedTestSet "OneOf" begin
    item = ArrayItem(rand(10, 10))
    tfm = OneOf([MapElem(x -> x + 1), MapElem(x -> x - 1)])
    @test apply(tfm, item; randstate = (1, nothing)).data ≈ item.data .+ 1
    @test apply(tfm, item; randstate = (2, nothing)).data ≈ item.data .- 1
end

@testset ExtendedTestSet "Maybe" begin
    item = ArrayItem(rand(10, 10))
    tfm = Maybe(MapElem(x -> x + 1))
    @test apply(tfm, item; randstate = (1, nothing)).data ≈ item.data .+ 1
    @test apply(tfm, item; randstate = (2, nothing)).data ≈ item.data
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
