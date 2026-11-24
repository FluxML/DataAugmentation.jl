include("imports.jl")


@testset "OneOf" begin
    item = ArrayItem(rand(10, 10))
    tfm = OneOf([MapElem(x -> x + 1), MapElem(x -> x - 1)])
    @test apply(tfm, item; randstate = (1, nothing)).data ≈ item.data .+ 1
    @test apply(tfm, item; randstate = (2, nothing)).data ≈ item.data .- 1
    testapply(tfm, ArrayItem)
    testapply!(tfm, ArrayItem)
end

@testset "Maybe" begin
    item = ArrayItem(rand(10, 10))
    tfm = Maybe(MapElem(x -> x + 1))
    @test apply(tfm, item; randstate = (1, nothing)).data ≈ item.data .+ 1
    @test apply(tfm, item; randstate = (2, nothing)).data ≈ item.data
    testapply(tfm, ArrayItem)
end


@testset "OneOfProjective" begin
    item = Image(rand(10, 10))
    tfm = Rotate(10) |> CenterCrop((5, 5))
    oneof = Maybe(tfm)
    @test oneof isa DataAugmentation.OneOfProjective
    @test_nowarn DataAugmentation.getprojection(oneof, getbounds(item); randstate = getrandstate(oneof))
    @test apply(oneof, item; randstate = (1, getrandstate(tfm))) isa Image
    @test apply(oneof, item; randstate = (2, nothing)) isa Image

    flipx, flipy = FlipX(), FlipY()
    horizontal = itemdata(apply(flipx, item))
    vertical = itemdata(apply(flipy, item))

    maybe = Maybe(flipx)
    @test maybe isa DataAugmentation.OneOfProjective
    @test itemdata(apply(maybe, item; randstate = (1, nothing))) == horizontal
    @test itemdata(apply(maybe, item; randstate = (2, nothing))) == itemdata(item)

    oneof = OneOf([flipx, flipy], Categorical([0.2, 0.8]))
    @test oneof isa DataAugmentation.OneOfProjective
    @test itemdata(apply(oneof, item; randstate = (1, nothing))) == horizontal
    @test itemdata(apply(oneof, item; randstate = (2, nothing))) == vertical
end


@testset "Sequence(oneofs...)" begin
    item = ArrayItem(rand(10, 10))
    tfm = Maybe(MapElem(x -> x + 1)) |> Maybe(MapElem(x -> x + 1))
    testapply(tfm, ArrayItem)
    testapply!(tfm, ArrayItem)
end


@testset "Sequence" begin
    tfm = Maybe(AdjustBrightness(0.1), .75) |> Maybe(AdjustContrast(0.1), .75)
    item = DataAugmentation.Image(rand(RGB{Float32}, 10, 10))
    testapply(tfm, Image)
    testapply!(tfm, Image)
end


@testset "RandomOrder" begin
    item = ArrayItem(fill(1.0, 2, 2))
    add, double = MapElem(x -> x + 1), MapElem(x -> 2x)
    tfm = RandomOrder(add, double)
    @test tfm.transforms == (add, double)
    @test RandomOrder([add, double]).transforms == (add, double)
    @test_throws ArgumentError RandomOrder()

    state = getrandstate(tfm)
    @test sort(collect(state[1])) == [1, 2]
    @test all(==(4.0), itemdata(apply(tfm, item; randstate = ((1, 2), (nothing, nothing)))))
    @test all(==(3.0), itemdata(apply(tfm, item; randstate = ((2, 1), (nothing, nothing)))))
    @test Set(only(unique(itemdata(apply(tfm, item)))) for _ in 1:50) == Set([3.0, 4.0])

    # Every item of a tuple goes through the same order, with the same random states
    rotation = RandomOrder(FlipX(), AdjustBrightness(0.5))
    image, mask = Image(rand(Float32, 6, 8)), MaskBinary(rand(Bool, 6, 8))
    state = getrandstate(rotation)
    out = apply(rotation, (image, mask); randstate = state)
    @test itemdata(out[1]) ≈ itemdata(apply(rotation, image; randstate = state))
    @test itemdata(out[2]) == itemdata(apply(FlipX(), mask))
    @test apply(rotation, Keypoints([SVector(1.0, 2.0)], (6, 8)); randstate = state) isa Keypoints
    @test_nowarn apply(Buffered(rotation), image)
end
