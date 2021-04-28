include("../imports.jl")

@testset ExtendedTestSet "`offsetcropbounds`" begin

    @test offsetcropbounds((50, 50), Bounds((1:100, 1:100)), (0., 0.)) == Bounds((1:50, 1:50))
    @test offsetcropbounds((50, 50), Bounds((1:100, 1:100)), (0.5, 0.5)) == Bounds((26:75, 26:75))
    @test offsetcropbounds((50, 50, 50), Bounds((1:100, 1:100, 1:100)), (1., 1., 1.)) == Bounds((51:100, 51:100, 51:100))

end


@testset ExtendedTestSet "CroppedProjectiveTransform" begin

    image = Image(rand(100, 100))
    tfm = Project(Translation(20, 20))
    crop = CenterCrop((50, 50))
    @test tfm |> crop isa CroppedProjectiveTransform

    cropped = tfm |> crop
    @test_nowarn apply(cropped, image)
end

@testset ExtendedTestSet "PadDivisible" begin
    img = rand(RGB, 64, 96)
    item = Image(img)
    tfm = ResizePadDivisible((32, 32), 4)
    titem = apply(tfm, item)
    @show size(titem.data)
    @test size(titem.data) == (32, 48)
end
