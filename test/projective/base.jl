include("../imports.jl")


@testset "`Bounds`" begin
    @test Bounds((2, 3)) == Bounds((1:2, 1:3))
    @test Bounds(axes(rand(2, 3))) == Bounds((1:2, 1:3))
    @test Bounds(()) isa Bounds{0}
end


@testset "`Project`" begin
    tfm = Project(Translation(2, 2))
    bounds = Bounds((50, 50))
    @test projectionbounds(tfm, getprojection(tfm, bounds), bounds) == Bounds((3:52, 3:52))

    image = Image(rand(50, 50))
    @test_nowarn apply(tfm, image)
    timage = apply(tfm, image)
    @test timage.data[1] == image.data[1]
end
