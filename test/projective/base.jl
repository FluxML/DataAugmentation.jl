include("../imports.jl")


@testset ExtendedTestSet "`Project`" begin
    tfm = Project(Translation(2, 2))
    bounds = makebounds((50, 50))
    @test cropindices(tfm, getprojection(tfm, bounds), bounds) == (3:52, 3:52)

    image = Image(rand(50, 50))
    @test_nowarn apply(tfm, image)
    timage = apply(tfm, image)
    @test timage.data[1] == image.data[1]
    @test timage.bounds[1] == SVector(2, 2)
end
