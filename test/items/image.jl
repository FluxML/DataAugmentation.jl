include("../imports.jl")


@testset ExtendedTestSet "projective" begin
    image = Image(rand(RGB, 32, 32))
    tfm = Project(Translation(20, 10))

    @test_nowarn apply(tfm, image)
    @show boundsranges(apply(tfm, image).bounds) == (21:52, 11:42)
end
