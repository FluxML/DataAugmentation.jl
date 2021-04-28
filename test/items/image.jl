include("../imports.jl")


@testset ExtendedTestSet "projective" begin
    image = Image(rand(RGB, 32, 32))
    tfm = Project(Translation(20, 10))

    @test_nowarn apply(tfm, image)
    timage = apply(tfm, image)
    @test timage.bounds.rs == (21:52, 11:42)

    @test_nowarn apply!(timage, tfm, image)

end
