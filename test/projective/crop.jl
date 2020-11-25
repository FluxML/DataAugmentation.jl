include("../imports.jl")

@testset ExtendedTestSet "`offsetcropindices`" begin

    @test offsetcropindices((50, 50), (1:100, 1:100), (0., 0.)) == (1:50, 1:50)
    @test offsetcropindices((50, 50), (1:100, 1:100), (0.5, 0.5)) == (26:75, 26:75)
    @test offsetcropindices((50, 50, 50), (1:100, 1:100, 1:100), (1., 1., 1.)) == (51:100, 51:100, 51:100)

end


@testset ExtendedTestSet "CroppedProjectiveTransform" begin

    image = Image(rand(100, 100))
    tfm = Project(Translation(20, 20))
    crop = CenterCrop((50, 50))

    @test tfm |> crop isa CroppedProjectiveTransform

    cropped = tfm |> crop
    @test_nowarn apply(cropped, image)

end
