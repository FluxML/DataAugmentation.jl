include("../imports.jl")


@testset ExtendedTestSet "projective" begin
    keypoints = Keypoints([SVector(2., 2)], (32, 32))
    tfm = Project(Translation(20, 10))

    @test_nowarn apply(tfm, keypoints)
    @test apply(tfm, keypoints).bounds.rs == (21:52, 11:42)

    @test_nowarn apply(tfm, Polygon(keypoints))
end
