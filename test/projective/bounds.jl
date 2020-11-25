include("../imports.jl")


@testset ExtendedTestSet "Bounds" begin
    @testset ExtendedTestSet "boundsranges" begin
        keypoints = Keypoints(
            [SVector(1, 1)],
            (50, 50)
        )
        image = Image(rand(RGB, 50, 50))
        @test boundsranges(getbounds(keypoints)) == boundsranges(getbounds(image))
    end
end
