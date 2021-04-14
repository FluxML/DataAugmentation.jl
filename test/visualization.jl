include("imports.jl")


@testset ExtendedTestSet "showitems" begin
    image = Image(rand(RGB, 100, 100))
    image2 = Image(rand(RGB, 200, 500))
    kps = Keypoints([SVector(1., 2.), SVector(150., 200.)], getbounds(image))
    poly = Polygon([SVector(1., 2.), SVector(150., 200.)], getbounds(image))
    bb = BoundingBox([SVector(1., 2.), SVector(150., 200.)], getbounds(image))

    items = [image, image2, kps, poly, bb]

    @test_nowarn showitems(items)
    @test_nowarn showitems(items, showbounds = true)

end
