include("imports.jl")


@testset "showitems" begin
    image = Image(rand(RGB, 100, 100))
    image2 = Image(rand(RGB, 200, 500))
    kps = Keypoints([SVector(1., 2.), SVector(150., 200.)], getbounds(image))
    poly = Polygon([SVector(1., 2.), SVector(150., 200.)], getbounds(image))
    bb = BoundingBox([SVector(1., 2.), SVector(150., 200.)], getbounds(image))

    items = [image, image2, kps, poly, bb]

    @test_nowarn showitems(items)
    @test_nowarn showitems(items, showbounds = true)

end

@testset "mask overlays" begin
    image = Image(fill(RGB{N0f8}(1, 0, 0), 1, 2))
    renderedimage = showitems(image)
    multiclass = MaskMulti([1 2], [2, 1])
    binary = MaskBinary(Bool[false true])

    for (mask, background) in ((multiclass, 1), (binary, false))
        renderedmask = showitems(mask)
        overlay = showitems((image, mask))
        expected = map(renderedmask) do pixel
            RGBA{N0f8}((1 + Float64(red(pixel))) / 2, Float64(green(pixel)) / 2,
                       Float64(blue(pixel)) / 2, 1)
        end
        @test overlay == expected
        @test showitems((image, mask); maskalpha = 0) == renderedimage
        @test showitems((image, mask); maskalpha = 1) == renderedmask
        @test showitems((mask, image)) == renderedimage

        withbackground = showitems((image, mask); maskbackground = background)
        @test withbackground[1, 1] == renderedimage[1, 1]
        @test withbackground[1, 2] == overlay[1, 2]
    end

    overlay = showitems((image, multiclass); maskalpha = 0.25, maskbackground = 1)
    grid = showgrid([(image, multiclass), (image, multiclass)];
                    ncol = 2, maskalpha = 0.25, maskbackground = 1)
    @test collect(grid) == hcat(parent(overlay), parent(overlay))
    @test_throws ArgumentError showitems((image, multiclass); maskalpha = 1.5)

    grayimage = Image(fill(0.25, 1, 2))
    @test parent(showitems((grayimage, binary); C = Float64)) == [0.125 0.625]
end

@testset "mask overlay with offset axes and transparency" begin
    pixel = RGBA{Float64}(0, 0, 1, 0.5)
    image = Image(OffsetArray(fill(pixel, 1, 2), 4:4, -2:-1), Bounds((4:4, -2:-1)))
    mask = MaskBinary(OffsetArray(trues(1, 2), 4:4, -1:0), Bounds((4:4, -1:0)))

    overlay = showitems((image, mask); C = RGBA{Float64})
    @test axes(overlay) == (4:4, -2:0)
    @test overlay[4, -2] == pixel
    @test overlay[4, -1] == RGBA{Float64}(2 / 3, 2 / 3, 1, 0.75)
    @test overlay[4, 0] == RGBA{Float64}(1, 1, 1, 0.5)
end
