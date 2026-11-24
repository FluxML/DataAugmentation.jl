include("imports.jl")


@testset "`AdjustContrast`" begin
    item = Image(rand(RGB{N0f8}, 64, 64))
    tfm = AdjustContrast(0.2)
    @test_nowarn apply(tfm, item)
    titem = apply(tfm, item)
    @test eltype(itemdata(titem)) == RGB{N0f8}
    testapply(tfm, Image)
    testapply!(tfm, Image)
end

@testset "`AdjustBrightness`" begin
    item = Image(rand(RGB{N0f8}, 64, 64))
    tfm = AdjustBrightness(0.2)
    @test_nowarn apply(tfm, item)
    titem = apply(tfm, item)
    @test eltype(itemdata(titem)) == RGB{N0f8}

    testapply(tfm, Image)
    testapply!(tfm, Image)
end


@testset "Color transforms leave non-image items unchanged" begin
    img = Image(rand(RGB{N0f8}, 16, 16))
    mask = MaskMulti(rand(1:3, 16, 16))
    maskbin = MaskBinary(rand(Bool, 16, 16))
    keypoints = Keypoints([SVector(1., 1)], (16, 16))
    bbox = BoundingBox([SVector(1., 1), SVector(8., 8)], (16, 16))
    polygon = Polygon([SVector(1., 1), SVector(8., 1), SVector(8., 8)], (16, 16))
    others = (mask, maskbin, keypoints, bbox, polygon)

    for tfm in (AdjustBrightness(0.4), AdjustContrast(0.4))
        for item in others
            @test apply(tfm, item) === item
        end

        titems = apply(tfm, (img, others...))
        @test itemdata(titems[1]) != itemdata(img)
        @test all(titems[i + 1] === others[i] for i in eachindex(others))

        pipeline = tfm |> AdjustBrightness(0.1)
        titems = apply(pipeline, (img, mask))
        @test titems[2] === mask

        buffered = Buffered(pipeline)
        titems = apply(buffered, (img, mask))
        @test titems[2] === mask
    end
end
