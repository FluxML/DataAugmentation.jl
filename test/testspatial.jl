include("./imports.jl")

@testset ExtendedTestSet "Crop" begin
    item = Image(rand(RGB, 50, 100))
    bounds = getbounds(item)

    @testset ExtendedTestSet "cropindices" begin
        c = CropFixed((50, 50), FromRandom())
        r = (rand(), rand())
        @test cropindices((50, 50), FromOrigin(), bounds, r) == (1:50, 1:50)
        @test cropindices((50, 50), FromCenter(), bounds, r) == (1:50, 26:75)
        @test cropindices((50, 50), FromRandom(), bounds, (1., 1.)) == (1:50, 51:100)
    end
end


@testset ExtendedTestSet "Affine helpers" begin
    @testset ExtendedTestSet "boundsranges" begin
        keypoints = Keypoints(
            [SVector(1, 1)],
            (50, 50)
        )
        image = Image(rand(RGB, 50, 50))
        @test boundsranges(getbounds(keypoints)) == boundsranges(getbounds(image))
    end
end


@testset ExtendedTestSet "Affine" begin
    A = CoordinateTransformations.Translation(10, 10)
    tfm = Affine(A)

    keypoints = Keypoints([SVector(1, 1)], (50, 50))
    image = Image(rand(RGB, 50, 50))


    @testset ExtendedTestSet "Affine Keypoints" begin
        @test getaffine(tfm, keypoints, nothing) == A
        @test_nowarn applyaffine(keypoints, getaffine(tfm, keypoints, nothing))
        tkeypoints = apply(tfm, keypoints)
        @test itemdata(tkeypoints) == [SVector(11, 11)]
        @test getbounds(tkeypoints)[1] == SVector(10, 10)
    end

    @testset ExtendedTestSet "Affine Image" begin
        @test getaffine(tfm, image, nothing) == A
        @test_nowarn applyaffine(image, getaffine(tfm, image, nothing))
        timage = apply(tfm, image)
        @test itemdata(timage)[11, 11] ≈ itemdata(image)[1, 1]
        @test getbounds(timage)[1] == SVector(10, 10)
    end

    tfmcropped = CroppedAffine(tfm, CropFixed(50, 50))

    @testset ExtendedTestSet "Affine Cropped Image" begin
        @test_nowarn apply(tfmcropped, keypoints)
        tkeypoints = apply(tfmcropped, keypoints)
        @test getbounds(tkeypoints)[1] == SVector(0, 0)
    end

    @testset ExtendedTestSet "Affine Cropped Image" begin
        @test_nowarn apply(tfmcropped, image)
        timage = apply(tfmcropped, image)
        @test getbounds(timage)[1] == SVector(0, 0)
    end

    @testset ExtendedTestSet "Composed" begin
        tfms = Affine(Translation(10, 10)) |> Affine(Translation(-10, -10))
        randstate = getrandstate(tfms)
        @test getaffine(tfms, getbounds(image), randstate) == Translation(0, 0)
        @test itemdata(apply(tfms, image)) == itemdata(image)
        @test itemdata(apply(tfms, keypoints)) == itemdata(keypoints)
    end


    @testset ExtendedTestSet "ScaleKeepAspect" begin
        image = Image(zeros(RGB, 100, 50))

        keypoints = Keypoints([SVector(0., 0)], (50, 50))

        imbounds = getbounds(image)
        kpbounds = getbounds(keypoints)
        tfm1 = ScaleKeepAspect(50)
        @test getaffine(tfm1, imbounds, nothing) ≈ getaffine(ScaleRatio((1, 1)), imbounds, nothing)
        @test getaffine(tfm1, kpbounds, nothing) ≈ getaffine(ScaleRatio((1, 1)), kpbounds, nothing)

        tfm2 = ScaleKeepAspect(25)
        @test getaffine(tfm2, imbounds, nothing) ≈ getaffine(ScaleRatio((1 / 2, 1 / 2)), imbounds, nothing)
        @test getaffine(tfm2, kpbounds, nothing) ≈ getaffine(ScaleRatio((1 / 2, 1 / 2)), kpbounds, nothing)
    end

    @testset ExtendedTestSet "Resize" begin
        keypoints = Keypoints([SVector(0., 0)], (50, 50))
        image = Image(rand(RGB, 50, 50))
        @testset ExtendedTestSet "ResizeFixed" begin
            tfm = ResizeFixed((25, 25))
            @test_nowarn apply(tfm, image)
            @test_nowarn apply(tfm, keypoints)
            timage = apply(tfm, image)
            tkeypoints = apply(tfm, keypoints)
            @test boundsranges(getbounds(timage)) == (1:25, 1:25)
            @test getbounds(timage) == getbounds(tkeypoints)

        end
        @testset ExtendedTestSet "ResizeRatio" begin
            tfm = ResizeRatio((1/2, 1/2))
            @test_nowarn apply(tfm, image)
            @test_nowarn apply(tfm, keypoints)
            timage = apply(tfm, image)
            tkeypoints = apply(tfm, keypoints)
            @test boundsranges(getbounds(timage)) == (1:25, 1:25)
            @test getbounds(timage) == getbounds(tkeypoints)
        end
    end

    @testset ExtendedTestSet "RandomResizeCrop" begin
        tfm = CenterResizeCrop((25, 40))
        keypoints = Keypoints([SVector(0., 0)], (50, 50))
        image = Image(rand(RGB, 50, 50))
        timage, tkeypoints = apply(tfm, (image, keypoints))
        @test getbounds(timage) == getbounds(tkeypoints)
        @test boundsranges(getbounds(timage)) == (1:25, 1:40)
    end

    @testset ExtendedTestSet "CroppedAffine inplace" begin
        image1 = Image(rand(RGB, 100, 100))
        image2 = Image(rand(RGB, 100, 100))
        tfm = Inplace(ResizeFixed((50, 50)), Image)
        buffer = makebuffer(tfm, image1)

        @test_nowarn apply!(buffer, tfm, image2)
    end
end
