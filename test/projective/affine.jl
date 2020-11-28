include("../imports.jl")

@testset ExtendedTestSet "Affine" begin
    P = CoordinateTransformations.Translation(10, 10)
    tfm = Project(P)

    keypoints = Keypoints([SVector(1., 1)], (50, 50))
    image = Image(rand(RGB, 50, 50))

    indices = boundsranges(getbounds(keypoints))

    @testset ExtendedTestSet "Affine Keypoints" begin
        @test getprojection(tfm, getbounds(keypoints)) == P
        @test_nowarn project(P, keypoints, indices)
        tkeypoints = apply(tfm, keypoints)
        @test itemdata(tkeypoints) == [SVector(11, 11)]
        @test getbounds(tkeypoints)[1] == SVector(10, 10)
    end


    @testset ExtendedTestSet "Affine Image" begin
        @test_nowarn project(P, image, indices)
        timage = apply(tfm, image)
        @test itemdata(timage)[11, 11] ≈ itemdata(image)[1, 1]
        @test getbounds(timage)[1] == SVector(10, 10)
    end



    tfmcropped = tfm |> CenterCrop((50, 50))

    @testset ExtendedTestSet "Cropped, Image" begin
        tfmcropped = tfm |> CenterCrop((50, 50))
        @test_nowarn apply(tfmcropped, keypoints)
        tkeypoints = apply(tfmcropped, keypoints)
        @test getbounds(tkeypoints)[1] == SVector(10, 10)
        @test itemdata(tkeypoints)[1] == SVector(11, 11)
    end

    @testset ExtendedTestSet "Cropped with PinOrigin, Image" begin
        tfmcropped = tfm |> PinOrigin() |> CenterCrop((50, 50))
        @test_nowarn apply(tfmcropped, keypoints)
        tkeypoints = apply(tfmcropped, keypoints)
        @test getbounds(tkeypoints)[1] == SVector(0, 0)
    end


    @testset ExtendedTestSet "Composed" begin
        tfms = Project(Translation(10, 10)) |> Project(Translation(-10, -10))
        randstate = getrandstate(tfms)
        @test getprojection(tfms, getbounds(image); randstate) == Translation(0, 0)
        @test itemdata(apply(tfms, image)) == itemdata(image)
        @test itemdata(apply(tfms, keypoints)) == itemdata(keypoints)
    end


    @testset ExtendedTestSet "ScaleKeepAspect" begin
        image = Image(zeros(RGB, 100, 50))
        keypoints = Keypoints([SVector(0., 0)], (50, 50))

        imbounds = getbounds(image)
        kpbounds = getbounds(keypoints)
        tfm1 = ScaleKeepAspect((50, 50))
        @test getprojection(tfm1, imbounds) ≈ getprojection(ScaleRatio((1, 1)), imbounds)
        @test getprojection(tfm1, kpbounds) ≈ getprojection(ScaleRatio((1, 1)), kpbounds,)

        tfm2 = ScaleKeepAspect((25, 25))
        @test getprojection(tfm2, imbounds) ≈ getprojection(ScaleRatio((1 / 2, 1 / 2)), imbounds)
        @test getprojection(tfm2, kpbounds) ≈ getprojection(ScaleRatio((1 / 2, 1 / 2)), kpbounds)
    end


    @testset ExtendedTestSet "Scale" begin
        keypoints = Keypoints([SVector(0., 0)], (50, 50))
        image = Image(rand(RGB, 50, 50))

        @testset ExtendedTestSet "ScaleFixed" begin
            tfm = ScaleFixed((25, 25))
            @test_nowarn apply(tfm, image)
            @test_nowarn apply(tfm, keypoints)
            timage = apply(tfm, image)
            tkeypoints = apply(tfm, keypoints)
            @test boundsranges(getbounds(timage)) == (0:25, 0:25)
            @test getbounds(timage) == getbounds(tkeypoints)

        end
        @testset ExtendedTestSet "ScaleRatio" begin
            tfm = ScaleRatio((1/2, 1/2))
            @test_nowarn apply(tfm, image)
            @test_nowarn apply(tfm, keypoints)
            timage = apply(tfm, image)
            tkeypoints = apply(tfm, keypoints)
            @test boundsranges(getbounds(timage)) == (0:25, 0:25)
            @test getbounds(timage) == getbounds(tkeypoints)
        end
    end

    @testset ExtendedTestSet "Rotate" begin
        tfm = Rotate(10)
        image = Image(rand(RGB, 50, 50))
        @test_nowarn apply(tfm, image)

    end

    @testset ExtendedTestSet "Reflect" begin
        tfm = Reflect(10)
        image = Image(rand(RGB, 50, 50))
        @test_nowarn apply(tfm, image)

    end

    @testset ExtendedTestSet "CenterResizeCrop" begin
        tfm = CenterResizeCrop((25, 40))
        keypoints = Keypoints([SVector(0., 0)], (50, 50))
        image = Image(rand(RGB, 50, 50))
        timage, tkeypoints = apply(tfm, (image, keypoints))

        @test getbounds(timage) ≈ getbounds(tkeypoints)
        @test boundsranges(getbounds(timage)) == (1:25, 1:40)
    end

    @testset ExtendedTestSet "CroppedAffine inplace" begin
        image1 = Image(rand(RGB, 100, 100))
        image2 = Image(rand(RGB, 100, 100))
        tfm = ScaleFixed((50, 50))
        buffer = makebuffer(tfm, image1)

        @test_nowarn apply!(buffer, tfm, image2)
    end
end


@testset ExtendedTestSet "Big pipeline" begin
    @testset ExtendedTestSet "2D" begin
        sz = (100, 100)
        items = (
            Image(rand(RGB, sz)),
            Keypoints(rand(SVector{2, Float32}, 50), sz),
            MaskBinary(rand(Bool, sz)),
            MaskMulti(rand(UInt8, sz)),
        )

        tfms = compose(
            Rotate(10),
            FlipX(), FlipY(),
            ScaleRatio((.8, .8)),
            RandomResizeCrop((50, 50)),
        )
        @test_nowarn apply(tfms, items)
    end

    @testset ExtendedTestSet "2D" begin
        sz = (50, 50, 50)
        items = (
            Image(rand(RGB, sz)),
            Keypoints(rand(SVector{3, Float32}, 50), sz),
            MaskBinary(rand(Bool, sz)),
            MaskMulti(rand(UInt8, sz)),
        )

        tfms = compose(
            ScaleRatio((.8, .8, .8)),
            RandomResizeCrop((25, 25, 25)),
        )
        @test_nowarn apply(tfms, items)
    end
end
