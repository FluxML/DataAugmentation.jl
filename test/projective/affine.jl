include("../imports.jl")

@testset ExtendedTestSet "Affine" begin
    P = CoordinateTransformations.Translation(10, 10)
    tfm = Project(P)

    keypoints = Keypoints([SVector(1., 1)], (50, 50))
    image = Image(rand(RGB, 50, 50))

    bounds = getbounds(keypoints)

    @testset ExtendedTestSet "Affine Keypoints" begin
        @test getprojection(tfm, getbounds(keypoints)) == P
        @test_nowarn project(P, keypoints, bounds)
        tkeypoints = apply(tfm, keypoints)
        @test itemdata(tkeypoints) == [SVector(11, 11)]
        testprojective(tfm)
    end


    @testset ExtendedTestSet "Affine Image" begin
        @test_nowarn project(P, image, bounds)
        timage = apply(tfm, image)
        @test itemdata(timage)[11, 11] ≈ itemdata(image)[1, 1]
    end



    tfmcropped = tfm |> CenterCrop((50, 50))

    @testset ExtendedTestSet "Cropped, Image" begin
        tfmcropped = tfm |> CenterCrop((50, 50))
        @test_nowarn apply(tfmcropped, keypoints)
        tkeypoints = apply(tfmcropped, keypoints)
        @test getbounds(tkeypoints).rs[1] |> first == 11
        @test itemdata(tkeypoints)[1] == SVector(11, 11)
    end

    @testset ExtendedTestSet "Cropped with PinOrigin, Image" begin
        tfmcropped = tfm |> PinOrigin() |> CenterCrop((50, 50))
        @test_nowarn apply(tfmcropped, keypoints)
        tkeypoints = apply(tfmcropped, keypoints)
        @test getbounds(tkeypoints).rs[1] |> first == 1
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
        tfm2 = ScaleKeepAspect((25, 25))
        testprojective(tfm1)
        testprojective(tfm2)
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
            @test length.(getbounds(timage).rs) == (25, 25)
            @test getbounds(timage) == getbounds(tkeypoints)
            testprojective(tfm)

        end
        @testset ExtendedTestSet "ScaleRatio" begin
            tfm = ScaleRatio((1/2, 1/2))
            @test_nowarn apply(tfm, image)
            @test_nowarn apply(tfm, keypoints)
            timage = apply(tfm, image)
            tkeypoints = apply(tfm, keypoints)
            @test getbounds(timage).rs == (0:25, 0:25)
            @test getbounds(timage) == getbounds(tkeypoints)
            testprojective(tfm)
        end

        @testset ExtendedTestSet "ScaleKeepAspect" begin
            tfm = ScaleKeepAspect((32, 32))

            img = rand(RGB{N0f8}, 64, 96)
            @test apply(tfm, Image(img)) |> itemdata |> size == (32, 48)

            img = rand(RGB{N0f8}, 196, 196)
            @test apply(tfm, Image(img)) |> itemdata |> size == (32, 32)
        end
    end

    @testset ExtendedTestSet "Rotate" begin
        tfm = Rotate(10)
        image = Image(rand(RGB, 50, 50))
        @test_nowarn apply(tfm, image)
        P = DataAugmentation.getprojection(tfm, getbounds(image))
        @test P isa AffineMap
        @test P.linear.mat[1] isa Float32
        timage = apply(tfm, image, randstate=180)
        @test itemdata(image) ≈ itemdata(timage)[end:-1:1, end:-1:1]
    end

    @testset ExtendedTestSet "Rotate3D" begin
        image = Image(rand(RGB, 10, 20, 30))
        tfm1 = Rotate(180, 180, 180)
        tfm2 = Rotate{3}(180)
        @test_nowarn apply(tfm1, image)
        @test_nowarn apply(tfm2, image)

        # Test equivalent rotations result in the same image. Both rotations
        # should invert the x and y axis.
        timage1 = apply(tfm1, image, randstate=[180, 180, 0])
        timage2 = apply(tfm2, image, randstate=[0, 0, 180])
        @test image.bounds == timage1.bounds
        @test image.bounds == timage2.bounds
        @test size(itemdata(image)) == size(itemdata(timage1))
        @test size(itemdata(image)) == size(itemdata(timage2))
        @test itemdata(image) ≈ itemdata(timage1)[end:-1:1, end:-1:1, :]
        @test itemdata(image) ≈ itemdata(timage2)[end:-1:1, end:-1:1, :]
    end

    @testset ExtendedTestSet "RotateX" begin
        tfm = RotateX(180)
        image = Image(rand(Float32, 10, 20, 30))
        @test_nowarn apply(tfm, image)
        transformed = apply(tfm, image, randstate=180)
        @test image.bounds == transformed.bounds
        @test size(itemdata(image)) == size(itemdata(transformed))
        @test itemdata(image) ≈ itemdata(transformed)[:, end:-1:1, end:-1:1]
    end

    @testset ExtendedTestSet "RotateY" begin
        tfm = RotateY(180)
        image = Image(rand(Float32, 10, 20, 30))
        @test_nowarn apply(tfm, image)
        transformed = apply(tfm, image, randstate=180)
        @test image.bounds == transformed.bounds
        @test size(itemdata(image)) == size(itemdata(transformed))
        @test itemdata(image) ≈ itemdata(transformed)[end:-1:1, :, end:-1:1]
    end

    @testset ExtendedTestSet "RotateZ" begin
        tfm = RotateZ(180)
        image = Image(rand(Float32, 10, 20, 30))
        @test_nowarn apply(tfm, image)
        transformed = apply(tfm, image, randstate=180)
        @test image.bounds == transformed.bounds
        @test size(itemdata(image)) == size(itemdata(transformed))
        @test itemdata(image) ≈ itemdata(transformed)[end:-1:1, end:-1:1, :]
    end

    @testset ExtendedTestSet "Zoom" begin
        tfm = Zoom((0.1, 2.))
        image = Image(rand(RGB, 50, 50))
        @test_nowarn apply(tfm, image)
    end

    @testset ExtendedTestSet "Reflect" begin
        tfm = Reflect(10)
        testprojective(tfm)
    end

    @testset ExtendedTestSet "CenterResizeCrop" begin
        tfm = CenterResizeCrop((25, 40))
        keypoints = Keypoints([SVector(0., 0)], (50, 50))
        image = Image(rand(RGB, 50, 50))
        timage, tkeypoints = apply(tfm, (image, keypoints))
        @test getbounds(timage).rs == (1:25, 1:40)
        testprojective(Project(IdentityTransformation()) |> CenterCrop((12, 12)))
    end

    @testset ExtendedTestSet "CroppedAffine inplace" begin
        image1 = Image(rand(RGB, 100, 100))
        image2 = Image(rand(RGB, 100, 100))
        tfm = ScaleFixed((50, 50))
        buffer = makebuffer(tfm, image1)

        @test_nowarn apply!(buffer, tfm, image2)
    end


    @testset ExtendedTestSet "FlipX 2D correct indices" begin
        tfm = FlipX{2}() |> RandomCrop((10,10)) |> PinOrigin()
        img = rand(RGB, 10, 10)
        item = Image(img)
        @test_nowarn titem = apply(tfm, item)
        titem = apply(tfm, item)
        @test itemdata(titem) == img[:, end:-1:1]
    end

    @testset ExtendedTestSet "FlipY 2D correct indices" begin
        tfm = FlipY{2}() |> RandomCrop((10,10)) |> PinOrigin()
        img = rand(RGB, 10, 10)
        item = Image(img)
        @test_nowarn titem = apply(tfm, item)
        titem = apply(tfm, item)
        @test itemdata(titem) == img[end:-1:1, :]
    end


    @testset ExtendedTestSet "FlipX 3D correct indices" begin
        tfm = FlipX{3}() |> RandomCrop((10,10,10)) |> PinOrigin()
        img = rand(RGB, 10, 10, 10)
        item = Image(img)
        @test_nowarn titem = apply(tfm, item)
        titem = apply(tfm, item)
        @test itemdata(titem) == img[end:-1:1, :, :]
    end

    @testset ExtendedTestSet "FlipY 3D correct indices" begin
        tfm = FlipY{3}() |> RandomCrop((10,10,10)) |> PinOrigin()
        img = rand(RGB, 10, 10, 10)
        item = Image(img)
        @test_nowarn titem = apply(tfm, item)
        titem = apply(tfm, item)
        @test itemdata(titem) == img[:, end:-1:1, :]
    end

    @testset ExtendedTestSet "FlipZ 3D correct indices" begin
        tfm = FlipZ{3}() |> RandomCrop((10,10,10)) |> PinOrigin()
        img = rand(RGB, 10, 10, 10)
        item = Image(img)
        @test_nowarn titem = apply(tfm, item)
        titem = apply(tfm, item)
        @test itemdata(titem) == img[:, :, end:-1:1]
    end

    @testset ExtendedTestSet "Double flip is identity" begin
        tfm = FlipZ{3}() |> FlipZ{3}() |> RandomCrop((10,10,10)) |> PinOrigin()
        img = rand(RGB, 10, 10, 10)
        item = Image(img)
        @test_nowarn titem = apply(tfm, item)
        titem = apply(tfm, item)
        @test itemdata(titem) == img
    end
end


@testset ExtendedTestSet "Big pipeline" begin
    @testset ExtendedTestSet "2D" begin
        tfms = compose(
            Rotate(10),
            FlipX{2}(),
            FlipY{2}(),
            ScaleRatio((.8, .8)),
            WarpAffine(0.1),
            Zoom((1., 1.2)),
            RandomCrop((10, 10)),
        )
        testprojective(tfms)
    end

    @testset ExtendedTestSet "3D" begin
        sz = (50, 50, 50)
        items = (
            Image(rand(RGB, sz)),
            Keypoints(rand(SVector{3, Float32}, 50), sz),
            MaskBinary(rand(Bool, sz)),
            MaskMulti(rand(1:8, sz)),
        )

        tfms = compose(
            FlipX{3}(),
            FlipY{3}(),
            FlipZ{3}(),
            ScaleFixed((30, 40, 50)),
            Rotate(10, 20, 30),
            ScaleRatio((.8, .8, .8)),
            ScaleKeepAspect((12, 10, 10)),
            Zoom((1., 1.2)),
            RandomCrop((10, 10, 10))
        )
        testprojective(tfms, items)

        buf = makebuffer(tfms, items)
        @test_nowarn apply!(buf, tfms, items)
    end
end
