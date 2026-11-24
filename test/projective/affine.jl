include("../imports.jl")

@testset "Affine" begin
    P = CoordinateTransformations.Translation(10, 10)
    tfm = Project(P)

    keypoints = Keypoints([SVector(1., 1)], (50, 50))
    image = Image(rand(RGB, 50, 50))

    bounds = getbounds(keypoints)

    @testset "Affine Keypoints" begin
        @test getprojection(tfm, getbounds(keypoints)) == P
        @test_nowarn project(P, keypoints, bounds)
        tkeypoints = apply(tfm, keypoints)
        @test itemdata(tkeypoints) == [SVector(11, 11)]
        testprojective(tfm)
    end


    @testset "Affine Image" begin
        @test_nowarn project(P, image, bounds)
        timage = apply(tfm, image)
        @test itemdata(timage)[11, 11] ≈ itemdata(image)[1, 1]
    end



    tfmcropped = tfm |> CenterCrop((50, 50))

    @testset "Cropped, Image" begin
        tfmcropped = tfm |> CenterCrop((50, 50))
        @test_nowarn apply(tfmcropped, keypoints)
        tkeypoints = apply(tfmcropped, keypoints)
        @test getbounds(tkeypoints).rs[1] |> first == 11
        @test itemdata(tkeypoints)[1] == SVector(11, 11)
    end

    @testset "Cropped with PinOrigin, Image" begin
        tfmcropped = tfm |> PinOrigin() |> CenterCrop((50, 50))
        @test_nowarn apply(tfmcropped, keypoints)
        tkeypoints = apply(tfmcropped, keypoints)
        @test getbounds(tkeypoints).rs[1] |> first == 1
    end


    @testset "Composed" begin
        tfms = Project(Translation(10, 10)) |> Project(Translation(-10, -10))
        randstate = getrandstate(tfms)
        @test getprojection(tfms, getbounds(image); randstate) == Translation(0, 0)
        @test itemdata(apply(tfms, image)) == itemdata(image)
        @test itemdata(apply(tfms, keypoints)) == itemdata(keypoints)
    end


    @testset "ScaleKeepAspect" begin
        image = Image(zeros(RGB, 100, 50))
        keypoints = Keypoints([SVector(0., 0)], (50, 50))

        imbounds = getbounds(image)
        kpbounds = getbounds(keypoints)
        tfm1 = ScaleKeepAspect((50, 50))
        tfm2 = ScaleKeepAspect((25, 25))
        testprojective(tfm1)
        testprojective(tfm2)
    end


    @testset "Scale" begin
        keypoints = Keypoints([SVector(0., 0)], (50, 50))
        image = Image(rand(RGB, 50, 50))

        @testset "ScaleFixed" begin
            tfm = ScaleFixed((25, 25))
            @test_nowarn apply(tfm, image)
            @test_nowarn apply(tfm, keypoints)
            timage = apply(tfm, image)
            tkeypoints = apply(tfm, keypoints)
            @test !any(isnan.(timage |> itemdata))
            @test getbounds(timage).rs == (1:25, 1:25)
            @test getbounds(timage) == getbounds(tkeypoints)
            testprojective(tfm)

        end

        @testset "ScaleRatio" begin
            tfm = ScaleRatio((1/2, 1/2))
            @test_nowarn apply(tfm, image)
            @test_nowarn apply(tfm, keypoints)
            timage = apply(tfm, image)
            tkeypoints = apply(tfm, keypoints)
            @test !any(isnan.(timage |> itemdata))
            @test getbounds(timage).rs == (1:25, 1:25)
            @test getbounds(timage) == getbounds(tkeypoints)
            testprojective(tfm)
        end

        @testset "ScaleRatioTwice" begin
            tfm = ScaleRatio((4/5, 4/5)) |> ScaleRatio((1/2, 1/2))
            @test_nowarn apply(tfm, image)
            @test_nowarn apply(tfm, keypoints)
            timage = apply(tfm, image)
            tkeypoints = apply(tfm, keypoints)
            @test !any(isnan.(timage |> itemdata))
            @test getbounds(timage).rs == (1:20, 1:20)
            @test getbounds(timage) == getbounds(tkeypoints)
            testprojective(tfm)
        end

        @testset "ScaleKeepAspect" begin
            tfm = ScaleKeepAspect((32, 32))

            img = rand(RGB{N0f8}, 64, 96)
            timg = apply(tfm, Image(img)) |> itemdata
            timg = apply(tfm, Image(img))
            @test getbounds(timg).rs == (1:32, 1:48)
            @test !any(isnan.(timg |> itemdata))
            img = rand(RGB{N0f8}, 196, 196)
            timg = apply(tfm, Image(img)) |> itemdata
            timg = apply(tfm, Image(img))
            @test getbounds(timg).rs == (1:32, 1:32)
            @test !any(isnan.(timg |> itemdata))
        end

        @testset "No black borders" begin
            for n in (7, 10, 50), s in (1, 2, 5, 13, 20, 40, 101)
                n == s && continue
                tfm = ScaleFixed((s, s + 1))
                image = Image(ones(Float32, n, n + 1))
                timage = apply(tfm, image)
                @test getbounds(timage).rs == (1:s, 1:(s + 1))
                @test all(>(0.99), itemdata(timage))

                shifted = apply(Project(Translation(10, 20)), image)
                @test getbounds(shifted).rs == (11:(10 + n), 21:(20 + n + 1))
                tshifted = apply(tfm, shifted)
                @test getbounds(tshifted).rs == (1:s, 1:(s + 1))
                @test all(>(0.99), itemdata(tshifted))

                corners = Keypoints([SVector(1., 1), SVector(n + 0., n + 1.)], (n, n + 1))
                tcorners = apply(tfm, corners)
                @test getbounds(tcorners) == getbounds(timage)
                @test all(p -> all(0.5 - 1e-4 .<= p .<= (s + 0.5, s + 1.5) .+ 1e-4), itemdata(tcorners))
            end
        end

        @testset "Unchanged axes are left alone" begin
            image = Image(rand(Float32, 20, 30))
            timage = apply(ScaleFixed((20, 10)), image)
            @test getbounds(timage).rs == (1:20, 1:10)
            tkeypoints = apply(ScaleFixed((20, 10)), Keypoints([SVector(3., 4.)], (20, 30)))
            @test itemdata(tkeypoints)[1][1] == 3
        end
    end

    @testset "Rotate" begin
        tfm = Rotate(10)
        image = Image(rand(RGB, 50, 50))
        @test_nowarn apply(tfm, image)
        P = DataAugmentation.getprojection(tfm, getbounds(image))
        @test P isa AffineMap
        @test P.linear.mat[1] isa Float32
        timage = apply(tfm, image, randstate=180)
        @test itemdata(image) ≈ itemdata(timage)[end:-1:1, end:-1:1]

        image = Image(reshape(Float32.(1:25), 5, 5))
        points = Keypoints(Union{Nothing, SVector{2, Float64}}[SVector(2., 1.), nothing], (5, 5))
        rotation = Rotate(90; expand = false, center = (1, 1), fill = -1)
        turned, moved = apply(rotation, (image, points); randstate = 90)
        @test getbounds(turned) == getbounds(moved) == getbounds(image)
        @test itemdata(turned)[1, 2] == itemdata(image)[2, 1]
        @test itemdata(turned)[5, 1] == -1
        @test itemdata(moved)[1] ≈ SVector(1., 2.)
        @test isnothing(itemdata(moved)[2])
        expanded = apply(Rotate(90; center = (1, 1)), image; randstate = 90)
        @test getbounds(expanded) != getbounds(image)
        @test getbounds(apply(Maybe(Rotate(45; expand = false), 1), image; randstate = (1, 45))) == getbounds(image)
    end

    @testset "Rotate3D" begin
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

    @testset "RotateX" begin
        tfm = RotateX(180)
        image = Image(rand(Float32, 10, 20, 30))
        @test_nowarn apply(tfm, image)
        transformed = apply(tfm, image, randstate=180)
        @test image.bounds == transformed.bounds
        @test size(itemdata(image)) == size(itemdata(transformed))
        @test itemdata(image) ≈ itemdata(transformed)[:, end:-1:1, end:-1:1]
    end

    @testset "RotateY" begin
        tfm = RotateY(180)
        image = Image(rand(Float32, 10, 20, 30))
        @test_nowarn apply(tfm, image)
        transformed = apply(tfm, image, randstate=180)
        @test image.bounds == transformed.bounds
        @test size(itemdata(image)) == size(itemdata(transformed))
        @test itemdata(image) ≈ itemdata(transformed)[end:-1:1, :, end:-1:1]
    end

    @testset "RotateZ" begin
        tfm = RotateZ(180)
        image = Image(rand(Float32, 10, 20, 30))
        @test_nowarn apply(tfm, image)
        transformed = apply(tfm, image, randstate=180)
        @test image.bounds == transformed.bounds
        @test size(itemdata(image)) == size(itemdata(transformed))
        @test itemdata(image) ≈ itemdata(transformed)[end:-1:1, end:-1:1, :]
    end

    @testset "Zoom" begin
        tfm = Zoom((0.1, 2.))
        image = Image(rand(RGB, 50, 50))
        @test_nowarn apply(tfm, image)
    end

    @testset "Reflect" begin
        tfm = Reflect(10)
        testprojective(tfm)
    end

    @testset "CenterResizeCrop" begin
        tfm = CenterResizeCrop((25, 40))
        keypoints = Keypoints([SVector(0., 0)], (50, 50))
        image = Image(rand(RGB, 50, 50))
        timage, tkeypoints = apply(tfm, (image, keypoints))
        @test getbounds(timage).rs == (1:25, 1:40)
        testprojective(Project(IdentityTransformation()) |> CenterCrop((12, 12)))
    end

    @testset "CroppedAffine inplace" begin
        image1 = Image(rand(RGB, 100, 100))
        image2 = Image(rand(RGB, 100, 100))
        tfm = ScaleFixed((50, 50))
        buffer = makebuffer(tfm, image1)

        @test_nowarn apply!(buffer, tfm, image2)
    end


    @testset "FlipX 2D correct indices" begin
        img = rand(RGB, 10, 10)
        item = Image(img)
        for flip in (FlipX(), FlipX{2}(), FlipX{3}(), FlipDim{2}(2), Reflect(0))
            flipped = itemdata(apply(flip, item))
            @test axes(flipped) == axes(img)
            @test flipped == img[:, end:-1:1]
        end
    end

    @testset "FlipY 2D correct indices" begin
        img = rand(RGB, 10, 10)
        item = Image(img)
        for flip in (FlipY(), FlipY{2}())
            flipped = itemdata(apply(flip, item))
            @test axes(flipped) == axes(img)
            @test flipped == img[end:-1:1, :]
        end
    end


    @testset "FlipX 3D correct indices" begin
        img = rand(RGB, 10, 10, 10)
        item = Image(img)
        for flip in (FlipX(), FlipX{3}())
            flipped = itemdata(apply(flip, item))
            @test axes(flipped) == axes(img)
            @test flipped == img[end:-1:1, :, :]
        end
    end

    @testset "FlipY 3D correct indices" begin
        img = rand(RGB, 10, 10, 10)
        item = Image(img)
        for flip in (FlipY(), FlipY{3}())
            flipped = itemdata(apply(flip, item))
            @test axes(flipped) == axes(img)
            @test flipped == img[:, end:-1:1, :]
        end
    end

    @testset "FlipZ 3D correct indices" begin
        img = rand(RGB, 10, 10, 10)
        item = Image(img)
        for flip in (FlipZ(), FlipZ{3}())
            flipped = itemdata(apply(flip, item))
            @test axes(flipped) == axes(img)
            @test flipped == img[:, :, end:-1:1]
        end
    end

    @testset "Flip with offset bounds and keypoints" begin
        img = reshape(Float32.(1:12), 3, 4)
        bounds = Bounds((-2:0, 41:44))
        image = Image(OffsetArray(img, bounds.rs), bounds)
        keypoints = Keypoints([SVector(-1f0, 41.5f0)], bounds)

        timage, tkeypoints = apply(FlipX(), (image, keypoints))
        @test axes(itemdata(timage)) == axes(itemdata(image))
        @test itemdata(timage) == OffsetArray(img[:, end:-1:1], bounds.rs)
        @test getbounds(tkeypoints) == bounds
        @test itemdata(tkeypoints) == [SVector(-1f0, 43.5f0)]
    end

    @testset "FlipZ on 2D data" begin
        item = Image(rand(RGB, 10, 10))
        for flip in (FlipZ(), FlipZ{2}())
            @test_throws ArgumentError("cannot flip the z-axis of 2-dimensional data") apply(flip, item)
        end
    end

    @testset "Double flip is identity" begin
        tfm = FlipZ() |> FlipZ() |> RandomCrop((10, 10, 10)) |> PinOrigin()
        img = rand(RGB, 10, 10, 10)
        item = Image(img)
        @test_nowarn titem = apply(tfm, item)
        titem = apply(tfm, item)
        @test itemdata(titem) == img
    end
end


@testset "FlipDim takes the dimensionality from the item" begin
    img = rand(Float32, 4, 5)
    vol = rand(Float32, 4, 5, 6)
    @test itemdata(apply(FlipDim(2), Image(img))) == img[:, end:-1:1]
    @test itemdata(apply(FlipDim(1), Image(vol))) == vol[end:-1:1, :, :]
    @test itemdata(apply(FlipDim(3), Image(vol))) == vol[:, :, end:-1:1]
    @test itemdata(apply(FlipDim{3}(2), Image(vol))) == itemdata(apply(FlipDim(2), Image(vol)))
    @test itemdata(apply(FlipDim{2}(1), Image(img))) == img[end:-1:1, :]
    @test_throws ArgumentError("cannot flip dimension 3 of 2-dimensional data") apply(FlipDim(3), Image(img))
    @test_throws ArgumentError FlipDim(0)
    @test_throws ErrorException FlipDim{2}(3)
end

@testset "Rotate takes the dimensionality from the item" begin
    img = Image(rand(Float32, 9, 7))
    vol = Image(rand(Float32, 9, 7, 5))
    tfm = Rotate(10)

    @testset "states" begin
        states = [getrandstate(tfm) for _ in 1:50]
        @test all(state -> length(state) == 3 && all(a -> abs(a) <= 10, state), states)
        @test length(unique(first.(states))) > 1
        @test length(unique(last.(states))) > 1
    end

    @testset "2D uses the first angle" begin
        expected = itemdata(apply(Rotate{2}(10), img; randstate = 30))
        @test itemdata(apply(tfm, img; randstate = (30, 99, 99))) == expected
        @test itemdata(apply(tfm, img; randstate = 30)) == expected
        @test getbounds(apply(tfm, img; randstate = 30)) == getbounds(apply(Rotate{2}(10), img; randstate = 30))
    end

    @testset "3D is Rotate{3}" begin
        angles = (12., -7., 25.)
        timage = apply(tfm, vol; randstate = angles)
        expected = apply(Rotate{3}(10), vol; randstate = angles)
        @test itemdata(timage) == itemdata(expected)
        @test getbounds(timage) == getbounds(expected)
        @test itemdata(apply(tfm, vol; randstate = collect(angles))) == itemdata(expected)
    end

    @testset "keywords" begin
        for item in (img, vol)
            @test getbounds(apply(Rotate(10; expand = false), item; randstate = (20, 30, 40))) == getbounds(item)
            @test getbounds(apply(Rotate(10), item; randstate = (20, 30, 40))) != getbounds(item)
        end
        center = (2, 3)
        @test itemdata(apply(Rotate(10; center), img; randstate = 30)) ==
            itemdata(apply(Rotate{2}(10; center), img; randstate = 30))
        @test_throws ArgumentError("center must contain 3 finite coordinates") apply(Rotate(10; center), vol)
        @test_throws ArgumentError Rotate(10; center = (1, 2, 3, 4))
        filled = apply(Rotate(90; expand = false, fill = -1), vol; randstate = (0, 0, 90))
        @test any(==(-1), itemdata(filled))
        @test itemdata(apply(Rotate(90; expand = false, fill = -1), img; randstate = 90)) ==
            itemdata(apply(Rotate{2}(90; expand = false, fill = -1), img; randstate = 90))
    end

    @testset "other forms" begin
        @test Rotate(0) isa DataAugmentation.Identity
        @test_throws ArgumentError apply(tfm, Image(rand(Float32, 4, 4, 4, 4)))
        @test itemdata(apply(Rotate(Uniform(-10, 10)), img; randstate = 30)) ==
            itemdata(apply(tfm, img; randstate = 30))
        @test itemdata(apply(Rotate(5, 6, 7), vol; randstate = (1, 2, 3))) ==
            itemdata(apply(RotateX(5) |> RotateY(6) |> RotateZ(7), vol; randstate = (1, 2, 3)))
    end

    @testset "composition" begin
        for item in (img, vol)
            composed = Rotate(10) |> FlipX() |> CenterCrop(4)
            @test composed isa CroppedProjectiveTransform
            @test size(itemdata(apply(composed, item))) == ntuple(_ -> 4, ndims(itemdata(item)))
        end
    end
end

@testset "A number for sizes and ratios" begin
    img = Image(rand(Float32, 12, 20))
    vol = Image(rand(Float32, 12, 20, 16))

    @testset "scaling $(name)" for (name, number, tuple) in (
            ("ScaleFixed", ScaleFixed(8), n -> ScaleFixed(ntuple(_ -> 8, n))),
            ("ScaleRatio", ScaleRatio(0.5), n -> ScaleRatio(ntuple(_ -> 0.5, n))),
            ("ScaleKeepAspect", ScaleKeepAspect(6), n -> ScaleKeepAspect(ntuple(_ -> 6, n))))
        for item in (img, vol)
            n = ndims(itemdata(item))
            titem = apply(number, item)
            @test itemdata(titem) == itemdata(apply(tuple(n), item))
            @test getbounds(titem) == getbounds(apply(tuple(n), item))
        end
    end

    @test getbounds(apply(ScaleFixed(8), vol)).rs == (1:8, 1:8, 1:8)
    @test getbounds(apply(ScaleRatio(0.5), vol)).rs == (1:6, 1:10, 1:8)
    @test getbounds(apply(ScaleKeepAspect(6), vol)).rs == (1:6, 1:10, 1:8)
    @test getbounds(apply(ScaleRatio(2), img)).rs == (1:24, 1:40)
    @test_throws ArgumentError ScaleFixed(0)
    @test_throws ArgumentError ScaleKeepAspect(-1)
    @test_throws ArgumentError ScaleRatio(0)
    @test_throws ArgumentError ScaleRatio(Inf)

    @testset "keypoints and masks follow the image" begin
        items = (
            Image(rand(Float32, 8, 8, 8)),
            Keypoints([SVector(2f0, 3f0, 4f0)], (8, 8, 8)),
            MaskBinary(rand(Bool, 8, 8, 8)),
        )
        for tfm in (ScaleFixed(4), ScaleRatio(0.5), ScaleKeepAspect(4))
            testprojective(tfm, items)
        end
    end

    @testset "resize crops" begin
        for item in (img, vol)
            n = ndims(itemdata(item))
            @test size(itemdata(apply(CenterResizeCrop(6), item))) == ntuple(_ -> 6, n)
            @test size(itemdata(apply(RandomResizeCrop(6), item))) == ntuple(_ -> 6, n)
        end
        @test size(itemdata(apply(ResizePadDivisible(6, 4), img))) == (8, 12)
        @test size(itemdata(apply(ResizePadDivisible(6, 4), vol))) == (8, 12, 8)
        @test itemdata(apply(CenterResizeCrop(6), img)) == itemdata(apply(CenterResizeCrop((6, 6)), img))
    end
end


@testset "Big pipeline" begin
    @testset "Batching optional flips" begin
        image = Image(rand(RGB{N0f8}, 32, 32))
        tensors = map((0.0, 1.0)) do p
            tfm = RandomResizeCrop((32, 32)) |> Maybe(FlipX(), p) |> ImageToTensor()
            itemdata(apply(tfm, image))
        end
        @test size(stack(tensors)) == (32, 32, 3, 2)
    end

    @testset "2D" begin
        tfms = compose(
            Rotate(10),
            FlipX(),
            FlipY(),
            ScaleRatio((.8, .8)),
            WarpAffine(0.1),
            Zoom((1., 1.2)),
            RandomCrop((10, 10)),
        )
        testprojective(tfms)
    end

    @testset "2D, dimension-free" begin
        tfms = compose(Rotate(10), FlipDim(1), ScaleRatio(.8), WarpAffine(0.1), ScaleKeepAspect(12), RandomCrop(10))
        testprojective(tfms)
    end

    @testset "3D" begin
        sz = (50, 50, 50)
        items = (
            Image(rand(RGB, sz)),
            Keypoints(rand(SVector{3, Float32}, 50), sz),
            MaskBinary(rand(Bool, sz)),
            MaskMulti(rand(1:8, sz)),
        )

        tfms = compose(
            FlipX(),
            FlipY(),
            FlipZ(),
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

        tfms = compose(Rotate(10), FlipDim(3), ScaleRatio(.8), WarpAffine(0.1), ScaleKeepAspect(12), RandomCrop(10))
        testprojective(tfms, items)

        buf = makebuffer(tfms, items)
        @test_nowarn apply!(buf, tfms, items)
    end
end
