include("../imports.jl")

@testset "`offsetcropbounds`" begin

    @test offsetcropbounds((50, 50), Bounds((1:100, 1:100)), (0., 0.)) == Bounds((1:50, 1:50))
    @test offsetcropbounds((50, 50), Bounds((1:100, 1:100)), (0.5, 0.5)) == Bounds((26:75, 26:75))
    @test offsetcropbounds((50, 50, 50), Bounds((1:100, 1:100, 1:100)), (1., 1., 1.)) == Bounds((51:100, 51:100, 51:100))

end


@testset "CroppedProjectiveTransform" begin

    @testset "apply" begin
        image = Image(rand(100, 100))
        tfm = Project(Translation(20, 20))
        crop = CenterCrop((50, 50))
        @test tfm |> crop isa CroppedProjectiveTransform

        cropped = tfm |> crop
        @test_nowarn apply(cropped, image)
    end

    @testset "multiple crops $(N)D" for N in 2:3
        image = Image(rand(ntuple(_->100, N)...))
        tfms = [
            Project(Translation(ntuple(_->10, N)...)),
            CenterCrop(ntuple(_->50, N)),
            Crop(ntuple(_->30, N)),
        ]

        # Apply transformatations as composed CroppedProjectiveTransform
        composed = compose(tfms...)
        @test composed isa CroppedProjectiveTransform
        @test_nowarn apply(composed, image)
        composedoutput = apply(composed, image) |> itemdata

        # Apply transformatations one at a time
        sequenceoutput = image
        for tfm in tfms
            sequenceoutput = apply(tfm, sequenceoutput)
        end
        sequenceoutput = sequenceoutput |> itemdata

        @test composedoutput ≈ sequenceoutput
    end

    @testset "compose cropped transforms" begin
        image = Image(rand(100, 100))
        tfm = (ScaleKeepAspect((50, 50)) |> CenterCrop((50, 50))) |> (Rotate(10) |> RandomCrop((20, 20)))
        @test tfm isa Sequence
        @test size(itemdata(apply(tfm, image))) == (20, 20)
    end
end

@testset "Crop padding" begin
    @testset "defaults" begin
        @test Crop((4, 4)).padding == (0, 0)
        @test RandomCrop((4, 4)).padding == (0, 0)
        @test CenterCrop((4, 4, 4); padding = 2).padding == (2, 2, 2)
        @test CenterCrop((4, 4); padding = (1, 3)).padding == (1, 3)
        @test_throws ArgumentError Crop((4, 4); padding = (1, 2, 3))
    end

    @testset "bounds" begin
        crop = RandomCrop((8, 8); padding = 4)
        bounds = Bounds((8, 8))
        @test DataAugmentation.cropbounds(crop, bounds; randstate = (0., 0.)) == Bounds((-3:4, -3:4))
        @test DataAugmentation.cropbounds(crop, bounds; randstate = (1., 1.)) == Bounds((5:12, 5:12))

        starts = Set(
            first.(DataAugmentation.cropbounds(crop, bounds).rs) for _ in 1:2000)
        @test starts == Set((i, j) for i in -3:5, j in -3:5)
    end

    @testset "identity for centered symmetric padding $(N)D" for N in 2:3
        sz = ntuple(i -> 6 + i, N)
        item = Image(rand(sz...))
        out = apply(CenterCrop(sz; padding = 3), item)
        @test itemdata(out) == itemdata(item)
    end

    @testset "output size" begin
        item = Image(rand(8, 8))
        out = apply(RandomCrop((8, 8); padding = (2, 5)), item)
        @test size(itemdata(out)) == (8, 8)
        out = apply(RandomCrop((12, 12); padding = 2), item)
        @test size(itemdata(out)) == (12, 12)
    end

    @testset "fill follows the extrapolate option of the item" begin
        tfm = Crop((8, 8), DataAugmentation.FromOrigin(); padding = 2)
        out = apply(tfm, Image(ones(8, 8)))
        @test count(==(1), itemdata(out)) == 36
        @test count(==(0), itemdata(out)) == 28

        out = apply(tfm, Image(ones(8, 8); extrapolate = Interpolations.Flat()))
        @test all(==(1), itemdata(out))
    end

    @testset "masks" begin
        tfm = Crop((4, 4), DataAugmentation.FromOrigin(); padding = 2)
        out = apply(tfm, MaskMulti(fill(2, 4, 4), 1:2))
        @test all(==(2), itemdata(out))

        out = apply(tfm, MaskMulti(fill(2, 4, 4), 1:2; extrapolate = 1))
        @test count(==(2), itemdata(out)) == 4
    end

    @testset "multiple items use the same window" begin
        tfm = RandomCrop((8, 8); padding = 4)
        image = Image(rand(8, 8))
        mask = MaskMulti(rand(1:2, 8, 8), 1:2)
        keypoints = Keypoints([SVector(4., 4.)], (8, 8))
        out = apply(tfm, (image, mask, keypoints))
        @test getbounds(out[1]) == getbounds(out[2]) == getbounds(out[3])
        @test size(itemdata(out[1])) == size(itemdata(out[2])) == (8, 8)
    end

    @testset "composed with a projective transformation" begin
        tfm = Rotate(10) |> RandomCrop((8, 8); padding = 4)
        @test tfm isa CroppedProjectiveTransform
        @test size(itemdata(apply(tfm, Image(rand(8, 8))))) == (8, 8)
    end
end

@testset "PadDivisible" begin
    img = rand(RGB, 64, 96)
    item = Image(img)
    tfm = ResizePadDivisible((32, 32), 4)
    titem = apply(tfm, item)
    @test size(titem.data) == (32, 48)
end
