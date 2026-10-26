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

@testset "PadDivisible" begin
    img = rand(RGB, 64, 96)
    item = Image(img)
    tfm = ResizePadDivisible((32, 32), 4)
    titem = apply(tfm, item)
    @test size(titem.data) == (32, 48)
end
