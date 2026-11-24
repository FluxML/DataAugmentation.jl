include("../imports.jl")

@testset "Pad" begin
    a = reshape(collect(1:6), 2, 3)
    padded = apply(Pad((1, 2, 3, 4); fill=0), Image(a))
    @test size(itemdata(padded)) == (8, 7)
    @test parent(itemdata(padded))[3:4, 2:4] == a
    @test parent(itemdata(padded))[1, 1] == 0
    @test parent(itemdata(apply(Pad(1; padding_mode = :edge), Image(a))))[1, 1] == a[1, 1]
    @test parent(itemdata(apply(Pad(1; padding_mode = :reflect), Image(a))))[1, 1] == a[2, 2]
    @test parent(itemdata(apply(Pad(2; padding_mode = :symmetric), Image([1, 2, 3, 4])))) ==
        [2, 1, 1, 2, 3, 4, 4, 3]

    mask = MaskMulti(fill(2, 2, 3), 1:2)
    @test all(==(2), itemdata(apply(Pad(1), mask)))
    @test all(==(2), itemdata(apply(Pad(1; fill = 1), mask)))
    @test parent(itemdata(apply(Pad(1), MaskMulti(fill(2, 2, 3), 1:2; extrapolate = 1))))[1, 1] == 1
    binary = MaskBinary(trues(2, 3))
    @test parent(itemdata(apply(Pad(1; fill=true), binary)))[1, 1]
    keypoints = Keypoints([SVector(1., 1.)], (2, 3))
    @test getbounds(apply(Pad((1, 2, 3, 4),), keypoints)).rs == (-1:6, 0:6)

    image = Image(ones(4, 4))
    mask = MaskBinary(trues(4, 4))
    points = Keypoints([SVector(2., 2.)], (4, 4))
    aligned = apply(Pad((1, 2, 3, 4)), (image, mask, points))
    @test getbounds(aligned[1]) == getbounds(aligned[2]) == getbounds(aligned[3])
    @test itemdata(aligned[3])[1] == SVector(2., 2.)

    zoom = RandomZoomOut(p = 1, side_range = (2, 2), fill = 0)
    zoomed = apply(zoom, (image, mask, points); randstate = (true, 2., (0., 1.)))
    @test size(itemdata(zoomed[1])) == size(itemdata(zoomed[2])) == (8, 8)
    @test getbounds(zoomed[1]) == getbounds(zoomed[2]) == getbounds(zoomed[3])
    @test itemdata(zoomed[3])[1] == SVector(2., 2.)
    @test size(itemdata(apply(RandomZoomOut(p = 1, side_range = (1.4, 1.4)), image))) == (5, 5)
    @test apply(RandomZoomOut(p = 0), image) === image

    @test size(itemdata(apply(RandomCrop((4, 5); padding = 1, pad_if_needed = true), Image(a)))) == (4, 5)
    undersized = ScaleFixed((2, 3)) |> RandomCrop((3, 3); pad_if_needed = true, fill = 7)
    top = apply(undersized, Image(a); randstate = (nothing, (0., 0.)))
    bottom = apply(undersized, Image(a); randstate = (nothing, (0.99, 0.)))
    @test all(==(7), itemdata(top)[0, :])
    @test all(==(7), itemdata(bottom)[3, :])

    @testset "fill applies to images only" begin
        image = Image(ones(Float32, 6, 6))
        multi = MaskMulti(rand(1:3, 6, 6), 1:3)
        binary = MaskBinary(trues(6, 6))
        cases = (
            (Pad(2; fill = 0), nothing, 0),
            (Pad(2; fill = 0.5), nothing, 0.5),
            (Rotate(15; fill = 0), 15, 0),
            (Rotate(15; fill = 0.5), 15, 0.5),
            (RandomZoomOut(p = 1, side_range = (2, 2), fill = 0), (true, 2., (0.5, 0.5)), 0),
            (RandomCrop((8, 8); pad_if_needed = true, fill = 0), (0., 0.), 0),
        )
        for (tfm, randstate, value) in cases
            img, labels, flags = apply(tfm, (image, multi, binary); randstate)
            @test any(==(value), itemdata(img))
            @test minimum(itemdata(labels)) >= 1
            @test all(itemdata(flags))
        end
    end

    @testset "periodic padding matches projection" begin
        for interpolate in (Interpolations.BSpline(Interpolations.Constant()), Interpolations.BSpline(Interpolations.Linear()))
            image = Image(reshape(Float32.(1:4), 1, 4); interpolate, extrapolate = Interpolations.Periodic())
            padded = apply(Pad((3, 0, 3, 0)), image)
            projected = project(IdentityTransformation(), image, getbounds(padded))
            outside = [j for j in axes(itemdata(padded), 2) if !(1 <= j <= 4)]
            @test itemdata(padded)[1, outside] == itemdata(projected)[1, outside]
            @test parent(itemdata(padded))[1, 4:7] == 1f0:4f0
        end
        row = Image(reshape(Float32.(1:4), 1, 4); extrapolate = Interpolations.Periodic())
        @test all(parent(itemdata(apply(Pad((0, 2, 0, 2)), row)))[:, 1] .== 1)
    end
end
