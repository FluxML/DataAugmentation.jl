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
    @test parent(itemdata(apply(Pad(1; fill = 1), mask)))[1, 1] == 1
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
end
