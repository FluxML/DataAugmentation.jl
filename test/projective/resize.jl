include("../imports.jl")

@testset "Resize transforms" begin
    image = Image(reshape(Float32.(1:80), 8, 10))
    keypoints = Keypoints([SVector(1., 1.), SVector(8., 10.)], (8, 10))
    mask = MaskBinary(mod.(reshape(1:80, 8, 10), 2) .== 0)

    @test getbounds(apply(Resize(4), image)) == Bounds((1:4, 1:5))
    @test getbounds(apply(Resize(4; max_size=4), image)) == Bounds((1:3, 1:4))
    @test getbounds(apply(Resize(nothing; max_size = 8), image)) == Bounds((1:6, 1:8))
    @test getbounds(apply(Resize((3, 7)), image)) == Bounds((1:3, 1:7))
    @test_throws ArgumentError Resize((3, 7); max_size=10)
    @test getbounds(apply(RandomResize(3, 4), image)) == Bounds((1:3, 1:3))

    for tfm in (Resize(5), Resize(5; max_size=7), Resize((6, 4)),
                ScaleJitter((6, 8)), RandomShortestSize((4, 6); max_size=9),
                RandomResize(3, 8), RandomResizedCrop((5, 7)))
        state = getrandstate(tfm)
        im = apply(tfm, image; randstate=state)
        kp = apply(tfm, keypoints; randstate=state)
        ma = apply(tfm, mask; randstate=state)
        @test getbounds(im) == getbounds(kp) == getbounds(ma)
        @test itemdata(apply(tfm, image; randstate=state)) == itemdata(im)
        @test size(itemdata(im)) == size(itemdata(ma))
        testprojective(tfm)
    end

    @test_throws ArgumentError Resize(0)
    @test_throws ArgumentError ScaleJitter((10, 10); scale_range=(0, 1))
    @test_throws ArgumentError RandomShortestSize((0, 2))
    @test_throws ArgumentError RandomResize(4, 3)
    @test_throws ArgumentError RandomResizedCrop((4, 4); scale=(1.2, 1.5))

    # Width-to-height ratio controls crop shape, and impossible ranges trigger fallback.
    tall = Image(ones(Float32, 12, 4))
    tfm = RandomResizedCrop((4, 4); ratio=(3, 4))
    @test getbounds(apply(tfm, tall; randstate=123)) == Bounds((1:4, 1:4))

    # This fallback crop is 8×8, so the no-resize case must still shift its origin.
    source = Image(reshape(Float32.(1:80), 8, 10))
    cropped = apply(RandomResizedCrop((8, 8); scale=(1, 1), ratio=(1, 1)), source; randstate=1)
    @test itemdata(cropped) == itemdata(source)[:, 2:9]

    # Expanding a singleton axis samples its only pixel throughout the output axis.
    resize = Resize((5, 8))
    strip = Image(reshape(Float32.(1:4), 1, 4))
    singleton = apply(resize, strip)
    @test all(itemdata(singleton)[i, :] == itemdata(singleton)[1, :] for i in 2:5)
    @test all(>(0), itemdata(singleton))
    @test itemdata(apply!(makebuffer(resize, strip), resize, strip)) == itemdata(singleton)

    strip_mask = MaskBinary(reshape([true, true, false, false], 1, 4))
    @test parent(itemdata(apply(resize, strip_mask))) == repeat([true true true true false false false false], 5)
    offset_strip = Image(OffsetArray(reshape(Float32.(1:4), 1, 4), -3:-3, 10:13);
                         extrapolate = Interpolations.Reflect())
    @test itemdata(apply(resize, offset_strip)) ≈ itemdata(singleton)

    extreme = apply(RandomResizedCrop((5, 8); scale = (1, 1), ratio = (1e12, 1e12)), Image(ones(4, 4)))
    @test all(==(1), itemdata(extreme))

    variable = Buffered(RandomResize(3, 9))
    @test size(itemdata(apply(variable, image; randstate = 3))) == (3, 3)
    @test size(itemdata(apply(variable, image; randstate = 7))) == (7, 8)
end
