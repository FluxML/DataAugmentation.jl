include("imports.jl")

@testset "RandomChannelPermutation" begin
    img = Image(fill(RGBA{Float32}(0.2, 0.4, 0.6, 0.25), 2, 3))
    tfm = RandomChannelPermutation()
    perm = (3, 1, 2)
    @test all(==(RGBA{Float32}(0.6, 0.2, 0.4, 0.25)), itemdata(apply(tfm, img; randstate = perm)))
    @test sort(collect(getrandstate(tfm))) == [1, 2, 3]

    grayimg = Image(fill(Gray{N0f8}(0.3), 2, 3))
    @test itemdata(apply(tfm, grayimg; randstate = perm)) == itemdata(grayimg)
end

@testset "RandomPhotometricDistort" begin
    img = Image(reshape([RGB{Float32}(0.1, 0.4, 0.5), RGB{Float32}(0.8, 0.2, 0.1)], 1, 2))
    mask = MaskBinary(trues(1, 2))
    tfm = RandomPhotometricDistort(p = 1)
    perm = (3, 1, 2)
    states = (1.1, 0.75, nothing, 0.05, perm)
    order = (1, 3, 4, 2, 5)
    state = (order, states)

    expected = apply(AdjustBrightness(), img; randstate = states[1])
    expected = apply(AdjustHue(), expected; randstate = states[4])
    expected = apply(AdjustContrast(), expected; randstate = states[2])
    expected = apply(RandomChannelPermutation(), expected; randstate = perm)
    @test itemdata(apply(tfm, img; randstate = state)) == itemdata(expected)

    outputs = apply(Buffered(tfm), (img, img, mask); randstate = state)
    @test itemdata(outputs[1]) == itemdata(outputs[2]) == itemdata(expected)
    @test outputs[3] === mask
    @test itemdata(apply(RandomPhotometricDistort(p = 0), img)) == itemdata(img)
    @test_throws ArgumentError RandomPhotometricDistort(p = 1.1)
end

@testset "JPEG before loading JpegTurbo" begin
    img = Image(zeros(Float32, 2, 3))
    tfm = JPEG()
    err = ArgumentError("JPEG requires JpegTurbo. Load it with `using JpegTurbo`.")
    @test_throws err apply(tfm, img)
    @test_throws err apply!(Image(similar(itemdata(img))), tfm, img)
end

using JpegTurbo

@testset "JPEG" begin
    data = [RGBA{N0f8}(isodd(i + j), isodd(j), isodd(i), j / 13) for i in 1:8, j in 1:13]
    bounds = Bounds((-3:4, 0:12))
    img = Image(OffsetArray(copy(data), bounds.rs), bounds)
    tfm = JPEG((10, 20))
    result = apply(tfm, img; randstate = 12)
    @test eltype(itemdata(result)) == eltype(data)
    @test axes(itemdata(result)) == axes(itemdata(img))
    @test getbounds(result) == bounds
    @test itemdata(img) == OffsetArray(data, bounds.rs)
    @test alpha.(itemdata(result)) == alpha.(itemdata(img))
    @test itemdata(result) != itemdata(img)
    @test getrandstate(tfm) in 10:20

    buf = makebuffer(tfm, img)
    @test itemdata(apply!(buf, tfm, img; randstate = 12)) === itemdata(buf)
    @test itemdata(buf) == itemdata(result)

    grayscale = Image(Float32[i / 8 for i in 1:8, _ in 1:13])
    grayresult = apply(JPEG(75), grayscale)
    @test eltype(itemdata(grayresult)) == Float32
    @test size(itemdata(grayresult)) == (8, 13)
    @test maximum(abs, itemdata(grayresult) - itemdata(grayscale)) < 0.03
    @test_throws ArgumentError JPEG(0)
    @test_throws ArgumentError JPEG((75, 50))
    @test_throws ArgumentError apply(tfm, Image(zeros(Float32, 3, 3, 3)))
end
