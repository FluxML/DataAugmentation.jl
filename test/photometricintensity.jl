include("imports.jl")

@testset "Intensity transforms" begin
    data = reshape(N0f8[0, 0.25, 0.5, 1], 2, 2)
    img = Image(data)
    @test itemdata(apply(AdjustGamma(2), img)) == N0f8.(Float32.(data) .^ 2)
    @test itemdata(apply(AdjustGamma(1; gain = 2), img))[end] == N0f8(1)
    @test itemdata(apply(Invert(), img)) == N0f8.(1 .- data)
    solarized = itemdata(apply(Solarize(0.5), img))
    @test solarized[1:2] == data[1:2]
    @test solarized[3] == N0f8(1) - data[3]
    @test solarized[4] == N0f8(0)
    @test itemdata(apply(Posterize(4), Image(N0f8[0, 0.5, 1]))) == N0f8[0, 128/255, 240/255]

    rgba = Image(fill(RGBA{N0f8}(0.2, 0.4, 0.6, 0.3), 2, 2))
    inverted = itemdata(apply(Invert(), rgba))
    @test red(inverted[1]) ≈ 0.8
    @test alpha(inverted[1]) == alpha(itemdata(rgba)[1])
    @test eltype(inverted) == eltype(itemdata(rgba))

    @test itemdata(apply(Autocontrast(), Image(fill(N0f8(0.4), 2, 2)))) == fill(N0f8(0.4), 2, 2)
    @test itemdata(apply(Equalize(), Image(fill(N0f8(0.4), 2, 2)))) == fill(N0f8(0.4), 2, 2)
    levels = Image(GrayA{Float32}.([0.2, 0.2, 0.4, 0.9], 0.15))
    @test itemdata(apply(Equalize(), levels)) == GrayA{Float32}.([0, 0, 0.5, 1], 0.15)
    narrow = Image([RGB{Float32}(0.4, 0.12345, 0.5), RGB{Float32}(0.4001, 0.12345, 0.3)])
    @test itemdata(apply(Autocontrast(), narrow)) == RGB{Float32}[RGB(0, 0.12345, 1), RGB(1, 0.12345, 0)]

    offset = OffsetArray(copy(data), -1:0, 3:4)
    offset_img = Image(offset, Bounds((-1:0, 3:4)))
    out = apply(Invert(), offset_img)
    @test axes(itemdata(out)) == axes(offset)
    apply!(offset_img, Invert(), offset_img)
    @test itemdata(offset_img) == itemdata(out)
end

@testset "Intensity transform validation" begin
    @test_throws ArgumentError AdjustGamma(-1)
    @test_throws ArgumentError Posterize(0)
    @test_throws ArgumentError Posterize(9)
end
