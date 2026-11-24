include("imports.jl")
using DataAugmentation: clamp01
using Distributions: Uniform

@testset "Brightness and contrast" begin
    rgb = RGB{Float32}(0.2, 0.4, 0.6)
    item = Image(fill(rgb, 4, 4))
    bright = apply(AdjustBrightness(), item; randstate=2f0)
    @test itemdata(bright)[1] ≈ RGB{Float32}(0.4, 0.8, 1.0)
    @test itemdata(apply(AdjustBrightness(0), item; randstate=1)) == itemdata(item)
    @test getrandstate(AdjustBrightness((1.5, 1.5))) == 1.5
    @test 0.8 <= getrandstate(AdjustBrightness(Uniform(0.8, 1.2))) <= 1.2
    @test itemdata(apply(AdjustContrast(), item; randstate=1)) == itemdata(item)
    @test first(itemdata(apply(AdjustContrast(), item; randstate=0.5f0))) ≈ rgb * 1.5f0
    @test eltype(itemdata(apply(AdjustBrightness(), item; randstate=1.1))) == RGB{Float32}
    testapply(AdjustBrightness(0.2), Image)
    testapply!(AdjustBrightness(0.2), Image)
    testapply(AdjustContrast(0.2), Image)
    testapply!(AdjustContrast(0.2), Image)
end

@testset "Brightness and contrast retain fixed-point rounding" begin
    for T in (RGB{N0f8}, Gray{N0f8})
        data = [T(i / 255) for i in 0:255]
        item = Image(data)
        brightness = itemdata(apply(AdjustBrightness(), item; randstate = 0.5))
        @test brightness == map(x -> T(clamp01(x * 0.5)), data)
        μ = mean(data)
        contrast = itemdata(apply(AdjustContrast(), item; randstate = 2.0))
        @test contrast == map(x -> T(clamp01(x - μ)), data)
    end
end

@testset "Brightness and contrast accept other color spaces" begin
    c = RGB{Float32}(0.2, 0.4, 0.6)
    for T in (HSV{Float32}, Lab{Float32})
        img = Image(fill(convert(T, c), 2, 3))
        bright = apply(AdjustBrightness(), img; randstate = 0.5f0)
        contrast = apply(AdjustContrast(), img; randstate = 1.5f0)
        @test eltype(itemdata(bright)) === T
        @test RGB(first(itemdata(bright))) ≈ 0.5f0 * c atol = 2f-5
        @test RGB(first(itemdata(contrast))) ≈ 0.5f0 * c atol = 2f-5
    end
end

@testset "Saturation, hue, and jitter" begin
    red = Image(fill(RGB{Float32}(1, 0, 0), 3, 3))
    gray = apply(AdjustSaturation(), red; randstate=0)
    @test isapprox(itemdata(gray)[1], RGB{Float32}(0.299, 0.299, 0.299); atol=0.01)
    shifted = apply(AdjustHue(), red; randstate=1/3)
    @test itemdata(shifted)[1] ≈ RGB{Float32}(0, 1, 0) atol = 1f-6
    jitter = ColorJitter()
    state = ((1, 2, 3, 4), (1.0, nothing, nothing, nothing))
    @test itemdata(apply(jitter, red; randstate=state)) == itemdata(red)
    pixel = Image(fill(RGB{Float32}(0.8, 0.4, 0.2), 1, 1))
    states = (2.0, 1.5, nothing, nothing)
    brighter_first = apply(jitter, pixel; randstate = ((1, 2, 3, 4), states))
    contrast_first = apply(jitter, pixel; randstate = ((2, 1, 3, 4), states))
    @test only(itemdata(brighter_first)) ≈ RGB{Float32}(0.5, 0.4, 0.2)
    @test only(itemdata(contrast_first)) ≈ RGB{Float32}(0.8, 0.4, 0.2)
    testapply(jitter, Image)
    testapply!(jitter, Image)
end

@testset "Grayscale and RGB conversion" begin
    rgb = Image(fill(RGB{Float32}(1, 0, 0), 3, 3))
    gray = apply(Grayscale(), rgb)
    @test eltype(itemdata(gray)) == Gray{Float32}
    @test itemdata(gray)[1].val ≈ 0.299 atol=0.01
    rgb3 = apply(Grayscale(3), rgb)
    @test eltype(itemdata(rgb3)) == RGB{Float32}
    @test apply(ToRGB(), gray) isa Image
    @test eltype(itemdata(apply(ToRGB(), gray))) == RGB{Float32}
    testapply!(Grayscale(), Image)
    testapply!(ToRGB(), Image)
    @test itemdata(apply(RandomGrayscale(0), rgb)) == itemdata(rgb)
    @test itemdata(apply(RandomGrayscale(1), rgb)) == itemdata(rgb3)

    pixel = RGBA{N0f8}(0.2, 0.4, 0.6, 0.3)
    rgba = Image(fill(pixel, 2, 3); extrapolate = pixel)
    gray = apply(Grayscale(), rgba)
    @test eltype(itemdata(gray)) == GrayA{N0f8}
    @test alpha(first(itemdata(gray))) == alpha(pixel)
    @test gray.extrapolate == first(itemdata(gray))
    @test getbounds(gray) == getbounds(rgba)
    restored = apply(ToRGB(), gray)
    @test eltype(itemdata(restored)) == RGBA{N0f8}
    @test alpha(first(itemdata(restored))) == alpha(pixel)
end

@testset "Alpha and non-color behavior" begin
    rgba = Image(fill(RGBA{Float32}(0.2, 0.4, 0.6, 0.3), 2, 2))
    @test itemdata(apply(AdjustBrightness(), rgba; randstate=2))[1].alpha ≈ 0.3
    mask = MaskBinary(fill(true, 2, 2))
    multimask = MaskMulti(fill(1, 2, 2))
    keypoints = Keypoints([SVector(1., 1)], (2, 2))
    for tfm in (AdjustBrightness(), AdjustContrast(), AdjustSaturation(), AdjustHue(),
                ColorJitter(), Grayscale(), ToRGB(), RandomGrayscale())
        @test apply(tfm, mask) === mask
        @test apply(tfm, keypoints) === keypoints
    end

    bbox = BoundingBox([SVector(1., 1), SVector(2., 2)], (2, 2))
    polygon = Polygon([SVector(1., 1), SVector(2., 1), SVector(2., 2)], (2, 2))
    others = (mask, multimask, keypoints, bbox, polygon)
    img = Image(rand(RGB{N0f8}, 2, 2))
    for tfm in (AdjustBrightness(), AdjustContrast(), AdjustSaturation(), AdjustHue(), ColorJitter())
        titems = apply(tfm, (img, others...))
        @test all(titems[i + 1] === others[i] for i in eachindex(others))
        @test apply(tfm |> AdjustBrightness(0.1), (img, mask))[2] === mask
        @test apply(Buffered(tfm), (img, mask))[2] === mask
    end
end
