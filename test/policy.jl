include("imports.jl")

using DataAugmentation: AugmentationOp
import Interpolations: BSpline, Constant

ops(tfm) = Dict(op.name => op for op in tfm.ops)
steps(tfm, name, magnitude) = [(ops(tfm)[name], float(magnitude))]

# A bright pixel that stays sharp under every interpolation
spike_image() = Image(setindex!(zeros(Float32, 21, 31), 1, 5, 7); interpolate = BSpline(Constant()))
spike_position(data) = Tuple(only(findall(>(0.5), data)))

@testset "Magnitude tables" begin
    table = ops(RandAugment())
    @test sort(collect(keys(table))) == sort([:Identity, :ShearX, :ShearY, :TranslateX, :TranslateY,
        :Rotate, :Brightness, :Color, :Contrast, :Sharpness, :Posterize, :Solarize, :AutoContrast, :Equalize])
    @test table[:ShearX].magnitudes[10] ≈ 0.09
    @test table[:Rotate].magnitudes[end] ≈ 30
    @test table[:Brightness].magnitudes[10] ≈ 0.27
    @test table[:TranslateX].magnitudes[end] ≈ 150 / 331
    @test table[:TranslateX].relative
    @test table[:Solarize].magnitudes[[1, end]] == [1, 0]
    @test table[:Posterize].magnitudes[[1, 5, 13, 20, 28, 31]] == [8, 7, 6, 5, 4, 4]
    @test isempty(table[:Equalize].magnitudes) && !table[:Equalize].signed

    wide = ops(TrivialAugmentWide())
    @test wide[:ShearY].magnitudes[end] ≈ 0.99
    @test wide[:Rotate].magnitudes[end] ≈ 135
    @test wide[:TranslateY].magnitudes[end] == 32 && !wide[:TranslateY].relative
    @test wide[:Posterize].magnitudes[1:5:31] == [8, 7, 6, 5, 4, 3, 2]
end

@testset "RandAugment" begin
    @test_throws ArgumentError RandAugment(-1)
    @test_throws ArgumentError RandAugment(2, 31)
    @test_throws ArgumentError RandAugment(2, 9; num_magnitude_bins = 1)
    @test_throws ArgumentError RandAugment(fill = "black")

    tfm = RandAugment(3, 9)
    state = getrandstate(tfm)
    @test length(state) == 3
    for (op, magnitude) in state
        @test abs(magnitude) == (isempty(op.magnitudes) ? 0 : op.magnitudes[10])
    end
    sampled = [m for _ in 1:100 for (_, m) in getrandstate(RandAugment(1, 9))]
    @test any(<(0), sampled) && any(>(0), sampled)

    image = Image(rand(RGB{N0f8}, 12, 16))
    @test apply(RandAugment(0), image) === image
    @test apply(tfm, image; randstate = steps(tfm, :Identity, 0)) === image

    # Steps apply in order
    state = [steps(tfm, :Brightness, 0.5); steps(tfm, :Solarize, 0.5); steps(tfm, :Equalize, 0)]
    expected = apply(Equalize(), apply(Solarize(0.5), apply(AdjustBrightness((1.5, 1.5)), image)))
    @test itemdata(apply(tfm, image; randstate = state)) == itemdata(expected)
    @test eltype(itemdata(apply(tfm, image))) == RGB{N0f8}
    @test itemdata(apply(Buffered(tfm), image; randstate = state)) == itemdata(expected)
end

@testset "Operations" begin
    tfm = RandAugment()
    image = Image(rand(RGB{Float32}, 12, 16))
    for (name, magnitude, direct) in [
            (:Brightness, -0.4, AdjustBrightness((0.6, 0.6))),
            (:Color, 0.3, AdjustSaturation((1.3, 1.3))),
            (:Contrast, 0.9, AdjustContrast((1.9, 1.9))),
            (:Sharpness, 0.9, AdjustSharpness(1.9)),
            (:Posterize, 5, Posterize(5)),
            (:Solarize, 0.3, Solarize(0.3)),
            (:AutoContrast, 0, Autocontrast()),
            (:Equalize, 0, Equalize())]
        out = apply(tfm, image; randstate = steps(tfm, name, magnitude))
        @test itemdata(out) == itemdata(apply(direct, image))
    end
    invert = AugmentationOp(:Invert, Float64[], false, false)
    @test itemdata(apply(AutoAugment(), image; randstate = [(invert, 0.0)])) == itemdata(apply(Invert(), image))

    # Geometric operations move images, masks and keypoints alike
    image = spike_image()
    keypoints = Keypoints([SVector(5.0, 7.0)], (21, 31))
    mask = MaskBinary(itemdata(image) .> 0.5)
    for (name, magnitude, position) in [
            (:TranslateX, 0.2, (5, 13)),    # trunc(0.2 * 31) = 6 pixels
            (:TranslateY, 0.3, (11, 7)),    # trunc(0.3 * 21) = 6 pixels
            (:ShearX, 0.5, (5, 9)),         # 7 + 0.5 * (5 - 1)
            (:ShearY, -0.5, (2, 7)),        # 5 - 0.5 * (7 - 1)
            (:Rotate, 180, (17, 25))]       # about the center (11, 16)
        out = apply(tfm, (image, keypoints, mask); randstate = steps(tfm, name, magnitude))
        @test spike_position(itemdata(out[1])) == position
        @test only(itemdata(out[2])) ≈ SVector(position...) atol = 1e-4
        @test Tuple(only(findall(itemdata(out[3])))) == position
        @test getbounds(out[1]) == getbounds(image)
    end

    # Zero magnitudes leave items as they are, and so do photometric operations on masks and keypoints
    out = apply(tfm, (image, keypoints, mask); randstate = steps(tfm, :Rotate, 0))
    @test all(map(===, out, (image, keypoints, mask)))
    out = apply(tfm, (image, keypoints, mask); randstate = steps(tfm, :Equalize, 0))
    @test out[2] === keypoints && out[3] === mask

    # Other dimensionalities are rejected whatever operations are drawn
    volume = Image(rand(Float32, 6, 6, 6))
    @test_throws ArgumentError apply(tfm, volume; randstate = steps(tfm, :ShearX, 0.1))
    @test_throws ArgumentError apply(tfm, volume; randstate = steps(tfm, :Brightness, 0.1))
    @test_throws ArgumentError apply(TrivialAugmentWide(), volume)
end

@testset "Fill" begin
    tfm = RandAugment(fill = 0.5f0)
    image = Image(zeros(Float32, 10, 10))
    mask = MaskBinary(falses(10, 10))
    out = apply(tfm, (image, mask); randstate = steps(tfm, :TranslateX, 0.3))
    @test itemdata(out[1])[:, 1:3] == fill(0.5f0, 10, 3)
    @test itemdata(out[1])[:, 4:end] == zeros(Float32, 10, 7)
    @test out[1].extrapolate == image.extrapolate
    @test !any(itemdata(out[2]))

    # Photometric steps do not change the fill of later geometric ones
    invert = AugmentationOp(:Invert, Float64[], false, false)
    tfm = RandAugment(fill = 0.2f0)
    rotate = (ops(tfm)[:Rotate], 45.0)
    out = apply(tfm, Image(fill(0.9f0, 12, 12)); randstate = [(invert, 0.0), rotate])
    @test itemdata(out)[1, 1] == 0.2f0
    @test itemdata(out)[6, 6] ≈ 0.1f0

    colored = Image(zeros(RGB{N0f8}, 10, 10))
    tfm = TrivialAugmentWide(fill = RGB{N0f8}(1, 0, 0))
    out = apply(tfm, colored; randstate = steps(tfm, :TranslateY, 3))
    @test itemdata(out)[1:3, :] == fill(RGB{N0f8}(1, 0, 0), 3, 10)
end

@testset "TrivialAugmentWide" begin
    tfm = TrivialAugmentWide()
    @test_throws ArgumentError TrivialAugmentWide(num_magnitude_bins = 1)
    @test_throws ArgumentError TrivialAugmentWide(fill = "black")
    for _ in 1:50
        (op, magnitude), = getrandstate(tfm)
        @test abs(magnitude) <= (isempty(op.magnitudes) ? 0 : maximum(op.magnitudes))
    end
    @test length(unique(only(getrandstate(tfm))[1].name for _ in 1:300)) == 14

    for item in (Image(rand(Gray{N0f8}, 12, 16)), Image(rand(RGBA{Float32}, 12, 16)))
        titem = apply(tfm, item)
        @test typeof(titem) == typeof(item) && size(itemdata(titem)) == size(itemdata(item))
    end
    testapply(tfm, Image)
end
