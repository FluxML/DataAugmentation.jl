@testset "RandomErasing" begin
    image = Image(fill(1f0, 20, 20))
    tfm = RandomErasing(p = 1, scale = (0.125, 0.125), ratio = (2, 2))
    state = (probability = 0.0, seed = UInt64(123))
    out = itemdata(apply(tfm, image; randstate = state))
    @test count(iszero, out) == 50
    @test count(row -> any(iszero, row), eachrow(out)) == 10
    @test count(col -> any(iszero, col), eachcol(out)) == 5
    @test all(==(1f0), itemdata(image))

    mask = MaskBinary(trues(20, 20))
    multiclass = MaskMulti(ones(Int, 20, 20), 1:2)
    keypoints = Keypoints([SVector(1.0, 2.0)], (20, 20))
    erased, annotations... = apply(tfm, (image, mask, multiclass, keypoints); randstate = state)
    @test itemdata(erased) == out
    @test annotations == (mask, multiclass, keypoints)

    buffer = Image(similar(itemdata(image)))
    @test itemdata(apply!(buffer, tfm, image; randstate = state)) === itemdata(buffer)
    @test itemdata(buffer) == out
    @test apply!(mask, tfm, mask; randstate = state) === mask

    skipped = itemdata(apply(RandomErasing(p = 0), image; randstate = state))
    @test skipped == itemdata(image)
    @test skipped !== itemdata(image)
    no_fit = RandomErasing(p = 1, scale = (0.5, 0.5), ratio = (1e100, 1e100))
    @test itemdata(apply(no_fit, image; randstate = state)) == itemdata(image)
    @test_throws ArgumentError apply(tfm, Image(fill(1f0, 4, 5, 3)); randstate = state)

    fixed = RGB{N0f8}(0, 0.5, 1)
    color_image = Image(fill(RGB{N0f8}(1, 1, 1), 10, 10))
    color_fill = RandomErasing(p = 1, scale = (1, 1), ratio = (1, 1), value = fixed)
    @test all(==(fixed), itemdata(apply(color_fill, color_image; randstate = state)))

    gray_image = Image(fill(Gray{Float32}(1), 10, 10))
    gray_fill = RandomErasing(p = 1, scale = (1, 1), ratio = (1, 1), value = 0.25)
    @test all(==(Gray{Float32}(0.25)), itemdata(apply(gray_fill, gray_image; randstate = state)))

    offset_data = OffsetArray(fill(1f0, 20, 20), -5:14, 7:26)
    offset = Image(offset_data, Bounds((-5:14, 7:26)))
    erased_offset = itemdata(apply(tfm, offset; randstate = state))
    @test axes(erased_offset) == axes(offset_data)
    @test parent(erased_offset) == out
end

@testset "RandomErasing noise" begin
    tfm = RandomErasing(p = 1, scale = (1, 1), ratio = (1, 1), value = :random)
    state = (probability = 0.0, seed = UInt64(123))
    image = Image(fill(1f0, 4, 4))
    noisy = itemdata(apply(tfm, image; randstate = state))
    @test all(x -> 0 <= x < 1, noisy)
    @test noisy == itemdata(apply(tfm, image; randstate = state))

    colors = reshape([RGBA{Float32}(1, 1, 1, i / 5) for i in 1:4], 2, 2)
    noisy_colors = itemdata(apply(tfm, Image(colors); randstate = state))
    @test alpha.(noisy_colors) == alpha.(colors)
    @test noisy_colors != colors

    noisy_ints = itemdata(apply(tfm, Image(fill(UInt8(4), 4, 4)); randstate = state))
    @test eltype(noisy_ints) == UInt8
    @test any(>(1), noisy_ints)
end

@testset "RandomErasing validation" begin
    @test_throws ArgumentError RandomErasing(p = 1.1)
    @test_throws ArgumentError RandomErasing(scale = (0, 1))
    @test_throws ArgumentError RandomErasing(scale = (0.1, 1.1))
    @test_throws ArgumentError RandomErasing(ratio = (2, 1))
    @test_throws ArgumentError RandomErasing(value = :unknown)
end
