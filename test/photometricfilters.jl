include("imports.jl")
using DataAugmentation: Random

@testset "GaussianBlur" begin
    @test_throws ArgumentError GaussianBlur(0)
    @test_throws ArgumentError GaussianBlur(1, 4)
    @test_throws ArgumentError GaussianBlur(1, (3, 0))

    data = fill(0.4f0, 7, 6)
    item = Image(data)
    blurred = itemdata(apply(GaussianBlur(0.8; kernel_size = (3, 5)), item))
    @test blurred ≈ data
    @test eltype(blurred) == Float32

    impulse = zeros(Float32, 7, 7)
    impulse[4, 4] = 1
    tfm = GaussianBlur(1, 3)
    expected = itemdata(apply(tfm, Image(impulse); randstate = 1.0))
    @test expected[4, 4] ≈ inv(1 + 2exp(-0.5))^2
    @test getrandstate(GaussianBlur((0.8, 0.8))) == 0.8
    inplace = Image(copy(impulse))
    @test itemdata(apply!(inplace, tfm, inplace; randstate = 1.0)) == expected
    testapply(tfm, Image)
    testapply!(tfm, Image)
end

@testset "GaussianBlur in 3D" begin
    constant = Image(fill(0.4f0, 5, 6, 7))
    @test itemdata(apply(GaussianBlur(0.8; kernel_size = 3), constant)) ≈ itemdata(constant)

    impulse = zeros(Float32, 7, 7, 7)
    impulse[4, 4, 4] = 1
    weight = inv(1 + 2exp(-0.5))
    blurred = itemdata(apply(GaussianBlur(1, 3), Image(impulse); randstate = 1.0))
    @test blurred[4, 4, 4] ≈ weight^3
    @test sum(blurred) ≈ 1
    @test blurred[4, 4, 4] > blurred[3, 4, 4] > blurred[3, 3, 4] > blurred[3, 3, 3] > 0
    @test iszero(blurred[1, 4, 4])

    # A pass is skipped for the dimensions with a kernel of one
    only_first = itemdata(apply(GaussianBlur(1, (3, 1, 1)), Image(impulse); randstate = 1.0))
    @test only_first[4, 4, 4] ≈ weight
    @test count(!iszero, only_first) == 3

    for item in (Image(rand(Float32, 6, 5, 4)), Image(rand(Gray{Float32}, 6, 5, 4)))
        number = apply(GaussianBlur(1, 3), item; randstate = 1.0)
        @test itemdata(number) == itemdata(apply(GaussianBlur(1, (3, 3, 3)), item; randstate = 1.0))
        @test eltype(itemdata(number)) == eltype(itemdata(item))
    end
    @test_throws ArgumentError apply(GaussianBlur(1, (3, 3)), Image(rand(Float32, 4, 4, 4)))
    @test_throws ArgumentError apply(GaussianBlur(1, (3, 3, 3)), Image(rand(Float32, 4, 4)))
    @test_throws ArgumentError GaussianBlur(1, (3, 4, 3))

    inplace = Image(copy(impulse))
    @test itemdata(apply!(inplace, GaussianBlur(1, 3), inplace; randstate = 1.0)) == blurred
end

@testset "GaussianNoise" begin
    @test_throws ArgumentError GaussianNoise(-0.1)
    @test_throws ArgumentError GaussianNoise((0.2, 0.1))

    item = Image(fill(0.5f0, 8, 8))
    tfm = GaussianNoise(0.2)
    state = (seed=UInt64(1234), sigma=0.2)
    a = apply(tfm, item; randstate=state)
    b = apply(tfm, item; randstate=state)
    @test itemdata(a) == itemdata(b)
    @test all(0 .<= itemdata(a) .<= 1)
    @test itemdata(apply(GaussianNoise(0), item; randstate=(seed=UInt64(1), sigma=0.0))) == itemdata(item)
    testapply(tfm, Image)
    testapply!(tfm, Image)
    buffered = Buffered(tfm)
    @test itemdata(apply(buffered, item; randstate=state)) == itemdata(apply(tfm, item; randstate=state))
    @test getrandstate(GaussianNoise((0.2, 0.2))).sigma == 0.2
    Random.seed!(876)
    state1 = getrandstate(tfm)
    Random.seed!(876)
    state2 = getrandstate(tfm)
    @test state1 == state2

    fixed = Image(rand(RGB{N0f8}, 5, 5))
    @test_nowarn apply(tfm, fixed)
end

@testset "AdjustSharpness" begin
    @test_throws ArgumentError AdjustSharpness(-1)
    item = Image(rand(Float32, 8, 8))
    data = itemdata(item)
    @test itemdata(apply(AdjustSharpness(1), item)) == data
    softened = itemdata(apply(AdjustSharpness(0), item))
    @test softened[1, :] == data[1, :]
    @test softened[end, :] == data[end, :]
    @test softened[:, 1] == data[:, 1]
    @test softened[:, end] == data[:, end]
    impulse = Image(reshape(Float32[0, 0, 0, 0, 1, 0, 0, 0, 0], 3, 3))
    @test itemdata(apply!(impulse, AdjustSharpness(0), impulse))[2, 2] ≈ 5f0 / 13
    testapply(AdjustSharpness(2), Image)
    testapply!(AdjustSharpness(2), Image)
end

@testset "AdjustSharpness in 3D" begin
    volume = rand(Float32, 5, 4, 6)
    item = Image(volume)
    @test itemdata(apply(AdjustSharpness(1), item)) == volume

    # Weight 5 for the center and 1 for each of the other 26 voxels of the neighborhood
    smoothed(v, i) = (4v[i] + sum(v[i + j] for j in CartesianIndices((-1:1, -1:1, -1:1)))) / 31
    interior = CartesianIndices((2:4, 2:3, 2:5))
    out = itemdata(apply(AdjustSharpness(0), item))
    @test all(i -> out[i] ≈ smoothed(volume, i), interior)
    border = [i for i in CartesianIndices(volume) if i ∉ interior]
    @test all(i -> out[i] == volume[i], border)

    factor = 2
    sharp = itemdata(apply(AdjustSharpness(factor), item))
    @test all(i -> sharp[i] ≈ clamp(smoothed(volume, i) + factor * (volume[i] - smoothed(volume, i)), 0, 1), interior)

    impulse = zeros(Float32, 3, 3, 3)
    impulse[2, 2, 2] = 1
    center = Image(impulse)
    @test itemdata(apply!(center, AdjustSharpness(0), center))[2, 2, 2] ≈ 5f0 / 31
    inplace = Image(copy(volume))
    @test itemdata(apply!(inplace, AdjustSharpness(2), inplace)) ≈ sharp

    thin = rand(Float32, 2, 8, 8)
    @test itemdata(apply(AdjustSharpness(0), Image(thin))) == thin
    colors = Image(rand(RGB{Float32}, 4, 4, 4))
    @test eltype(itemdata(apply(AdjustSharpness(2), colors))) == RGB{Float32}
end
