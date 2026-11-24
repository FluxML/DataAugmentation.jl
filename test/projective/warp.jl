
include("../imports.jl")

@testset "WarpAffine" begin
    tfm = WarpAffine(0.1)
    image = Image(rand(RGB, 50, 50))

    @test_nowarn apply(tfm, image)
    @test sum.(getbounds(apply(tfm, image)).rs) != (50, 500)
    buf = apply(tfm, image)
    ctfm = tfm |> CenterCrop((50, 50))
    @test_nowarn apply(ctfm, image)
    buf = apply(ctfm, image)
    @test_nowarn apply!(buf, ctfm, image)
end

@testset "WarpAffine is the same as before in 2D" begin
    # Values recorded before the generalization to N dimensions.
    P = getprojection(WarpAffine(0.1), Bounds((20, 30)); randstate = 1)
    @test P.linear ≈ Float32[0.9825539 0.0019312337; -0.02836518 1.0595665]
    @test P.translation ≈ Float32[-0.45383632, 1.0231587]
    P = getprojection(WarpAffine(0.1), Bounds((20, 30)); randstate = 7)
    @test P.linear ≈ Float32[1.0990084 -0.10806603; 0.02605944 1.0843933]
    @test P.translation ≈ Float32[0.6833911, -2.3728514]
end

@testset "WarpAffine in $N dimensions" for N in (3, 4)
    bounds = Bounds(ntuple(i -> 6 + i, N))

    @testset "no displacement is the identity" begin
        for seed in 1:20
            P = getprojection(WarpAffine(0), bounds; randstate = seed)
            @test P.linear ≈ I(N) atol = 1e-4
            @test P.translation ≈ zeros(N) atol = 1e-4
        end
    end

    @testset "projections are invertible" begin
        for seed in 1:100
            P = getprojection(WarpAffine(0.1), bounds; randstate = seed)
            @test P isa AffineMap
            @test all(isfinite, P.linear)
            @test abs(det(P.linear)) > 0.1
            @test P == getprojection(WarpAffine(0.1), bounds; randstate = seed)
        end
        @test length(unique(getprojection(WarpAffine(0.1), bounds; randstate = s) for s in 1:10)) == 10
    end
end

@testset "WarpAffine on volumes" begin
    sz = (8, 9, 10)
    items = (
        Image(rand(Float32, sz)),
        Keypoints([SVector(2f0, 3f0, 4f0)], sz),
        MaskBinary(rand(Bool, sz)),
        MaskMulti(rand(1:3, sz), 1:3),
    )
    tfm = WarpAffine(0.1)
    testprojective(tfm, items)
    cropped = tfm |> CenterCrop(6)
    @test size(itemdata(apply(cropped, items[1]))) == (6, 6, 6)
    buf = makebuffer(cropped, items)
    @test_nowarn apply!(buf, cropped, items)
    @test_nowarn apply(WarpAffine(0.1), Image(rand(RGB{N0f8}, 5, 5, 5)))
end
