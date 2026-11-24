include("../imports.jl")


@testset "projective" begin
    mask = MaskBinary(rand(Bool, 32, 32))
    tfm = Project(Translation(20, 10))

    @test_nowarn apply(tfm, mask)
    tmask = apply(tfm, mask)
    @test tmask.bounds.rs == (21:52, 11:42)

    @test_nowarn apply!(tmask, tfm, mask)
end

@testset "MaskMulti from IndirectArray" begin
    index = rand(1:3, 8, 8)
    mask = MaskMulti(IndirectArray(index, [10, 20, 30]))
    @test itemdata(mask) == index
    @test mask.classes == [10, 20, 30]
end

@testset "Projection keeps the interpolation settings" begin
    linear = Interpolations.BSpline(Interpolations.Linear())
    tfm = Project(Translation(1, 2))
    for mask in (MaskMulti(rand(1:3, 8, 8); interpolate = linear, extrapolate = 1),
                 MaskBinary(rand(Bool, 8, 8); interpolate = linear, extrapolate = true))
        out = apply(tfm, mask)
        @test out.interpolate == linear
        @test out.extrapolate == mask.extrapolate
        out = apply!(makebuffer(tfm, mask), tfm, mask)
        @test out.interpolate == linear
        @test out.extrapolate == mask.extrapolate
    end
end

