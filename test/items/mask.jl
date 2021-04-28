include("../imports.jl")


@testset ExtendedTestSet "projective" begin
    mask = MaskBinary(rand(Bool, 32, 32))
    tfm = Project(Translation(20, 10))

    @test_nowarn apply(tfm, mask)
    tmask = apply(tfm, mask)
    @test tmask.bounds.rs == (21:52, 11:42)

    @test_nowarn apply!(tmask, tfm, mask)
end
