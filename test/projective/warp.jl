
include("../imports.jl")

@testset ExtendedTestSet "WarpAffine" begin
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
