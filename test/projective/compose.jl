include("../imports.jl")


@testset ExtendedTestSet "`ComposedProjectiveTransform`" begin
    tfms = (Project(Translation(10, 10)), Project(Translation(-10, 10)))
    tfm = ComposedProjectiveTransform(tfms[1], tfms[2])
    item = Image(rand(10, 10))
    ## composition of `AbstractProjective`s creates a `ComposedProjectiveTransform`
    @test tfm == tfms[1] |> tfms[2]
    ## can be `apply`ed
    @test_nowarn apply(tfm, item)

    ## identical result to sequential application
    @test itemdata(apply(tfm, item)) == itemdata(apply(Sequence(tfms), item))
end
