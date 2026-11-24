include("../imports.jl")


@testset "projective" begin
    image = Image(rand(RGB, 32, 32))
    tfm = Project(Translation(20, 10))

    @test_nowarn apply(tfm, image)
    timage = apply(tfm, image)
    @test timage.bounds.rs == (21:52, 11:42)

    @test_nowarn apply!(timage, tfm, image)

end

@testset "Antialiasing affine downsampling" begin
    data = Float32[isodd(i + j) for i in 1:65, j in 1:65]
    resize = Resize((8, 8))
    raw = itemdata(apply(resize, Image(data)))
    image = Image(data; antialias = true)
    filtered = apply(resize, image)
    @test sum(abs, raw .- 0.5) > 0
    @test itemdata(filtered) ≈ fill(0.5f0, 8, 8)
    @test filtered.antialias
    @test itemdata(apply!(makebuffer(resize, image), resize, image)) == itemdata(filtered)

    rotated = apply(Rotate(15; expand = false), image; randstate = 15)
    @test itemdata(rotated) == itemdata(apply(Rotate(15; expand = false), Image(data); randstate = 15))
    colored = Image(OffsetArray(fill(RGBA{N0f8}(0.2, 0.4, 0.6, 0.8), 6, 10), -1:4, -3:6); antialias = true)
    @test all(==(first(itemdata(colored))), itemdata(apply(Resize((2, 3)), colored)))
    @test eltype(itemdata(apply(Resize((2, 3)), colored))) == RGBA{N0f8}
end
