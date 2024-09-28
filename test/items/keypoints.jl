include("../imports.jl")


@testset ExtendedTestSet "projective" begin
    keypoints = Keypoints([SVector(2., 2)], (32, 32))
    tfm = Project(Translation(20, 10))

    @test_nowarn apply(tfm, keypoints)
    @test apply(tfm, keypoints).bounds.rs == (21:52, 11:42)

    @test_nowarn apply(tfm, Polygon(keypoints))
end

@testset ExtendedTestSet "$(N)D BoundingBox" for N in 2:4
    bounds = Bounds{N}(ntuple(_ -> -50:50, N))

    min = SVector{N, Float64}(ntuple(i -> i==1 ? -20 : -10, N)) # [-20, -10, ...]
    max = SVector{N, Float64}(ntuple(i -> i==1 ? 20 : 10, N)) # [20, 10, ...]
    widebbox = BoundingBox(SVector{N, Float64}[min, max], bounds)

    min = SVector{N, Float64}(ntuple(i -> i==2 ? -20 : -10, N)) # [-10, -20, ...]
    max = SVector{N, Float64}(ntuple(i -> i==2 ? 20 : 10, N)) # [10, 20, ...]
    tallbbox = BoundingBox(SVector{N, Float64}[min, max], bounds)

    rotation = Matrix(1.0I, N, N)
    rotation[1:2, 1:2] = RotMatrix(pi/2.0)
    tfm90 = Project(LinearMap(rotation))
    @test_nowarn apply(tfm90, tallbbox)

    # Rotation by 90 degrees should make the wide bounding box equal to the
    # tall bounding box and vice versa.
    transformed = apply(tfm90, widebbox)
    @test itemdata(transformed) ≈ itemdata(tallbbox)
    transformed = apply(tfm90, transformed)
    @test itemdata(transformed) ≈ itemdata(widebbox)
    transformed = apply(tfm90, transformed)
    @test itemdata(transformed) ≈ itemdata(tallbbox)
    transformed = apply(tfm90, transformed)
    @test itemdata(transformed) ≈ itemdata(widebbox)
end
