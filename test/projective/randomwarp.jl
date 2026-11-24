include("../imports.jl")

@testset "RandomAffine" begin
    image = Image(reshape(Float32.(1:100), 10, 10))
    keypoints = Keypoints([SVector(3.0, 4.0)], (10, 10))
    tfm = RandomAffine(0; translate=(0, 0), scale=(1, 1), shear=(0, 0))
    state = getrandstate(tfm)
    transformed = apply(tfm, image; randstate=state)
    @test getbounds(transformed) == getbounds(image)
    @test itemdata(transformed) ≈ itemdata(image)
    @test itemdata(apply(tfm, keypoints; randstate=state)) == itemdata(keypoints)

    random = RandomAffine(20; translate=(0.1, 0.2), scale=(0.8, 1.2), shear=(-5, 5))
    p1 = getprojection(random, getbounds(image); randstate=123)
    p2 = getprojection(random, getbounds(image); randstate=123)
    @test p1(SVector(5.0, 5.0)) == p2(SVector(5.0, 5.0))
    @test getbounds(apply(random, image; randstate=123)) == getbounds(image)
    @test_throws ArgumentError RandomAffine(10; scale=(0, 1))
    @test_throws ArgumentError RandomAffine(10; translate=(1.1, 0))
    @test_throws ArgumentError RandomAffine(0; shear = (90, 90))
    shear = getprojection(RandomAffine(0; shear = (45, 45, 45, 45)), getbounds(image); randstate = 0)
    @test inv(shear)(shear(SVector(2., 3.))) ≈ SVector(2., 3.)
    points = Keypoints([SVector(2, 3)], (10, 10))
    moved = apply(RandomAffine((20, 20)), points)
    @test !all(isinteger, only(itemdata(moved)))
end

@testset "RandomPerspective" begin
    image = Image(reshape(Float32.(1:100), 10, 10))
    keypoints = Keypoints([SVector(2.0, 3.0)], (10, 10))
    identity_tfm = RandomPerspective(0; p=1)
    state = getrandstate(identity_tfm)
    @test itemdata(apply(identity_tfm, image; randstate=state)) ≈ itemdata(image)

    tfm = RandomPerspective(0.4; p=1)
    P = getprojection(tfm, getbounds(image); randstate=42)
    @test P isa DataAugmentation._Homography
    @test all(isfinite, P(SVector(5.0, 5.0)))
    corners = (SVector(1., 1.), SVector(1., 10.), SVector(10., 10.), SVector(10., 1.))
    @test all(p -> all(1 - 1e-8 .<= P(p) .<= 10 + 1e-8), corners)
    @test inv(P)(P(SVector(5.0, 5.0))) ≈ SVector(5.0, 5.0)
    @test getbounds(apply(tfm, image; randstate=42)) == getbounds(image)
    @test length(itemdata(apply(tfm, keypoints; randstate=42))) == 1
    @test getprojection(RandomPerspective(1; p=0), getbounds(image); randstate=2) isa IdentityTransformation
    @test getprojection(tfm, Bounds((1, 8)); randstate=42) isa IdentityTransformation
    @test_throws ArgumentError RandomPerspective(1.1)
    @test_throws ArgumentError RandomPerspective(0.5; p=-0.1)

    small = Bounds((-5:-3, 10:16))
    center = SVector(-4., 13.)
    for seed in 1:12
        P = getprojection(RandomPerspective(1; p = 1), small; randstate = seed)
        @test inv(P)(P(center)) ≈ center
    end

    pipeline = tfm |> RandomAffine(5)
    states = getrandstate(pipeline)
    @test itemdata(apply(pipeline, image; randstate=states)) ≈ itemdata(apply(pipeline, image; randstate=states))
end
