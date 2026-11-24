include("../imports.jl")

@testset "ElasticTransform" begin
    @test_throws ArgumentError ElasticTransform(-1)
    @test_throws ArgumentError ElasticTransform(1, 0)
    @test_throws ArgumentError ElasticTransform((1, -1), (2, 2))
    @test_throws ArgumentError ElasticTransform((1, 2, 3), 2)
    @test_throws ArgumentError ElasticTransform(1, (1, Inf))

    image = Image(reshape(Float32.(1:100), 10, 10))
    keypoints = Keypoints([SVector(3.0, 4.0), SVector(7.0, 8.0)], (10, 10))
    identity = ElasticTransform(0, 1)
    @test itemdata(apply(identity, image; randstate=10)) == itemdata(image)
    @test itemdata(apply(identity, keypoints; randstate=10)) == itemdata(keypoints)

    tfm = ElasticTransform((3, 5), (1.5, 2.5))
    @test getrandstate(tfm) isa UInt64
    P1 = getprojection(tfm, getbounds(image); randstate=123)
    P2 = getprojection(tfm, getbounds(image); randstate=123)
    point = SVector(5.25, 6.5)
    @test P1(point) ≈ P2(point)
    @test inv(P1)(P1(point)) ≈ point atol=1e-8

    state = 123
    transformed = apply(tfm, image; randstate=state)
    transformed_again = apply(tfm, image; randstate=state)
    @test getbounds(transformed) == getbounds(image)
    @test itemdata(transformed) == itemdata(transformed_again)
    transformed_points = apply(tfm, keypoints; randstate=state)
    @test itemdata(transformed_points) ≈ [P1(p) for p in itemdata(keypoints)]
    testapply(tfm, Image)
    testapply!(tfm, Image)
end
