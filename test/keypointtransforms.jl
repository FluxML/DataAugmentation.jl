include("imports.jl")

@testset "Keypoint bounds after cropping" begin
    image = Image(zeros(100, 100))
    data = SVector{2, Float32}[(1, 1), (50, 50), (100, 100)]
    keypoints = Keypoints(data, (100, 100))
    crop = CenterCrop((20, 20)) |> PinOrigin()
    cropped_image, cropped = apply(crop, (image, keypoints))
    @test itemdata(cropped) == SVector{2, Float32}[(-39, -39), (10, 10), (60, 60)]

    _, clamped = apply(crop |> ClampKeypoints(), (image, keypoints))
    @test itemdata(clamped) == SVector{2, Float32}[(1, 1), (10, 10), (20, 20)]
    @test getbounds(clamped) == getbounds(cropped_image)

    new_image, sanitized = apply(crop |> SanitizeKeypoints(), (image, keypoints))
    missing_point = SVector(NaN32, NaN32)
    @test isequal(itemdata(sanitized), [missing_point, SVector(10f0, 10f0), missing_point])
    @test getbounds(sanitized) == getbounds(new_image) == getbounds(cropped_image)
    @test itemdata(new_image) == itemdata(cropped_image)
    @test itemdata(keypoints) == SVector{2, Float32}[(1, 1), (50, 50), (100, 100)]

    visible = Keypoints([SVector(10f0, 10f0)], (20, 20))
    @test showitems((new_image, sanitized)) == showitems((new_image, visible))
end

@testset "Offset and fractional bounds in 3D" begin
    bounds = Bounds((-4:0, 3:7, 10:13))
    data = reshape(SVector{3, Float32}[
        (-4, 3, 10), (0.5, 7.5, 13.5), (1, 5, 11),
        (-4, 2.75, 11), (0, Inf, 13), (NaN, NaN, NaN),
    ], 2, 3)
    keypoints = Keypoints(data, bounds)
    clamped = itemdata(apply(ClampKeypoints(), keypoints))
    @test eltype(clamped) == eltype(data)
    @test isequal(clamped, reshape(SVector{3, Float32}[
        (-4, 3, 10), (0, 7, 13), (0, 5, 11),
        (-4, 3, 11), (0, 7, 13), (NaN, NaN, NaN),
    ], 2, 3))

    sanitized = apply(SanitizeKeypoints(), keypoints)
    @test getbounds(sanitized) == bounds
    @test size(itemdata(sanitized)) == size(data)
    @test eltype(itemdata(sanitized)) == eltype(data)
    @test itemdata(sanitized)[1:2] == data[1:2]
    @test all(p -> all(isnan, p), itemdata(sanitized)[3:6])
end

@testset "Integer and missing keypoints" begin
    P = Union{Nothing, SVector{2, Int}}
    keypoints = Keypoints(P[nothing, SVector(2, 3), SVector(0, 3)], (3, 4))
    @test itemdata(apply(ClampKeypoints(), keypoints)) == P[nothing, SVector(2, 3), SVector(1, 3)]
    sanitized = itemdata(apply(SanitizeKeypoints(), keypoints))
    @test eltype(sanitized) == Union{Nothing, SVector{2, Float64}}
    @test isequal(sanitized, [nothing, SVector(2.0, 3.0), SVector(NaN, NaN)])

    for tfm in (ClampKeypoints(), SanitizeKeypoints())
        empty = Keypoints(SVector{2, Int}[], (0, 4))
        @test isempty(itemdata(apply(tfm, empty)))
        missing = OffsetArray(P[nothing], 0:0)
        result = itemdata(apply(tfm, Keypoints(missing, (3, 4))))
        @test axes(result) == axes(missing)
        @test result[0] === nothing
    end

    no_pixels = Keypoints([SVector(1, 2)], (0, 4))
    @test_throws ArgumentError apply(ClampKeypoints(), no_pixels)
    @test all(isnan, only(itemdata(apply(SanitizeKeypoints(), no_pixels))))
end

@testset "Keypoint transforms with other items" begin
    image = Image(zeros(8, 8))
    mask = MaskBinary(trues(8, 8))
    box = BoundingBox([SVector(1.0, 1.0), SVector(8.0, 8.0)], (8, 8))
    polygon = Polygon([SVector(1.0, 1.0), SVector(8.0, 1.0), SVector(8.0, 8.0)], (8, 8))
    others = (image, mask, box, polygon)
    crop = CenterCrop((4, 4))
    for tfm in (ClampKeypoints(), SanitizeKeypoints())
        @test apply(tfm, others) === others
        for item in (box, polygon)
            @test apply(Maybe(tfm, 1.0), item) === item
            @test apply(Buffered(tfm), item) === item
            @test itemdata(apply(BufferedThreadsafe(tfm), item)) == itemdata(item)
            expected = apply(crop, item)
            pipeline = crop |> tfm
            @test itemdata(apply(pipeline, item)) == itemdata(expected)
            @test itemdata(apply(Buffered(pipeline), item)) == itemdata(expected)
        end
    end
end
