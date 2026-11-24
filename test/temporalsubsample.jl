include("./imports.jl")

@testset "UniformTemporalSubsample" begin
    # frame `t` of a test video holds the value `t`, so the values show which frames were kept
    framevalues(item, dim = 3) = [only(unique(selectdim(itemdata(item), dim, t))) for t in axes(itemdata(item), dim)]
    video(T; sz = (3, 2)) = cat((fill(Float32(t), sz...) for t in 1:T)...; dims = 3)

    @testset "frame selection" begin
        for (T, n, expected) in [
            (10, 4, [1, 4, 7, 10]),
            (10, 5, [1, 3, 5, 7, 10]),
            (10, 1, [1]),
            (10, 2, [1, 10]),
            (10, 10, 1:10),
            (3, 5, [1, 1, 2, 2, 3]),
            (1, 3, [1, 1, 1]),
        ]
            res = apply(UniformTemporalSubsample(n), ArrayItem(video(T)))
            @test framevalues(res) == expected
        end
    end

    @testset "items" begin
        tfm = UniformTemporalSubsample(4)

        tensor = ArrayItem(cat((fill(Float32(t), 3, 2, 1, 5) for t in 1:8)...; dims = 3))
        @test size(itemdata(apply(tfm, tensor))) == (3, 2, 4, 5)
        @test framevalues(apply(tfm, tensor)) == [1, 3, 5, 8]

        image = Image(reshape(Gray{Float32}.(video(8)), 3, 2, 8))
        res = apply(tfm, image)
        @test res isa Image{3}
        @test getbounds(res) == Bounds((3, 2, 4))
        @test getbounds(image) == Bounds((3, 2, 8))
        @test gray.(itemdata(res))[1, 1, :] == [1, 3, 5, 8]

        mask = MaskMulti(UInt8.(rand(1:3, 3, 2, 8)), 1:3)
        pair = apply(tfm, (image, mask))
        @test map(getbounds, pair) == (Bounds((3, 2, 4)), Bounds((3, 2, 4)))
        @test itemdata(pair[2]) == itemdata(mask)[:, :, [1, 3, 5, 8]]
        @test pair[2].classes == mask.classes

        binary = MaskBinary(rand(Bool, 3, 2, 8))
        @test itemdata(apply(tfm, binary)) == itemdata(binary)[:, :, [1, 3, 5, 8]]

        offset = Image(OffsetArray(rand(Float32, 3, 2, 8), 4:6, 1:2, 10:17))
        res = apply(tfm, offset)
        @test getbounds(res) == Bounds(axes(itemdata(res)))
        @test axes(itemdata(res), 1) == 4:6
        @test size(itemdata(res), 3) == 4
        @test itemdata(res)[:, :, end] == itemdata(offset)[:, :, 17]
    end

    @testset "dim" begin
        res = apply(UniformTemporalSubsample(3; dim = 1), ArrayItem(video(5; sz = (1, 2))))
        @test size(itemdata(res)) == (3, 2, 5)
        data = permutedims(video(7), (3, 1, 2))
        @test framevalues(apply(UniformTemporalSubsample(3; dim = 1), ArrayItem(data)), 1) == [1, 4, 7]
    end

    @testset "apply!" begin
        tfm = UniformTemporalSubsample(4)
        item1 = ArrayItem(rand(Float32, 3, 2, 8))
        item2 = ArrayItem(rand(Float32, 3, 2, 8))
        testapply(tfm, item1)
        testapply!(tfm, item1, item2)
        buf = makebuffer(tfm, item1)
        res = apply!(buf, tfm, item2)
        @test res === buf
        @test itemdata(res) == itemdata(apply(tfm, item2))

        image1 = Image(rand(Float32, 3, 2, 8))
        image2 = Image(rand(Float32, 3, 2, 8))
        testapply!(tfm, image1, image2)
        imagebuf = makebuffer(tfm, image1)
        @test apply!(imagebuf, tfm, image2) === imagebuf

        # a buffer of the wrong size or one aliasing the input falls back to `apply`
        wrong = ArrayItem(zeros(Float32, 3, 2, 3))
        @test itemdata(apply!(wrong, tfm, item2)) == itemdata(apply(tfm, item2))
        same = ArrayItem(copy(itemdata(item2)))
        @test itemdata(apply!(same, UniformTemporalSubsample(8), same)) == itemdata(item2)

        buffered = Buffered(tfm)
        @test itemdata(apply(buffered, item1)) == itemdata(apply(tfm, item1))
        @test itemdata(apply(buffered, item2)) == itemdata(apply(tfm, item2))
    end

    @testset "errors" begin
        @test_throws ArgumentError UniformTemporalSubsample(0)
        @test_throws ArgumentError UniformTemporalSubsample(2; dim = 0)
        @test_throws ArgumentError apply(UniformTemporalSubsample(2), ArrayItem(rand(3, 3)))
        @test_throws ArgumentError apply(UniformTemporalSubsample(2), ArrayItem(rand(3, 3, 0)))
    end
end
