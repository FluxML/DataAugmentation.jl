include("./imports.jl")


@testset "apply!(buf, ::Map, ...)" begin
    newitem() = ArrayItem(randn(5, 5))
    tfm = MapElem(x -> x + 1)

    buf = apply(tfm, newitem())
    buf_copy = deepcopy(buf)

    @test buf.data ≈ buf_copy.data
    @test_nowarn apply!(buf, tfm, newitem())
    @test !(buf.data ≈ buf_copy.data)
end


@testset "copyitemdata!" begin
    newitem() = ArrayItem(randn(5, 5))
    @testset "" begin
        a = newitem()
        b = newitem()
        @test !(a.data ≈ b.data)
        DataAugmentation.copyitemdata!(a, b)
        @test a.data ≈ b.data
    end


    @testset "" begin
        a = newitem()
        b = newitem()
        @test !(a.data ≈ b.data)
        DataAugmentation.copyitemdata!([a], [b])
        @test a.data ≈ b.data
    end

    @testset "" begin
        a = newitem()
        b = newitem()
        @test !(a.data ≈ b.data)
        DataAugmentation.copyitemdata!((a,), (b,))
        @test a.data ≈ b.data
    end
end


@testset "`Buffered`" begin
    newitem() = ArrayItem(randn(5, 5))
    T = typeof(newitem())
    # buffer should be created
    tb = Buffered(MapElem(x -> x + one(typeof(x))))
    @test !haskey(tb.buffers, T)
    @test_nowarn apply(tb, newitem())
    @test haskey(tb.buffers, T)

    # buffer should change
    buf = deepcopy(tb.buffers[T])
    @test_nowarn apply(tb, newitem())
    @test !(buf.data ≈ tb.buffers[T].data)

    # inplace version
    buf2 = deepcopy(buf)
    @test_nowarn apply!(buf, tb, newitem())
    @test !(buf2.data ≈ tb.buffers[T].data)
    testapply(Buffered(MapElem(x -> x + one(typeof(x)))), ArrayItem)
end

@testset "`BufferedThreadsafe`" begin
    newitem() = ArrayItem(randn(5, 5))
    tfm = MapElem(x -> x + 1)
    tbt = BufferedThreadsafe(tfm; n = 2)
    @test length(tbt.pool) == 2

    # the result is a copy, not the buffer
    item1, item2 = newitem(), newitem()
    res1 = apply(tbt, item1)
    res1_copy = deepcopy(res1)
    res2 = apply(tbt, item2)
    @test res1 !== res2
    @test itemdata(res1) == itemdata(res1_copy)
    @test itemdata(res2) ≈ itemdata(item2) .+ 1
    @test length(tbt.pool) == 2  # buffers are returned to the pool

    # inplace version writes into the given buffer
    buf = apply(tfm, newitem())
    @test apply!(buf, tbt, item1) === buf
    @test itemdata(buf) ≈ itemdata(item1) .+ 1

    testapply(BufferedThreadsafe(MapElem(x -> x + one(typeof(x)))), ArrayItem)
end

@testset "Ownership of buffered results" begin
    tfm = MapElem(x -> x + 1) |> MapElem(x -> 2x)
    items = [ArrayItem(fill(Float64(i), 4, 4)) for i in 1:8]
    expected = [2 * (i + 1) for i in 1:8]

    @testset "Buffered without copy returns the buffer" begin
        btfm = Buffered(tfm)
        res = [apply(btfm, item) for item in items]
        @test all(r -> r === res[1], res)
    end

    @testset "$name" for (name, btfm) in (
            "Buffered(copy = true)" => Buffered(tfm; copy = true),
            "BufferedThreadsafe" => BufferedThreadsafe(tfm),
        )
        res = [apply(btfm, item) for item in items]
        @test [itemdata(r)[1] for r in res] == expected
        @test allunique(objectid(itemdata(r)) for r in res)

        # tuples of items and item wrappers
        bbox = BoundingBox([SVector(10.0, 10.0), SVector(40.0, 50.0)], (64, 64))
        ttfm = ScaleRatio((0.5, 0.5))
        tbtfm = btfm isa Buffered ? Buffered(ttfm; copy = true) : BufferedThreadsafe(ttfm)
        bboxes = [BoundingBox([SVector(10.0 * i, 10.0), SVector(40.0, 50.0)], (64, 64)) for i in 1:4]
        tres = [apply(tbtfm, b) for b in bboxes]
        @test all(itemdata(t) ≈ itemdata(apply(ttfm, b)) for (t, b) in zip(tres, bboxes))
        pair = [apply(tbtfm, (bboxes[i], bboxes[i + 1])) for i in 1:3]
        @test all(itemdata(p[1]) ≈ itemdata(apply(ttfm, bboxes[i])) for (i, p) in enumerate(pair))
    end
end

@testset "`BufferedThreadsafe` from many tasks" begin
    tfm = MapElem(x -> x + 1)
    btfm = BufferedThreadsafe(tfm; n = 1)
    items = [ArrayItem(fill(Float64(i), 8, 8)) for i in 1:64]
    res = Vector{Any}(undef, length(items))
    @sync for i in eachindex(items)
        Threads.@spawn res[i] = apply(btfm, items[i])
    end
    @test all(itemdata(res[i]) == fill(Float64(i + 1), 8, 8) for i in eachindex(items))
    @test length(btfm.pool) >= 1
end

@testset "Buffered item wrappers" begin
    bbox = BoundingBox([SVector(10.0, 10.0), SVector(40.0, 50.0)], (64, 64))
    tfm = ScaleRatio((0.5, 0.5))
    @testset "$(nameof(typeof(btfm)))" for btfm in (Buffered(tfm), BufferedThreadsafe(tfm))
        tbbox = apply(btfm, bbox)
        @test tbbox isa BoundingBox
        @test itemdata(tbbox) ≈ itemdata(apply(tfm, bbox))
    end
end

@testset "Buffered projective transform on items with wrappers" begin
    img = Image(rand(64, 64))
    bbox = BoundingBox([SVector(10.0, 10.0), SVector(40.0, 50.0)], (64, 64))
    poly = Polygon([SVector(10.0, 10.0), SVector(40.0, 50.0), SVector(5.0, 30.0)], (64, 64))
    items = (img, bbox, poly)
    tfm = ScaleRatio((0.5, 0.5))
    expected = apply(tfm, items)
    @testset "$(nameof(typeof(btfm)))" for btfm in (Buffered(tfm), BufferedThreadsafe(tfm))
        for _ in 1:2  # second call reuses the buffers
            titems = apply(btfm, items)
            @test typeof.(titems) == typeof.(expected)
            @test all(itemdata(t) ≈ itemdata(e) for (t, e) in zip(titems, expected))
        end

        buf = apply(tfm, items)
        res = apply!(buf, btfm, items)
        @test all(itemdata(t) ≈ itemdata(e) for (t, e) in zip(res, expected))
        @test all(itemdata(t) ≈ itemdata(e) for (t, e) in zip(buf, expected))
    end
end

@testset "apply! with buffered transform on item collections" begin
    items = (ArrayItem(rand(5, 5)), ArrayItem(rand(5, 5)))
    tfm = MapElem(x -> x + 1)
    @testset "$(nameof(typeof(btfm)))" for btfm in (Buffered(tfm), BufferedThreadsafe(tfm))
        titems = apply!(apply(tfm, items), btfm, items)
        @test all(itemdata(t) ≈ itemdata(i) .+ 1 for (t, i) in zip(titems, items))
    end
end


struct PlusRand <: Transform
end
DataAugmentation.getrandstate(::PlusRand) = rand()
function DataAugmentation.apply(tfm::PlusRand, item::DataAugmentation.AbstractItem; randstate = getrandstate(tfm))
    return DataAugmentation.setdata(item, map(x -> x + randstate, itemdata(item)))
end
function DataAugmentation.apply!(buf, tfm::PlusRand, item::DataAugmentation.AbstractItem; randstate = getrandstate(tfm))
    map!(x -> x + randstate, itemdata(buf), itemdata(item))
    return buf
end

@testset "" begin
    item1 = ArrayItem(ones(10))
    item2 = ArrayItem(ones(10))
    tfm = PlusRand()
    @test_nowarn apply(tfm, item1)
    titem1, titem2 = apply(tfm, (item1, item2))
    a1, a2 = itemdata.((titem1, titem2))
    b1, b2 = copy.((a1, a2))
    @test a1 == a2
    apply!((titem1, titem2), tfm, (item1, item2))
    @test a1 == a2
    @test a1 != b1
end
