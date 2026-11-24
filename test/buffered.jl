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
    T = typeof(newitem())
    # buffer should be created
    tbt = BufferedThreadsafe(MapElem(x -> x + 1))
    tb = tbt.buffereds[1]
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

    testapply(BufferedThreadsafe(MapElem(x -> x + one(typeof(x)))), ArrayItem)
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
