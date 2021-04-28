include("./imports.jl")


@testset ExtendedTestSet "apply!(buf, ::Map, ...)" begin
    newitem() = ArrayItem(randn(5, 5))
    tfm = MapElem(x -> x + 1)

    buf = apply(tfm, newitem())
    buf_copy = deepcopy(buf)

    @test buf.data ≈ buf_copy.data
    @test_nowarn apply!(buf, tfm, newitem())
    @test !(buf.data ≈ buf_copy.data)
end


@testset ExtendedTestSet "copyitemdata!" begin
    newitem() = ArrayItem(randn(5, 5))
    @testset ExtendedTestSet "" begin
        a = newitem()
        b = newitem()
        @test !(a.data ≈ b.data)
        DataAugmentation.copyitemdata!(a, b)
        @test a.data ≈ b.data
    end


    @testset ExtendedTestSet "" begin
        a = newitem()
        b = newitem()
        @test !(a.data ≈ b.data)
        DataAugmentation.copyitemdata!([a], [b])
        @test a.data ≈ b.data
    end

    @testset ExtendedTestSet "" begin
        a = newitem()
        b = newitem()
        @test !(a.data ≈ b.data)
        DataAugmentation.copyitemdata!((a,), (b,))
        @test a.data ≈ b.data
    end
end


@testset ExtendedTestSet "`Buffered`" begin
    newitem() = ArrayItem(randn(5, 5))
    # buffer should be created
    tb = Buffered(MapElem(x -> x + one(typeof(x))))
    @test isnothing(tb.buffer)
    @test_nowarn apply(tb, newitem())
    @test !isnothing(tb.buffer)

    # buffer should change
    buf = deepcopy(tb.buffer)
    @test_nowarn apply(tb, newitem())
    @test !(buf.data ≈ tb.buffer.data)

    # inplace version
    buf2 = deepcopy(buf)
    @test_nowarn apply!(buf, tb, newitem())
    @test !(buf2.data ≈ tb.buffer.data)
    testapply(Buffered(MapElem(x -> x + one(typeof(x)))), ArrayItem)
end

@testset ExtendedTestSet "`BufferedThreadsafe`" begin
    newitem() = ArrayItem(randn(5, 5))
    # buffer should be created
    tbt = BufferedThreadsafe(MapElem(x -> x + 1))
    tb = tbt.buffereds[1]
    @test isnothing(tb.buffer)
    @test_nowarn apply(tbt, newitem())
    @test !isnothing(tb.buffer)

    # buffer should change
    buf = deepcopy(tb.buffer)
    @test_nowarn apply(tbt, newitem())
    @test !(buf.data ≈ tb.buffer.data)

    # inplace version
    buf2 = deepcopy(buf)
    @test_nowarn apply!(buf, tbt, newitem())
    @test !(buf2.data ≈ tb.buffer.data)

    testapply(BufferedThreadsafe(MapElem(x -> x + one(typeof(x)))), ArrayItem)
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

@testset ExtendedTestSet "" begin
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
