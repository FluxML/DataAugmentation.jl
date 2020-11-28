include("./imports.jl")


# TODO: test apply!(Sequential)

a = randn(5, 5)
item = ArrayItem(a)

tfm = MapElem(x -> x + 1)

buf = apply(tfm, item)

apply!(buf, tfm, item)

DataAugmentation.copyitemdata!([item], [buf])
item
buf

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
    tb = Buffered(MapElem(x -> x + 1))
    @test isnothing(tb.buffer)
    @test_nowarn apply(tb, newitem())
    @test !isnothing(tb.buffer)

    # buffer should change
    buf = deepcopy(tb.buffer)
    @test_nowarn apply(tb, newitem())
    @test !(buf[1].data ≈ tb.buffer[1].data)

    # inplace version
    buf2 = deepcopy(buf)
    @test_nowarn apply!(buf2, tb, newitem())
    @test !(buf2[1].data ≈ tb.buffer[1].data)
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
    @test !(buf[1].data ≈ tb.buffer[1].data)

    # inplace version
    buf2 = deepcopy(buf)
    @test_nowarn apply!(buf2, tbt, newitem())
    @test !(buf2[1].data ≈ tb.buffer[1].data)
end
