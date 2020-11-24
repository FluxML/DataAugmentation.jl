include("./imports.jl")


# TODO: test apply!(Sequential)

a = randn(5, 5)
item = ArrayItem(a)

tfm = Map(x -> x + 1)

buf = apply(tfm, item)

apply!(buf, tfm, item)

DataAugmentation.copyitemdata!([item], [buf])
item
buf

@testset ExtendedTestSet "apply!(buf, ::Map, ...)" begin
    newitem() = ArrayItem(randn(5, 5))
    tfm = Map(x -> x + 1)

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


@testset ExtendedTestSet "Inplace" begin
    newitem() = ArrayItem(randn(5, 5))
    # buffer should be created
    ti = DataAugmentation.Inplace(Map(x -> x + 1))
    @test isnothing(ti.buffer)
    @test_nowarn ti(newitem())
    @test !isnothing(ti.buffer)

    # buffer should change
    buf = deepcopy(ti.buffer)
    @test_nowarn ti(newitem())
    @test !(buf.data ≈ ti.buffer.data)

    # inplace version
    buf2 = deepcopy(buf)
    @test_nowarn ti(buf2, newitem())
    @test !(buf2.data ≈ buf.data)

end

@testset ExtendedTestSet "Inplace" begin
    newitem() = ArrayItem(randn(5, 5))
    # buffer should be created
    ti = DataAugmentation.InplaceThreadsafe(Map(x -> x + 1))
    @test isnothing(ti.inplaces[Threads.threadid()].buffer)
    @test_nowarn ti(newitem())
    @test !isnothing(ti.inplaces[Threads.threadid()].buffer)

    # buffer should change
    buf = deepcopy(ti.inplaces[Threads.threadid()].buffer)
    @test_nowarn ti(newitem())
    @test !(buf.data ≈ ti.inplaces[Threads.threadid()].buffer.data)

    # inplace version
    buf2 = deepcopy(buf)
    @test_nowarn ti(buf2, newitem())
    @test !(buf2.data ≈ buf.data)

end

tfm = Map(x -> x + 1)
DataAugmentation.InplaceThreadsafe(tfm)
