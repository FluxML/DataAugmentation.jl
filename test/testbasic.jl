include("./imports.jl")

struct TestItem <: Item
    data::Any
end

struct TestItemWrapper <: ItemWrapper{TestItem}
    item::TestItem
end

struct Add <: Transform
    n::Int
end
apply(tfm::Add, item::TestItem; randstate=getrandstate(tfm)) =
    TestItem(item.data + tfm.n)

compose(add1::Add, add2::Add) = Add(add1.n + add2.n)

@testset ExtendedTestSet "basic `apply` and `itemdata`" begin
    data = 1
    tfm = Add(0)
    item = TestItem(data)
    @testset ExtendedTestSet "`apply` single item" begin
        @test_nowarn apply(tfm, item)
        @test_nowarn apply(tfm, item, randstate=nothing)
    end

    @testset ExtendedTestSet "`apply` multiple items" begin
        @test_nowarn apply(tfm, (item, item); randstate=nothing)
    end

    @testset ExtendedTestSet "`itemdata` single item" begin
        res1 = apply(tfm, item)
        @test itemdata(res1) == data
    end

    @testset ExtendedTestSet "`itemdata` multiple items" begin
        res2 = apply(tfm, (item, item))
        @test res2 == (item, item)
        @test itemdata(res2) == (data, data)

    end
end


@testset ExtendedTestSet "`apply` ItemWrapper" begin
    data = 1
    tfm = Identity()
    item = TestItem(data)
    itemw = TestItemWrapper(item)

    @test_nowarn apply(tfm, itemw)

    res = apply(tfm, itemw)
    @test res isa TestItemWrapper
    @test itemdata(res) == data

end


@testset ExtendedTestSet "`compose`" begin
    data = 1
    item = TestItem(data)

    @test compose(Identity(), Identity()) == Identity()
    @test compose(Identity(), Add(10)) == Add(10)
    @test compose(Add(10), Add(10)) == Add(20)
    @test compose(Add(10), Add(10)) == Add(10) |> Add(10)

    t = Add(10)
    @test compose(t, t, t, t) == t |> t |> t |> t

end

@testset ExtendedTestSet "setdata, setwrapped" begin
    sz = (50, 50)
    keypoints1 = Keypoints([SVector(1., 1)], sz)
    keypoints2 = Keypoints([SVector(2., 2)], sz)

    @testset ExtendedTestSet "setdata" begin
        keypoints3 = setdata(keypoints1, itemdata(keypoints2))
        @test itemdata(keypoints3) == itemdata(keypoints2)
    end

    @testset ExtendedTestSet "setwrapped" begin
        bbox1 = BoundingBox(keypoints1)
        bbox3 = setwrapped(bbox1, keypoints1)
        @test getwrapped(bbox3) == getwrapped(bbox1)
    end
end

@testset ExtendedTestSet "Sequential" begin
    seq = Sequential([Add(10), Add(10)])
    @test_nowarn apply(seq, TestItem(10))
    @test_nowarn apply(seq, (TestItem(10),))

end


@testset ExtendedTestSet "`apply!`" begin
    data = rand(Float32, 10, 10)
    item = ArrayItem(data)
    tfm = MapElem(x->round(Int, x))
    buftfm = Inplace(tfm)

    @testset ExtendedTestSet "`makebuffer`" begin
        buffer = makebuffer(buftfm, item)
        @test eltype(itemdata(buffer)) === Int
    end

    @testset ExtendedTestSet "`apply!` default" begin
        buffer = makebuffer(tfm, item)
        @test_nowarn apply!(buffer, tfm, item)
    end

    @testset ExtendedTestSet "`apply!` default" begin
        buffer::ArrayItem = makebuffer(buftfm, item)
        a = itemdata(buffer)
        a[1] = -1
        @test_nowarn apply!(buffer, buftfm, item)
        # buffer should have been mutated
        @test itemdata(buffer)[1] != -1
    end

    @testset ExtendedTestSet "`apply!` pipeline" begin
        item1 = ArrayItem(rand(Float32, 10, 10))
        item2 = ArrayItem(rand(Float32, 10, 10))
        tfm1 = MapElem(round)
        tfm2 = MapElem(Int)
        pipeline = Inplace(tfm1) |> Inplace(tfm2)
        buffers = makebuffer(pipeline, item1)
        a = copy(itemdata(buffers[2]))
        @test_nowarn apply!(buffers, pipeline, item2)
        @test itemdata(buffers[2]) != a
    end
end
