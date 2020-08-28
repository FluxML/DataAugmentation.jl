include("./imports.jl")

struct TestItem <: Item
    data::Any
end

struct TestItemWrapper <: ItemWrapper
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
        @test_nowarn apply(tfm, [item, item]; randstate=nothing)
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
    bounds = [SVector(0, 0), SVector(50, 0), SVector(50, 50), SVector(0, 50)]
    keypoints1 = Keypoints([SVector(1, 1)], bounds)
    keypoints2 = Keypoints([SVector(2, 2)], bounds)

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
    @test_nowarn apply(seq, [TestItem(10)])

end


struct ArrayItem <: Item
    a::AbstractArray
end

struct MapArray <: Transform
    f
end

apply(tfm::MapArray, item::ArrayItem) = ArrayItem(map(tfm.f, item.a))
function apply!(buffer::ArrayItem, tfm::Buffered{MapArray}, item::ArrayItem; randstate=getrandstate(tfm))
    return ArrayItem(map!(tfm.transform.f, buffer.a, item.a))
end

@testset ExtendedTestSet "`apply!`" begin
    data = rand(Float32, 10, 10)
    item = ArrayItem(data)
    tfm = MapArray(x->round(Int, x))
    buftfm = Buffered(tfm)

    @testset ExtendedTestSet "`makebuffer`" begin
        @test_nowarn buffer = makebuffer(tfm, item)
        buffer = makebuffer(tfm, item)
        @test buffer === nothing

        buffer = makebuffer(buftfm, item)
        @test eltype(buffer.a) === Int
    end

    @testset ExtendedTestSet "`apply!` default" begin
        buffer::Nothing = makebuffer(tfm, item)
        @test_nowarn apply!(buffer, tfm, item)
    end

    @testset ExtendedTestSet "`apply!` default" begin
        buffer::ArrayItem = makebuffer(buftfm, item)
        buffer.a[1] = -1
        @test_nowarn apply!(buffer, buftfm, item)
        # buffer should have been mutated
        @test buffer.a[1] != -1
    end

    @testset ExtendedTestSet "`apply!` pipeline" begin
        item1 = ArrayItem(rand(Float32, 10, 10))
        item2 = ArrayItem(rand(Float32, 10, 10))
        tfm1 = MapArray(round)
        tfm2 = MapArray(Int)
        pipeline = Buffered(tfm1) |> Buffered(tfm2)
        buffers = makebuffer(pipeline, item1)
        a = copy(buffers[2].a)
        @test_nowarn apply!(buffers, pipeline, item2)
        @test buffers[2].a != a
    end

end


@testset ExtendedTestSet "`Pipeline`" begin
    @testset ExtendedTestSet "`ApplyStep`" begin
        sample = Dict(:x => TestItem(-10))
        step = ApplyStep(Add(10), :x)
        @test_nowarn applystep!(sample, step)
        @test itemdata(sample[:x]) == 0
    end

    @testset ExtendedTestSet "`CombineStep`" begin
        sample = Dict(:x => TestItem(-10), :y => TestItem(10))
        # combines to :z = :x + :y
        step = CombineStep((x, y)->TestItem(itemdata(x) + itemdata(y)), (:x, :y), :z)
        @test_nowarn applystep!(sample, step)
        @test itemdata(sample[:z]) == 0
    end

    @testset ExtendedTestSet "`MapStep`" begin
        sample = Dict(:x => TestItem(-10), :y => TestItem(10))
        step = MapStep(
            (x, y)->(TestItem(itemdata(x) + 10), TestItem(itemdata(y) - 10)),
            (:x, :y))
        @test_nowarn applystep!(sample, step)
        @test itemdata(sample[:x]) == itemdata(sample[:y])
    end

    @testset ExtendedTestSet "Pipeline" begin
        sample = Dict(:x => TestItem(-23), :y => TestItem(10))
        pipeline = Pipeline(
            ApplyStep(Add(13), :x),
            MapStep(
                (x, y)->(TestItem(itemdata(x) + 10), TestItem(itemdata(y) - 10)),
                (:x, :y)
            ),
            CombineStep((x, y)->TestItem(itemdata(x) + itemdata(y)), (:x, :y), :z),
        )
        sample = pipeline(sample)
        @test itemdata(sample[:z]) == 0

    end
end
