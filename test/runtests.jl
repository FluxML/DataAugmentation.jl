using DataAugmentation
using DataAugmentation:
    Affine, ApplyStep, CropFixed, CropRatio, Sequential,
    makebuffer, getrandstate,
    applyaffine, getaffine, applystep!, getwrapped, getcropsizes, index_ranges,
    ScaleFixed, ScaleRatio, ScaleKeepAspect, cropindices,
    CropFromOrigin, CropFromCenter, CropFromRandom
import DataAugmentation: apply, apply!, compose, makebuffer
using Images
using StaticArrays
using Test
using TestSetExtensions
using CoordinateTransformations

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


@testset ExtendedTestSet "Crop" begin
    item = Image(rand(RGB, 50, 100))
    bounds = getbounds(item)

    @testset ExtendedTestSet "cropindices" begin
        c = CropFixed((50, 50), CropFromRandom())
        r = (rand(), rand())
        @test cropindices((50, 50), CropFromOrigin(), bounds, r) == (1:50, 1:50)
        @test cropindices((50, 50), CropFromCenter(), bounds, r) == (1:50, 26:75)
        @test cropindices((50, 50), CropFromRandom(), bounds, (1., 1.)) == (1:50, 51:100)
    end
end


@testset ExtendedTestSet "Affine helpers" begin
    @testset ExtendedTestSet "index_ranges" begin
        keypoints = Keypoints(
            [SVector(1, 1)],
            [SVector(0, 0), SVector(50, 0), SVector(50, 50), SVector(0, 50)]
        )
        image = Image(rand(RGB, 50, 50))
        @test index_ranges(getbounds(keypoints)) == index_ranges(getbounds(image))
    end
end


@testset ExtendedTestSet "Affine" begin
    A = CoordinateTransformations.Translation(10, 10)
    tfm = Affine(A)

    keypoints = Keypoints(
            [SVector(1, 1)],
            [SVector(0, 0), SVector(50, 0), SVector(50, 50), SVector(0, 50)]
    )
    image = Image(rand(RGB, 50, 50))


    @testset ExtendedTestSet "Affine Keypoints" begin
        @test getaffine(tfm, keypoints, nothing) == A
        @test_nowarn applyaffine(keypoints, getaffine(tfm, keypoints, nothing))
        tkeypoints = apply(tfm, keypoints)
        @test itemdata(tkeypoints) == [SVector(11, 11)]
        @test getbounds(tkeypoints)[1] == SVector(10, 10)
    end

    @testset ExtendedTestSet "Affine Image" begin
        @test getaffine(tfm, image, nothing) == A
        @test_nowarn applyaffine(image, getaffine(tfm, image, nothing))
        timage = apply(tfm, image)
        @test itemdata(timage)[11, 11] ≈ itemdata(image)[1, 1]
        @test getbounds(timage)[1] == SVector(10, 10)
    end

    tfmcropped = CroppedAffine(tfm, Crop(50, 50))

    @testset ExtendedTestSet "Affine Cropped Image" begin
        @test_nowarn apply(tfmcropped, keypoints)
        tkeypoints = apply(tfmcropped, keypoints)
        @test getbounds(tkeypoints)[1] == SVector(0, 0)
    end

    @testset ExtendedTestSet "Affine Cropped Image" begin
        @test_nowarn apply(tfmcropped, image)
        timage = apply(tfmcropped, image)
        @test getbounds(timage)[1] == SVector(0, 0)
    end

    @testset ExtendedTestSet "Composed" begin
        tfms = Affine(Translation(10, 10)) |> Affine(Translation(-10, -10))
        randstate = getrandstate(tfms)
        @test getaffine(tfms, getbounds(image), randstate) == Translation(0, 0)
        @test itemdata(apply(tfms, image)) == itemdata(image)
        @test itemdata(apply(tfms, keypoints)) == itemdata(keypoints)
    end


    @testset ExtendedTestSet "ScaleKeepAspect" begin
        image = Image(zeros(RGB, 100, 50))

        keypoints = Keypoints(
            [SVector(0, 0)],
            [SVector(0, 0), SVector(50, 0), SVector(50, 50), SVector(0, 50)]
        )

        imbounds = getbounds(image)
        kpbounds = getbounds(keypoints)
        tfm1 = ScaleKeepAspect(50)
        @test getaffine(tfm1, imbounds, nothing) ≈ getaffine(ScaleRatio((1, 1)), imbounds, nothing)
        @test getaffine(tfm1, kpbounds, nothing) ≈ getaffine(ScaleRatio((1, 1)), kpbounds, nothing)

        tfm2 = ScaleKeepAspect(25)
        @test getaffine(tfm2, imbounds, nothing) ≈ getaffine(ScaleRatio((1 / 2, 1 / 2)), imbounds, nothing)
        @test getaffine(tfm2, kpbounds, nothing) ≈ getaffine(ScaleRatio((1 / 2, 1 / 2)), kpbounds, nothing)
    end

    @testset ExtendedTestSet "Resize" begin
        keypoints = Keypoints(
            [SVector(0, 0)],
            [SVector(0, 0), SVector(50, 0), SVector(50, 50), SVector(0, 50)]
        )
        image = Image(rand(RGB, 50, 50))
        @testset ExtendedTestSet "ResizeFixed" begin
            tfm = ResizeFixed((25, 25))
            @test_nowarn apply(tfm, image)
            @test_nowarn apply(tfm, keypoints)
            timage = apply(tfm, image)
            tkeypoints = apply(tfm, keypoints)
            @test index_ranges(getbounds(timage)) == (1:25, 1:25)
            @test getbounds(timage) == getbounds(tkeypoints)

        end
        @testset ExtendedTestSet "ResizeRatio" begin
            tfm = ResizeRatio((1/2, 1/2))
            @test_nowarn apply(tfm, image)
            @test_nowarn apply(tfm, keypoints)
            timage = apply(tfm, image)
            tkeypoints = apply(tfm, keypoints)
            @test index_ranges(getbounds(timage)) == (1:25, 1:25)
            @test getbounds(timage) == getbounds(tkeypoints)
        end
    end

    @testset ExtendedTestSet "RandomResizeCrop" begin
        tfm = CenterResizeCrop((25, 40))
        keypoints = Keypoints(
            [SVector(0, 0)],
            [SVector(0, 0), SVector(50, 0), SVector(50, 50), SVector(0, 50)]
        )
        image = Image(rand(RGB, 50, 50))
        timage, tkeypoints = apply(tfm, (image, keypoints))
        @test getbounds(timage) == getbounds(tkeypoints)
        @test index_ranges(getbounds(timage)) == (1:25, 1:40)
    end

    @testset ExtendedTestSet "CroppedAffine inplace" begin
        image1 = Image(rand(RGB, 100, 100))
        image2 = Image(rand(RGB, 100, 100))
        tfm = Buffered(ResizeFixed((50, 50)), Image)
        buffer = makebuffer(tfm, image1)

        @test_nowarn apply!(buffer, tfm, image2)
    end
end


#= TODO:
- test Keypoints transformation with Nothing

=#
