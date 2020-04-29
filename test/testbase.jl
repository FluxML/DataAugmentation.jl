using Test
using TestSetExtensions
using DataAugmentation
using DataAugmentation: AbstractDictTransform, DictTransformApplyAll, DictTransformCombine, SampleTransformLambda
import DataAugmentation: getparam
using Images

@testset ExtendedTestSet "Item" begin
    kp = [(1, 1), (10, 10), (25, 40)]
    kpitem = Keypoints(kp, (50, 50))
    img = rand(RGB, 50, 50)
    imgitem = Image(img)
    tensor = randn(50, 50)
    tensoritem = Tensor(tensor)

    @testset ExtendedTestSet "itemdata" begin
        @test itemdata(kpitem) == kp
        @test itemdata(imgitem) == img
        @test itemdata(tensoritem) == tensor

    end
    @testset ExtendedTestSet "getbounds" begin
        @test getbounds(kpitem) == (50, 50)
        @test getbounds(imgitem) == (50, 50)
    end

end

@testset ExtendedTestSet "AbstractTransform" begin
    struct MyTransform <: AbstractTransform end
    DataAugmentation.getparam(::MyTransform) = rand()
    (::MyTransform)(_, param) = param

    @testset ExtendedTestSet "AbstractTransform interface" begin
        myt = MyTransform()
        item = Tensor([1])

        # test if
        r = rand()
        @test myt(item, r) == r

        # test separate applications of transform receive different `param`
        @test myt(item) != myt(item)

        # test tuple of items receive same `param`
        titems = myt((item, item))
        @test titems[1] == titems[2]
    end

    @testset ExtendedTestSet "Pipeline" begin
        item = Tensor([1])
        t = LambdaTransform(t -> Tensor(itemdata(t) .+ 1))
        @test itemdata(t(item)) == [2]

        # test composition of (t, t) and (pipe, t)
        pipe = t |> t |> t
        @test itemdata(pipe(item)) == [4]

    end
end



@testset ExtendedTestSet "DictTransform" begin
    @test AbstractDictTransform((:a, :b), identity) isa DictTransformApplyAll
    @test AbstractDictTransform(:a, identity) isa DictTransformApplyAll
    @test AbstractDictTransform((:a, :b), :c, identity) isa DictTransformCombine
    @test AbstractDictTransform(identity) isa SampleTransformLambda

    spipe = DictPipeline([
        ((:a, :b), :c, +),
        (:c, xs -> abs.(xs))
        ])
    @test spipe(Dict(:a => -1, :b => -1))[:c] == 2
end

@testset ExtendedTestSet "XYPipeline" begin
    pipe = XYPipeline(Image, Label)
end
