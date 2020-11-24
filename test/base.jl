include("imports.jl")  #src
# ## Testing `base.jl`
# Since we haven't defined any items or real transforms
# yet, we test the interface with test versions of each.

struct TestItem <: Item
    data::Any
end

struct Add <: Transform
    n::Int
end

apply(tfm::Add, item::TestItem; randstate=getrandstate(tfm)) =
    TestItem(item.data + tfm.n)

compose(add1::Add, add2::Add) = Add(add1.n + add2.n)

# Now we can perform some sanity checks for the basic interface.

@testset ExtendedTestSet "Basic interface" begin
    tfm = Add(0)
    item = TestItem(10)

    @test itemdata(item) == 10
    @test itemdata(apply(tfm, item)) === itemdata(item)
    @test itemdata(apply(tfm, (item, item))) === itemdata((item, item))
    @test isnothing(getrandstate(tfm))
end

# And composition:

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

@testset ExtendedTestSet "`Sequence`" begin
    seq = Sequence(Add(10), Add(10))
    @test apply(seq, TestItem(10)) == TestItem(30)
    @test apply(seq, (TestItem(10),)) == (TestItem(30),)
end


# `setdata` should copy, not modify.

@testset ExtendedTestSet "`setdata`" begin
    item = TestItem(10)
    @test setdata(item, 20) == TestItem(20)
    @test setdata(item, 20) !== item
end
