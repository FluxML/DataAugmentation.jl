
"""
    testapply(tfm, item)
    testapply(tfm, I)

Test `apply` invariants of `tfm` on `item` or item type `I`.

1. With a constant `randstate` parameter, `apply` should always return the
    same result.
"""
function testapply(tfm::Transform, item::AbstractItem)
    # Invariant 1
    r = getrandstate(tfm)
    titems = [apply(tfm, item; randstate = r) for i = 1:8]
    @test all(map(titem -> itemdata(titem) == itemdata(titems[1]), titems))

    #
    @test_nowarn apply(tfm, (item, item))
end

testapply(tfm::Transform, I::Type{<:AbstractItem}) = testapply(tfm, testitem(I))
testapply(tfm, items::Tuple) = foreach(i -> testapply(tfm, i), items)


"""
    testapply!(tfm, Items)
    testapply!(tfm, Item)
    testapply!(tfm, item1, item2)

Test `apply!` invariants.

1. With a constant `randstate` parameter, `apply!` should always return the
    same result.
2. Given a different item than was used to create the buffer, the buffer's data
    should be modified.
"""
function testapply!(tfm::Transform, item1::I, item2::I) where I<:AbstractItem
    # Invariant 1
    r = getrandstate(tfm)
    titems = [apply(tfm, deepcopy(item1); randstate = r) for i = 1:8]
    @test allequal(itemdata.(titems))

    # Invariant 2
    buf = makebuffer(tfm, item1)
    cbuf = deepcopy(buf)
    apply!(buf, tfm, item2)
    if buf isa AbstractItem
        @test itemdata(buf) != itemdata(cbuf)
    end
end

testapply!(tfm::Transform, I::Type{<:AbstractItem}) = testapply!(tfm, testitem(I), testitem(I))
testapply!(tfm, items::Tuple) = foreach(i -> testapply!(tfm, i), items)



"""
    testprojective(tfm)

Test invariants of a `ProjectiveTransform`.

1. `getprojection` is defined, and, given a constant `randstate` parameter,
    always returns the same result.
2. It preserves the item type, i.e. `apply(tfm, ::I) -> I`.
3. Applying it to multiple items with the same bounds results in the same bounds
    for all items.
"""
function testprojective(tfm::ProjectiveTransform, items::Tuple)
    # All bounds must be equal for the test to work
    @assert allequal(getbounds.(items))

    # Invariant 1
    r = getrandstate(tfm)
    bs = getbounds(items[1])
    Ps = [getprojection(tfm, bs; randstate = r) for i = 1:10]
    @test allequal(Ps)

    # Invariant 2
    titems = apply(tfm, items)
    @test all(typeof.(items) .== typeof.(titems))

    # Invariant 3
    @test allequal(getbounds.(titems))
end


function testprojective(tfm::ProjectiveTransform, Is::NTuple{N, <:Type}) where {N}
    testprojective(tfm, testitem.(Is))
end

testprojective(tfm::ProjectiveTransform) =
    testprojective(tfm, (Image, MaskBinary, MaskMulti, Keypoints))

"""
    testitem(TItem)

Create an instance of an item with type `TItem`. If it has spatial bounds,
should return an instance with bounds with ranges (1:16, 1:16).
"""
function testitem end


testitem(::Type{ArrayItem}) = testitem(ArrayItem{2, Float32})
testitem(::Type{ArrayItem{N, T}}) where {N, T} = ArrayItem(rand(T, ntuple(i -> 16, N)))

testitem(::Type{Image}; kwargs...) = testitem(Image{2, RGB{N0f8}}; kwargs...)
testitem(::Type{Image{N}}; kwargs...) where {N} = testitem(Image{N, RGB{N0f8}}; kwargs...)
testitem(::Type{Image{N, T}}; kwargs...) where {N, T} = Image(rand(T, ntuple(i -> 16, N)); kwargs...)

testitem(::Type{MaskBinary}; kwargs...) = testitem(MaskBinary{2}; kwargs...)
testitem(::Type{MaskBinary{N}}; kwargs...) where {N} = MaskBinary(rand(Bool, ntuple(i -> 16, N)); kwargs...)

testitem(::Type{MaskMulti}; kwargs...) = testitem(MaskMulti{2, UInt8}; kwargs...)
testitem(::Type{MaskMulti{N}}; kwargs...) where {N} = testitem(MaskMulti{N, UInt8}; kwargs...)
function testitem(::Type{MaskMulti{N, T}}; kwargs...) where {N, T}
    n = rand(2:10)
    data = T.(rand(1:n, ntuple(i -> 16, N)))
    MaskMulti(data, 1:n; kwargs...)
end


testitem(::Type{Keypoints}) = testitem(Keypoints{2, Float32})
function testitem(::Type{Keypoints{N, T}}) where {N, T}
    n = rand(8:16)
    data = map(v -> v .* 15, rand(SVector{N, T}, n))
    return Keypoints(data, (16, 16))
end

testitem(::Type{Polygon}) = testitem(Polygon{2, Float32})
function testitem(::Type{Polygon{N, T}}) where {N, T}
    n = rand(8:16)
    data = map(v -> v .* 15, rand(SVector{N, T}, n))
    return Polygon(Keypoints(data, (16, 16)))
end

testitem(::Type{BoundingBox}) = testitem(BoundingBox{2, Float32})
function testitem(::Type{BoundingBox{N, T}}) where {N, T}
    n = 2
    data = map(v -> v .* 15, rand(SVector{N, T}, n))
    return BoundingBox(Keypoints(data, (16, 16)))
end

allequal(xs) = all(x == xs[1] for x in xs)
