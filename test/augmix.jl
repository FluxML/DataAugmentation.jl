include("imports.jl")

using DataAugmentation: AugmentationOp

@testset "AugMix" begin
    tfm = AugMix()
    @test length(tfm.ops) == 13 && length(AugMix(all_ops = false).ops) == 9
    @test tfm.ops[1].magnitudes[end] ≈ 0.3 && tfm.ops[3].magnitudes[end] ≈ 1 / 3
    posterize = only(op for op in tfm.ops if op.name == :Posterize)
    @test posterize.magnitudes == [4, 4, 3, 3, 2, 2, 1, 1, 0, 0]
    for kwargs in [(severity = 0,), (severity = 11,), (mixture_width = 0,), (chain_depth = 0,),
                   (alpha = 0,), (alpha = Inf,), (fill = "black",)]
        @test_throws ArgumentError AugMix(; kwargs...)
    end

    state = getrandstate(tfm)
    @test state.original + sum(state.weights) ≈ 1
    @test length(state.weights) == length(state.chains) == 3
    @test Set(length(chain) for _ in 1:50 for chain in getrandstate(tfm).chains) == Set(1:3)
    @test all(chain -> length(chain) == 2, getrandstate(AugMix(chain_depth = 2, mixture_width = 5)).chains)

    # Only the `severity` weakest magnitudes are used
    mild = AugMix(severity = 1)
    for _ in 1:20, chain in getrandstate(mild).chains, (op, magnitude) in chain
        @test abs(magnitude) == (isempty(op.magnitudes) ? 0 : op.magnitudes[1])
    end
end

@testset "AugMix mixing" begin
    image = Image(rand(Float32, 8, 10))
    invert = AugmentationOp(:Invert, Float64[], false, false)
    none = Tuple{AugmentationOp, Float64}[]
    state = (original = 0.25, weights = [0.5, 0.25], chains = [[(invert, 0.0)], none])
    out = apply(AugMix(), image; randstate = state)
    @test itemdata(out) ≈ 0.25 .* itemdata(image) .+ 0.5 .* (1 .- itemdata(image)) .+ 0.25 .* itemdata(image)
    @test itemdata(apply(AugMix(), image; randstate = state)) == itemdata(out)
    @test itemdata(apply(Buffered(AugMix()), image; randstate = state)) == itemdata(out)

    unchanged = (original = 1.0, weights = [0.0, 0.0], chains = [[(invert, 0.0)], none])
    @test itemdata(apply(AugMix(), image; randstate = unchanged)) ≈ itemdata(image)

    # Pixel types are preserved, and rounding errors in the weights do not overflow them
    colors = Image(rand(RGBA{N0f8}, 8, 10))
    out = apply(AugMix(severity = 10), colors)
    @test eltype(itemdata(out)) == RGBA{N0f8} && alpha.(itemdata(out)) == alpha.(itemdata(colors))
    white = Image(fill(Gray{N0f8}(1), 6, 6))
    thirds = (original = 0.1, weights = [0.3, 0.3, 0.3], chains = [none, none, none])
    @test all(==(Gray{N0f8}(1)), itemdata(apply(AugMix(), white; randstate = thirds)))
    for _ in 1:30
        @test apply(AugMix(severity = 10), white) isa Image{2, Gray{N0f8}}
    end

    offset = Image(OffsetArray(rand(Float32, 6, 6), -2:3, 0:5), Bounds((-2:3, 0:5)))
    @test axes(itemdata(apply(AugMix(), offset))) == axes(itemdata(offset))

    @test_throws ArgumentError apply(AugMix(), Image(rand(Float32, 4, 4, 4)))
    @test_throws ArgumentError apply(AugMix(), MaskBinary(trues(4, 4)))
    @test_throws ArgumentError apply(AugMix(), (image, Keypoints([SVector(1.0, 1.0)], (8, 10))))
end
