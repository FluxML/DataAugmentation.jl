include("imports.jl")

@testset "AutoAugment policies" begin
    for policy in (:imagenet, :cifar10, :svhn)
        tfm = AutoAugment(policy)
        @test length(tfm.subpolicies) == 25
        @test all(sub -> length(sub) == 2, tfm.subpolicies)
    end
    @test_throws ArgumentError AutoAugment(:coco)

    first_subpolicy = AutoAugment().subpolicies[1]
    @test [(op.name, p, m) for (op, p, m) in first_subpolicy] == [(:Posterize, 0.4, 4), (:Rotate, 0.6, 30)]
    last_subpolicy = AutoAugment(:cifar10).subpolicies[end]
    @test [(op.name, p) for (op, p, _) in last_subpolicy] == [(:TranslateY, 0.7), (:AutoContrast, 0.9)]
    @test last_subpolicy[1][3] ≈ 150 / 331
end

@testset "AutoAugment" begin
    image = Image(rand(RGB{Float32}, 12, 16))

    # Solarizing at threshold zero inverts the colors, and inverting again restores them
    tfm = AutoAugment([((:Solarize, 1.0, 9), (:Invert, 1.0, nothing))])
    state = getrandstate(tfm)
    @test [op.name for (op, _) in state] == [:Solarize, :Invert]
    @test itemdata(apply(tfm, image)) ≈ itemdata(image)

    never = AutoAugment([((:Rotate, 0.0, 5), (:Equalize, 0.0, nothing))])
    @test isempty(getrandstate(never))
    @test apply(never, image) === image

    # Signs of signed operations are random, magnitudes fixed
    shear = AutoAugment([((:ShearX, 1.0, 9), (:Brightness, 1.0, 1))])
    magnitudes = [m for _ in 1:100 for (_, m) in getrandstate(shear)]
    @test Set(magnitudes) == Set([0.3, -0.3, 0.1, -0.1])

    # One sub-policy is drawn per call, and used for every item
    two = AutoAugment([((:Invert, 1.0, nothing),), ((:Equalize, 1.0, nothing),)])
    @test Set(only(getrandstate(two))[1].name for _ in 1:50) == Set([:Invert, :Equalize])
    mask = MaskBinary(rand(Bool, 12, 16))
    out = apply(AutoAugment(:svhn), (image, mask))
    @test out[1] isa Image && out[2] isa MaskBinary

    @test_throws ArgumentError AutoAugment([])
    @test_throws ArgumentError AutoAugment([((:Blur, 1.0, 3),)])
    @test_throws ArgumentError AutoAugment([((:Rotate, 1.5, 3),)])
    @test_throws ArgumentError AutoAugment([((:Rotate, 1.0, 10),)])
    @test_throws ArgumentError AutoAugment([((:Rotate, 1.0, nothing),)])
    @test_throws ArgumentError AutoAugment([((:Equalize, 1.0, 3),)])
    @test_throws ArgumentError AutoAugment(fill = [1])

    for policy in (:imagenet, :cifar10, :svhn), _ in 1:20
        tfm = AutoAugment(policy)
        titem = apply(tfm, Image(rand(RGB{N0f8}, 16, 16)))
        @test eltype(itemdata(titem)) == RGB{N0f8}
    end
end
