include("../imports.jl")

@testset "FiveCrop and TenCrop" begin
    data = reshape(Float32.(1:20), 4, 5)
    image = Image(data)
    mask = MaskBinary(isodd.(Int.(data)))
    five = FiveCrop((2, 3))
    crops, masks = apply(five, (image, mask))
    @test length(crops) == length(masks) == 5
    @test getbounds.(crops) == getbounds.(masks)
    @test parent(itemdata(crops[2])) == data[1:2, 3:5]
    @test parent(itemdata(crops[3])) == data[3:4, 1:3]
    @test parent(itemdata(crops[5])) == data[2:3, 2:4]

    ten = apply(TenCrop((2, 3)), image)
    @test length(ten) == 10
    @test itemdata(ten[6]) == data[1:2, 5:-1:3]
    @test itemdata(apply(TenCrop((2, 3); vertical_flip = true), image)[6]) == data[4:-1:3, 1:3]
    @test_throws ArgumentError apply(FiveCrop(6), image)

    box = BoundingBox([SVector(1., 1.), SVector(3., 4.)], (4, 5))
    @test all(==(Bounds((2, 3))), map(getbounds, apply(five |> PinOrigin(), box)))
end

@testset "RandomIoUCrop" begin
    data = reshape(Float32.(1:64), 8, 8)
    image = Image(data)
    mask = MaskMulti(Int.(data), 1:64)
    boxes = [
        BoundingBox([SVector(1., 1.), SVector(3., 3.)], (8, 8)),
        BoundingBox([SVector(6., 6.), SVector(8., 8.)], (8, 8)),
    ]
    items = (image, mask, boxes)
    tfm = RandomIoUCrop(min_scale = 0.5, max_scale = 0.5, sampler_options = (0.1,), trials = 100)
    cropped, labels, new_boxes = apply(tfm, items; randstate = 4)
    bounds = getbounds(cropped)
    @test length.(bounds.rs) == (4, 4)
    @test bounds == getbounds(labels)
    @test getbounds.(new_boxes) == [bounds, bounds]
    @test itemdata(cropped) == itemdata(labels)
    @test parent(itemdata(cropped)) == data[bounds.rs...]
    @test count(box -> all(>(0), last(itemdata(box)) - first(itemdata(box))), new_boxes) == 1
    repeated = apply(tfm, items; randstate = 4)
    @test itemdata(repeated[1]) == itemdata(cropped)
    @test itemdata.(repeated[3]) == itemdata.(new_boxes)
    @test apply(RandomIoUCrop(sampler_options = (1,)), items) === items
    @test apply(tfm, image) === image

    @testset "labeled box collections" begin
        data = [1 6; 1 6; 3 8; 3 8]
        collection = box_convert(BoundingBoxes(data, (8, 8); labels = [:a, :b]), :CXCYWH)
        new_image, new_collection = apply(tfm, (image, collection); randstate = 4)
        xyxy = itemdata(box_convert(new_collection, :XYXY))
        legacy = [[lo[2], lo[1], hi[2], hi[1]] for (lo, hi) in itemdata.(new_boxes)]
        @test getbounds(new_image) == getbounds(new_collection) == bounds
        @test xyxy == hcat(legacy...)
        @test new_collection.format === :CXCYWH
        @test new_collection.labels == [:a, :b]
        kept = findall(i -> all(xyxy[1:2, i] .< xyxy[3:4, i]), 1:2)
        @test length(kept) == 1
        @test apply(SanitizeBoundingBoxes(), new_collection).labels == collection.labels[kept]
        @test itemdata(apply(tfm, collection; randstate = 4)) == itemdata(new_collection)

        empty_boxes = BoundingBoxes(zeros(4, 0), (8, 8))
        @test apply(tfm, (image, empty_boxes); randstate = 4) === (image, empty_boxes)
    end

    @testset "boxes are clipped to the crop edges" begin
        crop = Bounds((3:6, 2:5))
        spanning = BoundingBox([SVector(0., 0.), SVector(10., 10.)], (8, 8))
        clipped = DataAugmentation._crop_detection(spanning, Bounds((8, 8)), crop)
        @test itemdata(clipped) == [SVector(2.5, 1.5), SVector(6.5, 5.5)]
        inside = BoundingBox([SVector(3., 2.), SVector(6., 5.)], (8, 8))
        @test itemdata(DataAugmentation._crop_detection(inside, Bounds((8, 8)), crop)) == itemdata(inside)
        line = BoundingBox([SVector(4., 0.), SVector(4., 10.)], (8, 8))
        @test itemdata(DataAugmentation._crop_detection(line, Bounds((8, 8)), crop)) ==
            [SVector(4., 1.5), SVector(4., 5.5)]
        @test DataAugmentation._meets_iou_threshold(itemdata(spanning), crop, 0.1)
        full = (SVector(2.5, 1.5), SVector(6.5, 5.5))
        @test DataAugmentation._meets_iou_threshold(full, crop, 1)
    end
end
