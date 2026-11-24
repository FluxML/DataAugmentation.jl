@testset "Box formats" begin
    bounds = Bounds((8, 9))
    labels = [:cat, :dog]
    formats = (
        XYXY = [2 4; 3 2; 7 6; 7 5],
        XYWH = [2 4; 3 2; 5 2; 4 3],
        CXCYWH = [4.5 5; 5 3.5; 5 2; 4 3],
    )

    for (source, data) in pairs(formats), (target, expected) in pairs(formats)
        boxes = BoundingBoxes(data, bounds; format = source, labels)
        converted = box_convert(boxes, target)
        @test itemdata(converted) ≈ expected
        @test converted.format === target
        @test converted.labels == labels
        @test getbounds(converted) == bounds
    end

    empty_boxes = BoundingBoxes(zeros(Int, 4, 0), bounds; labels = Symbol[])
    @test size(itemdata(box_convert(empty_boxes, :CXCYWH))) == (4, 0)
    @test_throws DimensionMismatch BoundingBoxes(zeros(3, 2), bounds)
    @test_throws DimensionMismatch BoundingBoxes(zeros(4, 2), bounds; labels = [:cat])
    @test_throws ArgumentError BoundingBoxes(zeros(4, 2), bounds; format = :YX)
    @test_throws ArgumentError box_convert(empty_boxes, :YX)
end

@testset "Projecting boxes with labels" begin
    image = Image(zeros(8, 10))
    data = [3 5; 2 3; 9 9; 5 3]
    labels = [:valid, :degenerate]
    boxes = BoundingBoxes(data, (8, 10); labels)
    tfm = Project(LinearMap(SA[1 1; -1 1]))
    for format in (:XYXY, :XYWH, :CXCYWH)
        input = box_convert(boxes, format)
        projected_image, projected = apply(tfm, (image, input))
        @test getbounds(projected) == getbounds(projected_image)
        @test projected.format === format
        @test projected.labels == labels
        @test itemdata(box_convert(projected, :XYXY)) ≈ [-2 5; 5 3; 7 9; 14 3]
        @test apply(SanitizeBoundingBoxes(), projected).labels == [:valid]
        @test itemdata(apply!(projected, tfm, input)) == itemdata(projected)
    end

    @test apply(AdjustBrightness(0.5), boxes) === boxes
    @test any(!iszero, showitems(boxes))
    @test all(iszero, showitems(BoundingBoxes([11; 11; 12; 12;;], (8, 10))))

    buffered = Buffered(FlipX(); copy = true)
    first = apply(buffered, boxes)
    second = apply(buffered, BoundingBoxes(data, (8, 10); labels = [:a, :b]))
    @test first.labels == labels
    @test first.labels !== labels
    @test second.labels == [:a, :b]
    @test apply!(first, buffered, BoundingBoxes(data, (8, 10); labels = [:a, :b])) === first
    @test first.labels == second.labels

    crop = CenterCrop((4, 6))
    buf = apply(crop, boxes)
    new_data = data .+ 1
    input = BoundingBoxes(new_data, (8, 10); labels = [:a, :b])
    @test apply!(buf, Buffered(crop), input) === buf
    @test itemdata(buf) == new_data && buf.labels == [:a, :b]
    @test itemdata(boxes) == [3 5; 2 3; 9 9; 5 3] && boxes.labels == [:valid, :degenerate]
end

@testset "Clamping and sanitizing cropped boxes" begin
    image = Image(zeros(8, 10))
    data = [1 9 4 7 NaN 4; 2 7 4 3 4 4; 5 10 6 6 6 Inf; 5 8 6 4 6 6]
    original_data = copy(data)
    labels = [:partial, :outside, :inside, :inverted, :nan, :inf]
    boxes = BoundingBoxes(data, (8, 10); labels)
    crop = CenterCrop((4, 6))
    unpinned = apply(crop |> ClampBoundingBoxes(), boxes)
    @test getbounds(unpinned) == Bounds((3:6, 3:8))
    @test itemdata(unpinned)[:, 1] == [2.5, 2.5, 5, 5]
    tfm = crop |> PinOrigin()
    for format in (:XYXY, :XYWH, :CXCYWH)
        _, cropped = tfm((image, box_convert(boxes, format)))
        @test apply(SanitizeBoundingBoxes(), cropped).labels == [:inside]

        clamped = apply(ClampBoundingBoxes(), cropped)
        @test clamped.labels == labels
        @test itemdata(box_convert(clamped, :XYXY))[:, 2] == [6.5, 4.5, 6.5, 4.5]
        cleaned = apply(SanitizeBoundingBoxes(), clamped)
        @test cleaned.labels == [:partial, :inside]
        @test cleaned.format === format
        @test getbounds(cleaned) == Bounds((4, 6))
        @test itemdata(box_convert(cleaned, :XYXY)) ≈ [0.5 2; 0.5 2; 3 4; 3 4]
    end
    @test isequal(itemdata(boxes), original_data)

    unlabeled = BoundingBoxes(data[:, 2:2], (8, 10))
    _, empty_boxes = apply(tfm |> ClampBoundingBoxes() |> SanitizeBoundingBoxes(), (image, unlabeled))
    @test size(itemdata(empty_boxes)) == (4, 0)
    @test isnothing(empty_boxes.labels)
    @test apply(SanitizeBoundingBoxes(), empty_boxes) === empty_boxes
end

@testset "Box formats stay inside the edges after clamping" begin
    for (cols, x1, x2, format) in ((1:7, 0.53, 8.5, :CXCYWH), (-5:4, -5.46, 5.5, :XYWH),
                                   (5:20, 3.5, 11.51, :CXCYWH))
        boxes = BoundingBoxes([x1; 1.2; x2; 2.3;;], Bounds((1:5, cols)); labels = [:partial])
        clamped = apply(ClampBoundingBoxes(), box_convert(boxes, format))
        xyxy = box_convert(clamped, :XYXY)
        @test first(cols) - 0.5 <= itemdata(xyxy)[1] < itemdata(xyxy)[3] <= last(cols) + 0.5
        @test apply(SanitizeBoundingBoxes(), clamped).labels == [:partial]
    end
end

@testset "Boxes from instance masks" begin
    masks = OffsetArray(falses(3, 5, 3), -1:1, 4:8, 15:17)
    masks[-1, 4, 15] = true
    masks[0, 5, 16] = masks[1, 8, 16] = true
    boxes = masks_to_boxes(masks; labels = [1, 2, 3])
    @test getbounds(boxes) == Bounds((-1:1, 4:8))
    @test boxes.format === :XYXY
    @test itemdata(boxes) == [3.5 4.5 3.5; -1.5 -0.5 -1.5; 4.5 8.5 3.5; -0.5 1.5 -1.5]
    @test apply(SanitizeBoundingBoxes(), boxes).labels == [1, 2]
    mask = MaskBinary(selectdim(masks, 3, 16))
    @test itemdata(masks_to_boxes(mask)) == itemdata(boxes)[:, 2:2]
    @test size(itemdata(masks_to_boxes(falses(3, 5, 0)))) == (4, 0)
end

@testset "Pairwise box intersection over union" begin
    a = BoundingBoxes([0.5 2.5 1 2 NaN; 0.5 0.5 2 2 1; 2.5 4.5 1 1 3; 2.5 2.5 4 4 4], (6, 8))
    b = BoundingBoxes([1.5 0.5; 0.5 0.5; 3.5 2.5; 2.5 2.5], (6, 8))
    b = box_convert(b, :CXCYWH)
    @test box_iou(a, b) ≈ [1/3 1; 1/3 0; 0 0; 0 0; 0 0]
    @test box_iou(b, a) ≈ transpose(box_iou(a, b))
    @test size(box_iou(BoundingBoxes(zeros(4, 0), (6, 8)), b)) == (0, 2)
end
