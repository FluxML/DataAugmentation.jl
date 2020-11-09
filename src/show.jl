

function showitem(items::Tuple)
    return blend.(showitem.(items)...)
end


function showitem(image::Image{2, <:Colorant})
    return image.data
end

function showitem(image::Image{2, <:AbstractFloat})
    return colorview(Gray, image.data)
end


function showitem(keypoints::Keypoints{N}) where N
    img = zeros(RGBA{N0f8}, boundssize(getbounds(keypoints)))
    for point in filter(!isnothing, keypoints.data)
        drawkeypoint!(img, point, RGBA(1, 0, 0, 1))
    end
    return img
end


function showitem(polygon::Polygon)
    img = zeros(RGBA{N0f8}, boundssize(getbounds(polygon)))
    drawpolygon!(img, itemdata(polygon), RGBA(1, 0, 0, 1))
    return img
end

function showitem(bbox::BoundingBox{2})
    img = zeros(RGBA{N0f8}, boundssize(getbounds(bbox)))
    ul, br = itemdata(bbox)
    points = [
        ul,
        SVector(br[1], ul[2]),
        br,
        SVector(ul[1], br[2]),
    ]
    drawpolygon!(img, points, RGBA(1, 0, 0, 1))
    return img
end


showbounds(bounds) = showbounds!(zeros(RGBA{N0f8}, boundssize(bounds), bounds))

function showbounds!(img, bounds::AbstractArray{<:SVector{2}}) where N
    points = [
        bounds[1] + SVector(1, 1),
        bounds[2] + SVector(0, 1),
        bounds[4] + SVector(0, 0),
        bounds[3] + SVector(1, 0),
    ]
    drawpolygon!(img, points, RGBA(0, 0, 0, 1))
    return img
end


function drawkeypoint!(img, point::SVector{N}, C; sz = 3) where N
    I = CartesianIndex(Tuple(round.(Int, reverse(point))))
    offset = sz ÷ 2
    Is = I-offset*one(CartesianIndex{N}):I+offset*one(CartesianIndex{N})
    img[Is[[checkbounds(Bool, img, I) for I in Is]]] .= C
end


function drawpolygon!(img, points, C)
    polydraw = ImageDraw.Polygon([Tuple(round.(Int, reverse(point))) for point in reshape(points, :)])
    draw!(img, polydraw, C)
end
