
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
