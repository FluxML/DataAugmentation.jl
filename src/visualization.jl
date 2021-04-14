"""
    showitems(items)

Visualize `items`.
"""
function showitems(items; C=RGBA{N0f8}, showbounds = false)
    rs = boundingranges([DataAugmentation.getbounds(item) for item in items]...)
    a = OffsetArray(zeros(C, length.(rs)...), rs...)

    for item in items
        showitem!(a, item)
    end

    if showbounds
        for item in items
            showbounds!(a, getbounds(item))
        end
    end

    return a
end


## Visualization utilities

function showimage!(dst, img)
    for I in CartesianIndices(img)
        if checkbounds(Bool, dst, I)
            dst[I] = img[I]
        end
    end
end


showbounds(bounds) = showbounds!(zeros(RGBA{N0f8}, boundssize(bounds)), bounds)

function showbounds!(img, bounds::AbstractArray{<:SVector{2}})
    points = [
        bounds[1] + SVector(1, 1),
        bounds[2] + SVector(0, 1),
        bounds[4] + SVector(0, 0),
        bounds[3] + SVector(1, 0),
    ]
    showpolygon!(img, points, RGBA(0, 0, 0, 1))
    return img
end


function showkeypoint!(img, point::SVector{N}, C; sz = 3) where N
    I = CartesianIndex(Tuple(round.(Int, reverse(point))))
    offset = sz ÷ 2
    Is = I-offset*one(CartesianIndex{N}):I+offset*one(CartesianIndex{N})
    img[Is[[checkbounds(Bool, img, I) for I in Is]]] .= C
end


function showpolygon!(img, points, C)
    polydraw = ImageDraw.Polygon([Tuple(round.(Int, reverse(point))) for point in reshape(points, :)])
    draw!(img, polydraw, C)
end


"""
    boundingranges(ps)

Find bounding index ranges for points `ps`.
"""
function boundingranges(ps::Vararg{AbstractArray{<:SVector{N}}}) where N
    ranges = UnitRange[]
    for i ∈ 1:N
        min, max = extrema((p[i] for p in Iterators.flatten(ps)))
        r = floor(Int, min):ceil(Int, max)
        push!(ranges, r)
    end
    return ranges
end
