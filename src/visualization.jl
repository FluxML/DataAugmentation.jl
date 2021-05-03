"""
    showitems(items)

Visualize `items`.
"""
function showitems(items; C=RGBA{N0f8}, showbounds = false)
    bounds = boundsof([getbounds(item) for item in items])
    a = OffsetArray(zeros(C, length.(bounds.rs)...), bounds.rs...)

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


showitems(item::AbstractItem; kwargs...) = showitems((item,); kwargs...)

## Visualization utilities

function showimage!(dst, img)
    for I in CartesianIndices(img)
        if checkbounds(Bool, dst, I)
            dst[I] = img[I]
        end
    end
end


function showgrid(items; fillvalue = RGBA{N0f8}(0.,0.,0.,0.), kwargs...)
    imgs = [showitems(item) for item in items]
    mosaicview(imgs; fillvalue = fillvalue, kwargs...)
end

showbounds(bounds) = showbounds!(zeros(RGBA{N0f8}, sum.(bounds.rs)), bounds)

function showbounds!(img, bounds::Bounds{2})
    ry, rx = bounds.rs
    miny, maxy = extrema(ry)
    minx, maxx = extrema(rx)
    points = [
        SVector(miny, minx),
        SVector(miny, maxx),
        SVector(maxy, maxx),
        SVector(maxy, minx),
    ]
    showpolygon!(img, points, RGBA(0, 0, 0, 1))
    return img
end


function showkeypoint!(img, point::SVector{N}, C; sz = 3) where N
    I = CartesianIndex(Tuple(round.(Int, point)))
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
function boundsof(boundss::AbstractVector{<:Bounds{N}}) where N
    rs = map(1:N) do i
        minimum((minimum(bs.rs[i]) for bs in boundss)):maximum((maximum(bs.rs[i]) for bs in boundss))
    end
    return Bounds(Tuple(rs))
end
