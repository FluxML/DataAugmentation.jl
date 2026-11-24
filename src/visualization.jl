"""
    showitems(items; C = RGBA{N0f8}, showbounds = false, maskalpha = 0.5,
              maskbackground = nothing)

Draw a single spatial item or a collection of items in the given order.
Masks drawn after an [`Image`](@ref) are blended over the canvas with opacity
`maskalpha`, a number between 0 and 1. Set `maskbackground` to a class label
to leave the canvas visible at those pixels. For a binary mask, use `false`
for the background. By default, all mask labels are drawn. A mask drawn on
its own is opaque.
"""
function showitems(items; C = RGBA{N0f8}, showbounds = false, maskalpha::Real = 0.5,
                   maskbackground = nothing)
    0 <= maskalpha <= 1 || throw(ArgumentError("maskalpha must be between 0 and 1"))
    opacity = Float64(maskalpha)
    bounds = boundsof([getbounds(item) for item in items])
    a = OffsetArray(zeros(C, length.(bounds.rs)...), bounds.rs...)

    hasimage = false
    for item in items
        if hasimage && item isa Union{MaskMulti, MaskBinary}
            showitem!(a, item; opacity, background = maskbackground)
        else
            showitem!(a, item)
        end
        hasimage |= item isa Image
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


function showmask!(colorfor, dst, labels, opacity, background)
    iszero(opacity) && return
    for I in CartesianIndices(labels)
        checkbounds(Bool, dst, I) || continue
        label = labels[I]
        isequal(label, background) && continue
        maskpixel = colorfor(label)
        dst[I] = isone(opacity) ? maskpixel : blendmask(maskpixel, dst[I], opacity)
    end
end


function blendmask(maskpixel, pixel::Colorant, opacity)
    background_alpha = (1 - opacity) * alpha(pixel)
    output_alpha = opacity + background_alpha
    foreground = RGB{Float64}(maskpixel)
    iszero(background_alpha) && return RGBA(foreground, output_alpha)
    rgb = weighted_color_mean(opacity / output_alpha, foreground, RGB{Float64}(color(pixel)))
    return RGBA(rgb, output_alpha)
end

blendmask(maskpixel::AbstractGray, pixel::Real, opacity) =
    opacity * gray(maskpixel) + (1 - opacity) * pixel


"""
    showgrid(itemgroups; fillvalue = RGBA{N0f8}(0, 0, 0, 0), maskalpha = 0.5,
             maskbackground = nothing, kwargs...)

Visualize every element of `itemgroups` (an item or a tuple of items) with
[`showitems`](@ref), which receives `maskalpha` and `maskbackground`.
The results are arranged in a grid with `MosaicViews.mosaicview`, which receives
`fillvalue` and grid options such as `ncol` and `npad`.
"""
function showgrid(items; fillvalue = RGBA{N0f8}(0, 0, 0, 0), maskalpha = 0.5,
                  maskbackground = nothing, kwargs...)
    imgs = [showitems(item; maskalpha, maskbackground) for item in items]
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
    Is = I-offset*oneunit(CartesianIndex{N}):I+offset*oneunit(CartesianIndex{N})
    img[Is[[checkbounds(Bool, img, I) for I in Is]]] .= C
end


function showpolygon!(img, points, C)
    polydraw = ImageDraw.Polygon([Tuple(round.(Int, reverse(point))) for point in reshape(points, :)])
    draw!(img, polydraw, C)
end

function showbox!(img, lo, hi)
    points = [lo, SVector(hi[1], lo[2]), hi, SVector(lo[1], hi[2])]
    showpolygon!(img, points, RGBA(1, 0, 0, 1))
    return img
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
