showitem(item::Item) = summary(itemdata(item))

function showitem(item::Image)
    return RGB{N0f8}.(parent(itemdata(item)))
end


function showitem(item::Keypoints)
    img = zeros(RGB{N0f8}, getbounds(item)...)
    for keypoint in itemdata(item)
        if isnothing(keypoint)
            continue
        end

        I = CartesianIndex(Tuple(round.(Int, keypoint)))
        if checkbounds(Bool, img, I)
            img[I] = RGB(1, 1, 1)
        end
    end
    return img
end


function showsample(sample::Dict{Symbol, Item})
    error("Not implemented")
end
