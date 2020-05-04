function showitem(image::Image)
    return RGB{N0f8}.(parent(image.data))
end


function showitem(keypoints::Keypoints)
    img = zeros(RGB{N0f8}, getbounds(keypoints)...)
    for keypoint in keypoints.data
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
