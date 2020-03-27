showitem(item::Item) = summary(itemdata(item))

function showitem(item::Image)
    return RGBA.(itemdata(item))
end


function showitem(item::Keypoints)
    img = zeros(RGBA, getbounds(item)...)
    return img
end


function showsample(sample::Dict{Symbol, Item})
    error("Not implemented")
end