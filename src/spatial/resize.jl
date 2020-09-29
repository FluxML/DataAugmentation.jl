ResizeFixed(sz, from = CropFromOrigin()) = ScaleFixed(sz) |> CropFixed(sz, from)
ResizeRatio(ratios, from = CropFromOrigin()) = ScaleRatio(ratios) |> CropRatio((1, 1), from)


CenterResizeCrop(sz::Tuple{Int, Int}) = ScaleKeepAspect(sz) |> CropFixed(sz, CropFromCenter())
CenterResizeCrop(h, w) = CenterResizeCrop((h, w))
RandomResizeCrop(sz::Tuple{Int, Int}) = ScaleKeepAspect(sz) |> CropFixed(sz, CropFromRandom())
RandomResizeCrop(h, w) = RandomResizeCrop((h, w))

ResizeDivisible(sz::Tuple{Int, Int}; divisible = 1) = ScaleKeepAspect(sz) |> CropDivisible(divisible)
