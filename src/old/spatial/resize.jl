ResizeFixed(sz, from = FromOrigin()) = ScaleFixed(sz) |> CropFixed(sz, from)
ResizeRatio(ratios, from = FromOrigin()) = ScaleRatio(ratios) |> CropRatio((1, 1), from)


CenterResizeCrop(sz::Tuple{Int, Int}) = ScaleKeepAspect(sz) |> CropFixed(sz, FromCenter())
CenterResizeCrop(h, w) = CenterResizeCrop((h, w))
RandomResizeCrop(sz::Tuple{Int, Int}) = ScaleKeepAspect(sz) |> CropFixed(sz, FromRandom())
RandomResizeCrop(h, w) = RandomResizeCrop((h, w))

ResizeDivisible(sz::Tuple{Int, Int}; divisible = 1) = ScaleKeepAspect(sz) |> CropDivisible(divisible)
