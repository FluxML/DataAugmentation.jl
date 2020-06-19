# [Quick start](@id quickstart)


`DataAugmentation.jl` 

# Quick start

```@example
using TestImages

image = testimage("lighthouse")

item = Image(image)

tfm = RandomResizeCrop((128, 128)) |> FlipX()


```