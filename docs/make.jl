using Publish
using DataAugmentation
using Pkg.Artifacts


Publish.Themes.default() = artifact"flux-theme"

p = Publish.Project(DataAugmentation)

rm("dev", recursive = true, force = true)
rm(p.env["version"], recursive = true, force = true)

deploy(DataAugmentation; root = "/DataAugmentation.jl", force = true, label = "dev")
