using Artifacts
using Publish
using DataAugmentation


artifactsfile = download("https://raw.githubusercontent.com/darsnack/flux-theme/main/Artifacts.toml", joinpath(@__DIR__, "Artifacts.toml"))
Publish.Themes.default() = artifact"flux-theme"


p = Publish.Project(DataAugmentation)
rm("dev", recursive = true, force = true)
rm(p.env["version"], recursive = true, force = true)
deploy(DataAugmentation; root = "/DataAugmentation.jl", force = true, label = "dev")
