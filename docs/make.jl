using Publish
using DataAugmentation

p = Publish.Project(DataAugmentation)
rm("dev", recursive = true, force = true)
rm(p.env["version"], recursive = true, force = true)
deploy(DataAugmentation; root = "/DataAugmentation.jl", force = true, label = "dev")
