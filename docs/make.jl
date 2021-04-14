using Pollen
using DataAugmentation
using FilePathsBase

Pkg.add(PackageSpec(; url = "https://github.com/lorenzoh/Pollen.jl", rev="main"))
project = Pollen.documentationproject(DataAugmentation)
Pollen.fullbuild(project, Pollen.FileBuilder(Pollen.HTML(), p"dev/"))
