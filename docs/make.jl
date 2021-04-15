using Pollen
using DataAugmentation
using FilePathsBase

project = Pollen.documentationproject(DataAugmentation)
Pollen.fullbuild(project, Pollen.FileBuilder(Pollen.HTML(), p"dev/"))
