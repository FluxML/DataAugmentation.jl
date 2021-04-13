using Pollen
using DataAugmentation

project = Pollen.docmentationproject(DataAugmentation)


Pollen.fullbuild(project, Pollen.FileBuilder(Pollen.HTML()), Path("dev/"))
