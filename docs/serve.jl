using Pkg
using Pollen
using DataAugmentation

p = Pollen.documentationproject(DataAugmentation)
Pollen.serve(p)
