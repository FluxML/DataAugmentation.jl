using Pkg
using Pollen
using DataAugmentation

p = Pollen.documentationproject(DataAugmentation, executecode=true)
push!(p.rewriters, Pollen.PackageWatcher([DataAugmentation]))
server = Server(p)
mode = ServeFilesLazy()
runserver(server, mode)
