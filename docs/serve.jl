using Pkg
#Pkg.activate(@__DIR__)
using Pollen
using DataAugmentation

project = Pollen.documentationproject(DataAugmentation, executecode=true)
push!(project.rewriters, Pollen.PackageWatcher([DataAugmentation]))
server = Server(project)
mode = ServeFilesLazy()
runserver(server, mode)
