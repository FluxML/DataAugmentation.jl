# {cell=main}
using Pkg
Pkg.activate("../test")
Pkg.instantiate()

include("../test/runtests.jl")
