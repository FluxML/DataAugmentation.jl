using DataAugmentation
using Documenter

makedocs(;
    modules=[DataAugmentation],
    authors="lorenzoh <lorenz.ohly@gmail.com>",
    repo="https://github.com/lorenzoh/DataAugmentation.jl/blob/{commit}{path}#L{line}",
    sitename="DataAugmentation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lorenzoh.github.io/DataAugmentation.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lorenzoh/DataAugmentation.jl",
)
