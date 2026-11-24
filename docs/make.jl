using Documenter, DataAugmentation

DocMeta.setdocmeta!(DataAugmentation, :DocTestSetup, :(using DataAugmentation); recursive = true)

makedocs(;
    modules = [DataAugmentation],
    sitename="DataAugmentation.jl",
    pages = [
        "Home" => "index.md",
        "Quickstart" => "quickstart.md",
        "Guide" => [
            "Items" => "items.md",
            "Transformations" => "transformations.md",
            "Gallery" => "gallery.md",
            "Pipelines" => "pipelines.md",
            "Preprocessing" => "preprocessing.md",
            "Using with Flux" => "flux.md",
            "Buffering" => "buffering.md",
        ],
        "Concepts" => [
            "Design" => "design.md",
            "Projective transformations" => "projective.md",
        ],
        "Extending" => [
            "Defining items" => "iteminterface.md",
            "Defining transformations" => "tfminterface.md",
            "Defining projective transformations" => "projinterface.md",
        ],
        "API reference" => "ref.md",
    ],
    format = Documenter.HTML(canonical = "https://fluxml.ai/DataAugmentation.jl/stable/",
                             prettyurls = get(ENV, "CI", nothing) == "true"),
)

deploydocs(repo = "github.com/FluxML/DataAugmentation.jl.git",
           devbranch = "master",
           push_preview = true)
