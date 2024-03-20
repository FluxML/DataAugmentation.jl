using Documenter, DataAugmentation

makedocs(;
    modules = [DataAugmentation],
    sitename="DataAugmentation.jl",
    pages = [
        "index.md",
        "Quickstart" => "quickstart.md",
        "Transformations" => "transformations.md",
        "Build your transformations" =>[
            "Item Interface" => "iteminterface.md",
            "Transform Interface" => "tfminterface.md", 
            "Projective Interface" => [
                "Intro" => "projective/intro.md",
                "Data" => "projective/data.md",
                "Gallery" => "projective/gallery.md",
            ],
        ],
        "Misc" =>[
            "Buffering" => "buffering.md",
            "Preprocessing" => "preprocessing.md",
            "References" => "ref.md"
        ],
    ],
    warnonly = [:example_block, :missing_docs, :cross_references],
    format = Documenter.HTML(canonical = "https://fluxml.ai/DataAugmentation.jl/stable/",
                             assets = ["assets/flux.css"],
                             prettyurls = get(ENV, "CI", nothing) == "true")
)

deploydocs(repo = "github.com/FluxML/DataAugmentation.jl.git",
           target = "build",
           push_preview = true)
