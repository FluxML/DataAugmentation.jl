using Documenter, DataAugmentation

DocMeta.setdocmeta!(DataAugmentation, :DocTestSetup, :(using DataAugmentation); recursive = true)

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
    format = Documenter.HTML(canonical = "https://fluxml.ai/DataAugmentation.jl/stable/",
                             prettyurls = get(ENV, "CI", nothing) == "true"),
)

deploydocs(repo = "github.com/FluxML/DataAugmentation.jl.git",
           devbranch = "master",
           push_preview = true)
