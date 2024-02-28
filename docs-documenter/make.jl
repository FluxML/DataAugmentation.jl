using Documenter, DataAugmentation

makedocs(sitename="DataAugmentation.jl",
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
  # format = Documenter.HTML(prettyurls = false)
)
