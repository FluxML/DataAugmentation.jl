include("./imports.jl")


@testset ExtendedTestSet "DataAugmentation.jl" begin
    include("base.jl")
    include("buffered.jl")
    include("projective/base.jl")
    include("projective/bounds.jl")
    include("projective/compose.jl")
    include("projective/crop.jl")
    include("projective/affine.jl")
    include("items/image.jl")
    include("items/keypoints.jl")
    include("items/mask.jl")
    include("preprocessing.jl")
    include("visualization.jl")
end
