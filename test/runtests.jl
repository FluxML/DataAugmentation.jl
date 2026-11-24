include("imports.jl")

@testset "DataAugmentation.jl" begin
    for file in [
        "base.jl",
        "testing.jl",
        "buffered.jl",
        "projective/base.jl",
        "projective/compose.jl",
        "projective/crop.jl",
        "projective/affine.jl",
        "projective/warp.jl",
        "projective/randomwarp.jl",
        "projective/resize.jl",
        "projective/pad.jl",
        "projective/elastic.jl",
        "projective/geometric_crops.jl",
        "items/image.jl",
        "items/keypoints.jl",
        "items/mask.jl",
        "preprocessing.jl",
        "oneof.jl",
        "visualization.jl",
        "rowtransforms.jl",
        "colortransforms.jl",
        "photometricfilters.jl",
        "photometricintensity.jl",
        "photometricdistort.jl",
        "randomerasing.jl",
        "mixing.jl",
    ]
        @testset "$file" begin
            include(file)
        end
    end
    @testset "Aqua" begin
        Aqua.test_all(DataAugmentation)
    end
end
