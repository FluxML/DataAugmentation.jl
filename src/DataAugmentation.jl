module DataAugmentation

using ColorBlendModes
using CoordinateTransformations
using ImageDraw
using Images
using Images: Colorant, permuteddimsview
using ImageTransformations
using ImageTransformations: center, _center, box_extrapolation, warp!
using Interpolations
using LinearAlgebra: I
using Parameters
using Setfield
using StaticArrays


include("./base.jl")
include("./arrayitem.jl")
include("./projective/base.jl")
include("./projective/bounds.jl")
include("./projective/compose.jl")
include("./projective/crop.jl")
include("./projective/affine.jl")
include("./items/image.jl")

export Item,
    Transform,
    Identity,
    Sequence,
    Project,
    Image,
    itemdata,
    CropCenter,
    CropRandom


end # module
