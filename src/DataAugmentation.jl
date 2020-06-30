module DataAugmentation

using CoordinateTransformations
using ImageTransformations
using ImageTransformations: center, _center, box_extrapolation, warp!
using Images
using Images: Colorant
using StaticArrays
using Parameters
using DocStringExtensions

include("./item.jl")
include("./transform.jl")
include("./buffered.jl")
include("./samples.jl")

include("./spatial/affine.jl")
include("./spatial/crop.jl")
include("./spatial/translate.jl")
include("./spatial/scale.jl")
include("./spatial/resize.jl")

#= TODO


    Either,
    FlipX,
    FlipY,
    Lambda,
    Normalize,
    OneHot,
    Rotate,
    Rotate90,
    Rotate180,
    Rotate270,
    ToEltype,
    ToTensor, =#


    # items interface
export Item,
    ItemWrapper,
    getwrapped,
    getbounds,

    # items
    Image,
    Keypoints,
    Polygon,
    BoundingBox,
    Category,

    # pipelines
    ApplyStep,
    CombineStep,
    MapStep,
    Pipeline,

    # transforms
    Transform,
    Identity,
    Buffered,

    # affine transforms
    AbstractAffine,
    Affine,
    Crop,
    CropFixed,
    CropRatio,
    CroppedAffine,

    # scale & resize
    ScaleFixed,
    ScaleRatio,
    ScaleKeepAspect,
    ResizeFixed,
    ResizeRatio,
    CenterResizeCrop,
    RandomResizeCrop,

    # functions
    apply,
    apply!,
    compose,
    itemdata,
    makebuffer

end # module
