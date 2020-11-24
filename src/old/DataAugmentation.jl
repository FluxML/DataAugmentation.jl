module DataAugmentation

using ColorBlendModes
using CoordinateTransformations
using ImageDraw
using Images
using Images: Colorant, permuteddimsview
using ImageTransformations
using ImageTransformations: center, _center, box_extrapolation, warp!
using Interpolations
using Parameters
using SetField
using StaticArrays

include("./items/item.jl")
include("./items/wrapper.jl")

include("./transform.jl")
include("./buffered.jl")
include("./show.jl")

include("./spatial/affine.jl")
include("./spatial/crop.jl")
include("./spatial/translate.jl")
include("./spatial/scale.jl")
include("./spatial/resize.jl")

include("./transforms/preprocessing.jl")


export Item,
    # items interface
    ItemWrapper,
    getwrapped,
    getbounds,

    # items
    Many,
    ArrayItem,
    Image,
    Keypoints,
    Polygon,
    BoundingBox,
    Category,
    MaskBinary,
    MaskMulti,

    # transforms
    Transform,
    Identity,
    MapElem,

    # pre/postprocessing transforms
    ToEltype,
    Normalize,
    ImageToTensor,
    OneHotEncode,


    # affine transforms
    AbstractAffine,
    Affine,
    Crop,
    CropFixed,
    CropIndices,
    CropRatio,
    CropDivisible,
    CroppedAffine,

    # scale & resize
    ScaleFixed,
    ScaleRatio,
    ScaleKeepAspect,
    ResizeFixed,
    ResizeRatio,
    ResizeDivisible,
    CenterResizeCrop,
    RandomResizeCrop,

    # functions
    apply,
    apply!,
    compose,
    itemdata,
    makebuffer,
    showitem

end # module
