module DataAugmentation

using CoordinateTransformations
using ImageTransformations
using ImageTransformations: center, _center
using Images
using Images: Colorant
using StaticArrays

include("./item.jl")
include("./transform.jl")
include("./utils.jl")
include("./samples.jl")
include("./display.jl")

include("./transforms/spatial.jl")
include("./transforms/affinetransform.jl")
include("./transforms/preprocessing.jl")


export
    # Transforms
    Transform,
    AbstractAffine,
    Affine,
    CenterResize,
    CenterResizeCrop,
    Crop,
    CroppedAffine,
    Either,
    FlipX,
    FlipY,
    Identity,
    Lambda,
    Normalize,
    OneHot,
    Pipeline,
    Rotate,
    Rotate90,
    Rotate180,
    Rotate270,
    RandomResize,
    RandomResizeCrop,
    SamplePipeline,
    Scale,
    ScaleCrop,
    ToEltype,
    ToTensor,

    # Items
    Item,
    ItemWrapper,
    Image,
    Keypoints,
    Label,
    Tensor,

    # functions
    apply,
    compose,
    getbounds,
    showitem

end # module
