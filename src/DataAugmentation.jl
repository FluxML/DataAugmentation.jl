module DataAugmentation

using CoordinateTransformations
using ImageTransformations
using Images
using Images: Colorant
using NamedDims

include("./item.jl")
include("./transform.jl")
include("./samples.jl")
include("./display.jl")

include("./transforms/spatial.jl")
include("./transforms/affinetransform.jl")
include("./transforms/preprocessing.jl")


export
    # Transforms
    AbstractTransform,
    AbstractAffineTransform,
    AffineTransform,
    CenterResizedCrop,
    CropTransform,
    CroppedAffineTransform,
    Either,
    FlipX,
    FlipY,
    LambdaTransform,
    Normalize,
    OneHot,
    Pipeline,
    Rotate,
    Rotate90,
    Rotate180,
    Rotate270,
    RandomResizedCrop,
    RandomResizedTransform,
    DictPipeline,
    XYPipeline,
    Scale,
    ToEltype,
    ToTensor,

    # Items
    Item,
    ItemWrapper,
    Image,
    Keypoints,
    Label,
    Tensor,

    getbounds,
    itemdata,
    showitem

end # module
