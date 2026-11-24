module DataAugmentation

using Accessors: @set
using CoordinateTransformations
using Distributions: Beta, Categorical, Dirac, Dirichlet, Sampleable, Uniform
using IndirectArrays: IndirectArray
using ImageDraw
using ImageCore
using ImageTransformations
using ImageTransformations: center, _center, box_extrapolation, warp!
using Interpolations
using MosaicViews: mosaicview
using OffsetArrays: OffsetArray
using LinearAlgebra: I, det, mul!
using Random
using Rotations
using StaticArrays
using Statistics
using Test: @test, @test_nowarn


include("./base.jl")
include("./wrapper.jl")
include("./buffered.jl")
include("./sequence.jl")
include("./items/arrayitem.jl")
include("./projective/base.jl")
include("./items/image.jl")
include("./items/imagebatch.jl")
include("./items/table.jl")
include("./items/keypoints.jl")
include("./keypointtransforms.jl")
include("./items/mask.jl")
include("./items/boundingboxes.jl")
include("./boundingboxtransforms.jl")
include("./boxops.jl")
include("./projective/compose.jl")
include("./projective/crop.jl")
include("./projective/affine.jl")
include("./projective/resize.jl")
include("./projective/pad.jl")
include("./projective/warp.jl")
include("./projective/randomwarp.jl")
include("./projective/multicrop.jl")
include("./projective/ioucrop.jl")
include("./oneof.jl")
include("./preprocessing.jl")
include("./rowtransforms.jl")
include("./photometric.jl")
include("./colortransforms.jl")
include("./photometricfilters.jl")
include("./projective/elastic.jl")
include("./photometricintensity.jl")
include("./photometricdistort.jl")
include("./randomerasing.jl")
include("./mixing.jl")
include("./temporalsubsample.jl")
include("./policy.jl")
include("./autoaugment.jl")
include("./augmix.jl")
include("testing.jl")
include("./visualization.jl")


export Item,
    Transform,
    ArrayItem,
    MapElem,
    Identity,
    Sequence,
    Project,
    Image,
    ImageBatch,
    TabularItem,
    Keypoints,
    ClampKeypoints,
    SanitizeKeypoints,
    Polygon,
    ToEltype,
    ImageToTensor,
    PermuteDims,
    Normalize,
    NormalizeIntensity,
    LinearTransformation,
    UniformTemporalSubsample,
    MaskMulti,
    MaskBinary,
    BoundingBox,
    BoundingBoxes,
    box_convert,
    ClampBoundingBoxes,
    SanitizeBoundingBoxes,
    masks_to_boxes,
    box_iou,
    ScaleKeepAspect,
    ScaleRatio,
    itemdata,
    Crop,
    CenterCrop,
    RandomCrop,
    ScaleFixed,
    Resize,
    ScaleJitter,
    RandomShortestSize,
    RandomResize,
    Rotate,
    RotateX,
    RotateY,
    RotateZ,
    RandomResizeCrop,
    RandomResizedCrop,
    RandomIoUCrop,
    FiveCrop,
    TenCrop,
    CenterResizeCrop,
    Buffered,
    BufferedThreadsafe,
    OneHot,
    Zoom,
    OneOf,
    Maybe,
    apply,
    Reflect,
    WarpAffine,
    RandomAffine,
    RandomPerspective,
    ElasticTransform,
    FlipDim,
    FlipX,
    FlipY,
    FlipZ,
    PinOrigin,
    AdjustBrightness,
    AdjustContrast,
    AdjustSaturation,
    AdjustHue,
    ColorJitter,
    Grayscale,
    RandomGrayscale,
    ToRGB,
    GaussianBlur,
    GaussianNoise,
    AdjustSharpness,
    AdjustGamma,
    Solarize,
    Posterize,
    Invert,
    Equalize,
    Autocontrast,
    RandomPhotometricDistort,
    RandomChannelPermutation,
    JPEG,
    RandomErasing,
    MixUp,
    CutMix,
    RandAugment,
    TrivialAugmentWide,
    AutoAugment,
    AugMix,
    RandomOrder,
    apply!,
    Pad,
    RandomZoomOut,
    PadDivisible,
    ResizePadDivisible,
    onehot,
    showitems,
    showgrid,
    Bounds


end # module
