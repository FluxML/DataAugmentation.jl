using DataAugmentation
using DataAugmentation:
    Affine, ApplyStep, CropFixed, CropRatio, Sequential,
    makebuffer, getrandstate, normalize, denormalize, Normalize, setwrapped, setdata,
    applyaffine, getaffine, applystep!, getwrapped, getcropsizes, index_ranges,
    ScaleFixed, ScaleRatio, ScaleKeepAspect, cropindices, imagetotensor, tensortoimage,
    CropFromOrigin, CropFromCenter, CropFromRandom
import DataAugmentation: apply, apply!, compose, makebuffer
using Images
using StaticArrays
using Test
using TestSetExtensions
using CoordinateTransformations
using TestImages
