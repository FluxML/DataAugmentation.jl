using DataAugmentation
using DataAugmentation:
    Affine, CropFixed, CropRatio, Sequential,
    makebuffer, getrandstate, normalize, denormalize, Normalize, setwrapped, setdata,
    applyaffine, getaffine, getwrapped,
    ScaleFixed, ScaleRatio, ScaleKeepAspect, cropindices, imagetotensor, tensortoimage,
    FromOrigin, FromCenter, FromRandom, MapElem, Inplace, boundsranges
import DataAugmentation: apply, apply!, compose, makebuffer
using Images
using StaticArrays
using Test
using TestSetExtensions
using CoordinateTransformations
using TestImages
