using DataAugmentation
using Images
using StaticArrays
using Test
using TestSetExtensions
using CoordinateTransformations
using TestImages


using DataAugmentation: Item, Transform, getrandstate, itemdata, setdata, ComposedProjectiveTransform,
    cropindices, makebounds, getprojection, boundsextrema, boundsranges, boundssize, offsetcropindices,
    CroppedProjectiveTransform, getbounds, project, project!, makebuffer
import DataAugmentation: apply, compose
