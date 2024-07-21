using DataAugmentation
using StaticArrays
using Test
using TestSetExtensions
using CoordinateTransformations
using Colors
using FixedPointNumbers: N0f8
using LinearAlgebra
using Rotations


using DataAugmentation: Item, Transform, getrandstate, itemdata, setdata, ComposedProjectiveTransform,
    projectionbounds, getprojection, offsetcropbounds,
    CroppedProjectiveTransform, getbounds, project, project!, makebuffer, imagetotensor, imagetotensor!,
    normalize, normalize!, tensortoimage, denormalize, denormalize!,
    NormalizeRow, FillMissing, Categorify, TabularItem
using DataAugmentation: testitem, testapply, testapply!, testprojective
import DataAugmentation: apply, compose
