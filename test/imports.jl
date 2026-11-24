using DataAugmentation
using StaticArrays
using Test
using Aqua
using CoordinateTransformations
using Colors
using Distributions: Categorical
using FixedPointNumbers: N0f8
using IndirectArrays: IndirectArray
using LinearAlgebra
using Rotations
using Statistics
using OffsetArrays
import Interpolations


using DataAugmentation: Item, Transform, getrandstate, itemdata, setdata, ComposedProjectiveTransform,
    projectionbounds, getprojection, offsetcropbounds,
    CroppedProjectiveTransform, getbounds, project, project!, makebuffer, imagetotensor, imagetotensor!,
    normalize, normalize!, tensortoimage, denormalize, denormalize!,
    NormalizeRow, FillMissing, Categorify, TabularItem
using DataAugmentation: testitem, testapply, testapply!, testprojective
import DataAugmentation: apply, compose
