# # Gallery
#
# Let's visualize what these projective transformations look like.
#
# You can apply them to [`Image`](#)s and
# the keypoint-based items [`Keypoints`](#), [`Polygon`](#), and [`BoundingBox`](#).
#
# Let's take this picture of a light house:
# {cell=main style="display:none;" result=false output=false}
using DataAugmentation
using MosaicViews
using Images
using TestImages
using StaticArrays

imagedata = testimage("lighthouse")
imagedata = imresize(imagedata, ratio = 196 / size(imagedata, 1))
#
# ```julia
# imagedata = testimage("lighthouse")
# ```
# {cell=main style="display:none;"}
imagedata
# To apply a transformation `tfm` to it, wrap it in
# `Image`, apply the transformation and unwrap it using [`itemdata`](#):
# {cell=main }

tfm = CenterCrop((196, 196))
image = Image(imagedata)
apply(tfm, image) |> itemdata

# Now let's say we want to train a light house detector and have a bounding box
# for the light house. We can use the [`BoundingBox`](#) item to represent it.
# It takes the two corners of the bounding rectangle as the first argument. As
# the second argument we have to pass the size of the corresponding image.
#
# {cell=main}
points = SVector{2, Float32}[SVector(23., 120.), SVector(120., 150.)]
bbox = BoundingBox(points, size(imagedata))

# [`showitems`](#) visualizes the two items:
# {cell=main}

showitems((image, bbox))

# If we apply transformations like translation and cropping
# to the image, then the same transformations have to be applied to the bounding
# box. Otherwise, the bounding box will no longer match up with the light house.
#
# Another problem can occur with stochastic transformations like [`RandomResizeCrop`](#).
# If we apply it separately to the image and the bounding box, they will be cropped from
# slightly different locations:
#
# {cell=main}

tfm = RandomResizeCrop((128, 128))
showitems((
    apply(tfm, image),
    apply(tfm, bbox)
))

# Instead, pass a tuple of the items to a single `apply` call so the same
# random state will be used for both image and bounding box:
#
# {cell=main}

apply(tfm, (image, bbox)) |> showitems

# !!! info "3D Projective dimensions"
#
#     We'll use a 2-dimensional [`Image`](#) and [`BoundingBox`](#) here, but you can apply
#     most projective transformations to any spatial item (including [`Keypoints`](#),
#     [`MaskBinary`](#) and [`MaskMulti`](#)) in 3 dimensions.
#
#     Of course, you have to create a 3-dimensional transformation, i.e.
#     `CenterCrop((128, 128, 128))` instead of `CenterCrop((128, 128))`.
#
#
# ## Gallery
# {cell=main style="display:none;" result=false}

function showtransform(tfm, item, n = 8; ncol = 4)
    return mosaicview(
        [showitems(apply(tfm, item)) for _ in 1:n],
        fillvalue = RGBA(1, 1, 1, 0),
        npad = 8,
        rowmajor = true,
        ncol = ncol)
end

function showtransforms(tfms, item; ncol = length(tfms))
    return mosaicview(
        [parent(showitems(apply(tfm, item))) for tfm in tfms],
        fillvalue = RGBA(1, 1, 1, 0),
        npad = 8,
        rowmajor = true,
        ncol = ncol)
end

# ### [`RandomResizeCrop`](#)`(sz)`
#
# Resizes the sides so that one of them is no longer than `sz` and
# crops a region of size `sz` *from a random location*.
#
# {cell=main result=false}

tfm = RandomResizeCrop((128, 128))

# {cell=main style="display:none;"}

o = showtransform(tfm, (image, bbox), 6, ncol=6)

# ### [`CenterResizeCrop`](#)
#
# Resizes the sides so that one of them is no longer than `sz` and
# crops a region of size `sz` *from the center*.
#
# {cell=main result=false}

tfm = CenterResizeCrop((128, 128))

# {cell=main style="display:none;"}

o = showtransform(tfm, (image, bbox), 1)

# ### [`Crop`](#)`(sz[, from])`
#
# Crops a region of size `sz` from the image, *without resizing* the image first.
#
# {cell=main result=false}

using DataAugmentation: FromOrigin, FromCenter, FromRandom
tfms = [
    Crop((128, 128), FromOrigin()),
    Crop((128, 128), FromCenter()),
    Crop((128, 128), FromRandom()),
    Crop((128, 128), FromRandom()),
    Crop((128, 128), FromRandom()),
    Crop((128, 128), FromRandom()),
]

# {cell=main style="display:none;"}

o = showtransforms(tfms, (image, bbox))

# ### [`FlipX`](#), [`FlipY`](#), [`Reflect`](#)
#
# Flip the data on the horizontally and vertically, respectively. More generally, reflect
# around an angle from the x-axis.
#
# {cell=main result=false}

tfms = [
    FlipX(),
    FlipY(),
    Reflect(30),
]

# {cell=main style="display:none;"}

o = showtransforms(tfms, (image, bbox))

# ### [`Rotate`](#)
#
# Rotate counter-clockwise by an angle.
#
# {cell=main result=false}

tfm = Rotate(20) |> CenterCrop((256, 256))

# {cell=main style="display:none;"}

o = showtransform(tfm, (image, bbox), 1)
