# ## `base.jl`
#
# First, we define the abstract types for each concept.

"""
    abstract type AbstractItem

Abstract supertype for all items. To implement items, subtype
either [`Item`](#) to create a new item or [`ItemWrapper`](#)
to wrap an existing item.
"""
abstract type AbstractItem end


"""
    abstract type Item

Abstract supertype of concrete items.

Subtype if you want to create a new item. If you want to wrap
an existing item, see [`ItemWrapper`](#).
"""
abstract type Item <: AbstractItem end


"""
    showitem(item)

Visualize `item`. Should return an image.
"""
function showitem end

function showitem(items::Tuple)
    return blend.(showitem.(items)...)
end


"""
    abstract type Transform

Abstract supertype for all transformations.
"""
abstract type Transform end

# Plus the default implementations of [`itemdata`](#) and
# [`getrandstate`](#).

"""
    itemdata(item)
    itemdata(items)

Access the data wrapped in `item` or a tuple of items.
"""
itemdata(item::Item) = item.data
itemdata(items::Tuple) = itemdata.(items)

"""
    getrandstate(transform)

Generates random state for stochastic transformations.
Calling `apply(tfm, item)` is equivalent to
`apply(tfm, item; randstate = getrandstate(tfm))`. It
defaults to `nothing`, so you it only needs to be implemented
for stochastic `Transform`s.
"""
getrandstate(::Transform) = nothing

# The following 2 methods are convenience for `apply` so that
# you don't have to explicitly handle the random state. Passing
# a tuple of `Item`s to `apply` will also use the same random
# state for each item.

"""
    apply(tfm, item[; randstate])
    apply(tfm, items[; randstate])

Apply `tfm` to an `item` or a tuple `items`.

"""
apply(tfm::Transform, items) = apply(tfm, items; randstate = getrandstate(tfm))



function apply(tfm::Transform, items::Tuple; randstate = getrandstate(tfm))
    map(item -> apply(tfm, item; randstate = randstate), items)
end


#


"""
    compose(transforms...)

Compose tranformations. Use `|>` as an alias.

Defaults to creating a [`Sequence`](#) of transformations,
but smarter behavior can be implemented.
For example, `MapElem(f) |> MapElem(g) == MapElem(g ∘ f)`.
"""
function compose end
compose(tfm) = tfm
compose(tfms...) = compose(compose(tfms[1], tfms[2]), tfms[3:end]...)
Base.:(|>)(tfm1::Transform, tfm2::Transform) = compose(tfm1, tfm2)

# [`Identity`](#) is the identity transformation.

"""
    Identity()

The identity transformation.
"""
struct Identity <: Transform end

apply(::Identity, item::Item; randstate = nothing) = item

compose(::Identity, ::Identity) = Identity()
compose(tfm::Transform, ::Identity) = tfm
compose(::Identity, tfm::Transform) = tfm

# Lastly, [`setdata`](#) provides a convenient way to create a copy
# of an item, replacing only the wrapped data. This relies on the
# wrapped data field being named `data`, though.

function setdata(item::Item, data)
    item = Setfield.@set item.data = data
    return item
end
