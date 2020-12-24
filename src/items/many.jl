

"""
    Many(items)

Wrapper item for variable-length collections of items.

Use if you get errors in `apply!` when the length of the `bufs` and
`items` is not the same.
"""
struct Many{I<:Item} <: Item
    items
end

itemdata(many::Many) = itemdata.(many.items)

apply(::Transform)
