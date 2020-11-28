
"""
    abstract type ItemWrapper{Item}

"""
abstract type ItemWrapper{Item} <: AbstractItem end


itemdata(wrapper::ItemWrapper) = itemdata(getwrapped(wrapper))
itemfield(wrapped::ItemWrapper) = :item

getwrapped(wrapped::ItemWrapper) = getfield(wrapped, itemfield(wrapped))

function setwrapped(wrapped::ItemWrapper, item)
    wrapped = Setfield.@set wrapped.item = item
    return wrapped
end


function apply(tfm::Transform, itemw::ItemWrapper; randstate = getrandstate(tfm))
    item = apply(tfm, getwrapped(itemw); randstate = randstate)
    itemw = setwrapped(itemw, item)
    return itemw
end
