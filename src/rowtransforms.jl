struct NormalizeRow{T, S} <: Transform
    dict::T
    cols::S
end

struct FillMissing{T, S} <: Transform
    dict::T
    cols::S
end

struct Categorify{T, S}
    dict::T
    cols::S
    function Categorify{T, S}(dict::T, cols::S) where {T, S}
        newdict = Dict()
        for (col, vals) in dict
            newdict[col] = DataStructures.SortedSet{Union{Missing, eltype(vals)}}(Base.Order.ReverseOrdering(), vals)
            push!(newdict[col], missing)
        end
        new{typeof(newdict), S}(newdict, cols)
    end
end

Categorify(dict::T, cols::S) where {T, S} = Categorify{T, S}(dict, cols)

function apply(tfm::NormalizeRow, item::TabularItem; randstate=nothing)
    x = NamedTuple(Iterators.map(item.columns, item.data) do col, val
        if col in tfm.cols
            colmean, colstd = tfm.dict[col]
            val = (val - colmean)/colstd
        end
        (col, val)
    end)
    TabularItem(x, item.columns)
end

function apply(tfm::FillMissing, item::TabularItem; randstate=nothing)
    x = NamedTuple(Iterators.map(item.columns, item.data) do col, val
        if col in tfm.cols && ismissing(val)
            val = tfm.dict[col]
        end
        (col, val)
    end)
    TabularItem(x, item.columns)
end

function apply(tfm::Categorify, item::TabularItem; randstate=nothing)
    x = NamedTuple(Iterators.map(item.columns, item.data) do col, val
        if col in tfm.cols
            val = ismissing(val) ? 1 : findfirst(val .== skipmissing(tfm.dict[col])) + 1
            # val = findkey(tfm.dict[col], val).address
        end
        (col, val)
    end)
    TabularItem(x, item.columns)
end