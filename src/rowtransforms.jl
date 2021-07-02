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
        for (col, vals) in dict
            if any(ismissing, vals)
                dict[col] = filter(!ismissing, vals)
                @warn "There is a missing value present for category '$col' which will be removed from Categorify dict"
            end
        end
        new{T, S}(dict, cols)
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
            val = ismissing(val) ? 1 : findfirst(val .== tfm.dict[col]) + 1
        end
        (col, val)
    end)
    TabularItem(x, item.columns)
end
