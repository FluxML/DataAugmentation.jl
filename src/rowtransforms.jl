struct NormalizeRow{T, S} <: DataAugmentation.Transform
        normstats::T
        normcols::S
end
NormalizeRow(normstats::T, normcols::S) where {T, S} = NormalizeRow{T, S}(normstats, normcols)

struct Categorify{T, S} <: DataAugmentation.Transform
        catdict::T
        categorycols::S
end
Categorify(catdict::T, categorycols::S) where {T, S} = Categorify{T, S}(catdict, categorycols)

struct FillMissing{T, S} <: DataAugmentation.Transform
        fmvals::T
        fmcols::S
end
FillMissing(fmvals::T, fmcols::S) where {T, S} = FillMissing{T, S}(fmvals, fmcols)

function DataAugmentation.apply(tfm::FillMissing, item::TabularItem; randstate=nothing)
    x = [val for val in item.data]
    for col in tfm.fmcols
            idx = findfirst(col .== item.columns)
            if ismissing(x[idx])
                x[idx] = tfm.fmvals[col]
            end
    end
    x = (; zip(item.columns, x)...)
    TabularItem(x, item.columns)
end

function DataAugmentation.apply(tfm::NormalizeRow, item::TabularItem; randstate=nothing)
    x = [val for val in item.data]
    for col in tfm.normcols
            idx = findfirst(col .== item.columns)
            colmean, colstd = tfm.normstats[col]
            x[idx] = (x[idx] - colmean)/colstd
    end
    x = (; zip(item.columns, x)...)
    TabularItem(x, item.columns)
end

function DataAugmentation.apply(tfm::Categorify, item::TabularItem; randstate=nothing)
    x = [val for val in item.data]
    for col in tfm.categorycols
        idx = findfirst(col .== item.columns)
        x[idx] = ismissing(x[idx]) ? 1 : findfirst(skipmissing(x[idx] .== tfm.catdict[col])) + 1
    end
    x = (; zip(item.columns, x)...)
    TabularItem(x, item.columns)
end
