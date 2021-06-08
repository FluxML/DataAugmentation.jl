struct NormalizeRow <: DataAugmentation.Transform
	normstats
	normidxs
end

struct Categorify <: DataAugmentation.Transform
	pooldict
	categoryidxs
end

struct FillMissing <: DataAugmentation.Transform
	fmvals
	contidxs
	catidxs
end

function DataAugmentation.apply(tfm::FillMissing, item::TabularItem; randstate=nothing)
	x = (; zip(item.columns, [data for data in item.data])...)
	for idx in tfm.contidxs
		if ismissing(x[item.idxcolmap[idx]])
			Setfield.@set! x[item.idxcolmap[idx]] = tfm.fmvals[idx]
		end
	end
	for idx in tfm.catidxs
		if ismissing(x[item.idxcolmap[idx]])
			Setfield.@set! x[item.idxcolmap[idx]] = "missing"
		end
	end
	TabularItem(x, item.columns, item.idxcolmap)
end

function DataAugmentation.apply(tfm::NormalizeRow, item::TabularItem; randstate=nothing)
	x = (; zip(item.columns, [data for data in item.data])...)
	for idx in tfm.normidxs
		colmean, colstd = tfm.normstats[idx]
		Setfield.@set! x[item.idxcolmap[idx]] = (x[item.idxcolmap[idx]] - colmean)/colstd
	end
	TabularItem(x, item.columns, item.idxcolmap)
end

function DataAugmentation.apply(tfm::Categorify, item::TabularItem; randstate=nothing)
	x = (; zip(item.columns, [data for data in item.data])...)
	for idx in tfm.categoryidxs
		if ismissing(x[idx])
			Setfield.@set! x[item.idxcolmap[idx]] = "missing"
		end
		Setfield.@set! x[item.idxcolmap[idx]] = tfm.pooldict[idx].invindex[x[item.idxcolmap[idx]]]
	end
	TabularItem(x, item.columns, item.idxcolmap)
end

function getcategorypools(catdict, catidxs)
	pooldict = Dict()
	for idx in catidxs
		catarray = CategoricalArrays.categorical(catdict[idx])
        CategoricalArrays.levels!(catarray, ["missing", CategoricalArrays.levels(catarray)...])
        pooldict[idx] = catarray.pool
	end
	pooldict
end