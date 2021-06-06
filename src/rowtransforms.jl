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
	itemx = TabularItem(x, item.columns, item.idxcolmap)
	for idx in tfm.contidxs
		if ismissing(itemx.data[itemx.idxcolmap[idx]])
			itemx = Setfield.@set itemx.data[itemx.idxcolmap[idx]] = tfm.fmvals[idx]
		end
	end
	for idx in tfm.catidxs
		if ismissing(itemx.data[itemx.idxcolmap[idx]])
			itemx = Setfield.@set itemx.data[itemx.idxcolmap[idx]] = "missing"
		end
	end
	itemx
end

function DataAugmentation.apply(tfm::NormalizeRow, item::TabularItem; randstate=nothing)
	x = (; zip(item.columns, [data for data in item.data])...)
	itemx = TabularItem(x, item.columns, item.idxcolmap)
	for idx in tfm.normidxs
		colmean, colstd = tfm.normstats[idx]
		itemx = Setfield.@set itemx.data[itemx.idxcolmap[idx]] = (item.data[idx] - colmean)/colstd
	end
	itemx
end

function DataAugmentation.apply(tfm::Categorify, item::TabularItem; randstate=nothing)
	x = (; zip(item.columns, [data for data in item.data])...)
	itemx = TabularItem(x, item.columns, item.idxcolmap)
	for idx in tfm.categoryidxs
		if ismissing(item.data[idx])
			itemx = Setfield.@set itemx.data[itemx.idxcolmap[idx]] = "missing"
		end
		itemx = Setfield.@set itemx.data[itemx.idxcolmap[idx]] = tfm.pooldict[idx].invindex[x[item.idxcolmap[idx]]]
	end
	itemx
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