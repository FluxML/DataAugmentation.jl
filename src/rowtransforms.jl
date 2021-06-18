struct NormalizeRow <: DataAugmentation.Transform
	normstats
	normcols
end

struct Categorify <: DataAugmentation.Transform
	pooldict
	categorycols
end

struct FillMissing <: DataAugmentation.Transform
	fmvals
	contcols
	catcols
end

function DataAugmentation.apply(tfm::FillMissing, item::TabularItem; randstate=nothing)
	x = (; zip(item.columns, [data for data in item.data])...)
	for col in tfm.contcols
		if ismissing(x[col])
			Setfield.@set! x[col] = tfm.fmvals[col]
		end
	end
	for col in tfm.catcols
		if ismissing(x[col])
			Setfield.@set! x[col] = "missing"
		end
	end
	TabularItem(x, item.columns)
end

function DataAugmentation.apply(tfm::NormalizeRow, item::TabularItem; randstate=nothing)
	x = (; zip(item.columns, [data for data in item.data])...)
	for col in tfm.normcols
		colmean, colstd = tfm.normstats[col]
		Setfield.@set! x[col] = (x[col] - colmean)/colstd
	end
	TabularItem(x, item.columns)
end

function DataAugmentation.apply(tfm::Categorify, item::TabularItem; randstate=nothing)
	x = (; zip(item.columns, [data for data in item.data])...)
	for col in tfm.categorycols
		if ismissing(x[col])
			Setfield.@set! x[col] = "missing"
		end
		Setfield.@set! x[col] = tfm.pooldict[col].invindex[x[col]]
	end
	TabularItem(x, item.columns)
end

function getcategorypools(catdict, catcols)
	pooldict = Dict()
	for col in catcols
		catarray = CategoricalArrays.categorical(catdict[col])
        CategoricalArrays.levels!(catarray, ["missing", CategoricalArrays.levels(catarray)...])
        pooldict[col] = catarray.pool
	end
	pooldict
end