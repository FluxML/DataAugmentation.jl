struct SamplePipeline
    steps
    getxy
    SamplePipeline(steps, getxy = identity) = new(steps, getxy)
end

function apply(pipeline::SamplePipeline, sample::Dict)
    sample = copy(sample)
    for (tfm, names) in pipeline.steps
        sample = applystep(tfm, sample, names)
    end
    return return pipeline.getxy(sample)
end

(pipeline::SamplePipeline)(sample) = apply(pipeline, sample)

"""
    applystep(tfm, items, names)

Applies `tfm` inplace to items in `items` given by keys `names`.
"""
function applystep(tfm::Transform, items, names::NTuple{N, Symbol}) where N
    results = apply(tfm, Tuple(items[name] for name in names))
    for (name, result) in zip(names, results)
        items[name] = result
    end
    return items
end


function applystep(tfm::Transform, items, names::Pair{Symbol, Symbol})
    arg, res = names
    items[res] = apply(tfm, items[arg])
    return items
end


function applystep(
        f,
        items,
        names::Pair{NTuple{N, Symbol}, Symbol}) where N
    args, res = names
    items[res] = f([items[arg] for arg in args]...)
    return items
end


function applystep(f, items, names::NTuple{N, Symbol}) where N
    results = f(Tuple(items[name] for name in names)...)
    for (name, result) in zip(names, results)
        items[name] = result
    end
    return items
end
