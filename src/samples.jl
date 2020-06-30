struct Pipeline
    steps
    itemfn
    outfn
    Pipeline(steps...; itemfn = identity, outfn = identity) = new(steps, itemfn, outfn)
end

# TODO: make `Buffered` work with `Pipeline`

function (pipeline::Pipeline)(sample)
    sample = pipeline.itemfn(sample)
    for step in pipeline.steps
        sample = applystep!(sample, step)
    end
    return pipeline.outfn(sample)
end


abstract type PipelineStep end

"""
    ApplyStep(tfm, (:field1, :field2, …))
    ApplyStep(tfm, (:field1, :infield2 => outfield2, …)

`Pipeline` step that applies `tfm` to every `field` in `fields`.

If `field` is a symbol, the field is updated.

If `field` isa `Pair{Symbol, Symbol} = infield => outfield` then
the result is stored in `sample.outfield`.
"""
struct ApplyStep <: PipelineStep
    tfm::Transform
    fields
    ApplyStep(tfm, fields...) = new(tfm, fields)
end

function applystep!(sample, step::ApplyStep)
    infields = (field isa Symbol ? field : field[1] for field in step.fields)
    outfields = (field isa Symbol ? field : field[2] for field in step.fields)

    items = apply(step.tfm, [sample[infield] for infield in infields])
    for (item, outfield) in zip(items, outfields)
        sample[outfield] = item
    end

    return sample
end

"""
    CombineStep(f, (:field1, :field2, …), :outfield)
"""
struct CombineStep <: PipelineStep
    f
    infields
    outfield
end

function applystep!(sample, step::CombineStep)
    sample[step.outfield] = step.f([sample[field] for field in step.infields]...)
    return sample
end

"""
    MapStep(f, fields)

Updates `fields` with function `f(items...) = (newitems...)`
"""
struct MapStep <: PipelineStep
    f
    fields
end

function applystep!(sample, step::MapStep)
    outs = step.f([sample[field] for field in step.fields]...)
    for (field, out) in zip(step.fields, outs)
        sample[field] = out
    end
    return sample
end
