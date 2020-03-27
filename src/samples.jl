
# SampleTransform
abstract type AbstractSampleTransform end

struct SampleTransformApplyAll <: AbstractSampleTransform
    ins::NTuple{N, Symbol} where N
    tfm
end
SampleTransformApplyAll(in::Symbol, tfm) = SampleTransformApplyAll((in,), tfm);

struct SampleTransformCombine <: AbstractSampleTransform
    ins::Tuple
    out::Symbol
    tfm
end

struct SampleTransformLambda <: AbstractSampleTransform
    f
end


AbstractSampleTransform(ins::NTuple{N, Symbol}, tfm) where N = SampleTransformApplyAll(ins, tfm)
AbstractSampleTransform(in_::Symbol, tfm) = SampleTransformApplyAll((in_,), tfm)
AbstractSampleTransform(ins::Tuple, out::Symbol, tfm) = SampleTransformCombine(ins, out, tfm)
AbstractSampleTransform(in_, out::Symbol, tfm) = SampleTransformCombine((in_,), out, tfm)
AbstractSampleTransform(f)::AbstractSampleTransform = SampleTransformLambda(f)


function (st::SampleTransformApplyAll)(sample)
    args = Tuple(sample[in_] for in_ in st.ins)
    outs = st.tfm(args)
    for (in_, out) in zip(st.ins, outs)
        sample[in_] = out
    end 
    
    return sample
end

function (st::SampleTransformCombine)(sample)
    args = Tuple(sample[in_] for in_ in st.ins)
    sample[st.out] = st.tfm(args...)
    return sample
end

function (st::SampleTransformLambda)(sample) 
    return st.f(sample)
end


struct SamplePipeline <: AbstractSampleTransform
    sampletransforms::NTuple{N, AbstractSampleTransform} where N
end

SamplePipeline(argss::AbstractVector) = SamplePipeline(
    Tuple(AbstractSampleTransform(args...) for args in argss)
)

(pipeline::SamplePipeline)(sample) = foldl((sample, f) -> f(sample), pipeline.sampletransforms; init = sample)
