

function scaleprojection(scales::NTuple{N}, T = Float32) where N
    a = zeros(Float32, N, N)
    a[I(N)] = SVector{N}(scales)
    return SArray{Tuple{N, N}}(a)
end
