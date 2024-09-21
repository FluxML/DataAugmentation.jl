include("imports.jl")


const ITEMS = (ArrayItem, Image, MaskBinary, MaskMulti, Keypoints, Polygon, BoundingBox)

@testset ExtendedTestSet "testitem" begin
    for I in ITEMS
        @test testitem(I) isa I
    end
end

@testset ExtendedTestSet "$(N)D $(I) extrapolate (constant)" for N in (2, 3), I in (Image, MaskBinary, MaskMulti)
    item = testitem(I{N}; extrapolate=1)
    data = item |> itemdata

    bounds = getbounds(item)
    sizes = length.(bounds.rs)

    tfm = CenterCrop(ntuple(i -> 3*sizes[i], N))
    transformed = apply(tfm, item) |> itemdata

    @test data ≈ transformed[bounds.rs...]

    expected_value = convert(eltype(data), 1)
    for d in 1:N
        extrapolated = selectdim(transformed, d, (1-sizes[d]):0)
        @test all(v -> v ≈ expected_value, extrapolated)
        extrapolated = selectdim(transformed, d, (sizes[d]+1):2*sizes[d])
        @test all(v -> v ≈ expected_value, extrapolated)
    end
end

@testset ExtendedTestSet "$(N)D $(I) extrapolate (flat)" for N in (2, 3), I in (Image, MaskBinary, MaskMulti)
    item = testitem(I{N}; extrapolate=Interpolations.Flat())
    data = item |> itemdata

    bounds = getbounds(item)
    sizes = length.(bounds.rs)

    tfm = CenterCrop(ntuple(i -> 2*sizes[i], N))
    transformed = apply(tfm, item) |> itemdata

    expected = similar(transformed)
    for index in CartesianIndices(expected)
        clamped = clamp.(Tuple(index), bounds.rs)
        expected[index] = data[clamped...]
    end
    @test transformed ≈ expected
end

@testset ExtendedTestSet "$(N)D $(I) extrapolate (reflect)" for N in (2, 3), I in (Image, MaskBinary, MaskMulti)
    item = testitem(I{N}; extrapolate=Interpolations.Reflect())
    data = item |> itemdata

    bounds = getbounds(item)
    sizes = length.(bounds.rs)

    tfm = Crop(ntuple(i -> i==1 ? 2 * sizes[i] - 1 : sizes[i], N))
    transformed = apply(tfm, item) |> itemdata

    @test transformed[ntuple(i -> bounds.rs[i], N)...] ≈ data
    @test transformed[ntuple(i -> i==1 ? (2*sizes[i]-1:-1:sizes[i]) : bounds.rs[i], N)...] ≈ data
end

@testset ExtendedTestSet "$(N)D $(I) interpolate (nearest)" for N in (2, 3), I in (Image, MaskBinary, MaskMulti)
    item = testitem(I{N}; interpolate=Interpolations.BSpline(Interpolations.Constant()))
    data = item |> itemdata

    tfm = Project(LinearMap(UniformScaling(2)))
    transformed = apply(tfm, item) |> itemdata

    expected = similar(transformed)
    for index in CartesianIndices(expected)
        original = map(i -> ceil(Int, i/2), Tuple(index))
        expected[index] = data[original...]
    end
    @test transformed ≈ expected
end

@testset ExtendedTestSet "$(N)D $(I) interpolate (linear)" for N in (2, 3), I in (MaskBinary, MaskMulti)
    tfm = Project(LinearMap(UniformScaling(2)))
    item = testitem(I{N}; interpolate=Interpolations.BSpline(Interpolations.Linear()))
    @test_throws InexactError apply(tfm, item)
end

@testset ExtendedTestSet "$(N)D $(I) interpolate (linear)" for N in (2,3), I in (Image,)
    item = testitem(I{N}; interpolate=Interpolations.BSpline(Interpolations.Linear()))
    data = item |> itemdata

    tfm = Project(LinearMap(UniformScaling(2)))
    transformed = apply(tfm, item) |> itemdata

    expected = similar(transformed)
    for index in CartesianIndices(expected)
        neighborhood = map(Tuple(index)) do i
            (floor(Int, i/2) : ceil(Int, i/2))
        end
        expected[index] = mean(data[neighborhood...])
    end
    @test all(t ≈ e for (t, e) in zip(transformed, expected))
end
