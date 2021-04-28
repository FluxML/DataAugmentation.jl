include("imports.jl")


const ITEMS = (ArrayItem, Image, MaskBinary, MaskMulti, Keypoints, Polygon, BoundingBox)

@testset ExtendedTestSet "testitem" begin
    for I in ITEMS
        @test testitem(I) isa I
    end
end
