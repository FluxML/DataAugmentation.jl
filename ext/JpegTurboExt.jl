module JpegTurboExt

using DataAugmentation: JPEG, _map_channels, _photometric_color, _restore_photometric_color
using ImageCore: AbstractGray, Gray, N0f8, RGB, clamp01, color_type
using JpegTurbo: jpeg_encode, jpeg_decode
using OffsetArrays: no_offset_view

import DataAugmentation: photometric!

function photometric!(dst, ::JPEG, src::AbstractArray{T}, quality) where {T}
    ndims(src) == 2 || throw(ArgumentError("JPEG requires a two-dimensional image"))
    isempty(src) && return copyto!(dst, src)
    maximum(size(src)) <= 65500 || throw(ArgumentError("JPEG dimensions cannot exceed 65500 pixels"))
    C = T <: Real || color_type(T) <: AbstractGray ? Gray{N0f8} : RGB{N0f8}
    input = map(no_offset_view(src)) do pixel
        convert(C, _map_channels(clamp01, _photometric_color(pixel)))
    end
    bytes = jpeg_encode(input; quality = Int(quality))
    n = length(bytes)
    if n < 624
        # JpegTurbo rejects short streams. Pad the EOI marker with JPEG fill bytes.
        bytes = vcat(@view(bytes[1:(n - 1)]), fill(0xff, 624 - n), 0xd9)
    end
    output = jpeg_decode(C, bytes)
    for (i, j) in zip(eachindex(dst, src), eachindex(output))
        dst[i] = _restore_photometric_color(src[i], output[j])
    end
    return dst
end

end # module
