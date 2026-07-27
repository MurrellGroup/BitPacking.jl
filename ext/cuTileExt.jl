module cuTileExt

using BitPacking: NarrowArray, bitwidth

import Adapt
import cuTile as ct

struct ReinterpretTileArray{T,N,A<:ct.TileArray{UInt8,N}} <: ct.AbstractTileArray{T,N}
    parent::A

    function ReinterpretTileArray{T,N,A}(parent::A) where {T,N,A<:ct.TileArray{UInt8,N}}
        values_per_byte(T)
        return new{T,N,A}(parent)
    end
end

# Number of logical values stored per byte. The leading dimension is rescaled by
# this ratio, so element types that do not pack a whole number of values into a
# byte (6-bit floats, say) have no consistent shape on this side, even though
# `NarrowArray` stores them fine by chunking several values into whole bytes.
# `T` is a type parameter of the wrapper, so the check folds away in kernels.
function values_per_byte(::Type{T}) where T
    bits = bitwidth(T)
    (0 < bits <= 8 && 8 % bits == 0) || throw(ArgumentError(
        "cannot view a NarrowArray{$T} as tiles: $bits-bit values do not pack a whole number per byte"))
    return 8 ÷ bits
end

Base.parent(arr::ReinterpretTileArray) = arr.parent
Base.eltype(::ReinterpretTileArray{T}) where T = T
Base.ndims(::ReinterpretTileArray{T,N}) where {T,N} = N

function Base.size(arr::ReinterpretTileArray, i::Integer)
    ratio = values_per_byte(eltype(arr))
    return i == 1 ? size(parent(arr), i) * ratio : size(parent(arr), i)
end
Base.size(arr::ReinterpretTileArray) = ntuple(i -> size(arr, i), Val(ndims(arr)))

function Adapt.adapt_structure(to::ct.KernelAdaptor, arr::NarrowArray)
    parent = Adapt.adapt(to, reinterpret(UInt8, arr))
    return ReinterpretTileArray{eltype(arr),ndims(parent),typeof(parent)}(parent)
end

function ct.store(arr::ReinterpretTileArray, index, tile; kws...)
    return ct.store(parent(arr), index, reinterpret(UInt8, tile); kws...)
end

function ct.load(arr::ReinterpretTileArray, index, shape; kws...)
    ratio = values_per_byte(eltype(arr))
    shape′ = ntuple(Val(ndims(arr))) do i
        i == 1 ? shape[i] ÷ ratio : shape[i]
    end
    byte_tile = ct.load(parent(arr), index, shape′; kws...)
    tile = reinterpret(eltype(arr), byte_tile)
    return tile
end

end
