module cuTileExt

using BitPacking: NarrowArray, bitwidth

import Adapt
import cuTile as ct
using cuTile: KernelAdaptor, TileArray

struct ReinterpretTileArray{T,N,A<:TileArray{UInt8,N}}
    parent::A
end

Base.parent(arr::ReinterpretTileArray) = arr.parent
Base.eltype(::ReinterpretTileArray{T}) where T = T
Base.ndims(::ReinterpretTileArray{T,N}) where {T,N} = N

function Base.size(arr::ReinterpretTileArray, i::Integer)
    ratio = 8 ÷ bitwidth(eltype(arr))
    return i == 1 ? size(parent(arr), i) * ratio : size(parent(arr), i)
end
Base.size(arr::ReinterpretTileArray) = ntuple(i -> size(arr, i), Val(ndims(arr)))

function Adapt.adapt_structure(to::KernelAdaptor, arr::NarrowArray)
    parent = Adapt.adapt(to, reinterpret(UInt8, arr))
    return ReinterpretTileArray{eltype(arr),ndims(parent),typeof(parent)}(parent)
end

function ct.store(arr::ReinterpretTileArray, index, tile; kws...)
    return ct.store(parent(arr), index, reinterpret(UInt8, tile); kws...)
end

function ct.load(arr::ReinterpretTileArray, index, shape; kws...)
    ratio = 8 ÷ bitwidth(eltype(arr))
    shape′ = ntuple(Val(ndims(arr))) do i
        i == 1 ? shape[i] ÷ ratio : shape[i]
    end
    byte_tile = ct.load(parent(arr), index, shape′; kws...)
    tile = reinterpret(eltype(arr), byte_tile)
    return tile
end

end
