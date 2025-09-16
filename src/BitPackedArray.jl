struct BitPackedArray{W,T,N,P<:AbstractArray{UInt8,N}} <: AbstractArray{T,N}
    packed_bytes::P
end

BitPackedArray(args...; kws...) = throw(ArgumentError("Use `bitpacked(x, W)` to create a `BitPackedArray`"))

function BitPackedArray{W,T,N}(packed_bytes::P) where {W,T,N,P<:AbstractArray{UInt8,N}}
    BitPackedArray{W,T,N,P}(packed_bytes)
end

function bitpacked(x::AbstractArray{T,N}, w::Val{W}; kws...) where {T,N,W}
    packed_bytes = packbits(w, x; kws...)
    return BitPackedArray{W,T,N}(packed_bytes)
end

bitpacked(x::BitPackedArray{W}, ::Val{W}; kws...) where W = x
bitpacked(x::AbstractArray{T}, W::Int=bitwidth(T); kws...) where T = bitpacked(x, Val(W); kws...)

bitunpacked(x::BitPackedArray{W,T}) where {W,T} = unpackbits(Val(W), x.packed_bytes, T)
bitunpacked(x::AbstractArray) = x

Base.size(x::BitPackedArray{W}) where W = (size(x.packed_bytes, 1) * 8 ÷ W, size(x.packed_bytes)[2:end]...)

packed_chunk_index(w::Val, i::Int) = cld(i, chunk_size(w))
index_in_chunk(w::Val, i::Int) = mod1(i, chunk_size(w))

function Base.getindex(x::BitPackedArray{W}, (i, js...)::Int...) where W
    @boundscheck checkbounds(x, i, js...)
    packed_chunks = get_packed_chunks(Val(W), x.packed_bytes)
    packed_chunk = packed_chunks[packed_chunk_index(Val(W), i), js...]
    chunk = unpackchunk(Val(W), packed_chunk)
    value = chunk[index_in_chunk(Val(W), i)]
    return reinterpret(eltype(x), value)
end

function Base.setindex!(x::BitPackedArray{W}, value, (i, js...)::Int...) where W
    @boundscheck checkbounds(x, i, js...)
    packed_chunks = get_packed_chunks(Val(W), x.packed_bytes)
    packed_chunk = packed_chunks[packed_chunk_index(Val(W), i), js...]
    chunk = unpackchunk(Val(W), packed_chunk)
    new_chunk = Base.setindex(chunk, reinterpret(UInt8, value), index_in_chunk(Val(W), i))
    packed_chunks[packed_chunk_index(Val(W), i), js...] = packchunk(Val(W), new_chunk)
    return x
end

packbits!(dest::BitPackedArray{W}, x::AbstractArray{T}) where {W,T} = packbits!(Val(W), dest.packed_bytes, x)

Base.copy(x::BitPackedArray{W,T,N}) where {W,T,N} = BitPackedArray{W,T,N}(copy(x.packed_bytes))

Base.print_array(io::IO, x::BitPackedArray{W,T}) where {W,T} = Base.print_array(io, bitunpacked(x))

"""
    BitPackedArray{W,T,N,P<:AbstractArray{UInt8,N}} <: AbstractArray{T,N}

A wrapper for an array of packed bytes.

Use [`bitpacked`](@ref) to create a `BitPackedArray`, and [`bitunpacked`](@ref) to unpack it.

!!! warning
    Attempting to index into a `BitPackedArray` on GPUs may lead to scalar indexing errors.
"""
BitPackedArray

"""
    bitpacked(x::AbstractArray, W=bitwidth(T))

Returns a [`BitPackedArray`](@ref) of the same size as `x`, with each element packed into `W` bits.

`bitpacked` is a no-op if `x` is already a `BitPackedArray` with the same bitwidth.

# Examples

```jldoctest
julia> x = rand(Bool, 8, 2);

julia> packed = bitpacked(x, 1);

julia> packed == x
true

julia> packed .= [true false];

julia> all(packed[:,1])
true

julia> !any(packed[:,2])
true
```
"""
bitpacked

"""
    bitunpacked(x::BitPackedArray)

Returns an array of the same size as `x`, with each element unpacked from `W` bits.
"""
bitunpacked

### Broadcasting

Broadcast.broadcastable(x::BitPackedArray) = bitunpacked(x)

struct BitPackedArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end

Broadcast.BroadcastStyle(::Type{<:BitPackedArray{W,T,N}}) where {W,T,N} = BitPackedArrayStyle{N}()
(::Type{<:BitPackedArrayStyle})(::Val{N}) where N = BitPackedArrayStyle{N}()

function Base.copyto!(x::BitPackedArray{W}, bc::Broadcast.Broadcasted) where W
    packbits!(x, Broadcast.materialize(bc))
    return x
end

using FillArrays

function Base.materialize!(x::BitPackedArray{W}, bc::Broadcast.Broadcasted{<:Broadcast.AbstractArrayStyle{0}}) where W
    v, = bc.args
    packbits!(x, Fill(v, size(x)))
    return x
end
