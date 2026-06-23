"""
    NarrowArray{T}(array::AbstractArray)
    NarrowArray{T,N,L}(array::AbstractArray)

Array wrapper whose parent stores packed [`NArray`](@ref) chunks.

`NarrowArray{T}` presents an `AbstractArray{T}` interface while storing groups
of logical values in the first dimension as `NVector` chunks. For an element type
`T`, the default chunk length is `pack_count(T) == 8 ÷ gcd(bitwidth(T), 8)`.
Use `NarrowArray{T,N,L}` or `NarrowVector{T,L}` to choose a wider chunk length.

The constructor packs an existing logical array:

```julia
julia> x = NarrowArray{Bool}([true, false, true, false, true, false, true, false])
8-element NarrowVector{Bool, 8, Vector{NVector{Bool, 8, UInt8}}}:
 1
 0
 1
 0
 1
 0
 1
 0
```

The input's first dimension must contain whole chunks. For example,
`NarrowArray{Bool}` requires the first dimension to be divisible by 8, and a
4-bit element type requires it to be divisible by 2.

`NarrowArray{T}(x)` converts values to `T` before packing when `eltype(x) != T`.
`parent(x)` exposes the packed chunk array. `copy(x)` materializes the logical
values, while `reinterpret(T, x)` reinterprets the whole logical bit buffer and
rescales the leading dimension by `bitwidth`. Type broadcasts such as `T.(x)`
convert packed chunks to dense logical `T` values. Display uses a host-adapted
copy so GPU-backed parents do not print through scalar indexing.
"""
struct NarrowArray{T,N,L,A<:AbstractArray{<:NVector{T,L},N}} <: AbstractArray{T,N}
    parent::A
end

pack_count(::Type{T}) where T = 8 ÷ gcd(bitwidth(T), 8)

function check_chunk_length(::Type{T}, L) where T
    L isa Integer ||
        throw(ArgumentError("NarrowArray{$T} chunk length must be an integer, got $L"))
    L >= 1 ||
        throw(ArgumentError("NarrowArray{$T} chunk length must be positive, got $L"))

    bits = bitwidth(T) * Int(L)
    bits % 8 == 0 ||
        throw(ArgumentError("NarrowArray{$T,$L} chunks must contain a whole number of bytes, got $bits bits"))
    return Int(L)
end

NarrowArray{T}(arr::AbstractArray{S,N}) where {T,S,N} =
    NarrowArray{T,N,pack_count(T)}(arr)

NarrowArray{T,N}(arr::AbstractArray{S,N}) where {T,N,S} =
    NarrowArray{T,N,pack_count(T)}(arr)

function NarrowArray{T,N,L}(arr::AbstractArray{S,N}) where {T,N,L,S}
    isbitstype(S) ||
        throw(ArgumentError("NarrowArray{$T} input element type must be a bitstype, got $S"))

    chunk_length = check_chunk_length(T, L)
    size(arr, 1) % chunk_length == 0 ||
        throw(ArgumentError("the first dimension of a NarrowArray{$T,$N,$L} input must be divisible by $chunk_length, got $(size(arr, 1))"))

    chunks = if S === T
        NVector{T,chunk_length}.(reinterpret(NTuple{chunk_length,T}, arr))
    else
        NVector{T,chunk_length}.(SVector{chunk_length,S}.(reinterpret(NTuple{chunk_length,S}, arr)))
    end
    return NarrowArray(chunks)
end

NarrowArray{T}(arr::NarrowArray{T}) where T = arr
NarrowArray{T,N}(arr::NarrowArray{T,N}) where {T,N} = arr
NarrowArray{T,N,L}(arr::NarrowArray{T,N,L}) where {T,N,L} = arr
NarrowArray{T}(arr::NarrowArray) where T = NarrowArray{T}(T.(arr))
NarrowArray{T,N,L}(arr::NarrowArray{S,N}) where {T,N,L,S} = NarrowArray{T,N,L}(T.(arr))

const NarrowVector{T,L,A<:AbstractVector{<:NVector{T,L}}} = NarrowArray{T,1,L,A}
const NarrowMatrix{T,L,A<:AbstractMatrix{<:NVector{T,L}}} = NarrowArray{T,2,L,A}

Base.parent(arr::NarrowArray) = arr.parent

Adapt.adapt_structure(to, arr::NarrowArray) = NarrowArray(Adapt.adapt(to, parent(arr)))

inner_size(arr::NarrowArray, i::Integer) = size(eltype(parent(arr)), i)
inner_size(arr::NarrowArray{<:Any,N}) where N = ntuple(i -> inner_size(arr, i), Val(N))

Base.size(arr::NarrowArray, i::Integer) = size(parent(arr), i) * size(eltype(parent(arr)), i)
Base.size(arr::NarrowArray{<:Any,N}) where N = ntuple(i -> size(arr, i), Val(N))

Base.IndexStyle(::Type{<:NarrowArray}) = IndexCartesian()
function Base.getindex(arr::NarrowArray{T,N}, i::Vararg{Int,N}) where {T,N}
    outer_inner_i = ntuple(j -> fldmod1(i[j], inner_size(arr, j)), Val(N))
    outer_i = first.(outer_inner_i)
    inner_i = last.(outer_inner_i)
    return parent(arr)[outer_i...][inner_i...]
end

function narrow_chunk_storage_type(::Type{T}, L) where T
    chunk_length = check_chunk_length(T, L)
    bits = bitwidth(T) * chunk_length
    U = _unsigned_type_for(bits)
    return U !== nothing && 8 * sizeof(U) == bits ? U : NTuple{bits ÷ 8, UInt8}
end

narrow_chunk_type(::Type{T}, L=pack_count(T)) where T =
    NArray{T,1,Tuple{check_chunk_length(T, L)},narrow_chunk_storage_type(T, L)}

function reinterpret_bytes(arr::NarrowArray)
    E = eltype(parent(arr))
    bitwidth(E) == 8 * sizeof(E) ||
        throw(ArgumentError("cannot reinterpret a NarrowArray with padded chunk storage"))
    return reinterpret(UInt8, parent(arr))
end

function Base.reinterpret(::Type{T}, arr::NarrowArray{S}) where {T,S}
    source_bits = bitwidth(S)
    target_bits = bitwidth(T)
    target_bits > 0 ||
        throw(ArgumentError("cannot reinterpret to $T with non-positive bitwidth $target_bits"))
    (size(arr, 1) * source_bits) % target_bits == 0 ||
        throw(ArgumentError("cannot reinterpret $(size(arr, 1)) $source_bits-bit value(s) as $target_bits-bit $T"))

    bytes = reinterpret_bytes(arr)
    if target_bits == 8 * sizeof(T)
        return reinterpret(T, bytes)
    elseif target_bits < 8 && sizeof(T) == 1
        return NarrowArray(reinterpret(narrow_chunk_type(T), bytes))
    else
        throw(ArgumentError("cannot reinterpret packed bytes as $T with bitwidth $target_bits and sizeof $(sizeof(T))"))
    end
end

function Base.Broadcast.broadcasted(::Type{T}, arr::NarrowArray) where T
    isbitstype(T) || return Base.Broadcast.Broadcasted(T, (arr,))

    L = length(eltype(parent(arr)))
    chunks = SVector{L,T}.(parent(arr))
    return reinterpret(T, chunks)
end

Base.copy(arr::NarrowArray) = reinterpret(eltype(arr), map(SArray, parent(arr)))

function Base.print_array(io::IO, arr::NarrowArray)
    host = Adapt.adapt(Array, arr)
    if host isa AbstractVecOrMat
        return invoke(Base.print_array, Tuple{IO, AbstractVecOrMat}, io, host)
    else
        return invoke(Base.print_array, Tuple{IO, AbstractArray}, io, host)
    end
end
