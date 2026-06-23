using StaticArrays

function unsigned_type(N::Int)
    N ==  1 ? UInt8   :
    N ==  2 ? UInt16  :
    N ==  4 ? UInt32  :
    N ==  8 ? UInt64  :
    N == 16 ? UInt128 :
    error("Could not find a $N-byte Unsigned type")
end

const Bytes{N} = NTuple{N,UInt8}
const Storage = Union{Unsigned, Bytes}

_prev_bitwidth(::Type{UInt8}) = 0
_prev_bitwidth(::Type{T}) where T<:Unsigned = bitwidth(T) ÷ 2
_prev_bitwidth(::Type{NTuple{N,UInt8}}) where N = 8 * (N - 1)

"""
    NArray{T,N,S,D} <: StaticArray{S,T,N}
    NArray{T,N,S}(values::NTuple)
    NVector{T,L}
    NMatrix{T,M,N}

Small statically-sized array whose elements are packed into a single storage
value.

`NArray` behaves like a `StaticArrays.StaticArray`, but stores all lanes in the
`data` field as either an unsigned integer or an `NTuple` of bytes. The number of
bits used for each lane is `bitwidth(T)`, so the total packed width is
`bitwidth(T) * length(x)`.

Use `NVector{T,L}` and `NMatrix{T,M,N}` for the common one- and two-dimensional
forms. Construct from an `NTuple` of logical values:

```julia
julia> v = NVector(true, false, true, false, true, false, true, false)
8-element NVector{Bool, 8, UInt8} with indices SOneTo(8):
 1
 0
 1
 0
 1
 0
 1
 0

julia> reinterpret(UInt8, v)
0x55
```

Broadcasting treats an `NArray` as the unpacked static array of its logical
values, while `reinterpret` can be used to access the packed storage.
"""
struct NArray{T,N,S<:NTuple{N,Any},D<:Storage} <: StaticArray{S,T,N}
    data::D
    function NArray{T,N,S,D}(data::D) where {T,N,S<:NTuple{N,Any},D<:Storage}
        arr = new{T,N,S,D}(data)
        _prev_bitwidth(D) < bitwidth(arr) <= bitwidth(data) ||
            throw(ArgumentError("Cannot fit $(length(arr)) $(bitwidth(T))-bit element(s) in $(D)"))
        return arr
    end
end

NArray{T,N,S}(data::D) where {T,N,S<:NTuple{N,Any},D<:Unsigned} =
    NArray{T,N,S,D}(data)

bitwidth(::Type{T}) where T<:NArray = bitwidth(eltype(T)) * length(T)

function NArray{T,N,S}(xs::NTuple{L,T}) where {T,N,S<:NTuple{N,Any},L}
    unpacked_bytes = reinterpret.(unsigned_type(sizeof(T)), xs)
    data = pack(Val(bitwidth(T)), unpacked_bytes)
    arr = NArray{T,N,S,typeof(data)}(data)
    length(arr) == L || throw(ArgumentError("Type expects $(length(arr)) element(s), but got $L."))
    return arr
end
NArray{T,N,S}(xs::Tuple) where {T,N,S<:NTuple{N,Any}} =
    NArray{T,N,S}(convert.(T, xs))
NArray{T,N,S}(xs...) where {T,N,S<:NTuple{N,Any}} = NArray{T,N,S}(xs)

function StaticArrays.similar_type(
    ::Type{<:NArray{T,<:Any,S}},
    ::Type{T′}=T,
    ::Size{S′}=Size(S)
) where {T,S,T′,S′}
    return NArray{T′,length(S′),Tuple{S′...}}
end

const NVector{T,L} = NArray{T,1,Tuple{L}}
const NMatrix{T,S₁,S₂} = NArray{T,2,Tuple{S₁,S₂}}

NVector(xs::NTuple{L,T}) where {L,T} = NVector{T,L}(xs)

function Base.Tuple(v::NArray)
    unpacked_bytes = unpack(Val(bitwidth(eltype(v))), Val(length(v)), v.data)
    xs = reinterpret.(eltype(v), unpacked_bytes)::NTuple{length(v)}
    return xs
end

NArray(xs::StaticArray{S,T}) where {N,S<:NTuple{N,Any},T} = NArray{T,N,S}(Tuple(xs))
NArray{T,N,S}(xs::StaticArray{S,T,N}) where {T,N,S<:NTuple{N,Any}} =
    NArray{T,N,S}(Tuple(xs))
NArray{T,N,S}(xs::StaticArray{S,<:Any,N}) where {T,N,S<:NTuple{N,Any}} =
    NArray{T,N,S}(convert.(T, Tuple(xs)))

(::Type{SArray{S,T,N,L}})(xs::NArray{T,N,S}) where {S,T,N,L} =
    SArray{S,T,N,L}(Tuple(xs))
(::Type{SArray{S,T,N,L}})(xs::NArray{<:Any,N,S}) where {S,T,N,L} =
    SArray{S,T,N,L}(convert.(T, Tuple(xs)))

Base.IndexStyle(::Type{<:NArray}) = IndexLinear()

function Base.getindex(arr::NArray{T}, i::Int) where T
    W = bitwidth(T)
    S = length(arr)
    u = packed_getindex(Val(W), Val(S), arr.data, i)
    return reinterpret(T, u)
end

Base.broadcastable(v::NArray{<:Any,<:Any,S}) where S = SArray{S}(Tuple(v))

Base.reinterpret(::Type{T}, v::NArray) where T = reinterpret(T, v.data)
Base.reinterpret(::Type{Unsigned}, v::NArray) = reinterpret(unsigned_type(sizeof(v)), v)
