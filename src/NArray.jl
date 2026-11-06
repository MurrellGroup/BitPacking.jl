using StaticArrays

# Tags the inner constructor that takes packed storage rather than values
struct FromStorage end

"""
    NArray{T,N,S,D} <: StaticArray{S,T,N}
    NArray{T,N,S}(values::Tuple)
    NVector{T,L}
    NMatrix{T,M,N}

Small statically sized array whose elements are packed into a single storage
value.

`NArray` behaves like a `StaticArrays.StaticArray`, but stores its elements
LSB-first in the `data` field: element 1 occupies the lowest
[`bitwidth`](@ref)`(T)` bits, element 2 the next, and so on, in linear index
order. The storage `D` is the unsigned integer of the total bits rounded up to
whole bytes when there is one, and an `NTuple` of bytes otherwise; padding bits
past the last element are always zero. Elements may be 1 to 128 bits wide.

Use `NVector{T,L}` and `NMatrix{T,M,N}` for the common one- and two-dimensional
forms. Constructors always take values:

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

julia> reinterpret(Unsigned, v)
0x55
```

`reinterpret(T, x)` follows the array rule, as for Base arrays: the bits are kept,
and the length of dimension 1 is rescaled so that the total bit count stays the
same, throwing when it does not divide. `reinterpret(Unsigned, x)` returns the
storage word, and `reinterpret(NVector{T,L}, word)` is its inverse.

```julia
julia> reinterpret(UInt8, v)
1-element NVector{UInt8, 1, UInt8} with indices SOneTo(1):
 0x55

julia> reinterpret(NVector{Bool,8}, 0x0f) == NVector{Bool,8}(1, 1, 1, 1, 0, 0, 0, 0)
true
```

`map` keeps an `NArray` packed, while broadcasting treats it as the unpacked
`SArray` of its values.
"""
struct NArray{T,N,S<:NTuple{N,Any},D<:Storage} <: StaticArray{S,T,N}
    data::D
    function NArray{T,N,S,D}(::FromStorage, data::D) where {T,N,S<:NTuple{N,Any},D<:Storage}
        D′ = narray_storage(T, S)
        D === D′ || throw(ArgumentError("$(NArray{T,N,S}) stores its bits in $D′, not $D"))
        return new{T,N,S,D}(data)
    end
end

NArray{T,N,S}(tag::FromStorage, data::D) where {T,N,S<:NTuple{N,Any},D<:Storage} =
    NArray{T,N,S,D}(tag, data)

narray_storage(::Type{T}, ::Type{S}) where {T,S} =
    storage_type(Val(check_bitwidth(T) * StaticArrays.tuple_prod(S)))

bitwidth(::Type{<:NArray{T,N,S}}) where {T,N,S} = bitwidth(T) * StaticArrays.tuple_prod(S)

function NArray{T,N,S,D}(xs::Tuple) where {T,N,S<:NTuple{N,Any},D}
    L = StaticArrays.tuple_prod(S)
    length(xs) == L ||
        throw(ArgumentError("$(NArray{T,N,S}) has $L element(s), got $(length(xs))"))
    raws = map(x -> raw_bits(convert(T, x)), xs)
    return NArray{T,N,S,D}(FromStorage(), pack(Val(check_bitwidth(T)), D, raws))
end
NArray{T,N,S}(xs::Tuple) where {T,N,S<:NTuple{N,Any}} = NArray{T,N,S,narray_storage(T, S)}(xs)

function StaticArrays.similar_type(
    ::Type{<:NArray{T,<:Any,S}},
    ::Type{T′}=T,
    ::Size{S′}=Size(S)
) where {T,S,T′,S′}
    return NArray{T′,length(S′),Tuple{S′...}}
end

const NVector{T,L} = NArray{T,1,Tuple{L}}
const NMatrix{T,S₁,S₂} = NArray{T,2,Tuple{S₁,S₂}}

NVector(xs::Tuple{T,Vararg{T}}) where T = NVector{T,length(xs)}(xs)
NVector(xs::Tuple{Any,Vararg}) = NVector(promote(xs...))

function Base.Tuple(v::NArray{T,N,S}) where {T,N,S}
    raws = unpack(Val(bitwidth(T)), Val(StaticArrays.tuple_prod(S)), raw_type(T), v.data)
    return map(bits -> from_raw_bits(T, bits), raws)
end

NArray(xs::StaticArray{S,T}) where {N,S<:NTuple{N,Any},T} = NArray{T,N,S}(Tuple(xs))
NArray{T,N,S}(xs::StaticArray{S,<:Any,N}) where {T,N,S<:NTuple{N,Any}} =
    NArray{T,N,S}(Tuple(xs))

(::Type{SArray{S,T,N,L}})(xs::NArray{T,N,S}) where {S,T,N,L} =
    SArray{S,T,N,L}(Tuple(xs))
(::Type{SArray{S,T,N,L}})(xs::NArray{<:Any,N,S}) where {S,T,N,L} =
    SArray{S,T,N,L}(convert.(T, Tuple(xs)))

Base.IndexStyle(::Type{<:NArray}) = IndexLinear()

@inline function Base.getindex(v::NArray{T}, i::Int) where T
    @boundscheck checkbounds(v, i)
    return from_raw_bits(T, packed_getindex(Val(bitwidth(T)), raw_type(T), v.data, i))
end

Base.broadcastable(v::NArray{<:Any,<:Any,S}) where S = SArray{S}(Tuple(v))

# The size of `S` with dimension 1 rescaled from `W`-bit to `W′`-bit elements
@generated function reinterpret_size(::Type{S}, ::Val{W}, ::Val{W′}) where {S,W,W′}
    dims = collect(Int, S.parameters)
    if isempty(dims)
        W == W′ && return S
        return :(throw(ArgumentError($("cannot reinterpret a zero-dimensional array of a $W-bit value as $W′-bit values"))))
    end
    bits = dims[1] * W
    bits % W′ == 0 || return :(throw(ArgumentError($(
        "cannot reinterpret $(dims[1]) $W-bit value(s) along dimension 1 as $W′-bit values"))))
    return Tuple{bits ÷ W′,dims[2:end]...}
end

function Base.reinterpret(::Type{T′}, v::NArray{T,N,S,D}) where {T′,T,N,S,D}
    S′ = reinterpret_size(S, Val(bitwidth(T)), Val(check_bitwidth(T′)))
    return NArray{T′,N,S′,D}(FromStorage(), v.data)
end

function Base.reinterpret(::Type{Unsigned}, v::NArray{T,N,S,D}) where {T,N,S,D}
    D <: Unsigned ||
        throw(ArgumentError("$(NArray{T,N,S}) stores its bits in $D, not in an unsigned integer"))
    return v.data
end

function Base.reinterpret(::Type{A}, word::U) where {T,N,S,A<:NArray{T,N,S},U<:Unsigned}
    D = narray_storage(T, S)
    NArray{T,N,S,D} <: A ||
        throw(ArgumentError("$(NArray{T,N,S}) stores its bits in $D, so it is not a $A"))
    U === D || throw(ArgumentError("$(NArray{T,N,S}) stores its bits in $D, not $U"))
    bits = bitwidth(NArray{T,N,S})
    padding_is_zero(word, bits) ||
        throw(ArgumentError("$word has bits set past the $bits value bits of $(NArray{T,N,S})"))
    return NArray{T,N,S,D}(FromStorage(), word)
end
