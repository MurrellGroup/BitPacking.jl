# Packed storage is one unsigned word or a tuple of unsigned limbs. Values sit
# LSB-first: value `i` (0-based) of width `W` occupies bits `W*i` to `W*i + W - 1`
# of the bit stream, and bit `k` of the stream is bit `k % b` of limb `k ÷ b`
# for `b`-bit limbs. Bits past the last value are padding and always zero.

const Storage = Union{Unsigned,Tuple{Vararg{Unsigned}}}

const MAX_BITWIDTH = 128

function unsigned_type(N::Integer)
    N ==  1 ? UInt8   :
    N ==  2 ? UInt16  :
    N ==  4 ? UInt32  :
    N ==  8 ? UInt64  :
    N == 16 ? UInt128 :
    throw(ArgumentError("there is no $N-byte unsigned integer type"))
end

# The unsigned type `raw_bits` returns for `T`
function raw_type(::Type{T}) where T
    sizeof(T) <= 16 ||
        throw(ArgumentError("$T is $(sizeof(T)) bytes, more than the 16 bytes of the widest raw bits"))
    return unsigned_type(nextpow(2, max(sizeof(T), 1)))
end

limb_type(::Type{U}) where U<:Unsigned = U
limb_type(::Type{Tuple{}}) = UInt8
limb_type(::Type{<:Tuple{U,Vararg{U}}}) where U<:Unsigned = U

limb_count(::Type{<:Unsigned}) = 1
limb_count(::Type{<:Tuple{Vararg{Any,K}}}) where K = K

@inline get_limb(data::Unsigned, ::Integer) = data
Base.@propagate_inbounds get_limb(data::Tuple, j::Integer) = data[(j % Int) + 1]

function check_bitwidth(::Type{T}) where T
    isbitstype(T) ||
        throw(ArgumentError("element type $T is not a concrete bitstype"))
    W = bitwidth(T)
    W isa Integer && 1 <= W <= MAX_BITWIDTH ||
        throw(ArgumentError("element bitwidth must be from 1 to $MAX_BITWIDTH bits, got $W for $T"))
    return Int(W)
end

"""
    storage_type(bits)

The storage for `bits` packed bits, rounded up to whole bytes: the unsigned
integer of that many bytes when there is one, and otherwise a tuple of the
widest of `UInt64`, `UInt32`, `UInt16` and `UInt8` that divides them, so that
the storage is no larger than the bytes it holds.
"""
function storage_type(bits::Integer)
    bits >= 0 || throw(ArgumentError("packed storage needs a non-negative bit count, got $bits"))
    nbytes = cld(bits, 8)
    nbytes in (1, 2, 4, 8, 16) && return unsigned_type(nbytes)
    limb = nbytes % 8 == 0 ? 8 : nbytes % 4 == 0 ? 4 : nbytes % 2 == 0 ? 2 : 1
    return NTuple{nbytes ÷ limb,unsigned_type(limb)}
end

@generated storage_type(::Val{B}) where B = storage_type(B)

function _or_expr(::Type{U}, terms) where U
    isempty(terms) && return :(zero($U))
    return foldl((a, b) -> :($a | $b), terms)
end

# Bits `first:first + width - 1` of `source`, as a `U`, moved up to bit `dest`
function _field_expr(source, first, width, dest, ::Type{U}) where U
    expr = source
    first > 0 && (expr = :($expr >> $first))
    expr = :($expr % $U)
    width < 8 * sizeof(U) && (expr = :($expr & $(low_mask(U, width))))
    dest > 0 && (expr = :($expr << $dest))
    return expr
end

"""
    pack(Val(W), D, raws::NTuple{L,<:Unsigned})::D

Pack the low `W` bits of each of `raws`, in order, into storage of type `D`.
"""
@generated function pack(::Val{W}, ::Type{D}, raws::Tuple{Vararg{Unsigned,L}}) where {W,D,L}
    U = limb_type(D)
    b = 8 * sizeof(U)
    W * L <= b * limb_count(D) ||
        return :(throw(ArgumentError($("$D has too few bits for $L $W-bit values"))))
    limbs = map(0:(limb_count(D) - 1)) do j
        lo, hi = b * j, b * j + b - 1
        terms = Any[]
        for i in 0:(L - 1)
            first, last = W * i, W * i + W - 1
            f, l = max(lo, first), min(hi, last)
            f <= l || continue
            push!(terms, _field_expr(:(raws[$(i + 1)]), f - first, l - f + 1, f - lo, U))
        end
        _or_expr(U, terms)
    end
    return D <: Unsigned ? only(limbs) : Expr(:tuple, limbs...)
end

"""
    unpack(Val(W), Val(L), R, data)::NTuple{L,R}

The first `L` packed `W`-bit values of `data`, each in the low bits of an `R`.
"""
@generated function unpack(::Val{W}, ::Val{L}, ::Type{R}, data::Storage) where {W,L,R}
    U = limb_type(data)
    b = 8 * sizeof(U)
    W * L <= b * limb_count(data) ||
        return :(throw(ArgumentError($("$data has too few bits for $L $W-bit values"))))
    limb(j) = data <: Unsigned ? :data : :(data[$(j + 1)])
    values = map(0:(L - 1)) do i
        first, last = W * i, W * i + W - 1
        terms = Any[]
        for j in (first ÷ b):(last ÷ b)
            lo = b * j
            f, l = max(lo, first), min(lo + b - 1, last)
            push!(terms, _field_expr(limb(j), f - lo, l - f + 1, f - first, R))
        end
        _or_expr(R, terms)
    end
    return Expr(:tuple, values...)
end

"""
    packed_getindex(Val(W), R, data, i)::R

Packed `W`-bit value `i` (1-based, unchecked) of `data`, in the low bits of an `R`.
"""
@inline function packed_getindex(::Val{W}, ::Type{R}, data::D, i::Int) where {W,R,D<:Storage}
    b = 8 * sizeof(limb_type(D))
    # unsigned, so that the division by the limb width is a shift and a mask
    j, shift = divrem(UInt(W) * ((i - 1) % UInt), UInt(b))
    @inbounds value = (get_limb(data, j) >> shift) % R
    # A width dividing the limb never crosses into the next limb. Otherwise the
    # value continues into the next limbs, which exist for an index in bounds.
    if !(W <= b && b % W == 0)
        for k in 1:cld(W - 1, b)
            dest = UInt(k * b) - shift
            dest < W || break
            @inbounds value |= (get_limb(data, j + k) % R) << dest
        end
    end
    return value & low_mask(R, W)
end
