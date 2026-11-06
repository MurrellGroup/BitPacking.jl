module BitPacking

using Republic
import Adapt

@public bitwidth, raw_bits, from_raw_bits
export NArray, NVector, NMatrix
export NarrowArray, NarrowVector, NarrowMatrix
export Narrow

@static if isdefined(Core, :bitsizeof)
    # Julia 1.14 lets a primitive type declare fewer bits than it stores, as in
    # `primitive type Float4 4 end`, which occupies a byte.
    bitsizeof(::Type{T}) where T = Core.bitsizeof(T)
else
    bitsizeof(::Type{T}) where T = 8 * sizeof(T)
end

has_unused_bits(::Type{T}) where T = isprimitivetype(T) && bitsizeof(T) < 8 * sizeof(T)

# Whether Base's `reinterpret` refuses arrays of `T`: a primitive type with unused
# bits (Julia 1.14+), or one whose `sizeof` is less than its array stride, such
# as a 24-bit primitive type before Julia 1.14.
array_reinterpret_fails(::Type{T}) where T =
    has_unused_bits(T) || sizeof(T) != Base.aligned_sizeof(T)

"""
    bitwidth(T)::Int
    bitwidth(x)::Int

Return the number of value bits used by `T` or by the type of `x`.

The default for bitstypes is `Core.bitsizeof(T)` on Julia 1.14 and later, which
is the declared width of a primitive type such as `primitive type Float4 4 end`,
and `8 * sizeof(T)` before that. `Bool` is specialized to one bit. Packages and
users can overload this for narrow types declared with whole bytes and for
packed containers.
"""
bitwidth(::Type{T}) where T = isbitstype(T) ? bitsizeof(T) : error("$T is not a bitstype")
bitwidth(::T) where T = bitwidth(T)
bitwidth(::Type{Bool}) = 1

# The `U` with its low `width` bits set
@inline low_mask(::Type{U}, width::Integer) where U<:Unsigned =
    width >= 8 * sizeof(U) ? typemax(U) : (one(U) << width) - one(U)

"""
    raw_bits(x)::Unsigned

Return the value of `x` as an unsigned integer: its [`bitwidth`](@ref) value bits
in the low bits, and zeros above them. The integer is the narrowest of `UInt8`
to `UInt128` that holds `sizeof(x)` bytes.

This is how BitPacking reads every element it packs, and it differs from
`reinterpret` in two ways. It changes width where needed: a primitive type that
declares fewer bits than it stores, such as `primitive type Float4 4 end` on
Julia 1.14 and later, or a 3-byte type, is zero-extended, where `reinterpret`
refuses it. And a type declared with whole bytes but a narrower `bitwidth`, such
as `Bool` or a 4-bit float stored in a byte, has the bits above its width
cleared, whatever they held.

Together with `bitwidth` and [`from_raw_bits`](@ref) this is the element
protocol: any bitstype of at most 16 bytes takes part by default, and a type can
specialize all three. An [`NArray`](@ref) does, returning its storage word, so
that it can itself be an element.
"""
@inline function raw_bits(x::T) where T
    U = raw_type(T)
    bits = to_word(U, x)::U
    return bitwidth(T) < 8 * sizeof(U) ? bits & low_mask(U, bitwidth(T)) : bits
end

"""
    from_raw_bits(T, bits::Unsigned)::T

Return the `T` whose value bits are the low [`bitwidth`](@ref)`(T)` bits of
`bits`; the bits above them are ignored, whatever the width of `bits`. The
inverse of [`raw_bits`](@ref): `from_raw_bits(T, raw_bits(x)) === x` for every
`x` whose bits above its width are zero.
"""
@inline function from_raw_bits(::Type{T}, bits::Unsigned) where T
    U = raw_type(T)
    word = bits % U
    bitwidth(T) < 8 * sizeof(U) && (word &= low_mask(U, bitwidth(T)))
    return from_word(T, word)::T
end

# The bytes of `x` zero-extended to a `U`, and back
@inline function to_word(::Type{U}, x::T) where {U,T}
    if isprimitivetype(T)
        bitsizeof(T) == 8 * sizeof(U) && return Core.Intrinsics.bitcast(U, x)
        return Core.Intrinsics.zext_int(U, x)
    end
    sizeof(T) == sizeof(U) && return reinterpret(U, x)
    bytes = reinterpret(NTuple{sizeof(T),UInt8}, x)
    return foldl(|, ntuple(i -> (bytes[i] % U) << (8 * (i - 1)), Val(sizeof(T))))
end

@inline function from_word(::Type{T}, word::U) where {T,U}
    if isprimitivetype(T)
        bitsizeof(T) == 8 * sizeof(U) && return Core.Intrinsics.bitcast(T, word)
        return Core.Intrinsics.trunc_int(T, word)
    end
    sizeof(T) == sizeof(U) && return reinterpret(T, word)
    return reinterpret(T, ntuple(i -> (word >> (8 * (i - 1))) % UInt8, Val(sizeof(T))))
end

include("packing.jl")
include("NArray.jl")
include("NarrowArray.jl")

end
