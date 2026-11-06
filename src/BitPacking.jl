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

Return the value of `x` as an unsigned integer of `sizeof(x)` bytes: its
[`bitwidth`](@ref) value bits in the low bits, and zeros above them.

This is how BitPacking reads every element it packs, and it differs from
`reinterpret` in two ways. A primitive type that declares fewer bits than it
stores, such as `primitive type Float4 4 end` on Julia 1.14 and later, is
zero-extended, where `reinterpret` refuses it. A type declared with whole bytes
but a narrower `bitwidth`, such as `Bool` or a 4-bit float stored in a byte, has
the bits above its width cleared, whatever they held.

See also [`from_raw_bits`](@ref).
"""
@inline function raw_bits(x::T) where T
    U = unsigned_type(sizeof(T))
    has_unused_bits(T) && return Core.Intrinsics.zext_int(U, x)
    bits = reinterpret(U, x)
    return bitwidth(T) < 8 * sizeof(T) ? bits & low_mask(U, bitwidth(T)) : bits
end

"""
    from_raw_bits(T, bits::Unsigned)::T

Return the `T` whose value bits are the low [`bitwidth`](@ref)`(T)` bits of
`bits`; the bits above them are ignored, whatever the width of `bits`. The
inverse of [`raw_bits`](@ref): `from_raw_bits(T, raw_bits(x)) === x` for every
`x` whose bits above its width are zero.
"""
@inline function from_raw_bits(::Type{T}, bits::Unsigned) where T
    U = unsigned_type(sizeof(T))
    word = bits % U
    has_unused_bits(T) && return Core.Intrinsics.trunc_int(T, word)
    bitwidth(T) < 8 * sizeof(T) && (word &= low_mask(U, bitwidth(T)))
    return reinterpret(T, word)
end

include("packing.jl")
include("NArray.jl")
include("NarrowArray.jl")

end
