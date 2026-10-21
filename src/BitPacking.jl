module BitPacking

using Republic
import Adapt

@public bitwidth, raw_bits, from_raw_bits
export NArray, NVector, NMatrix
export NarrowArray, NarrowVector, NarrowMatrix
export Narrow
export NarrowTuple, @NarrowTuple
@public Pad, ZeroPad, OnePad

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
users can overload this for narrow types declared with whole bytes, packed
containers, and layout markers such as [`ZeroPad`](@ref).
"""
bitwidth(::Type{T}) where T = isbitstype(T) ? bitsizeof(T) : error("$T is not a bitstype")
bitwidth(::T) where T = bitwidth(T)
bitwidth(::Type{Bool}) = 1

"""
    raw_bits(x)::Unsigned

Return the bits of `x` as the unsigned integer type of the same size, like
`reinterpret(UInt8, x)` for a one-byte `x`.

Unlike `reinterpret`, this also accepts primitive types that declare fewer bits
than they store, such as `primitive type Float4 4 end` on Julia 1.14 and later.
Their unused high bits read as zero.

See also [`from_raw_bits`](@ref).
"""
@inline function raw_bits(x::T) where T
    U = unsigned_type(sizeof(T))
    return has_unused_bits(T) ? Core.Intrinsics.zext_int(U, x) : reinterpret(U, x)
end

"""
    from_raw_bits(T, bits::Unsigned)::T

Return the `T` whose bits are `bits`, an unsigned integer of `sizeof(T)` bytes.
The inverse of [`raw_bits`](@ref).

For a primitive type that declares fewer bits than it stores, bits of `bits`
beyond its width are dropped.
"""
@inline function from_raw_bits(::Type{T}, bits::Unsigned) where T
    return has_unused_bits(T) ? Core.Intrinsics.trunc_int(T, bits) : reinterpret(T, bits)
end

include("packing.jl")
include("NArray.jl")
include("NarrowTuple.jl")
include("NarrowArray.jl")

end
