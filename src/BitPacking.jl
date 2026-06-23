module BitPacking

import Adapt
using Republic

@public bitwidth
export NArray, NVector, NMatrix
export NarrowArray, NarrowVector, NarrowMatrix
export NarrowTuple, @NarrowTuple
@public Pad, ZeroPad, OnePad

"""
    bitwidth(T)::Int
    bitwidth(x)::Int

Return the number of value bits used by `T` or by the type of `x`.

The default for bitstypes is `8 * sizeof(T)`, with `Bool` specialized to
one bit. Packages and users can overload this for narrow primitive types,
packed containers, and layout markers such as [`ZeroPad`](@ref).
"""
bitwidth(::Type{T}) where T = isbitstype(T) ? sizeof(T) * 8 : error("$T is not a bitstype")
bitwidth(::T) where T = bitwidth(T)
bitwidth(::Type{Bool}) = 1

include("packing.jl")
include("NArray.jl")
include("NarrowTuple.jl")
include("NarrowArray.jl")

end
