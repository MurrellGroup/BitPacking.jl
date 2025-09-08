bitwidth(::Type{T}) where T = isbitstype(T) ? sizeof(T) * 8 : error("$T is not a bitstype")
bitwidth(::Type{Bool}) = 1

"""
    bitwidth(T::Type)

Returns the number of used bits in `T`.

The bit representation of `T` is expected to not take on any value equal to or
greater than `2^bitwidth(T)`. For example, a 1-bit type should be limited to
`0b00000000` and `0b00000001`.

# Examples

```jldoctest
julia> bitwidth(UInt8)
8

julia> bitwidth(Bool)
1
```
"""
bitwidth
