packbits(w::Int, args...; kws...) = packbits(Val(w), args...; kws...)
unpackbits(w::Int, args...; kws...) = unpackbits(Val(w), args...; kws...)

function packbits!(w::Val{W}, packed_bytes::AbstractArray{UInt8}, bytes::AbstractArray{UInt8}; groups::Val{M}=Val(1)) where {W,M}
    M <= 16 || error("The number of groups must be less than or equal to 16")
    N = chunk_size(w)
    size(bytes, 1) % (M * N) == 0 ||
        error("The number of bytes in the first dimension ($(size(bytes, 1))) must be divisible by $N times the number of groups ($M). " *
              "The maximum allowed number of groups is 16, but too few or too many groups are often less efficient, especially on GPUs. " *
              "To change the number of groups, call `packbits(...; groups=Val(M))`")
    map!(group -> packgroup(w, group), grouped_packed_chunks(groups, w, packed_bytes), grouped_chunks(groups, w, bytes))
    return packed_bytes
end

function packbits!(w::Val{W}, packed_bytes::AbstractArray{UInt8}, x::AbstractArray{T}; kws...) where {W,T}
    sizeof(T) === 1 || error("Bitpacking only supported for 8-bit types")
    bytes = reinterpret(UInt8, x)
    packbits!(w, packed_bytes, bytes; kws...)
    return packed_bytes
end

function packbits(w::Val{W}, x::AbstractArray; kws...) where W
    packed_bytes = similar(x, UInt8, size(x, 1) * W ÷ 8, size(x)[2:end]...)
    packbits!(w, packed_bytes, x; kws...)
    return packed_bytes
end

function unpackbits!(w::Val{W}, bytes::AbstractArray{UInt8}, packed_bytes::AbstractArray{UInt8}; groups::Val{M}=Val(1)) where {W,M}
    M <= 16 || error("The number of groups must be less than or equal to 16")
    N = packed_chunk_size(w)
    size(packed_bytes, 1) % (M * N) == 0 ||
        error("The number of packed bytes in the first dimension ($(size(packed_bytes, 1))) must be divisible by $N times the number of groups ($M). " *
              "The maximum allowed number of groups is 16, but too few or too many groups are often less efficient, especially on GPUs. " *
              "To change the number of groups, call `unpackbits(...; groups=Val(M))`")
    map!(group -> unpackgroup(w, group), grouped_chunks(groups, w, bytes), grouped_packed_chunks(groups, w, packed_bytes))
    return bytes
end

function unpackbits!(w::Val{W}, x::AbstractArray{T}, packed_bytes::AbstractArray{UInt8}; kws...) where {W,T}
    sizeof(T) === 1 || error("Bitpacking only supported for 8-bit types")
    bytes = reinterpret(UInt8, x)
    unpackbits!(w, bytes, packed_bytes; kws...)
    return x
end

function unpackbits(w::Val{W}, packed_bytes::AbstractArray{UInt8}, T::Type=UInt8; kws...) where W
    x = similar(packed_bytes, T, size(packed_bytes, 1) * 8 ÷ W, size(packed_bytes)[2:end]...)
    unpackbits!(w, x, packed_bytes; kws...)
    return x
end

"""
    packbits!(w::Int, packed_bytes::AbstractArray{UInt8}, x::AbstractArray; groups::Val=Val(1))

Mutates `packed_bytes` in-place to pack `x` into `W` bits per element.

The keyword argument `groups` can be increased to improve performance,
while limiting the allowed sizes of the first dimension of `x`.

See also [`packbits`](@ref), [`unpackbits!`](@ref), [`unpackbits`](@ref).
"""
packbits!

"""
    packbits(w::Int, x::AbstractArray; groups::Val=Val(1))

Returns a `BitPackedArray` of the same size as `x`, with each element packed into `W` bits per element.

See also [`packbits!`](@ref), [`unpackbits!`](@ref), [`unpackbits`](@ref).
"""
packbits

"""
    unpackbits!(w::Int, x::AbstractArray, packed_bytes::AbstractArray{UInt8}; groups::Val=Val(1))

Mutates `x` in-place to unpack `packed_bytes` into byte-sized elements of type `eltype(x)`.

The keyword argument `groups` can be increased to improve performance,
while limiting the allowed sizes of the first dimension of `x`.

See also [`packbits!`](@ref), [`packbits`](@ref), [`unpackbits`](@ref).
"""
unpackbits!

"""
    unpackbits(w::Int, packed_bytes::AbstractArray{UInt8}, T::Type=UInt8; groups::Val=Val(1))

Returns an array of the same size as `packed_bytes`, into byte-sized elements of type `T`.

See also [`packbits!`](@ref), [`packbits`](@ref), [`unpackbits!`](@ref).
"""
unpackbits
