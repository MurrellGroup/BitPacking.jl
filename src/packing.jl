function _check_pack_bitwidth(W)
    W isa Integer || throw(ArgumentError("bitwidth must be an integer in 1:8, got $W"))
    1 <= W <= 8 || throw(ArgumentError("bitwidth must be in 1:8, got $W"))
    return Int(W)
end

function _check_pack_count(S)
    S isa Integer || throw(ArgumentError("element count must be an integer, got $S"))
    S >= 0 || throw(ArgumentError("element count must be non-negative, got $S"))
    return Int(S)
end

function _or_expr(terms)
    isempty(terms) && return :(0x00)

    expr = first(terms)
    for term in Iterators.drop(terms, 1)
        expr = :($expr | $term)
    end
    return expr
end

_low_mask(width) = 0x01 << width - 0x01

function _unsigned_type_for(total_bits)
    total_bits <=   8 ? UInt8   :
    total_bits <=  16 ? UInt16  :
    total_bits <=  32 ? UInt32  :
    total_bits <=  64 ? UInt64  :
    total_bits <= 128 ? UInt128 :
    nothing
end

function _pack_part_expr(name, index, source_shift, width, dest_shift)
    expr = :(Core.getfield($name, $index))
    source_shift > 0 && (expr = :($expr >> $source_shift))
    width < 8 && (expr = :($expr & $(_low_mask(width))))
    dest_shift > 0 && (expr = :($expr << $dest_shift))
    return expr
end

@generated function pack(::Val{W}, xs::NTuple{S,UInt8}) where {W,S}
    width_bits = _check_pack_bitwidth(W)
    count = _check_pack_count(S)
    total_bits = width_bits * count

    U = _unsigned_type_for(total_bits)
    if U !== nothing && count > 0 && 8 * sizeof(U) == total_bits
        mask = _low_mask(width_bits)
        terms = Any[]
        for i in 0:(count - 1)
            shift = width_bits * i
            expr = :(Core.getfield(xs, $(i + 1)))
            width_bits < 8 && (expr = :($expr & $mask))
            expr = :($(U)($expr))
            shift > 0 && (expr = :($expr << $shift))
            push!(terms, expr)
        end
        return _or_expr(terms)
    end

    N = cld(total_bits, 8)
    bytes = Vector{Any}(undef, N)
    for byte_index in 0:(N - 1)
        byte_first_bit = 8 * byte_index
        byte_last_bit = byte_first_bit + 7
        first_source = byte_first_bit ÷ width_bits
        last_source = min(count - 1, byte_last_bit ÷ width_bits)
        terms = Any[]

        for source_index in first_source:last_source
            source_first_bit = width_bits * source_index
            source_last_bit = source_first_bit + width_bits - 1
            first_bit = max(byte_first_bit, source_first_bit)
            last_bit = min(byte_last_bit, source_last_bit)
            first_bit <= last_bit || continue

            source_shift = first_bit - source_first_bit
            dest_shift = first_bit - byte_first_bit
            width = last_bit - first_bit + 1
            push!(terms, _pack_part_expr(:xs, source_index + 1, source_shift, width, dest_shift))
        end

        bytes[byte_index + 1] = _or_expr(terms)
    end

    return Expr(:tuple, bytes...)
end

@generated function unpack(::Val{W}, ::Val{S}, data::U) where {W,S,U<:Unsigned}
    width_bits = _check_pack_bitwidth(W)
    count = _check_pack_count(S)
    width_bits * count <= 8 * sizeof(U) ||
        throw(ArgumentError("$U has $(8*sizeof(U)) bits, need $(width_bits * count) for $count × $width_bits-bit values"))

    mask = _low_mask(width_bits)
    values = Vector{Any}(undef, count)
    for i in 0:(count - 1)
        shift = width_bits * i
        expr = :data
        shift > 0 && (expr = :($expr >> $shift))
        values[i + 1] = :(UInt8($expr & $mask))
    end

    return Expr(:tuple, values...)
end

@generated function unpack(::Val{W}, ::Val{S}, bytes::NTuple{N,UInt8}) where {W,S,N}
    width_bits = _check_pack_bitwidth(W)
    count = _check_pack_count(S)
    expected = cld(width_bits * count, 8)
    N == expected || throw(ArgumentError("expected $expected packed bytes for $count values of bitwidth $width_bits, got $N"))

    values = Vector{Any}(undef, count)
    for value_index in 0:(count - 1)
        value_first_bit = width_bits * value_index
        value_last_bit = value_first_bit + width_bits - 1
        first_byte = value_first_bit ÷ 8
        last_byte = value_last_bit ÷ 8
        terms = Any[]

        for byte_index in first_byte:last_byte
            byte_first_bit = 8 * byte_index
            byte_last_bit = byte_first_bit + 7
            first_bit = max(value_first_bit, byte_first_bit)
            last_bit = min(value_last_bit, byte_last_bit)
            first_bit <= last_bit || continue

            source_shift = first_bit - byte_first_bit
            dest_shift = first_bit - value_first_bit
            width = last_bit - first_bit + 1
            push!(terms, _pack_part_expr(:bytes, byte_index + 1, source_shift, width, dest_shift))
        end

        values[value_index + 1] = _or_expr(terms)
    end

    return Expr(:tuple, values...)
end

@inline function packed_getindex(::Val{W}, ::Val{S}, data::Unsigned, i::Integer) where {W,S}
    @boundscheck 1 <= i <= S || throw(BoundsError(data, i))
    shift = (Int(i) - 1) * W
    mask = UInt8((UInt16(1) << W) - UInt16(1))
    return UInt8((data >>> shift) & mask)
end

@inline function packed_getindex(::Val{W}, ::Val{S}, bytes::NTuple{N,UInt8}, i::Integer) where {W,S,N}
    @boundscheck 1 <= i <= S || throw(BoundsError(bytes, i))

    bit = (Int(i) - 1) * W
    byte = (bit >>> 3) + 1
    shift = bit & 0x07

    word = UInt16(bytes[byte])
    byte < N && (word |= UInt16(bytes[byte + 1]) << 8)

    mask = UInt16((UInt16(1) << W) - 1)
    return UInt8((word >>> shift) & mask)
end
