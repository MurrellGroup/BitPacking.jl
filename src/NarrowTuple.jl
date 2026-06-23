"""
    Pad{N}

Abstract supertype for padding fields in a [`NarrowTuple`](@ref).

Padding fields contribute `N` bits to the packed layout. `ZeroPad{N}` and
`OnePad{N}` are abstract layout marker families; use `ZeroPad(N)` and
`OnePad(N)` to construct zero-size values for introspection.

Use [`ZeroPad`](@ref) or [`OnePad`](@ref) in layouts.
"""
abstract type Pad{N} end

"""
    ZeroPad{N} <: Pad{N}
    ZeroPad(N)

Insert `N` zero bits into a [`NarrowTuple`](@ref) layout.

`ZeroPad{N}` is an abstract layout marker. `ZeroPad(N)` returns an
introspectable zero-size value whose type is a concrete subtype of
`ZeroPad{N}`:

```julia
T = @NarrowTuple{UInt8, Bool, BitPacking.ZeroPad{7}}
nt = T(0x00, true)

nt.data    # 0x0100
Tuple(nt)  # (0x00, true, BitPacking.ZeroPad(7))
```

Raw-storage constructors validate that the corresponding padding bits are
zero.
"""
abstract type ZeroPad{N} <: Pad{N} end

"""
    OnePad{N} <: Pad{N}
    OnePad(N)

Insert `N` one bits into a [`NarrowTuple`](@ref) layout.

`OnePad{N}` is an abstract layout marker. `OnePad(N)` returns a zero-size
value whose type is a concrete subtype of `OnePad{N}`:

```julia
T = @NarrowTuple{UInt8, Bool, BitPacking.OnePad{7}}
nt = T(0x00, true)

nt.data    # 0xff00
Tuple(nt)  # (0x00, true, BitPacking.OnePad(7))
```

Raw-storage constructors validate that the corresponding padding bits are
one.
"""
abstract type OnePad{N} <: Pad{N} end

struct ZeroPadValue{N} <: ZeroPad{N} end
struct OnePadValue{N} <: OnePad{N} end

function check_pad_bitwidth(N)
    N isa Integer || throw(ArgumentError("padding bitwidth must be an integer, got $N"))
    N >= 1 || throw(ArgumentError("padding bitwidth must be positive, got $N"))
    return Int(N)
end

ZeroPad(N) = ZeroPadValue{check_pad_bitwidth(N)}()
OnePad(N) = OnePadValue{check_pad_bitwidth(N)}()
(::Type{ZeroPad{N}})() where N = ZeroPadValue{check_pad_bitwidth(N)}()
(::Type{OnePad{N}})() where N = OnePadValue{check_pad_bitwidth(N)}()

bitwidth(::Type{<:Pad{N}}) where N = check_pad_bitwidth(N)

is_pad_type(::Type{T}) where T = T <: Pad
is_zero_pad_type(::Type{T}) where T = T <: ZeroPad
is_one_pad_type(::Type{T}) where T = T <: OnePad

layout_fieldtype(::Type{T}) where {N,T<:ZeroPad{N}} = ZeroPad{N}
layout_fieldtype(::Type{T}) where {N,T<:OnePad{N}} = OnePad{N}
layout_fieldtype(::Type{T}) where T = T

function layout_tuple_type(::Type{Ts}) where Ts<:Tuple
    return Tuple{map(layout_fieldtype, fieldtypes(Ts))...}
end

materialized_fieldtype(::Type{T}) where {N,T<:ZeroPad{N}} = ZeroPadValue{N}
materialized_fieldtype(::Type{T}) where {N,T<:OnePad{N}} = OnePadValue{N}
materialized_fieldtype(::Type{T}) where T = T

function materialized_fieldtypes(::Type{Ts}) where Ts<:Tuple
    return Tuple(materialized_fieldtype(T) for T in fieldtypes(Ts))
end

function materialized_tuple_type(::Type{Ts}) where Ts<:Tuple
    return Tuple{materialized_fieldtypes(Ts)...}
end

function unpadded_fieldtypes(::Type{Ts}) where Ts<:Tuple
    return Tuple(T for T in fieldtypes(Ts) if !is_pad_type(T))
end

function unpadded_tuple_type(::Type{Ts}) where Ts<:Tuple
    return Tuple{unpadded_fieldtypes(Ts)...}
end

pad_value(::Type{T}) where {N,T<:ZeroPad{N}} = ZeroPadValue{N}()
pad_value(::Type{T}) where {N,T<:OnePad{N}} = OnePadValue{N}()

function check_narrow_tuple_type(::Type{Ts}) where Ts<:Tuple
    for T in fieldtypes(Ts)
        T <: Tuple && throw(ArgumentError("nested Tuple field types are not supported in NarrowTuple, got $T"))

        W = bitwidth(T)
        W isa Integer ||
            throw(ArgumentError("bitwidth($T) must be an integer, got $W"))

        if is_pad_type(T)
            (is_zero_pad_type(T) || is_one_pad_type(T)) ||
                throw(ArgumentError("NarrowTuple padding field type must be ZeroPad or OnePad, got $T"))
            W >= 1 || throw(ArgumentError("padding bitwidth must be positive, got $W"))
        else
            isbitstype(T) ||
                throw(ArgumentError("NarrowTuple field type $T is not a bitstype"))
            1 <= W <= 8 * sizeof(T) ||
                throw(ArgumentError("bitwidth($T) must be in 1:$(8 * sizeof(T)), got $W"))
            unsigned_type(sizeof(T))
        end
    end

    return nothing
end

function field_bitwidths(::Type{Ts}) where Ts<:Tuple
    check_narrow_tuple_type(Ts)
    return map(T -> Int(bitwidth(T)), fieldtypes(Ts))
end

function bitwidth(::Type{Ts}) where Ts<:Tuple
    return sum(field_bitwidths(Ts); init=0)
end

function exact_unsigned_type(total_bits::Int)
    total_bits ==   8 ? UInt8   :
    total_bits ==  16 ? UInt16  :
    total_bits ==  32 ? UInt32  :
    total_bits ==  64 ? UInt64  :
    total_bits == 128 ? UInt128 :
    nothing
end

function narrow_storage_type(total_bits::Int)
    U = exact_unsigned_type(total_bits)
    return U === nothing ? NTuple{cld(total_bits, 8), UInt8} : U
end

function narrow_storage_type(::Type{Ts}) where Ts<:Tuple
    return narrow_storage_type(bitwidth(Ts))
end

function narrow_mask(::Type{U}, width::Int) where U<:Unsigned
    width <= 0 && return zero(U)
    width >= 8 * sizeof(U) && return typemax(U)
    return U((big(1) << width) - 1)
end

function field_raw_expr(name, index, T)
    U = unsigned_type(sizeof(T))
    return :($U(reinterpret($U, Core.getfield($name, $index))))
end

function _typed_or_expr(::Type{U}, terms) where U<:Unsigned
    isempty(terms) && return :($U(0))

    expr = first(terms)
    for term in Iterators.drop(terms, 1)
        expr = :($expr | $term)
    end
    return expr
end

"""
    NarrowTuple{Ts,D}
    NarrowTuple(xs::Tuple)
    NarrowTuple{Ts}(xs...)

A packed, tuple-like value whose fields are stored without Julia field
alignment.

`Ts` is a concrete `Tuple` layout type. Each non-padding field must be a
bitstype with a valid [`bitwidth`](@ref). Padding fields such as
[`ZeroPad`](@ref) and [`OnePad`](@ref) contribute bits to storage and unpack
as zero-size marker values like `BitPacking.ZeroPad(7)`.

The storage type `D` is chosen from the total layout bitwidth. Exact widths of
8, 16, 32, 64, and 128 bits use the matching unsigned integer type. Other
widths use `NTuple{N,UInt8}` with enough bytes to hold the layout.

Fields are packed in declaration order from least-significant bit upward.
For byte-tuple storage, byte 1 contains the least-significant byte.

```julia
nt = NarrowTuple((0x00, true))

typeof(nt)      # @NarrowTuple{UInt8, Bool}
nt.data         # (0x00, 0x01)
Tuple(nt)       # (0x00, true)
bitwidth(nt)    # 9
```

Nested `NarrowTuple`s work like any other bitwidth-aware bitstype:

```julia
nt = @NarrowTuple(true, @NarrowTuple(0x00, true))
bitwidth(nt)  # 10
```

Padding fields are visible in the logical tuple representation:

```julia
@NarrowTuple{UInt8, BitPacking.ZeroPad{7}, Bool}(0xff, true)
# @NarrowTuple(0xff, BitPacking.ZeroPad(7), true)
```

See also [`@NarrowTuple`](@ref).
"""
struct NarrowTuple{Ts<:Tuple,D<:Storage}
    data::D
    function NarrowTuple{Ts,D}(data::D) where {Ts<:Tuple,D<:Storage}
        expected = narrow_storage_type(Ts)
        D === expected ||
            throw(ArgumentError("storage for NarrowTuple{$Ts} must be $expected, got $D"))
        check_narrow_tuple_padding(Val(field_bitwidths(Ts)), Ts, data)
        return new{Ts,D}(data)
    end
end

bitwidth(::Type{<:NarrowTuple{Ts}}) where Ts = bitwidth(Ts)

@generated function narrow_tuple_pack(::Val{Ws}, ::Type{Layout}, ::Type{D}, xs::Values) where {Ws,Layout<:Tuple,D<:Storage,Values<:Tuple}
    types = fieldtypes(Layout)
    value_types = Tuple(materialized_fieldtype(T) for T in types)
    Values === Tuple{value_types...} ||
        throw(ArgumentError("values for NarrowTuple{$Layout} must be Tuple{$(join(value_types, ", "))}, got $Values"))

    widths = Int[Ws...]
    length(widths) == length(types) ||
        throw(ArgumentError("expected $(length(types)) field bitwidths for $Layout, got $(length(widths))"))
    for (T, W) in zip(types, widths)
        is_pad_type(T) && continue
        1 <= W <= 8 * sizeof(T) ||
            throw(ArgumentError("bitwidth($T) must be in 1:$(8 * sizeof(T)), got $W"))
    end

    expected = narrow_storage_type(sum(widths))
    D === expected ||
        throw(ArgumentError("storage for NarrowTuple{$Layout} must be $expected, got $D"))

    starts = cumsum([0; widths[1:(end - 1)]])

    if D <: Unsigned
        terms = Any[]
        for (index, (T, W, start)) in enumerate(zip(types, widths, starts))
            if is_pad_type(T)
                is_one_pad_type(T) || continue
                expr = :($D($(narrow_mask(D, W))))
            else
                U = unsigned_type(sizeof(T))
                expr = field_raw_expr(:xs, index, T)
                W < 8 * sizeof(U) && (expr = :($expr & $(narrow_mask(U, W))))
                expr = :($D($expr))
            end
            start > 0 && (expr = :($expr << $start))
            push!(terms, expr)
        end
        return _typed_or_expr(D, terms)
    end

    N = length(fieldtypes(D))
    bytes = Vector{Any}(undef, N)
    for byte_index in 0:(N - 1)
        byte_first_bit = 8 * byte_index
        byte_last_bit = byte_first_bit + 7
        terms = Any[]

        for (field_index, (T, W, field_first_bit)) in enumerate(zip(types, widths, starts))
            field_last_bit = field_first_bit + W - 1
            first_bit = max(byte_first_bit, field_first_bit)
            last_bit = min(byte_last_bit, field_last_bit)
            first_bit <= last_bit || continue

            source_shift = first_bit - field_first_bit
            dest_shift = first_bit - byte_first_bit
            width = last_bit - first_bit + 1

            if is_pad_type(T)
                is_one_pad_type(T) || continue
                expr = :(UInt8($(_low_mask(width))))
            else
                expr = field_raw_expr(:xs, field_index, T)
                source_shift > 0 && (expr = :($expr >>> $source_shift))
                width < 8 && (expr = :($expr & $(_low_mask(width))))
                expr = :(UInt8($expr))
            end
            dest_shift > 0 && (expr = :(UInt8($expr << $dest_shift)))
            push!(terms, expr)
        end

        bytes[byte_index + 1] = _typed_or_expr(UInt8, terms)
    end

    return Expr(:tuple, bytes...)
end

function materialize_narrow_tuple_values(::Type{Ts}, xs::Tuple) where Ts<:Tuple
    Full = materialized_tuple_type(Ts)
    xs isa Full && return xs

    Unpadded = unpadded_tuple_type(Ts)
    values = xs isa Unpadded ? xs : convert(Unpadded, xs)
    result = Any[]
    value_index = 1

    for T in fieldtypes(Ts)
        if is_pad_type(T)
            push!(result, pad_value(T))
        else
            push!(result, values[value_index])
            value_index += 1
        end
    end

    return convert(Full, Tuple(result))
end

function NarrowTuple{Ts,D}(xs::Tuple) where {Ts<:Tuple,D<:Storage}
    values = materialize_narrow_tuple_values(Ts, xs)
    data = narrow_tuple_pack(Val(field_bitwidths(Ts)), Ts, D, values)
    return NarrowTuple{Ts,D}(data)
end

NarrowTuple{Ts,D}(xs...) where {Ts<:Tuple,D<:Storage} = NarrowTuple{Ts,D}(xs)
NarrowTuple{Ts}(xs::Tuple) where {Ts<:Tuple} = NarrowTuple{Ts,narrow_storage_type(Ts)}(xs)
NarrowTuple{Ts}(xs...) where {Ts<:Tuple} = NarrowTuple{Ts}(xs)
NarrowTuple(xs::Ts) where Ts<:Tuple = NarrowTuple{layout_tuple_type(Ts)}(xs)

@generated function narrow_tuple_unpack(::Val{Ws}, ::Type{Ts}, data::D) where {Ws,Ts<:Tuple,D<:Storage}
    types = fieldtypes(Ts)
    widths = Int[Ws...]
    length(widths) == length(types) ||
        throw(ArgumentError("expected $(length(types)) field bitwidths for $Ts, got $(length(widths))"))
    for (T, W) in zip(types, widths)
        is_pad_type(T) && continue
        1 <= W <= 8 * sizeof(T) ||
            throw(ArgumentError("bitwidth($T) must be in 1:$(8 * sizeof(T)), got $W"))
    end

    expected = narrow_storage_type(sum(widths))
    D === expected ||
        throw(ArgumentError("storage for NarrowTuple{$Ts} must be $expected, got $D"))

    starts = cumsum([0; widths[1:(end - 1)]])
    values = Any[]

    if D <: Unsigned
        for (index, (T, W, start)) in enumerate(zip(types, widths, starts))
            if is_pad_type(T)
                push!(values, pad_value(T))
                continue
            end
            U = unsigned_type(sizeof(T))
            expr = :data
            start > 0 && (expr = :($expr >>> $start))
            W < 8 * sizeof(D) && (expr = :($expr & $(narrow_mask(D, W))))
            expr = :($U($expr))
            push!(values, :(reinterpret($T, $expr)))
        end
        return Expr(:tuple, values...)
    end

    for (field_index, (T, W, field_first_bit)) in enumerate(zip(types, widths, starts))
        if is_pad_type(T)
            push!(values, pad_value(T))
            continue
        end
        U = unsigned_type(sizeof(T))
        field_last_bit = field_first_bit + W - 1
        first_byte = field_first_bit ÷ 8
        last_byte = field_last_bit ÷ 8
        terms = Any[]

        for byte_index in first_byte:last_byte
            byte_first_bit = 8 * byte_index
            byte_last_bit = byte_first_bit + 7
            first_bit = max(field_first_bit, byte_first_bit)
            last_bit = min(field_last_bit, byte_last_bit)
            first_bit <= last_bit || continue

            source_shift = first_bit - byte_first_bit
            dest_shift = first_bit - field_first_bit
            width = last_bit - first_bit + 1

            expr = :(Core.getfield(data, $(byte_index + 1)))
            source_shift > 0 && (expr = :($expr >>> $source_shift))
            width < 8 && (expr = :($expr & $(_low_mask(width))))
            expr = :($U($expr))
            dest_shift > 0 && (expr = :($expr << $dest_shift))
            push!(terms, expr)
        end

        expr = _typed_or_expr(U, terms)
        push!(values, :(reinterpret($T, $expr)))
    end

    return Expr(:tuple, values...)
end

@generated function check_narrow_tuple_padding(::Val{Ws}, ::Type{Ts}, data::D) where {Ws,Ts<:Tuple,D<:Storage}
    types = fieldtypes(Ts)
    widths = Int[Ws...]
    starts = cumsum([0; widths[1:(end - 1)]])
    checks = Any[]

    if D <: Unsigned
        for (T, W, start) in zip(types, widths, starts)
            is_pad_type(T) || continue

            actual = :data
            start > 0 && (actual = :($actual >>> $start))
            actual = :($actual & $(narrow_mask(D, W)))
            expected = is_one_pad_type(T) ? narrow_mask(D, W) : zero(D)
            message = "packed data does not match padding bits for $T"
            push!(checks, :($actual == $expected || throw(ArgumentError($message))))
        end
    else
        for (T, W, field_first_bit) in zip(types, widths, starts)
            is_pad_type(T) || continue

            field_last_bit = field_first_bit + W - 1
            first_byte = field_first_bit ÷ 8
            last_byte = field_last_bit ÷ 8

            for byte_index in first_byte:last_byte
                byte_first_bit = 8 * byte_index
                byte_last_bit = byte_first_bit + 7
                first_bit = max(field_first_bit, byte_first_bit)
                last_bit = min(field_last_bit, byte_last_bit)
                first_bit <= last_bit || continue

                source_shift = first_bit - byte_first_bit
                width = last_bit - first_bit + 1

                actual = :(Core.getfield(data, $(byte_index + 1)))
                source_shift > 0 && (actual = :($actual >>> $source_shift))
                width < 8 && (actual = :($actual & $(_low_mask(width))))
                expected = is_one_pad_type(T) ? _low_mask(width) : 0x00
                message = "packed data does not match padding bits for $T"
                push!(checks, :($actual == $expected || throw(ArgumentError($message))))
            end
        end
    end

    return Expr(:block, checks..., nothing)
end

Base.Tuple(nt::NarrowTuple{Ts}) where Ts = narrow_tuple_unpack(Val(field_bitwidths(Ts)), Ts, nt.data)

Base.length(::NarrowTuple{Ts}) where Ts = fieldcount(Ts)
Base.length(::Type{<:NarrowTuple{Ts}}) where Ts = fieldcount(Ts)
Base.firstindex(::NarrowTuple) = 1
Base.lastindex(nt::NarrowTuple) = length(nt)
Base.getindex(nt::NarrowTuple, i::Int) = Tuple(nt)[i]
Base.getindex(nt::NarrowTuple, ::Val{I}) where I = Core.getfield(Tuple(nt), I)
Base.iterate(nt::NarrowTuple, state...) = iterate(Tuple(nt), state...)
Base.convert(::Type{Values}, nt::NarrowTuple) where Values<:Tuple = convert(Values, Tuple(nt))
Base.reinterpret(::Type{T}, nt::NarrowTuple) where T = reinterpret(T, nt.data)

Base.show(io::IO, ::ZeroPadValue{N}) where N = print(io, "BitPacking.ZeroPad(", N, ")")
Base.show(io::IO, ::OnePadValue{N}) where N = print(io, "BitPacking.OnePad(", N, ")")

function Base.show(io::IO, nt::NarrowTuple)
    print(io, "@NarrowTuple(")
    for (i, x) in enumerate(Tuple(nt))
        i > 1 && print(io, ", ")
        show(io, x)
    end
    print(io, ")")
    return nothing
end

function show_narrow_tuple_fields(io::IO, ::Type{Ts}) where Ts<:Tuple
    for (i, T) in enumerate(fieldtypes(Ts))
        i > 1 && print(io, ", ")
        show(io, T)
    end
end

function Base.show(io::IO, ::Type{T}) where {T<:NarrowTuple}
    T_unwrapped = Base.unwrap_unionall(T)
    if !(T_unwrapped isa DataType && length(T_unwrapped.parameters) == 2)
        return invoke(show, Tuple{IO,Type}, io, T)
    end

    Ts, D = T_unwrapped.parameters
    if !(Ts isa Type && Ts <: Tuple && D isa Type && D <: Storage)
        return invoke(show, Tuple{IO,Type}, io, T)
    end

    print(io, "@NarrowTuple{")
    show_narrow_tuple_fields(io, Ts)
    print(io, "}")
end

"""
    @NarrowTuple{T1, T2, ...}
    @NarrowTuple(x1, x2, ...)
    @NarrowTuple((x1, x2, ...))

Convenience syntax for [`NarrowTuple`](@ref).

The braced form returns the packed tuple type for a layout:

```julia
T = @NarrowTuple{UInt8, Bool}
# @NarrowTuple{UInt8, Bool}
```

The parenthesized form constructs a value from logical tuple fields:

```julia
@NarrowTuple(0x00, true)
# @NarrowTuple(0x00, true)
```

Padding markers are written in the braced layout form. Constructor arguments
may either omit padding values, in which case they are inserted, or include
the matching zero-size values explicitly:

```julia
T = @NarrowTuple{UInt8, Bool, BitPacking.ZeroPad{7}}
T(0x00, true)
# @NarrowTuple(0x00, true, BitPacking.ZeroPad(7))

T(0x00, true, BitPacking.ZeroPad(7))
# @NarrowTuple(0x00, true, BitPacking.ZeroPad(7))
```
"""
macro NarrowTuple(args...)
    if length(args) == 1 && args[1] isa Expr && args[1].head === :braces
        types = args[1].args
        isempty(types) && throw(ArgumentError("@NarrowTuple requires at least one field type"))

        tuple_type = Expr(:curly, GlobalRef(Core, :Tuple), map(esc, types)...)
        storage_type = Expr(:call, GlobalRef(BitPacking, :narrow_storage_type), tuple_type)
        return Expr(:curly, GlobalRef(BitPacking, :NarrowTuple), tuple_type, storage_type)
    end

    isempty(args) && throw(ArgumentError("@NarrowTuple requires at least one value"))

    if length(args) == 1 && args[1] isa Expr && args[1].head === :tuple
        return Expr(:call, GlobalRef(BitPacking, :NarrowTuple), esc(args[1]))
    end

    values = Expr(:tuple, map(esc, args)...)
    return Expr(:call, GlobalRef(BitPacking, :NarrowTuple), values)
end
