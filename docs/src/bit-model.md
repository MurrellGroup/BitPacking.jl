```@meta
CurrentModule = BitPacking
DocTestSetup = :(using BitPacking)
```

# The bit model

Every BitPacking container, width and conversion follows the same four rules,
on every supported Julia version.

1. **A value of `T` has [`bitwidth`](@ref)`(T)` value bits.** Containers store
   value bits LSB-first in linear index order: element 1 occupies the lowest
   bits. Bits past the last element are padding, and padding is always zero.
2. **`reinterpret` keeps bits and never changes width.** When the bit counts
   don't match, it throws; nothing truncates or zero-extends. Width changes are
   [`raw_bits`](@ref) and [`from_raw_bits`](@ref), which are a zero-extension
   and a truncation, not reinterprets.
3. **Packed arrays follow the array rule**, as Base arrays do: `reinterpret`
   keeps the total bit count and rescales the length of dimension 1.
4. **`NArray` is the value layout, `NarrowArray` the memory layout, and
   `SArray` or a tuple the compute layout.**

## Elements

An element type takes part through three functions:

| function                | meaning                                                                   |
|:------------------------|:--------------------------------------------------------------------------|
| `bitwidth(T)`           | the number of value bits, from 1 to 128                                   |
| `raw_bits(x)`           | the value bits in the low bits of an unsigned integer, zeros above them   |
| `from_raw_bits(T, bits)`| the `T` whose value bits are the low `bitwidth(T)` bits; the rest ignored |

The defaults cover most types:

- A primitive type has the width it declares: `Core.bitsizeof(T)` on Julia
  1.14 and later, where `primitive type Float4 4 end` is allowed, and
  `8 * sizeof(T)` before.
- `Bool` is 1 bit. Julia declares it with 8, so this is an overload.
- A narrow type declared with whole bytes, such as a 4-bit float stored in a
  byte before Julia 1.14, overloads `bitwidth`; `raw_bits` then clears the bits
  above the width, whatever they held.
- Any other bitstype of up to 16 bytes, such as a struct of three `UInt8`s,
  packs as all of its bytes.
- An [`NArray`](@ref) is an element too, so vectors of vectors pack.

`raw_bits` returns the narrowest of `UInt8` to `UInt128` that holds the value's
bytes, so a 24-bit type gives a `UInt32`. It is also how to get the bits of an
`NArray` as one integer:

```jldoctest
julia> v = NVector(true, false, true, false, true, false, true, false);

julia> BitPacking.raw_bits(v)
0x55

julia> BitPacking.from_raw_bits(NVector{Bool,8}, 0x0f) == NVector{Bool,8}(1, 1, 1, 1, 0, 0, 0, 0)
true
```

An unsigned integer is the bit carrier because it is the one Julia, LLVM and
the GPU intrinsics already use: `UInt8` to `UInt128` are LLVM's `i8` to `i128`
and PTX's `.b8` to `.b64` registers.

## `reinterpret`

`reinterpret(T, x)` on a packed array keeps every bit, and rescales dimension 1
so that the total bit count stays the same:

```jldoctest
julia> v = NVector{UInt8,2}(0x21, 0x43);

julia> reinterpret(UInt16, v)
1-element NVector{UInt16, 1, UInt16} with indices SOneTo(1):
 0x4321

julia> reinterpret(Bool, v) |> size
(16,)

julia> reinterpret(UInt16, NVector{UInt8,3}(0x01, 0x02, 0x03))
ERROR: ArgumentError: cannot reinterpret 3 8-bit value(s) along dimension 1 as 16-bit values
[...]
```

For an `NArray` the result is always an `NArray`. For a
[`NarrowArray`](@ref) it is a plain view when `T` fills its bytes, as `UInt8`
and `Float32` do, and a `NarrowArray{T}` otherwise. Going the other way, from
plain bytes or chunks to a packed array, is
[`reinterpret(Narrow(T), data)`](@ref Narrow).

Base's own `reinterpret` on plain arrays is unchanged: on Julia 1.14 it refuses
arrays of a type that declares fewer bits than it stores, which is Base's
decision. Broadcast `raw_bits.(x)` and `from_raw_bits.(T, x)` instead, or pack
the values with `NarrowArray{T}(x)`.

## `NArray`: the value layout

An `NArray` packs its elements into one storage value, which is the unsigned
integer of its bit count rounded up to whole bytes when one exists, and
otherwise a tuple of the widest unsigned integers that divide the bytes:

| elements                  | bits | storage            |
|:--------------------------|-----:|:-------------------|
| 8 × `Bool`                |    8 | `UInt8`            |
| 9 × `Bool`                |    9 | `UInt16`           |
| 4 × 6-bit float           |   24 | `NTuple{3,UInt8}`  |
| 3 × `UInt16`              |   48 | `NTuple{3,UInt16}` |
| 4 × `Float32`             |  128 | `UInt128`          |
| 8 × `Float32`             |  256 | `NTuple{4,UInt64}` |

The storage is never larger than the bytes it holds, which is what lets a
`NarrowArray` of 6-bit floats be a dense byte stream: Julia stores a primitive
type at a power-of-two stride, so only a tuple gives a 3-byte element.

Where the storage fits a register, an `NArray` is exactly the packed register
types that GPU instructions take, since PTX also puts lane 1 in the low bits:

| `NArray`                    | storage  | PTX                                   |
|:----------------------------|:---------|:--------------------------------------|
| `NVector{Float4_E2M1FN,2}`  | `UInt8`  | `e2m1x2` in a `.b8`                   |
| `NVector{Float8_E4M3FN,2}`  | `UInt16` | `e4m3x2` in a `.b16`                  |
| `NVector{BFloat16,2}`       | `UInt32` | `bf16x2` in a `.b32`                  |
| `NVector{Float4_E2M1FN,8}`  | `UInt32` | an FP4 MMA operand register           |

In `cvt.rn.satfinite.e4m3x2.f32 d, a, b`, lane 1, in the low byte, is operand
`b`. Where a register layout is not dense, such as PTX's `e2m3x2`, which keeps
each 6-bit value in its own byte, the register form is an `SVector`, and only
the memory form is packed.

For element types that fill their bytes, an `NArray` holds exactly the bytes of
the matching `SArray`, since Julia lays tuples out LSB-first too: converting
between them moves the same bytes, with no reordering.

## `NarrowArray`: the memory layout

A `NarrowArray{T}` stores its values as an array of `NVector` chunks along
dimension 1. The default chunk length is the smallest whole number of bytes,
`8 ÷ gcd(bitwidth(T), 8)` values: 8 for `Bool`, 2 for 4-bit and 12-bit types, 4
for 6-bit types, 1 for types that fill their bytes. Chunks have no padding, so
`reinterpret(UInt8, x)` is the dense packed byte stream that libraries such as
cuDNN and cuBLASLt read.

## Computing

`map` keeps an `NArray` packed, while broadcasting treats it as the unpacked
`SArray` of its values. Broadcasts are mostly elementwise arithmetic, and
repacking after each one would cost more than packing saves:

```jldoctest
julia> v = NVector{Bool,8}(1, 0, 1, 0, 1, 0, 1, 0);

julia> map(!, v) isa NVector{Bool,8}
true

julia> (v .& v) isa BitPacking.SVector{8,Bool}
true
```

For a `NarrowArray`, `T.(x)` and `copy(x)` unpack into a dense array on the
same backend, `Narrow(T).(x)` packs, and `dest .= x` packs into an existing
`NarrowArray`, on the GPU as well.

## Interoperability

These consumers are known to share the LSB-first order: PTX `cvt` with packed
results (`e2m1x2`, `e4m3x2`, `bf16x2`), and cuBLASLt and cuDNN FP4 operands.
XLA's packed `f4` order has not been checked.

[PackedStructs.jl](https://github.com/JuliaData/PackedStructs.jl) packs records
of mixed field types, which BitPacking does not. It uses a different layout,
though: the first field takes the highest bits, and fields are grouped into 8-,
16-, 32- or 64-bit words. It takes widths from `EmulatedBitIntegers.bits`
rather than `bitwidth`, and packs non-integer fields through
`PackedStructs.pack`, so a narrow float type needs both overloaded. Its bytes
are not interchangeable with BitPacking's.

## Upgrading from 0.2

- `reinterpret(T, ::NArray)` returns an `NArray` with dimension 1 rescaled,
  rather than reinterpreting the storage word. For the storage word as an
  integer, use `raw_bits(v)`, and `from_raw_bits(NVector{T,L}, bits)` for the
  reverse.
- Constructor arguments are always values: `NVector{Bool,8}(0x55)` is an error,
  not the storage word. Use `from_raw_bits(NVector{Bool,8}, 0x55)`.
- Element widths above 8 bits work, up to 128. Storage that is not exactly 1, 2,
  4, 8 or 16 bytes is a tuple of the widest unsigned integers that divide it,
  rather than always of bytes, and storage whose bit count rounds up to such a
  size is that unsigned integer: `NVector{Bool,9}` stores a `UInt16`.
- `reinterpret(T, ::NarrowArray)` works for any width `T`.
- `copy(x)` and `T.(x)` on a `NarrowArray` return a plain array, not a
  `ReinterpretArray` over an array of `SVector` chunks.
- `NarrowArray` has `setindex!`.
- `NarrowTuple`, `@NarrowTuple`, `Pad`, `ZeroPad` and `OnePad` are removed; see
  PackedStructs.jl. The deprecated `Narrow{T}` forms, `Narrow{T}.(x)` and
  `reinterpret(Narrow{T}, x)`, are removed; use `Narrow(T)`.
