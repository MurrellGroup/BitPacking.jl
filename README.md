# BitPacking.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/BitPacking.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/BitPacking.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/BitPacking.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/BitPacking.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/BitPacking.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/BitPacking.jl)

BitPacking.jl packs narrow values densely, for array code on the CPU and GPU:

- `NArray`, `NVector`, and `NMatrix` pack a static group of values into one
  storage value: the value layout, which for FP4, FP8 and BF16 pairs is the
  packed register layout PTX instructions take.
- `NarrowArray`, `NarrowVector`, and `NarrowMatrix` store an array of values as
  packed chunks: the memory layout, a dense byte stream for libraries such as
  cuDNN and cuBLASLt.

Elements may be 1 to 128 bits wide. Values are stored LSB-first, `reinterpret`
keeps bits and never changes width, and packed arrays follow the array rule, as
Base arrays do; the [bit model](https://MurrellGroup.github.io/BitPacking.jl/dev/bit-model/)
page has the details.

An element type needs a width, `BitPacking.bitwidth(::Type)`. Primitive types
have the width they declare: on Julia 1.14 and later, `primitive type Float4 4
end` needs nothing more, and a narrow type declared with whole bytes overloads
`bitwidth`. `BitPacking.raw_bits` and `BitPacking.from_raw_bits` convert between
values and their bits.

## Examples

```julia
using BitPacking

v = NVector(true, false, true, false, true, false, true, false)

Tuple(v)                 # (true, false, true, false, true, false, true, false)
BitPacking.raw_bits(v)   # 0x55: the packed bits as one integer
reinterpret(UInt8, v)    # NVector{UInt8,1}: the same bits as one UInt8
map(!, v)                # NVector{Bool,8}: map stays packed
v .& v                   # SVector{8,Bool}: broadcasting unpacks

w = NVector{UInt16,3}(0x0123, 0x0456, 0x0789)
reinterpret(UInt8, w)    # NVector{UInt8,6}, any width works
```

```julia
x = Bool[1, 0, 1, 0, 1, 0, 1, 0]
packed = NarrowArray{Bool}(x)   # or Narrow(Bool).(x)

copy(packed) == x               # true
reinterpret(UInt8, packed)      # [0x55], a view of the packed bytes
packed[2] = true                # repacks the chunk
UInt8.(packed)                  # unpacks to a dense Vector{UInt8}
```

For packed records of mixed field types, see
[PackedStructs.jl](https://github.com/JuliaData/PackedStructs.jl). Its layout
differs from BitPacking's: the first field takes the highest bits.

## Installation

```julia
using Pkg
Pkg.add("BitPacking")
```
