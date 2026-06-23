# BitPacking.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/BitPacking.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/BitPacking.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/BitPacking.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/BitPacking.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/BitPacking.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/BitPacking.jl)

BitPacking.jl provides small bit-packed building blocks for Julia array code:

- `NArray`, `NVector`, and `NMatrix` pack a static group of narrow values into one storage value.
- `NarrowArray`, `NarrowVector`, and `NarrowMatrix` view an array of packed groups as a logical array of values.
- `NarrowTuple` and `@NarrowTuple` pack heterogeneous isbits values, with optional padding fields.

Packages can extend `BitPacking.bitwidth(::Type)` for their own narrow scalar types.

## Examples

```julia
using BitPacking

v = NVector(true, false, true, false, true, false, true, false)

Tuple(v)                 # (true, false, true, false, true, false, true, false)
reinterpret(UInt8, v)    # 0x55
```

```julia
x = Bool[1, 0, 1, 0, 1, 0, 1, 0]
packed = NarrowArray{Bool}(x)

copy(packed) == x        # true
parent(packed)           # Vector{NVector{Bool,8}}
```

```julia
t = @NarrowTuple(0x12, true)

Tuple(t)                 # (0x12, true)
BitPacking.bitwidth(t)   # 9
```

## Installation

```julia
using Pkg
Pkg.add("BitPacking")
```
