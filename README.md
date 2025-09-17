# BitPacking.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/BitPacking.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/BitPacking.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/BitPacking.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/BitPacking.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/BitPacking.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/BitPacking.jl)

BitPacking.jl is a Julia package that implements bitpacking and unpacking arrays with element bit-widths from 1 to 8, imposing mild size limitations on the first dimension. Bitpacking methods are designed to be device-agnostic, meaning they run fast on GPUs.

## Usage

```julia
using BitPacking

x = rand(UInt8, 32) .& 0b1111 # 4-bit unpacked values
y = bitpacked(x, 4)           # half the memory size
x == y == bitunpacked(y)      # true

# broadcasting assignment works
y .= rand(UInt8, 32) .& 0b1111 # assign new values
```

## Limitations

- The first dimension of the input array must currently be divisible by the least common multiple of 8 and the bitwidth.
- Broadcasting assignment first materializes the unpacked result.

## Installation

```julia
using Pkg
Pkg.add("BitPacking")
```

## Acknowledgements

BitPacking.jl partially inspired by [IntArrays.jl](https://github.com/bicycle1885/IntArrays.jl)
