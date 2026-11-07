```@meta
CurrentModule = BitPacking
```

# BitPacking

Documentation for [BitPacking](https://github.com/MurrellGroup/BitPacking.jl).

BitPacking packs values narrower than a byte, or of any width up to 128 bits,
densely: [`NArray`](@ref) packs a static group of values into one storage value,
and [`NarrowArray`](@ref) presents an array of such groups as an array of
values. The [bit model](bit-model.md) page states the rules they share.


## Element protocol

```@docs
BitPacking.bitwidth
BitPacking.raw_bits
BitPacking.from_raw_bits
```

## Narrow-element arrays

```@docs
NArray
NarrowArray
Narrow
```
