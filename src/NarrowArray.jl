"""
    NarrowArray{T}(array::AbstractArray{T})

Array wrapper whose parent stores packed [`NArray`](@ref) chunks.

`NarrowArray{T}` presents an `AbstractArray{T}` interface while storing groups
of logical values in the first dimension as `NVector` chunks. For an element type
`T`, each parent element contains `pack_count(T)` logical values, where
`pack_count(T) == 8 ÷ gcd(bitwidth(T), 8)`.

The constructor packs an existing logical array:

```julia
julia> x = NarrowArray{Bool}([true, false, true, false, true, false, true, false])
8-element NarrowVector{Bool, NVector{Bool, 8, UInt8}, Vector{NVector{Bool, 8, UInt8}}}:
 1
 0
 1
 0
 1
 0
 1
 0
```

The input's first dimension must contain whole chunks. For example,
`NarrowArray{Bool}` requires the first dimension to be divisible by 8, and a
4-bit element type requires it to be divisible by 2.

`parent(x)` exposes the packed chunk array. `copy(x)` materializes the logical
values for vector-backed narrow arrays, and display uses a host-adapted copy so
GPU-backed parents do not print through scalar indexing.
"""
struct NarrowArray{T,N,E<:NArray{T},A<:AbstractArray{E,N}} <: AbstractArray{T,N}
    parent::A
end

pack_count(::Type{T}) where T = 8 ÷ gcd(bitwidth(T), 8)
function NarrowArray{T}(arr::AbstractArray{T}) where T
    S = pack_count(T)
    size(arr, 1) % S == 0 ||
        throw(ArgumentError("the first dimension of a NarrowArray{$T} input must be divisible by $S, got $(size(arr, 1))"))
    return NarrowArray(NVector.(reinterpret(NTuple{S,T}, arr)))
end

const NarrowVector{T} = NarrowArray{T,1}
const NarrowMatrix{T} = NarrowArray{T,2}

Base.parent(arr::NarrowArray) = arr.parent

Adapt.adapt_structure(to, arr::NarrowArray) = NarrowArray(Adapt.adapt(to, parent(arr)))

inner_size(arr::NarrowArray, i::Integer) = size(eltype(parent(arr)), i)
inner_size(arr::NarrowArray{<:Any,N}) where N = ntuple(i -> inner_size(arr, i), Val(N))

Base.size(arr::NarrowArray, i::Integer) = size(parent(arr), i) * size(eltype(parent(arr)), i)
Base.size(arr::NarrowArray{<:Any,N}) where N = ntuple(i -> size(arr, i), Val(N))

Base.IndexStyle(::Type{<:NarrowArray}) = IndexCartesian()
function Base.getindex(arr::NarrowArray{T,N}, i::Vararg{Int,N}) where {T,N}
    outer_inner_i = ntuple(j -> fldmod1(i[j], inner_size(arr, j)), Val(N))
    outer_i = first.(outer_inner_i)
    inner_i = last.(outer_inner_i)
    return parent(arr)[outer_i...][inner_i...]
end

function Base.reinterpret(::Type{T}, arr::NarrowArray) where T
    return reinterpret(T, parent(arr))
end

Base.copy(arr::NarrowArray) = reinterpret(eltype(arr), map(SArray, parent(arr)))

function Base.print_array(io::IO, arr::NarrowArray)
    host = Adapt.adapt(Array, arr)
    if host isa AbstractVecOrMat
        return invoke(Base.print_array, Tuple{IO, AbstractVecOrMat}, io, host)
    else
        return invoke(Base.print_array, Tuple{IO, AbstractArray}, io, host)
    end
end
