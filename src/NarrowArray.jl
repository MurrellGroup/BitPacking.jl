"""
    NarrowArray{T}(array::AbstractArray)
    NarrowArray{T,N,L}(array::AbstractArray)

Array wrapper whose parent stores packed [`NArray`](@ref) chunks.

`NarrowArray{T}` presents an `AbstractArray{T}` interface while storing groups
of logical values in the first dimension as `NVector` chunks. For an element type
`T`, the default chunk length is `pack_count(T) == 8 ÷ gcd(bitwidth(T), 8)`.
Use `NarrowArray{T,N,L}` or `NarrowVector{T,L}` to choose a wider chunk length.

The constructor packs an existing logical array:

```julia
julia> x = NarrowArray{Bool}([true, false, true, false, true, false, true, false])
8-element NarrowVector{Bool, 8, Vector{NVector{Bool, 8, UInt8}}}:
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

`NarrowArray{T}(x)` converts values to `T` before packing when `eltype(x) != T`.
`parent(x)` exposes the packed chunk array. `copy(x)` materializes the logical
values, while `reinterpret(T, x)` reinterprets the whole logical bit buffer and
rescales the leading dimension by `bitwidth`. Type broadcasts such as `T.(x)`
convert packed chunks to dense logical `T` values. Display uses a host-adapted
copy so GPU-backed parents do not print through scalar indexing.
"""
struct NarrowArray{T,N,L,A<:AbstractArray{<:NVector{T,L},N}} <: AbstractArray{T,N}
    parent::A
end

pack_count(::Type{T}) where T = 8 ÷ gcd(bitwidth(T), 8)

function check_chunk_length(::Type{T}, L) where T
    L isa Integer ||
        throw(ArgumentError("NarrowArray{$T} chunk length must be an integer, got $L"))
    L >= 1 ||
        throw(ArgumentError("NarrowArray{$T} chunk length must be positive, got $L"))

    bits = bitwidth(T) * Int(L)
    bits % 8 == 0 ||
        throw(ArgumentError("NarrowArray{$T,$L} chunks must contain a whole number of bytes, got $bits bits"))
    return Int(L)
end

NarrowArray{T}(arr::AbstractArray{S,N}) where {T,S,N} =
    NarrowArray{T,N,pack_count(T)}(arr)

NarrowArray{T,N}(arr::AbstractArray{S,N}) where {T,N,S} =
    NarrowArray{T,N,pack_count(T)}(arr)

function NarrowArray{T,N,L}(arr::AbstractArray{S,N}) where {T,N,L,S}
    isbitstype(S) ||
        throw(ArgumentError("NarrowArray{$T} input element type must be a bitstype, got $S"))

    chunk_length = check_chunk_length(T, L)
    size(arr, 1) % chunk_length == 0 ||
        throw(ArgumentError("the first dimension of a NarrowArray{$T,$N,$L} input must be divisible by $chunk_length, got $(size(arr, 1))"))

    chunks = if S === T
        NVector{T,chunk_length}.(dense_chunks(Val(chunk_length), arr))
    else
        NVector{T,chunk_length}.(SVector{chunk_length,S}.(dense_chunks(Val(chunk_length), arr)))
    end
    return NarrowArray(chunks)
end

# `reinterpret` refuses arrays of a primitive type with unused bits (Julia
# 1.14+) or with an array stride above its `sizeof` (a 24-bit primitive type
# before 1.14), so for those the conversions between dense values and chunks go
# through raw bits instead, at the cost of a temporary. They map closures over
# `T` because `from_raw_bits.(T, x)` does not infer when `T` is a static
# parameter, and GPU broadcasts need it to.

# The first dimension of `dense` as runs of `L` values, each an `NTuple`.
function dense_chunks(::Val{L}, dense::AbstractArray{T}) where {L,T}
    array_reinterpret_fails(T) || return reinterpret(NTuple{L,T}, dense)
    raw = reinterpret(NTuple{L,raw_type(T)}, raw_bits.(dense))
    return Broadcast.broadcasted(chunk -> map(bits -> from_raw_bits(T, bits), chunk), raw)
end

# The inverse of `dense_chunks`: runs of values, flattened into the first
# dimension.
function dense_values(::Type{T}, chunks::AbstractArray) where T
    array_reinterpret_fails(T) || return reinterpret(T, chunks)
    raw = reinterpret(raw_type(T), raw_chunk.(chunks))
    return map(bits -> from_raw_bits(T, bits), raw)
end

raw_chunk(xs) = raw_bits.(Tuple(xs))

NarrowArray{T}(arr::NarrowArray{T}) where T = arr
NarrowArray{T,N}(arr::NarrowArray{T,N}) where {T,N} = arr
NarrowArray{T,N,L}(arr::NarrowArray{T,N,L}) where {T,N,L} = arr
NarrowArray{T}(arr::NarrowArray) where T = NarrowArray{T}(T.(arr))
NarrowArray{T,N,L}(arr::NarrowArray{S,N}) where {T,N,L,S} = NarrowArray{T,N,L}(T.(arr))

const NarrowVector{T,L,A<:AbstractVector{<:NVector{T,L}}} = NarrowArray{T,1,L,A}
const NarrowMatrix{T,L,A<:AbstractMatrix{<:NVector{T,L}}} = NarrowArray{T,2,L,A}

Base.parent(arr::NarrowArray) = arr.parent

Adapt.adapt_structure(to, arr::NarrowArray) = NarrowArray(Adapt.adapt(to, parent(arr)))

inner_size(arr::NarrowArray, i::Integer) = size(eltype(parent(arr)), i)
inner_size(arr::NarrowArray{<:Any,N}) where N = ntuple(i -> inner_size(arr, i), Val(N))

Base.size(arr::NarrowArray, i::Integer) = size(parent(arr), i) * size(eltype(parent(arr)), i)
Base.size(arr::NarrowArray) = size(parent(arr)) .* inner_size(arr)

Base.IndexStyle(::Type{<:NarrowArray}) = IndexCartesian()
function Base.getindex(arr::NarrowArray{T,N}, i::Vararg{Int,N}) where {T,N}
    outer_inner_i = ntuple(j -> fldmod1(i[j], inner_size(arr, j)), Val(N))
    outer_i = first.(outer_inner_i)
    inner_i = last.(outer_inner_i)
    return parent(arr)[outer_i...][inner_i...]
end

function narrow_chunk_type(::Type{T}, L=pack_count(T)) where T
    chunk_length = check_chunk_length(T, L)
    return NArray{T,1,Tuple{chunk_length},storage_type(bitwidth(T) * chunk_length)}
end

# The same bits along dimension 1 as `T`: a plain array when `T` fills its
# bytes, and a `NarrowArray{T}` otherwise. Chunks are whole bytes, so the bits
# along dimension 1 are a multiple of 8, and when they are also a multiple of
# `bitwidth(T)` they are whole `T` chunks.
function Base.reinterpret(::Type{T′}, arr::NarrowArray{S}) where {T′,S}
    T = concrete_element(T′)
    source_bits = bitwidth(S)
    target_bits = check_bitwidth(T)
    T === S && target_bits != 8 * sizeof(T) && return arr
    (size(arr, 1) * source_bits) % target_bits == 0 ||
        throw(ArgumentError("cannot reinterpret $(size(arr, 1)) $source_bits-bit value(s) as $target_bits-bit $T"))

    bytes = reinterpret(UInt8, parent(arr))
    if target_bits == 8 * sizeof(T)
        return reinterpret(T, bytes)
    else
        return NarrowArray(reinterpret(narrow_chunk_type(T), bytes))
    end
end

function Broadcast.broadcasted(::Type{T}, arr::NarrowArray) where T
    isbitstype(T) || return Broadcast.Broadcasted(T, (arr,))

    L = length(eltype(parent(arr)))
    chunks = SVector{L,T}.(parent(arr))
    return dense_values(T, chunks)
end

Base.copy(arr::NarrowArray{T,<:Any,L}) where {T,L} = dense_values(T, SVector{L,T}.(parent(arr)))

"""
    Narrow(T)

Representation tag for the packed form of logical element type `T`. A `Narrow`
instance substitutes for a `Type` in array operations whose result is a
[`NarrowArray{T}`](@ref):

| operation     | to a `NarrowArray{T}`             | from a `NarrowArray`         |
|:--------------|:----------------------------------|:-----------------------------|
| `reinterpret` | `reinterpret(Narrow(T), data)`    | `reinterpret(T, narr)`       |
| broadcast     | `Narrow(T).(array)`               | `T.(narr)`                   |
| `similar`     | `similar(array, Narrow(T), dims)` | `similar(narr, T, dims)`     |

For broadcast these are value conversions: `Narrow(T).(array)` packs values into
a `NarrowArray{T}` and `T.(narr)` unpacks to dense `T` values. `reinterpret`
never converts and never unpacks: `reinterpret(Narrow(T), data)` views the bytes
or packed chunks of `data` as a `NarrowArray{T}`, and `reinterpret(T, narr)`
views the packed bits of `narr` as `T`, which is a plain array when `T` fills
its bytes (`UInt8`, `Float32`) and a `NarrowArray{T}` otherwise (`Bool`, a 4-bit
float). Both keep the bits and rescale dimension 1. `similar(array, Narrow(T),
dims)` allocates an uninitialized `NarrowArray{T}` of logical size `dims` whose
chunk parent follows the backend of `array`, and `similar(narr, T, dims)` a dense
array on the backend of `narr`.

`Narrow(T).(array)` makes the narrowing explicit where `NarrowArray{T}(array)`
hides it; the equivalent in-place form is `dest .= expr` for a preallocated
`NarrowArray{T}` destination. All forms use the default chunk length
`pack_count(T)`, so the leading dimension must be a whole number of chunks.
"""
struct Narrow{T} end

Narrow(::Type{T}) where T = Narrow{T}()

# `dest .= expr` evaluates the (fused) broadcast into a dense temporary of `T`
# on the backend of `dest`, with the usual broadcast rules for its shape, then
# packs it straight into the existing parent at `dest`'s own chunk length `L`.
# This takes over `materialize!` rather than `copyto!`, because a GPU source
# gives the broadcast a GPU style whose `copyto!` would write `dest` element by
# element.
function Broadcast.materialize!(dest::NarrowArray{T,N,L}, bc::Broadcast.Broadcasted) where {T,N,L}
    dense = similar(parent(dest), T, size(dest))
    dense .= bc
    parent(dest) .= NVector{T,L}.(dense_chunks(Val(L), dense))
    return dest
end

# The dense forms delegate to the parent so the result follows its backend.
Base.similar(arr::NarrowArray, T::Type, dims::Dims) = similar(parent(arr), T, dims)

# The eltype-less forms stay narrow and preserve the array's own chunk length.
Base.similar(arr::NarrowArray) = NarrowArray(similar(parent(arr)))
function Base.similar(arr::NarrowArray{T,<:Any,L}, dims::Dims) where {T,L}
    first(dims) % L == 0 ||
        throw(ArgumentError("the first dimension of a NarrowArray{$T} with chunk length $L must be divisible by $L, got $(first(dims))"))
    return NarrowArray(similar(parent(arr), (first(dims) ÷ L, Base.tail(dims)...)))
end

# `dims` is the logical size: `size(similar(arr, Narrow(T), dims)) == dims`.
function Base.similar(arr::AbstractArray, ::Narrow{T}, dims::Dims) where T
    isempty(dims) &&
        throw(ArgumentError("a NarrowArray{$T} needs at least one dimension to chunk along"))
    L = pack_count(T)
    first(dims) % L == 0 ||
        throw(ArgumentError("the first dimension of a NarrowArray{$T} must be divisible by $L, got $(first(dims))"))
    return NarrowArray(similar(arr, narrow_chunk_type(T), (first(dims) ÷ L, Base.tail(dims)...)))
end

Base.similar(arr::AbstractArray, n::Narrow) = similar(arr, n, size(arr))
Base.similar(arr::AbstractArray, n::Narrow, dims::Integer...) = similar(arr, n, map(Int, dims))

# `Narrow(T).(x)` packs the (fused) broadcast `x` into a NarrowArray{T}. Routing
# through the constructor reuses its vectorized, backend-generic packing, so the
# result follows the backend of `x` rather than allocating a host `Array`.
Broadcast.broadcasted(::Narrow{T}, x) where T = NarrowArray{T}(Broadcast.materialize(x))

Base.reinterpret(::Narrow{T}, arr::AbstractArray) where T =
    NarrowArray(reinterpret(narrow_chunk_type(T), arr))

function Base.print_array(io::IO, arr::NarrowArray)
    host = Adapt.adapt(Array, arr)
    if host isa AbstractVecOrMat
        return @invoke Base.print_array(io::IO, host::AbstractVecOrMat)
    else
        return @invoke Base.print_array(io::IO, host::AbstractArray)
    end
end
