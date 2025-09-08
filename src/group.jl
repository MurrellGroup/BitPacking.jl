packgroup(w::Val, group::NTuple{M,NTuple{N,UInt8}}) where {M,N} = map(chunk -> packchunk(w, chunk), group)
unpackgroup(w::Val, group::NTuple{M,NTuple{N,UInt8}}) where {M,N} = map(chunk -> unpackchunk(w, chunk), group)

grouped_chunks(::Val{M}, w::Val, x::AbstractArray) where M = reinterpret(NTuple{M,NTuple{chunk_size(w),UInt8}}, x)
grouped_packed_chunks(::Val{M}, w::Val, x::AbstractArray) where M = reinterpret(NTuple{M,NTuple{packed_chunk_size(w),UInt8}}, x)

#=
autogroup(::Val) = Val(1)
autogroup(::Val{1}) = Val(2)
autogroup(::Val{2}) = Val(8)
autogroup(::Val{3}) = Val(2)
autogroup(::Val{4}) = Val(8)
autogroup(::Val{5}) = Val(1)
autogroup(::Val{6}) = Val(2)
autogroup(::Val{7}) = Val(1)
=#
