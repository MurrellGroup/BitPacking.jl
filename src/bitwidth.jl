bitwidth(::Type{T}) where T = isbitstype(T) ? sizeof(T) * 8 : error("$T is not a bitstype")
bitwidth(::Type{Bool}) = 1
