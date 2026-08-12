module cuDNNExt

import BitPacking
import cuDNN

function cuDNN.checked_array_pointer(t::cuDNN.Tensor, a::BitPacking.NarrowArray)
    parent(a) isa cuDNN.DenseCuArray || throw(ArgumentError(
        "binding for $(t.name) needs GPU-backed storage, got $(typeof(parent(a)))"))
    cuDNN.graph_dtype(eltype(a)) == t.dtype || throw(ArgumentError(
        "binding for $(t.name) has eltype $(eltype(a)), expected $(t.dtype)"))
    length(a) == prod(t.dims) || throw(DimensionMismatch(
        "binding for $(t.name) has $(length(a)) elements, expected $(prod(t.dims))"))
    return pointer(parent(a))
end

end
