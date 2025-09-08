module BitPacking

include("chunk.jl")
include("group.jl")

include("bitwidth.jl")
export bitwidth

include("packbits.jl")
export packbits, packbits!
export unpackbits, unpackbits!

include("BitPackedArray.jl")
export BitPackedArray
export bitpacked, bitunpacked

end
