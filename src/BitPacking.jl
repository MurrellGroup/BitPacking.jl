module BitPacking

include("bitwidth.jl")
export bitwidth

include("chunk.jl")

include("group.jl")

include("packbits.jl")
export packbits, packbits!
export unpackbits, unpackbits!

include("BitPackedArray.jl")
export BitPackedArray
export bitpacked, bitunpacked

end
