chunk_size(::Val{W}) where W = abs(lcm(8, W) ÷ W)
packed_chunk_size(::Val{W}) where W = abs(lcm(8, W) ÷ 8)

get_chunks(w::Val, x::AbstractArray) = reinterpret(NTuple{chunk_size(w),UInt8}, x)
get_packed_chunks(w::Val, x::AbstractArray) = reinterpret(NTuple{packed_chunk_size(w),UInt8}, x)

# 1-bit
# _______a _______b _______c _______d _______e _______f _______g _______h
# hgfedcba
# [h1 g1 f1 e1 d1 c1 b1 a1]

function packchunk(::Val{1}, (a, b, c, d, e, f, g, h)::NTuple{8,UInt8})
    b1 = h << 7 | g << 6 | f << 5 | e << 4 | d << 3 | c << 2 | b << 1 | a
    return (b1,)
end

function unpackchunk(::Val{1}, (b1,)::NTuple{1,UInt8})
    a = b1      & 0b1
    b = b1 >> 1 & 0b1
    c = b1 >> 2 & 0b1
    d = b1 >> 3 & 0b1
    e = b1 >> 4 & 0b1
    f = b1 >> 5 & 0b1
    g = b1 >> 6 & 0b1
    h = b1 >> 7
    return (a, b, c, d, e, f, g, h)
end


# 2-bit
# ______aa ______bb ______cc ______dd
# ddccbbaa
# [d2 d1 c2 c1 b2 b1 a2 a1]

function packchunk(::Val{2}, (a, b, c, d)::NTuple{4,UInt8})
    b1 = d << 6 | c << 4 | b << 2 | a
    return (b1,)
end

function unpackchunk(::Val{2}, (b1,)::NTuple{1,UInt8})
    a = b1      & 0b11
    b = b1 >> 2 & 0b11
    c = b1 >> 4 & 0b11
    d = b1 >> 6
    return (a, b, c, d)
end


# 3-bit
# _____aaa _____bbb _____ccc _____ddd _____eee _____fff _____ggg _____hhh
# ccbbbaaa feeedddc hhhgggff
# [c2 c1 b3 b2 b1 a3 a2 a1]
# [f1 e3 e2 e1 d3 d2 d1 c3]
# [h3 h2 h1 g3 g2 g1 f3 f2]

function packchunk(::Val{3}, (a, b, c, d, e, f, g, h)::NTuple{8,UInt8})
    b1 = c << 6 | b << 3 | a
    b2 = f << 7 | e << 4 | d << 1 | c >> 2
    b3 = h << 5 | g << 2 | f >> 1
    return (b1, b2, b3)
end

function unpackchunk(::Val{3}, (b1, b2, b3)::NTuple{3,UInt8})
    a = b1      & 0b111
    b = b1 >> 3 & 0b111
    c = b2 << 2 & 0b100 | b1 >> 6
    d = b2 >> 1 & 0b111
    e = b2 >> 4 & 0b111
    f = b3 << 1 & 0b110 | b2 >> 7
    g = b3 >> 2 & 0b111
    h =                   b3 >> 5
    return (a, b, c, d, e, f, g, h)
end


# 4-bit
# ____aaaa _____bbbb
# bbbbaaaa
# [b4 b3 b2 b1 a4 a3 a2 a1]

function packchunk(::Val{4}, (a, b)::NTuple{2,UInt8})
    b1 = b << 4 | a
    return (b1,)
end

function unpackchunk(::Val{4}, (b1,)::NTuple{1,UInt8})
    a = b1 & 0b1111
    b = b1 >> 4
    return (a, b)
end


# 5-bit
# ___aaaaa ___bbbbb ___ccccc ___ddddd ___eeeee ___ffffff ___gggggg ___hhhhhh
# bbbaaaaa dcccccbb eeeedddd ggfffffe hhhhhggg
# [b3 b2 b1 a5 a4 a3 a2 a1]
# [d1 c5 c4 c3 c2 c1 b5 b4]
# [e4 e3 e2 e1 d5 d4 d3 d2]
# [g2 g1 f5 f4 f3 f2 f1 e5]
# [h5 h4 h3 h2 h1 g5 g4 g3]

function packchunk(::Val{5}, (a, b, c, d, e, f, g, h)::NTuple{8,UInt8})
    b1 = b << 5 | a
    b2 = d << 7 | c << 2 | b >> 3
    b3 = e << 4 | d >> 1
    b4 = g << 6 | f << 1 | e >> 4
    b5 = h << 3 | g >> 2
    return (b1, b2, b3, b4, b5)
end

function unpackchunk(::Val{5}, (b1, b2, b3, b4, b5)::NTuple{5,UInt8})
    a = b1      & 0b11111
    b = b2 << 3 & 0b11000 | b1 >> 5
    c = b2 >> 2 & 0b11111
    d = b3 << 1 & 0b11110 | b2 >> 7
    e = b4 << 4 & 0b10000 | b3 >> 4
    f = b4 >> 1 & 0b11111
    g = b5 << 2 & 0b11100 | b4 >> 6
    h =                     b5 >> 3
    return (a, b, c, d, e, f, g, h)
end


# 6-bit
# __aaaaaa __bbbbbb __cccccc __dddddd
# bbaaaaaa ccccbbbb ddddddcc
# [b2 b1 a6 a5 a4 a3 a2 a1]
# [c4 c3 c2 c1 b6 b5 b4 b3]
# [d6 d5 d4 d3 d2 d1 c6 c5]

function packchunk(::Val{6}, (a, b, c, d)::NTuple{4,UInt8})
    b1 = b << 6 | a
    b2 = c << 4 | b >> 2
    b3 = d << 2 | c >> 4
    return (b1, b2, b3)
end

function unpackchunk(::Val{6}, (b1, b2, b3)::NTuple{3,UInt8})
    a = b1      & 0b111111
    b = b2 << 2 & 0b111100 | b1 >> 6
    c = b3 << 4 & 0b110000 | b2 >> 4
    d =                      b3 >> 2
    return (a, b, c, d)
end


# 7-bit
# _aaaaaaa _bbbbbbb _ccccccc _ddddddd _eeeeeee _fffffff _ggggggg _hhhhhhh
# baaaaaaa ccbbbbbb dddccccc eeeedddd fffffeee ggggggff hhhhhhhg
# [b1 a7 a6 a5 a4 a3 a2 a1]
# [c2 c1 b7 b6 b5 b4 b3 b2]
# [d3 d2 d1 c7 c6 c5 c4 c3]
# [e4 e3 e2 e1 d7 d6 d5 d4]
# [f5 f4 f3 f2 f1 e7 e6 e5]
# [g6 g5 g4 g3 g2 g1 f7 f6]
# [h7 h6 h5 h4 h3 h2 h1 g7]

function packchunk(::Val{7}, (a, b, c, d, e, f, g, h)::NTuple{8,UInt8})
    b1 = b << 7 | a
    b2 = c << 6 | b >> 1
    b3 = d << 5 | c >> 2
    b4 = e << 4 | d >> 3
    b5 = f << 3 | e >> 4
    b6 = g << 2 | f >> 5
    b7 = h << 1 | g >> 6
    return (b1, b2, b3, b4, b5, b6, b7)
end

function unpackchunk(::Val{7}, (b1, b2, b3, b4, b5, b6, b7)::NTuple{7,UInt8})
    a = b1      & 0b1111111
    b = b2 << 1 & 0b1111110 | b1 >> 7
    c = b3 << 2 & 0b1111100 | b2 >> 6
    d = b4 << 3 & 0b1111000 | b3 >> 5
    e = b5 << 4 & 0b1110000 | b4 >> 4
    f = b6 << 5 & 0b1100000 | b5 >> 3
    g = b7 << 6 & 0b1000000 | b6 >> 2
    h =                       b7 >> 1
    return (a, b, c, d, e, f, g, h)
end


# 8-bit
# aaaaaaaa
# aaaaaaaa
# [a8 a7 a6 a5 a4 a3 a2 a1]

packchunk(::Val{8}, x::NTuple{1,UInt8}) = x
unpackchunk(::Val{8}, x::NTuple{1,UInt8}) = x
