using BitPacking
using Test

primitive type Float4_E2M1FN 8 end
primitive type Float6_E3M2FN 8 end

BitPacking.bitwidth(::Type{Float4_E2M1FN}) = 4
Float4_E2M1FN(x::UInt8) = reinterpret(Float4_E2M1FN, x)
Base.convert(::Type{Float4_E2M1FN}, x::UInt8) = Float4_E2M1FN(x)
Base.convert(::Type{UInt8}, x::Float4_E2M1FN) = reinterpret(UInt8, x)

BitPacking.bitwidth(::Type{Float6_E3M2FN}) = 6
Float6_E3M2FN(x::UInt8) = reinterpret(Float6_E3M2FN, x)
Base.convert(::Type{Float6_E3M2FN}, x::UInt8) = Float6_E3M2FN(x)

float4(x::UInt8) = reinterpret(Float4_E2M1FN, x)
float6(x::UInt8) = reinterpret(Float6_E3M2FN, x)
bits(x::Float4_E2M1FN) = reinterpret(UInt8, x)
bits(x::Float6_E3M2FN) = reinterpret(UInt8, x)
bits(x) = x

@testset "BitPacking.jl" begin
    @testset "NArray" begin
        bools = (true, false, true, false, true, false, true, false)
        v = NVector(bools)

        @test v isa NVector{Bool,8}
        @test length(v) == 8
        @test size(v) == (8,)
        @test v.data == 0x55
        @test Tuple(v) == bools
        @test collect(v) == Bool[1, 0, 1, 0, 1, 0, 1, 0]
        @test v[1] === true
        @test v[8] === false
        @test BitPacking.bitwidth(v) == 8
        @test BitPacking.bitwidth(typeof(v)) == 8
        @test reinterpret(UInt8, v) == 0x55
        @test reinterpret(Unsigned, v) == 0x55

        b = Broadcast.broadcastable(v)
        @test b isa BitPacking.SVector{8,Bool}
        @test Tuple(b) == bools
        @test (!).(v) == BitPacking.SVector{8,Bool}((false, true, false, true, false, true, false, true))

        f4 = NVector((float4(0x0a), float4(0x03)))
        @test f4 isa NVector{Float4_E2M1FN,2}
        @test f4.data == 0x3a
        @test bits.(Tuple(f4)) == (0x0a, 0x03)
        @test bits(f4[1]) == 0x0a
        @test bits(f4[2]) == 0x03
        @test BitPacking.bitwidth(f4) == 8
        @test reinterpret(UInt8, f4) == 0x3a

        bools9 = (true, false, true, false, true, false, true, false, true)
        v9 = NVector(bools9)
        @test v9 isa NVector{Bool,9}
        @test v9.data == (0x55, 0x01)
        @test Tuple(v9) == bools9
        @test v9[1] === true
        @test v9[9] === true
        @test reinterpret(UInt16, v9) == 0x0155

        f6 = NVector((float6(0x01), float6(0x02), float6(0x03), float6(0x04)))
        @test f6 isa NVector{Float6_E3M2FN,4}
        @test f6.data == (0x81, 0x30, 0x10)
        @test bits.(Tuple(f6)) == (0x01, 0x02, 0x03, 0x04)
        @test bits(f6[1]) == 0x01
        @test bits(f6[2]) == 0x02
        @test bits(f6[3]) == 0x03
        @test bits(f6[4]) == 0x04
        @test_throws BoundsError f6[5]

        m = NMatrix{Bool,2,4}(bools)
        @test m isa NMatrix{Bool,2,4}
        @test IndexStyle(typeof(m)) === IndexLinear()
        @test size(m) == (2, 4)
        @test Tuple(m) == bools
        @test m[1, 1] === true
        @test m[2, 4] === false

        sv = BitPacking.SVector{2,UInt8}(0x01, 0x02)
        n = NArray(sv)
        @test n isa NVector{UInt8,2}
        @test n.data == 0x0201
        @test Tuple(n) == (0x01, 0x02)
        @test reinterpret(UInt16, n) == 0x0201

        same_uint = NVector{UInt8,2}(sv)
        @test same_uint isa NVector{UInt8,2}
        @test same_uint.data == 0x0201
        @test Tuple(same_uint) == (0x01, 0x02)

        same_bool = NVector{Bool,8}(BitPacking.SVector{8,Bool}(bools))
        @test same_bool isa NVector{Bool,8}
        @test same_bool.data == 0x55
        @test Tuple(same_bool) == bools

        svec_same = BitPacking.SVector{8,Bool}(v)
        @test svec_same isa BitPacking.SVector{8,Bool}
        @test Tuple(svec_same) == bools

        svec_cross = BitPacking.SVector{8,Int}(v)
        @test svec_cross isa BitPacking.SVector{8,Int}
        @test Tuple(svec_cross) == (1, 0, 1, 0, 1, 0, 1, 0)

        bool_from_int = NVector{Bool,8}(BitPacking.SVector{8,Int}(1, 0, 1, 0, 1, 0, 1, 0))
        @test bool_from_int isa NVector{Bool,8}
        @test bool_from_int.data == 0x55
        @test Tuple(bool_from_int) == bools
        @test NVector{Bool,8}((1, 0, 1, 0, 1, 0, 1, 0)) == bool_from_int
        @test NVector{Bool,8}(1, 0, 1, 0, 1, 0, 1, 0) == bool_from_int

        uint_from_bool = NVector{UInt8,8}(BitPacking.SVector{8,Bool}(bools))
        @test uint_from_bool isa NVector{UInt8,8}
        @test Tuple(uint_from_bool) == (0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00)

        f4_from_uint = NVector{Float4_E2M1FN,2}(BitPacking.SVector{2,UInt8}(0x0a, 0x03))
        @test f4_from_uint isa NVector{Float4_E2M1FN,2}
        @test f4_from_uint.data == 0x3a
        @test bits.(Tuple(f4_from_uint)) == (0x0a, 0x03)
        @test NVector{Float4_E2M1FN,2}((0x0a, 0x03)) == f4_from_uint
        @test NVector{Float4_E2M1FN,2}(0x0a, 0x03) == f4_from_uint

        @test BitPacking.StaticArrays.similar_type(typeof(v), Int, BitPacking.StaticArrays.Size(2)) === NVector{Int,2}
        @test_throws ArgumentError NVector{Bool,4}((true, false))
        @test_throws ArgumentError NVector{Bool,9}(0x01)
    end

    @testset "NarrowTuple" begin
        T3 = @NarrowTuple{Float4_E2M1FN,UInt8,Float4_E2M1FN}
        @test T3 === NarrowTuple{Tuple{Float4_E2M1FN,UInt8,Float4_E2M1FN},UInt16}
        @test BitPacking.bitwidth(T3) == 16
        @test BitPacking.bitwidth(Tuple{Float4_E2M1FN,UInt8,Float4_E2M1FN}) == 24

        x3 = (float4(0x0a), 0xbc, float4(0x03))
        nt3 = T3(x3)
        @test nt3 isa T3
        @test nt3.data == 0x3bca
        @test bits.(Tuple(nt3)) == (0x0a, 0xbc, 0x03)
        @test bits(nt3[1]) == 0x0a
        @test nt3[2] == 0xbc
        @test bits(nt3[Val(3)]) == 0x03

        T4 = @NarrowTuple{Float4_E2M1FN,UInt8,Float4_E2M1FN,UInt8}
        @test T4 === NarrowTuple{
            Tuple{Float4_E2M1FN,UInt8,Float4_E2M1FN,UInt8},
            NTuple{3,UInt8},
        }

        x4 = (float4(0x01), 0x23, float4(0x04), 0x56)
        nt4 = NarrowTuple(x4)
        @test nt4 isa T4
        @test nt4.data == (0x31, 0x42, 0x56)
        @test bits.(Tuple(nt4)) == (0x01, 0x23, 0x04, 0x56)
        @test bits.(collect(nt4)) == [0x01, 0x23, 0x04, 0x56]

        nt_bool = NarrowTuple((0x00, true))
        @test nt_bool.data == (0x00, 0x01)
        @test BitPacking.bitwidth(nt_bool) == 9
        @test BitPacking.bitwidth(typeof(nt_bool)) == 9
        @test sprint(show, nt_bool) == "@NarrowTuple(0x00, true)"
        @test sprint(show, typeof(nt_bool)) == "@NarrowTuple{UInt8, Bool}"
        @test sprint(show, T3) == "@NarrowTuple{Float4_E2M1FN, UInt8, Float4_E2M1FN}"
        @test sprint(show, NarrowTuple) == "NarrowTuple"
        @test sprint(show, NarrowTuple{Tuple{UInt8,Bool}}) == "NarrowTuple{Tuple{UInt8, Bool}}"
        @test startswith(sprint(show, Union{typeof(nt_bool),T3}), "Union{")
        @test contains(sprint(show, methods(show, Tuple{IO,Type{<:NarrowTuple}})), "T<:NarrowTuple")
        @test length(typeof(nt_bool)) == 2
        @test firstindex(nt_bool) == 1
        @test lastindex(nt_bool) == 2
        @test convert(Tuple{UInt8,Bool}, nt_bool) == (0x00, true)
        @test NarrowTuple{Tuple{UInt8,Bool}}(0x00, true) == nt_bool
        @test @NarrowTuple(0x00, true) == nt_bool
        @test @NarrowTuple((0x00, true)) == nt_bool

        nt_nested = @NarrowTuple(true, nt_bool)
        @test BitPacking.bitwidth(nt_nested) == 10
        @test nt_nested.data == (0x01, 0x02)
        @test Tuple(nt_nested) == (true, nt_bool)
        @test sprint(show, typeof(nt_nested)) == "@NarrowTuple{Bool, @NarrowTuple{UInt8, Bool}}"

        @test BitPacking.bitwidth(BitPacking.ZeroPad{7}) == 7
        @test BitPacking.bitwidth(BitPacking.OnePad{7}) == 7
        @test BitPacking.bitwidth(BitPacking.ZeroPad(7)) == 7
        @test BitPacking.bitwidth(BitPacking.OnePad(7)) == 7
        @test sprint(show, BitPacking.ZeroPad(7)) == "BitPacking.ZeroPad(7)"
        @test sprint(show, BitPacking.OnePad(7)) == "BitPacking.OnePad(7)"

        Tzero = @NarrowTuple{UInt8,Bool,BitPacking.ZeroPad{7}}
        @test Tzero === NarrowTuple{Tuple{UInt8,Bool,BitPacking.ZeroPad{7}},UInt16}
        @test sprint(show, Tzero) == "@NarrowTuple{UInt8, Bool, BitPacking.ZeroPad{7}}"
        nt_zero = Tzero(0x00, true)
        @test nt_zero.data == 0x0100
        @test BitPacking.bitwidth(nt_zero) == 16
        @test Tuple(nt_zero) == (0x00, true, BitPacking.ZeroPad(7))
        @test length(nt_zero) == 3
        @test sprint(show, nt_zero) == "@NarrowTuple(0x00, true, BitPacking.ZeroPad(7))"
        @test_throws ArgumentError Tzero(0xffff)

        Tmidpad = @NarrowTuple{UInt8,BitPacking.ZeroPad{7},Bool}
        nt_midpad = Tmidpad(0xff, true)
        @test nt_midpad.data == 0x80ff
        @test Tuple(nt_midpad) == (0xff, BitPacking.ZeroPad(7), true)
        @test sprint(show, nt_midpad) == "@NarrowTuple(0xff, BitPacking.ZeroPad(7), true)"
        @test sprint(show, @NarrowTuple(BitPacking.ZeroPad(8))) == "@NarrowTuple(BitPacking.ZeroPad(8))"
        nt_midpad_value = Tmidpad(0xff, BitPacking.ZeroPad(7), true)
        @test nt_midpad_value == nt_midpad
        @test @NarrowTuple(0xff, BitPacking.ZeroPad(7), true) == nt_midpad

        Tone = @NarrowTuple{UInt8,Bool,BitPacking.OnePad{7}}
        nt_one = Tone(0x00, true)
        @test nt_one.data == 0xff00
        @test Tuple(nt_one) == (0x00, true, BitPacking.OnePad(7))
        @test NarrowTuple((0x00, true, BitPacking.OnePad(7))) == nt_one
        @test NarrowTuple{Tuple{UInt8,Bool,BitPacking.OnePad{7}}}(0x00, true, BitPacking.OnePad(7)) == nt_one
        @test_throws ArgumentError Tone(0x0100)

        Tbytes = @NarrowTuple{Float4_E2M1FN,BitPacking.ZeroPad{5},UInt8}
        nt_bytes = Tbytes(float4(0x0a), 0xbc)
        @test nt_bytes.data == (0x0a, 0x78, 0x01)
        nt_bytes_tuple = Tuple(nt_bytes)
        @test bits(nt_bytes_tuple[1]) == 0x0a
        @test nt_bytes_tuple[2] == BitPacking.ZeroPad(5)
        @test nt_bytes_tuple[3] == 0xbc
        @test_throws ArgumentError Tbytes((0xfa, 0x78, 0x01))

        Tonebytes = @NarrowTuple{Float4_E2M1FN,BitPacking.OnePad{5},UInt8}
        nt_onebytes = Tonebytes(float4(0x0a), 0xbc)
        @test nt_onebytes.data == (0xfa, 0x79, 0x01)
        nt_onebytes_tuple = Tuple(nt_onebytes)
        @test bits(nt_onebytes_tuple[1]) == 0x0a
        @test nt_onebytes_tuple[2] == BitPacking.OnePad(5)
        @test nt_onebytes_tuple[3] == 0xbc
    end

    @testset "NarrowArray" begin
        values = Bool[1, 0, 1, 0, 1, 0, 1, 0]
        narrow = NarrowArray{Bool}(values)

        @test narrow isa NarrowVector{Bool}
        @test size(narrow) == (8,)
        @test copy(narrow) == values
        @test contains(sprint(show, MIME("text/plain"), narrow), "8-element NarrowVector")

        host_narrow = BitPacking.Adapt.adapt(Array, narrow)
        @test host_narrow isa NarrowVector{Bool}
        @test parent(host_narrow) isa Vector
        @test eltype(parent(host_narrow)) <: NVector{Bool}
        @test copy(host_narrow) == values

        int_values = Int.(narrow)
        @test size(int_values) == size(narrow)
        @test parent(int_values) isa Vector{BitPacking.SVector{8,Int}}
        @test collect(int_values) == Int[1, 0, 1, 0, 1, 0, 1, 0]

        uint_values = UInt8[1, 0, 1, 0, 1, 0, 1, 0]
        bool_from_uint = NarrowArray{Bool}(uint_values)
        @test bool_from_uint isa NarrowVector{Bool}
        @test collect(reinterpret(UInt8, bool_from_uint)) == UInt8[0x55]
        @test copy(bool_from_uint) == values
        @test NarrowArray{Bool}(bool_from_uint) === bool_from_uint

        values16 = Bool[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        narrow16 = NarrowArray{Bool}(values16)
        bytes16 = reinterpret(UInt8, narrow16)
        @test bytes16 isa AbstractVector{UInt8}
        @test collect(bytes16) == UInt8[0x55, 0x33]
        @test collect(reinterpret(UInt16, narrow16)) == UInt16[0x3355]

        f4_values = Float4_E2M1FN[float4(0x01), float4(0x02), float4(0x03), float4(0x04)]
        f4_narrow = NarrowArray{Float4_E2M1FN}(f4_values)
        @test size(f4_narrow) == (4,)
        @test collect(reinterpret(UInt8, f4_narrow)) == UInt8[0x21, 0x43]
        @test collect(UInt8.(f4_narrow)) == UInt8[0x01, 0x02, 0x03, 0x04]
        @test NarrowArray{Float4_E2M1FN}(UInt8[0x01, 0x02, 0x03, 0x04]) == f4_narrow
        @test copy(NarrowArray{UInt8}(f4_narrow)) == UInt8[0x01, 0x02, 0x03, 0x04]

        f4x4_narrow = NarrowArray{Float4_E2M1FN,1,4}(f4_values)
        @test f4x4_narrow isa NarrowVector{Float4_E2M1FN,4}
        @test eltype(parent(f4x4_narrow)) <: NVector{Float4_E2M1FN,4}
        @test collect(reinterpret(UInt8, f4x4_narrow)) == UInt8[0x21, 0x43]
        @test bits.(copy(f4x4_narrow)) == bits.(f4_values)
        @test NarrowArray{Float4_E2M1FN,1,4}(f4x4_narrow) === f4x4_narrow
        @test NarrowVector{Float4_E2M1FN,4}(f4_narrow) == f4x4_narrow

        f4_bits = reinterpret(Bool, f4_narrow)
        @test f4_bits isa NarrowVector{Bool}
        @test size(f4_bits) == (16,)
        @test collect(reinterpret(UInt8, f4_bits)) == UInt8[0x21, 0x43]
        @test bits.(copy(reinterpret(Float4_E2M1FN, f4_bits))) == bits.(f4_values)
        host_f4_bits = BitPacking.Adapt.adapt(Array, f4_bits)
        @test host_f4_bits isa NarrowVector{Bool}
        @test eltype(parent(host_f4_bits)) <: NVector{Bool}
        @test collect(reinterpret(UInt8, host_f4_bits)) == UInt8[0x21, 0x43]

        f4_matrix = NarrowArray{Float4_E2M1FN}(reshape(f4_values, 2, 2))
        f4_matrix_explicit = NarrowArray{Float4_E2M1FN,2}(reshape(f4_values, 2, 2))
        @test f4_matrix_explicit == f4_matrix
        @test NarrowArray{Float4_E2M1FN,2}(f4_matrix) === f4_matrix
        f4_bytes = reinterpret(UInt8, f4_matrix)
        @test size(f4_bytes) == (1, 2)
        @test collect(f4_bytes) == UInt8[0x21 0x43]
        f4_uints = UInt8.(f4_matrix)
        @test size(f4_uints) == size(f4_matrix)
        @test parent(f4_uints) isa Matrix{BitPacking.SVector{2,UInt8}}
        @test collect(f4_uints) == UInt8[0x01 0x03; 0x02 0x04]

        f6_values = Float6_E3M2FN[float6(0x01), float6(0x02), float6(0x03), float6(0x04)]
        f6_narrow = NarrowArray{Float6_E3M2FN}(f6_values)
        @test collect(reinterpret(UInt8, f6_narrow)) == UInt8[0x81, 0x30, 0x10]
        f6_bits = reinterpret(Bool, f6_narrow)
        @test f6_bits isa NarrowVector{Bool}
        @test size(f6_bits) == (24,)
        @test collect(reinterpret(UInt8, f6_bits)) == UInt8[0x81, 0x30, 0x10]
        @test bits.(copy(reinterpret(Float6_E3M2FN, f6_bits))) == bits.(f6_values)

        @test_throws ArgumentError NarrowArray{Bool}(values[1:4])
        @test_throws ArgumentError NarrowArray{Float4_E2M1FN,1,3}(f4_values[1:3])
        @test_throws ArgumentError NarrowArray{Float4_E2M1FN,1,4}(f4_values[1:2])
        @test bits.(copy(reinterpret(Float4_E2M1FN, narrow))) == UInt8[0x05, 0x05]
        @test_throws ArgumentError reinterpret(UInt16, narrow)
    end

    @testset "Narrow" begin
        values = Bool[1, 0, 1, 0, 1, 0, 1, 0]

        # Narrow{T}.(arr) packs into a NarrowArray{T}
        packed = Narrow{Bool}.(values)
        @test packed isa NarrowVector{Bool}
        @test copy(packed) == values
        @test packed == NarrowArray{Bool}(values)

        # the inner expression fuses, narrowing happens at the boundary
        fused = Narrow{Bool}.(.!values)
        @test fused isa NarrowVector{Bool}
        @test copy(fused) == .!values

        # reinterpret(Narrow{T}, bytes) views packed bytes without copying
        bytes = UInt8[0x55]
        viewed = reinterpret(Narrow{Bool}, bytes)
        @test viewed isa NarrowVector{Bool}
        @test copy(viewed) == values
        bytes[1] = 0x00
        @test copy(viewed) == falses(8)

        # destination broadcasting packs into preallocated narrow storage
        dest = similar(NarrowArray{Bool}(values))
        @test dest isa NarrowVector{Bool}
        @test size(dest) == size(values)
        dest .= values
        @test copy(dest) == values
        dest .= .!values
        @test copy(dest) == .!values

        # float4 round trip via both packing forms
        f4_values = Float4_E2M1FN[float4(0x01), float4(0x02), float4(0x03), float4(0x04)]
        f4_packed = Narrow{Float4_E2M1FN}.(f4_values)
        @test f4_packed isa NarrowVector{Float4_E2M1FN}
        @test collect(reinterpret(UInt8, f4_packed)) == UInt8[0x21, 0x43]
        @test bits.(copy(reinterpret(Narrow{Float4_E2M1FN}, UInt8[0x21, 0x43]))) == bits.(f4_values)

        # cross-type packing converts before packing, matching the constructor
        @test Narrow{Float4_E2M1FN}.(UInt8[0x01, 0x02, 0x03, 0x04]) == f4_packed

        # matrix destination chunks along the first dimension
        src = repeat(values, 1, 2)
        mat = similar(NarrowArray{Bool}(src))
        mat .= src
        @test copy(mat) == src

        # cross-type destination broadcast converts before packing
        f4_dest = similar(f4_packed)
        f4_dest .= UInt8[0x01, 0x02, 0x03, 0x04]
        @test collect(reinterpret(UInt8, f4_dest)) == UInt8[0x21, 0x43]

        # Narrow{T}.(narr) dispatches on a NarrowArray source
        @test Narrow{Bool}.(packed) == packed

        # print_array handles arrays beyond vectors and matrices
        arr3 = NarrowArray{Bool}(reshape(repeat(values, 4), 8, 2, 2))
        @test contains(sprint(show, MIME("text/plain"), arr3), "NarrowArray")
    end

end
