using BitPacking
using Test

primitive type Float4_E2M1FN 8 end

BitPacking.bitwidth(::Type{Float4_E2M1FN}) = 4
Float4_E2M1FN(x::UInt8) = reinterpret(Float4_E2M1FN, x)
Base.convert(::Type{Float4_E2M1FN}, x::UInt8) = Float4_E2M1FN(x)

float4(x::UInt8) = reinterpret(Float4_E2M1FN, x)
bits(x::Float4_E2M1FN) = reinterpret(UInt8, x)
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
        @test reinterpret(UInt16, v9) == 0x0155

        m = NMatrix{Bool,2,4}(bools)
        @test m isa NMatrix{Bool,2,4}
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

        uint_from_bool = NVector{UInt8,8}(BitPacking.SVector{8,Bool}(bools))
        @test uint_from_bool isa NVector{UInt8,8}
        @test Tuple(uint_from_bool) == (0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00)

        f4_from_uint = NVector{Float4_E2M1FN,2}(BitPacking.SVector{2,UInt8}(0x0a, 0x03))
        @test f4_from_uint isa NVector{Float4_E2M1FN,2}
        @test f4_from_uint.data == 0x3a
        @test bits.(Tuple(f4_from_uint)) == (0x0a, 0x03)

        @test BitPacking.StaticArrays.similar_type(typeof(v), Int, BitPacking.StaticArrays.Size(2)) === NVector{Int,2}
        @test_throws ArgumentError NVector{Bool,4}((true, false))
        @test_throws ArgumentError NVector{Bool,9}(0x01)
    end

    @testset "NarrowTuple" begin
        T3 = @NarrowTuple{Float4_E2M1FN,UInt8,Float4_E2M1FN}
        @test T3 === NarrowTuple{Tuple{Float4_E2M1FN,UInt8,Float4_E2M1FN},UInt16}
        @test BitPacking.bitwidth(Tuple{Float4_E2M1FN,UInt8,Float4_E2M1FN}) == 16

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
        @test @NarrowTuple(0x00, true) == nt_bool
        @test @NarrowTuple((0x00, true)) == nt_bool

        nt_nested = @NarrowTuple(true, nt_bool)
        @test BitPacking.bitwidth(nt_nested) == 10
        @test nt_nested.data == (0x01, 0x02)
        @test Tuple(nt_nested) == (true, nt_bool)
        @test sprint(show, typeof(nt_nested)) == "@NarrowTuple{Bool, @NarrowTuple{UInt8, Bool}}"

        @test isabstracttype(BitPacking.ZeroPad{7})
        @test isabstracttype(BitPacking.OnePad{7})
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
        @test copy(host_narrow) == values

        @test_throws ArgumentError NarrowArray{Bool}(values[1:4])
    end

end
