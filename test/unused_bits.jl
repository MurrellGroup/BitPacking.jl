# Julia 1.14 lets a primitive type declare fewer bits than it stores. These
# need no `bitwidth` overload, and `reinterpret` refuses them.
primitive type NativeFloat4 4 end
primitive type NativeFloat6 6 end
primitive type NativeUInt24 24 end

native4(x::UInt8) = Core.Intrinsics.trunc_int(NativeFloat4, x)
native6(x::UInt8) = Core.Intrinsics.trunc_int(NativeFloat6, x)
Base.convert(::Type{NativeFloat4}, x::UInt8) = native4(x)
Base.convert(::Type{UInt8}, x::NativeFloat4) = BitPacking.raw_bits(x)

@testset "Primitive types with unused bits" begin
    @testset "bitwidth and raw bits" begin
        @test BitPacking.bitwidth(NativeFloat4) == 4
        @test BitPacking.bitwidth(NativeFloat6) == 6
        @test BitPacking.bitwidth(NativeUInt24) == 24
        @test BitPacking.bitwidth(native4(0x01)) == 4

        x = native4(0xfa)
        @test BitPacking.raw_bits(x) === 0x0a
        @test BitPacking.from_raw_bits(NativeFloat4, 0x0a) === x
        @test BitPacking.from_raw_bits(NativeFloat4, 0xfa) === x
        @test BitPacking.raw_bits(Core.Intrinsics.trunc_int(NativeUInt24, 0xffbcdef0)) === 0x00bcdef0

        @test BitPacking.raw_bits(0x0a) === 0x0a
        @test BitPacking.raw_bits(true) === 0x01
        @test BitPacking.from_raw_bits(Bool, 0x01) === true
        @test BitPacking.from_raw_bits(Int16, 0xffff) === Int16(-1)
    end

    @testset "NArray" begin
        v = NVector((native4(0x0a), native4(0x03)))
        @test v isa NVector{NativeFloat4,2}
        @test v.data == 0x3a
        @test Tuple(v) === (native4(0x0a), native4(0x03))
        @test v[1] === native4(0x0a)
        @test v[2] === native4(0x03)

        v6 = NVector{NativeFloat6,4}(native6.((0x01, 0x02, 0x03, 0x04)))
        @test v6.data == (0x81, 0x30, 0x10)
        @test Tuple(v6) === native6.((0x01, 0x02, 0x03, 0x04))
    end

    @testset "NarrowTuple" begin
        x24 = Core.Intrinsics.trunc_int(NativeUInt24, 0x00bcdef0)
        T = @NarrowTuple{NativeFloat4,NativeUInt24,NativeFloat4}
        @test T === NarrowTuple{Tuple{NativeFloat4,NativeUInt24,NativeFloat4},UInt32}
        nt = T(native4(0x0a), x24, native4(0x03))
        @test nt.data == 0x3bcdef0a
        @test Tuple(nt) === (native4(0x0a), x24, native4(0x03))

        nt6 = NarrowTuple((native6(0x2a), 0xbc))
        @test nt6.data == (0x2a, 0x2f)
        @test Tuple(nt6) === (native6(0x2a), 0xbc)
    end

    @testset "NarrowArray" begin
        values = native4.(UInt8[0x01, 0x02, 0x03, 0x04])
        narrow = NarrowArray{NativeFloat4}(values)
        @test narrow isa NarrowVector{NativeFloat4}
        @test collect(reinterpret(UInt8, narrow)) == UInt8[0x21, 0x43]
        @test narrow[3] === native4(0x03)
        @test copy(narrow) isa Vector{NativeFloat4}
        @test copy(narrow) == values
        @test NativeFloat4.(narrow) == values

        # GPU broadcasts need these to infer
        @test @inferred(NarrowArray{NativeFloat4}(values)) == narrow
        @test @inferred(copy(narrow)) == values
        @test @inferred(broadcast(NativeFloat4, narrow)) == values

        # conversions on either side of the packing
        @test NarrowArray{NativeFloat4}(UInt8[0x01, 0x02, 0x03, 0x04]) == narrow
        @test copy(NarrowArray{UInt8}(values)) == UInt8[0x01, 0x02, 0x03, 0x04]
        @test copy(NarrowArray{UInt8}(narrow)) == UInt8[0x01, 0x02, 0x03, 0x04]

        matrix = NarrowArray{NativeFloat4}(reshape(values, 2, 2))
        @test collect(reinterpret(UInt8, matrix)) == UInt8[0x21 0x43]
        @test copy(matrix) == reshape(values, 2, 2)

        values6 = native6.(UInt8[0x01, 0x02, 0x03, 0x04])
        narrow6 = NarrowArray{NativeFloat6}(values6)
        @test collect(reinterpret(UInt8, narrow6)) == UInt8[0x81, 0x30, 0x10]
        @test copy(narrow6) == values6
        @test copy(reinterpret(NativeFloat6, reinterpret(Bool, narrow6))) == values6

        @test Narrow(NativeFloat4).(values) == narrow
        @test copy(reinterpret(Narrow(NativeFloat4), UInt8[0x21, 0x43])) == values

        dest = similar(narrow)
        dest .= reverse(values)
        @test copy(dest) == reverse(values)
        dest .= UInt8[0x01, 0x02, 0x03, 0x04]
        @test copy(dest) == values
    end
end
