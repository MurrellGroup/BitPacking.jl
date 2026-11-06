# Element widths from 1 to 128 bits, as primitive types declared with whole
# bytes and a `bitwidth` overload, so that they work on every Julia version.
const WIDTHS = (1:24..., 31, 32, 33, 48, 63, 64, 65, 100, 127, 128)

for W in WIDTHS
    name = Symbol(:Bits, W)
    @eval primitive type $name $(8 * nextpow(2, cld(W, 8))) end
    @eval BitPacking.bitwidth(::Type{$name}) = $W
end

primitive type Odd24 24 end
primitive type Odd40 40 end
primitive type Odd48 48 end
primitive type Odd56 56 end
struct RGB
    r::UInt8
    g::UInt8
    b::UInt8
end

const WIDTH_TYPES = map(W -> getfield(@__MODULE__, Symbol(:Bits, W)), WIDTHS)

width_type(W) = WIDTH_TYPES[findfirst(==(W), WIDTHS)]

random_value(rng, ::Type{T}) where T = BitPacking.from_raw_bits(T, rand(rng, UInt128))

# The packed bits of `data` as one integer, limb 1 lowest
storage_integer(data::Unsigned) = big(data)
storage_integer(data::Tuple) =
    sum(big(limb) << (8 * sizeof(limb) * (j - 1)) for (j, limb) in enumerate(data); init=big(0))

# The bits of `xs` laid out LSB-first, element 1 lowest
reference_integer(xs) = sum(
    big(BitPacking.raw_bits(x)) << (BitPacking.bitwidth(x) * (i - 1)) for (i, x) in enumerate(xs);
    init=big(0))

@testset "widths" begin
    rng = Random.MersenneTwister(0)

    @testset "storage types" begin
        @test BitPacking.storage_type(0) === Tuple{}
        @test BitPacking.storage_type(1) === UInt8
        @test BitPacking.storage_type(9) === UInt16
        @test BitPacking.storage_type(24) === NTuple{3,UInt8}
        @test BitPacking.storage_type(48) === NTuple{3,UInt16}
        @test BitPacking.storage_type(96) === NTuple{3,UInt32}
        @test BitPacking.storage_type(128) === UInt128
        @test BitPacking.storage_type(129) === NTuple{17,UInt8}
        @test BitPacking.storage_type(256) === NTuple{4,UInt64}
        @test BitPacking.storage_type(384) === NTuple{6,UInt64}
        @test NVector(ntuple(Float32, 8)).data isa NTuple{4,UInt64}
    end

    @testset "$W-bit elements" for (W, T) in zip(WIDTHS, WIDTH_TYPES)
        for L in (1, 2, 3, 5, 8, 9, 17)
            xs = ntuple(_ -> random_value(rng, T), L)
            v = NVector{T,L}(xs)
            @test v.data isa BitPacking.storage_type(W * L)
            @test sizeof(v) == sizeof(BitPacking.storage_type(W * L))
            @test storage_integer(v.data) == reference_integer(xs)
            @test Tuple(v) === xs
            @test all(v[i] === xs[i] for i in 1:L)
            @test_throws BoundsError v[L + 1]
            @test SVector(v) === SVector(xs)
            @test NArray(SVector(xs)) === v
        end
        xs = ntuple(_ -> random_value(rng, T), 3)
        v = @inferred NVector{T,3}(xs)
        @test @inferred(Tuple(v)) === xs
        @test @inferred(v[2]) === xs[2]
    end

    @testset "NarrowArray of $W-bit elements" for W in (3, 10, 12, 16, 24, 48, 100)
        T = width_type(W)
        L = BitPacking.pack_count(T)
        xs = [random_value(rng, T) for _ in 1:(3L), _ in 1:2]
        narrow = NarrowArray{T}(xs)
        @test size(narrow) == size(xs)
        @test copy(narrow) == xs
        @test all(narrow[i, j] === xs[i, j] for i in axes(xs, 1), j in axes(xs, 2))
        bytes = reinterpret(UInt8, narrow)
        @test size(bytes) == (3L * W ÷ 8, 2)
        @test storage_integer(Tuple(bytes[:, 1])) == reference_integer(xs[:, 1])
        @test copy(reinterpret(Narrow(T), collect(bytes))) == xs
    end

    @testset "NarrowArray reinterpret across widths" begin
        xs = [random_value(rng, width_type(12)) for _ in 1:8, _ in 1:2]
        narrow = NarrowArray{width_type(12)}(xs)
        @test reinterpret(width_type(12), narrow) === narrow
        bytes = collect(reinterpret(UInt8, narrow))
        for (T, n) in ((Bool, 96), (width_type(3), 32), (width_type(24), 4), (UInt16, 6),
                       (width_type(48), 2), (Float6_E3M2FN, 16))
            r = reinterpret(T, narrow)
            @test size(r) == (n, 2)
            @test collect(reinterpret(UInt8, r)) == bytes
            if 8 * sizeof(T) == BitPacking.bitwidth(T)
                # a plain array, which goes back to packed form through `Narrow`
                @test r isa AbstractMatrix{T} && !(r isa NarrowArray)
                @test copy(reinterpret(Narrow(width_type(12)), r)) == xs
            else
                @test r isa NarrowMatrix{T}
                @test copy(reinterpret(width_type(12), r)) == xs
            end
        end
        @test_throws ArgumentError reinterpret(width_type(10), narrow)
        @test_throws ArgumentError reinterpret(UInt64, narrow)
    end

    @testset "broadcasting into a NarrowArray" begin
        T = width_type(12)
        xs = [random_value(rng, T) for _ in 1:4, _ in 1:3]
        dest = similar(NarrowArray{T}(xs))
        dest .= xs
        @test copy(dest) == xs
        dest .= xs[:, 1]
        @test copy(dest) == repeat(xs[:, 1], 1, 3)
        dest .= Ref(xs[1, 1])
        @test all(==(xs[1, 1]), copy(dest))
        @test_throws DimensionMismatch dest .= xs[1:2, :]
    end

    @testset "values beyond 128 bits" begin
        @test_throws ArgumentError NVector{NTuple{3,UInt64},1}(((0x1, 0x2, 0x3),))
    end
end

@testset "NArray reinterpret" begin
    rng = Random.MersenneTwister(1)
    types = (Bool, UInt8, UInt16, Float4_E2M1FN, Float6_E3M2FN, Float32,
             width_type(3), width_type(10), width_type(12), width_type(24), width_type(48))

    @testset "$A to $B" for A in types, B in types
        WA, WB = BitPacking.bitwidth(A), BitPacking.bitwidth(B)
        L = lcm(WA, WB) ÷ WA
        v = NVector{A,L}(ntuple(_ -> random_value(rng, A), L))
        r = @inferred reinterpret(B, v)
        @test r isa NVector{B,lcm(WA, WB) ÷ WB}
        @test r.data === v.data
        @test reinterpret(A, r) === v
        # the same bits as a matrix, with dimension 1 rescaled
        m = NMatrix{A,L,2}((Tuple(v)..., Tuple(v)...))
        @test size(reinterpret(B, m)) == (length(r), 2)
        @test reinterpret(A, reinterpret(B, m)) === m
        if WA != WB
            # a bit count along dimension 1 that does not divide throws
            short = lcm(WA, WB) ÷ WA - 1
            short > 0 && WA * short % WB != 0 && @test_throws ArgumentError reinterpret(
                B, NVector{A,short}(ntuple(_ -> random_value(rng, A), short)))
        end
    end

    @testset "identity" begin
        v = NVector{UInt8,3}(0x01, 0x02, 0x03)
        @test reinterpret(UInt8, v) === v
    end

    @testset "zero-dimensional" begin
        z = NArray{UInt16,0,Tuple{}}((0x0102,))
        @test reinterpret(Int16, z) === NArray{Int16,0,Tuple{}}((Int16(0x0102),))
        @test_throws ArgumentError reinterpret(UInt8, z)
    end

    @testset "NArray as an element" begin
        raw_bits, from_raw_bits = BitPacking.raw_bits, BitPacking.from_raw_bits
        v = NVector{Float6_E3M2FN,2}(float6(0x01), float6(0x3f))
        @test @inferred(raw_bits(v)) === 0x0fc1
        @test @inferred(from_raw_bits(NVector{Float6_E3M2FN,2}, 0x0fc1)) === v
        @test from_raw_bits(typeof(v), 0x0fc1) === v
        # bits above the width are ignored, per the element protocol
        @test from_raw_bits(NVector{Float6_E3M2FN,2}, 0xffffffc1) === v
        @test_throws ArgumentError from_raw_bits(NVector{Float6_E3M2FN,2,UInt32}, 0x0fc1)

        # storage as a tuple of limbs is joined into the next unsigned integer
        v3 = NVector{UInt8,3}(0x01, 0x02, 0x03)
        @test v3.data === (0x01, 0x02, 0x03)
        @test raw_bits(v3) === 0x00030201
        @test from_raw_bits(NVector{UInt8,3}, 0xff030201) === v3
        v12 = NVector{UInt32,3}(0x01020304, 0x05060708, 0x090a0b0c)
        @test v12.data isa NTuple{3,UInt32}
        @test raw_bits(v12) === 0x00000000_090a0b0c_05060708_01020304
        @test from_raw_bits(NVector{UInt32,3}, raw_bits(v12)) === v12
        @test_throws ArgumentError raw_bits(NVector(ntuple(Float32, 8)))

        # vectors of vectors pack, and the bits are the same as flat
        f4x2 = NVector(float4(0x1), float4(0x2))
        nested = NVector(f4x2, f4x2, f4x2, f4x2)
        @test nested isa NVector{typeof(f4x2),4,UInt32}
        @test raw_bits(nested) === 0x21212121
        @test nested[3] === f4x2
        @test reinterpret(Float4_E2M1FN, nested) === NVector(ntuple(i -> float4(isodd(i) ? 0x1 : 0x2), 8))
        @test reinterpret(NVector{Float4_E2M1FN,2}, NVector{UInt8,2}(0x21, 0x43)) ===
              NVector(NVector(float4(0x1), float4(0x2)), NVector(float4(0x3), float4(0x4)))
        b6 = NVector{Bool,6}(1, 0, 1, 1, 0, 1)
        @test Tuple(NVector(b6, b6, b6, b6)) === (b6, b6, b6, b6)
        narrow = NarrowArray{typeof(b6)}(fill(b6, 8))
        @test copy(narrow) == fill(b6, 8)
        @test collect(reinterpret(UInt8, narrow)) == repeat([0x6d, 0xdb, 0xb6], 2)

        # an element type must be concrete
        u = NVector{Bool,8}(ntuple(_ -> true, 8))
        @test_throws ArgumentError NVector{NVector{Bool,8},2}((u, u))
        @test NVector{NVector{Bool,8,UInt8},2}((u, u)).data === 0xffff
        @test_throws ArgumentError reinterpret(String, v)
        @test_throws ArgumentError reinterpret(String, NarrowArray{Bool}(trues(8)))
    end

    @testset "odd sizes" begin
        raw_bits, from_raw_bits = BitPacking.raw_bits, BitPacking.from_raw_bits
        # before Julia 1.14 these have a `sizeof` of 3, 5, 6 and 7 bytes, and an
        # array stride of 4 or 8
        for (T, W) in ((Odd24, 24), (Odd40, 40), (Odd48, 48), (Odd56, 56))
            x = Core.Intrinsics.trunc_int(T, 0x0123456789abcdef)
            @test BitPacking.bitwidth(T) == W
            @test raw_bits(x) === 0x0123456789abcdef & BitPacking.low_mask(UInt64, W) % BitPacking.raw_type(T)
            @test from_raw_bits(T, raw_bits(x)) === x
            v = NVector(x, x, x)
            # elementwise: on Julia 1.10, `===` of tuples of these compares padding
            @test all(Tuple(v) .=== (x, x, x))
            xs = fill(x, 4)
            narrow = NarrowArray{T}(xs)
            @test copy(narrow) == xs
            @test length(reinterpret(UInt8, narrow)) == 4W ÷ 8
        end
        rgb = RGB(0x01, 0x02, 0x03)
        @test raw_bits(rgb) === 0x00030201
        @test from_raw_bits(RGB, 0xff030201) === rgb
        @test Tuple(NVector(rgb, rgb)) === (rgb, rgb)
        @test NVector(rgb, rgb).data === (0x0201, 0x0103, 0x0302)
    end

    @testset "constructors take values" begin
        @test NVector{UInt8,1}(0x05).data === 0x05
        @test NVector{UInt16,1}(0x05).data === 0x0005
        @test_throws ArgumentError NVector{Bool,8}(0x55)
        @test_throws InexactError NVector{Bool,1}(0x55)
        @test_throws ArgumentError NArray{UInt8,1,Tuple{2},UInt32}((0x01, 0x02))
    end
end
