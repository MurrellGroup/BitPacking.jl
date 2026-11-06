# Element widths from 1 to 128 bits, as primitive types declared with whole
# bytes and a `bitwidth` overload, so that they work on every Julia version.
const WIDTHS = (1:24..., 31, 32, 33, 48, 63, 64, 65, 100, 127, 128)

for W in WIDTHS
    name = Symbol(:Bits, W)
    @eval primitive type $name $(8 * nextpow(2, cld(W, 8))) end
    @eval BitPacking.bitwidth(::Type{$name}) = $W
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

    @testset "storage word" begin
        v = NVector{Float6_E3M2FN,2}(float6(0x01), float6(0x3f))
        word = @inferred reinterpret(Unsigned, v)
        @test word === 0x0fc1
        @test @inferred(reinterpret(NVector{Float6_E3M2FN,2}, word)) === v
        @test reinterpret(typeof(v), word) === v

        # padding bits must be zero, and the word must be the storage type
        @test_throws ArgumentError reinterpret(NVector{Float6_E3M2FN,2}, 0x1fc1)
        @test_throws ArgumentError reinterpret(NVector{Float6_E3M2FN,2}, 0x00000fc1)
        @test_throws ArgumentError reinterpret(NVector{Float6_E3M2FN,2,UInt32}, 0x0fc1)

        # storage that is not a single unsigned integer has no word
        v3 = NVector{UInt8,3}(0x01, 0x02, 0x03)
        @test v3.data === (0x01, 0x02, 0x03)
        @test_throws ArgumentError reinterpret(Unsigned, v3)
    end

    @testset "constructors take values" begin
        @test NVector{UInt8,1}(0x05).data === 0x05
        @test NVector{UInt16,1}(0x05).data === 0x0005
        @test_throws ArgumentError NVector{Bool,8}(0x55)
        @test_throws InexactError NVector{Bool,1}(0x55)
        @test_throws ArgumentError NArray{UInt8,1,Tuple{2},UInt32}((0x01, 0x02))
    end
end
