using BitPacking
using Test

using BitPacking: packchunk, unpackchunk, chunk_size, packed_chunk_size

mask(W) = (0x01 << W) - 0x01

@testset "BitPacking.jl" begin

    @testset "chunk" begin

        @testset for W in 1:8
            N = chunk_size(Val(W))
            values = ntuple(i -> rand(UInt8) & mask(W), N)
            packed = packchunk(Val(W), values)
            @test length(packed) == packed_chunk_size(Val(W))
            @test values == unpackchunk(Val(W), packed)
        end

    end

    @testset "packbits" begin

        @testset for W in 1:8
            N = chunk_size(Val(W))
            values = rand(UInt8, N * 2, 3, 5) .& mask(W)
            packed = packbits(W, values)
            @test length(packed) == length(values) * W ÷ 8
            @test unpackbits(W, packed) == values
        end

    end

    @testset "BitPackedArray" begin

        @testset for W in 1:8
            values = rand(UInt8, 32, 10) .& mask(W)
            packed = bitpacked(values, Val(W))
            @test packed == values
            @test bitunpacked(packed) == values
        end

        @testset "BitArray" begin
            values = rand(Bool, 32, 10)
            packed = bitpacked(values, Val(1))
            @test packed isa AbstractMatrix{Bool}
            @test packed == values
            @test bitunpacked(packed) == values
            @test vec(packed.packed_bytes) == reinterpret(UInt8, BitArray(values).chunks)
        end

        @testset "Broadcasting" begin
            values = rand(UInt8, 32, 10) .& 0b111
            packed = bitpacked(values, 3)
            @test packed == values
            @test bitunpacked(packed) == values
            new_values = rand(UInt8, 32) .& 0b111
            packed .= new_values
            @test all(==(new_values), eachcol(packed))
            @test all(==(new_values), eachcol(bitunpacked(packed)))
        end

    end

end
