# Tests for the cuTile extension, run by `test/runtests.jl` when
# `BITPACKING_TEST_CUTILE=true` has set up the environment they need.
#
# No GPU is needed: the kernel-side tests stop at `code_tiled`, which emits Tile
# IR for an explicit `sm_arch` without touching a device. Running the packed
# kernels end to end needs an architecture with FP4 support, and that coverage
# lives downstream in Microscaling.jl, where CUDA is already a dependency.

using BitPacking
using Test

import cuTile as ct
import Microfloats
import CUDACore
using CUDACore: CuArray, @cuda

# Microfloats provides both halves of what these tests need: `bitwidth` for
# BitPacking's packing, and the Tile IR dtype mapping through cuTile's own
# Microfloats extension. Referenced qualified, since the main test suite defines
# its own stand-in types under these names.
const FP4 = Microfloats.Float4_E2M1FN
const FP8 = Microfloats.Float8_E4M3FN

const ext = Base.get_extension(BitPacking, :cuTileExt)
@assert !isnothing(ext)

const ReinterpretTileArray = ext.ReinterpretTileArray

# A `TileArray` of the given element type and size, standing in for what
# `Adapt.adapt(ct.KernelAdaptor(), arr)` produces for a device-backed array.
# Nothing here dereferences the pointer: the tests only exercise the wrapper's
# shape logic and the IR emitted for its type.
function tile_array(::Type{T}, sizes::Dims{N}) where {T,N}
    strides = ntuple(i -> prod(sizes[1:i-1]; init=1), Val(N))
    return ct.TileArray(Ptr{T}(0), Int32.(sizes), Int32.(strides))
end

# The packed form: `sizes` counts stored bytes, and the logical size follows from
# `bitwidth(T)`, as it does for the adapted `NarrowArray`.
narrow_tile_array(::Type{T}, sizes::Dims) where T =
    ReinterpretTileArray{T,length(sizes),typeof(tile_array(UInt8, sizes))}(tile_array(UInt8, sizes))

@testset "cuTileExt" begin

    @testset "ReinterpretTileArray" begin
        arr = narrow_tile_array(FP4, (8,))

        # the `AbstractTileArray` supertype is what lets kernels take these
        # arguments without an `::Any` annotation
        @test arr isa ct.AbstractTileArray{FP4,1}
        @test eltype(arr) === FP4
        @test ndims(arr) == 1
        @test parent(arr) isa ct.TileArray{UInt8,1}

        # two 4-bit values per stored byte
        @test size(arr) == (16,)
        @test size(arr, 1) == 16
        @test size(parent(arr)) == (Int32(8),)

        # only the leading dimension is rescaled
        matrix = narrow_tile_array(FP4, (8, 4))
        @test size(matrix) == (16, 4)
        @test size(matrix, 1) == 16
        @test size(matrix, 2) == 4

        # the ratio follows `bitwidth`: 8 bits per byte, and byte-wide element
        # types are a plain pass-through
        @test size(narrow_tile_array(Bool, (8,))) == (64,)
        @test size(narrow_tile_array(FP8, (8,))) == (8,)

        # `NarrowArray` packs 6-bit values four to three bytes, but a tile view
        # rescales the leading dimension by whole bytes, so it has no shape here
        @test_throws ArgumentError narrow_tile_array(Microfloats.Float6_E3M2FN, (8,))
        @test_throws ArgumentError narrow_tile_array(Float32, (8,))
    end

    @testset "codegen" begin
        # `load` unpacks bytes into logical values and `store` packs them back,
        # so a copy kernel exercises both directions of the extension.
        function copy_kernel(X, Y)
            i = ct.bid(1)
            ct.store(Y, i, ct.load(X, i, (32,)))
            return
        end

        function copy_kernel_2d(X, Y)
            i, j = ct.bid(1), ct.bid(2)
            ct.store(Y, (i, j), ct.load(X, (i, j), (32, 8)))
            return
        end

        # the usual consumer side: unpack a packed tile and widen it
        function widen_kernel(X, Y)
            i = ct.bid(1)
            ct.store(Y, i, convert(ct.Tile{Float32}, ct.load(X, i, (32,))))
            return
        end

        # `pack`/`unpack` and the FP4 tile type need bytecode v13.3; pinning it
        # along with `sm_arch` keeps the emitted IR independent of the host.
        tile_ir(f, argtypes) = sprint() do io
            ct.code_tiled(io, f, argtypes; sm_arch=v"10.0", bytecode_version=v"13.3")
        end

        vector_arg = typeof(narrow_tile_array(FP4, (8,)))
        ir = tile_ir(copy_kernel, Tuple{vector_arg,vector_arg})

        # a 32-element FP4 tile is 16 bytes in memory, so both the partitioning
        # and the transfers happen at half the logical extent
        @test contains(ir, "partition_view<tile=(16), tensor_view<?xi8")
        @test contains(ir, "-> tile<16xi8>")
        @test occursin(r"unpack .*: tile<16xi8> -> tile<32xf4E2M1FN>", ir)
        @test occursin(r"pack .*: tile<32xf4E2M1FN> -> tile<16xi8>", ir)

        matrix_arg = typeof(narrow_tile_array(FP4, (8, 4)))
        ir_2d = tile_ir(copy_kernel_2d, Tuple{matrix_arg,matrix_arg})

        # only the leading dimension is halved (Tile IR prints shapes with the
        # leading Julia dimension last, so a (16, 8) byte tile reads as 8x16)
        @test contains(ir_2d, "partition_view<tile=(8x16), tensor_view<?x?xi8")
        @test contains(ir_2d, "-> tile<8x16xi8>")
        @test occursin(r"unpack .*: tile<128xi8> -> tile<256xf4E2M1FN>", ir_2d)
        @test occursin(r"pack .*: tile<256xf4E2M1FN> -> tile<128xi8>", ir_2d)

        float_arg = typeof(tile_array(Float32, (32,)))
        ir_widen = tile_ir(widen_kernel, Tuple{vector_arg,float_arg})
        @test occursin(r"unpack .*: tile<16xi8> -> tile<32xf4E2M1FN>", ir_widen)
        @test occursin(r"ftof .*tile<32xf4E2M1FN> -> tile<32xf32>", ir_widen)

        # a kernel taking the packed argument compiles the same way when it is
        # annotated with the supertype the extension hooks into
        function annotated_kernel(X::ct.AbstractTileArray, Y::ct.AbstractTileArray)
            i = ct.bid(1)
            ct.store(Y, i, ct.load(X, i, (32,)))
            return
        end
        @test contains(tile_ir(annotated_kernel, Tuple{vector_arg,vector_arg}), "unpack")
    end

    # What the host tests cannot reach: adapting a device-backed `NarrowArray`,
    # and running the packed load/store against real memory.
    @testset "device" begin
        if CUDACore.functional()
            function copy_kernel!(X, Y)
                i = ct.bid(1)
                ct.store(Y, i, ct.load(X, i, (64,)))
                return
            end

            round_trip(src, dst) = Array(reinterpret(UInt8, parent(dst))) ==
                                   Array(reinterpret(UInt8, parent(src)))

            # Adapting a device-backed `NarrowArray` is the piece the host tests
            # cannot cover; it works on any device, packed element type or not.
            src = NarrowArray{FP4}(CuArray(FP4.(Float32.(1:64) ./ 8)))
            adapted = BitPacking.Adapt.adapt(ct.KernelAdaptor(), src)
            @test adapted isa ct.AbstractTileArray{FP4,1}
            @test size(adapted) == size(src) == (64,)
            @test parent(adapted) isa ct.TileArray{UInt8,1}
            @test size(parent(adapted)) == (Int32(32),)

            # A byte-wide element type makes the wrapper a pass-through, and its
            # tile type is one every Tile IR device supports, so the load/store
            # round trip runs here rather than only on Blackwell.
            bytes_src = NarrowArray{UInt8}(CuArray(UInt8.(1:64)))
            bytes_dst = similar(bytes_src)
            @cuda backend=ct blocks=(1,) copy_kernel!(bytes_src, bytes_dst)
            CUDACore.synchronize()
            @test round_trip(bytes_src, bytes_dst)

            # The packed round trip needs a device whose backend supports the FP4
            # tile type, which `tileiras` rejects below Blackwell — as it does
            # every microfloat type, FP8 included.
            capability = CUDACore.capability(CUDACore.device())
            if capability >= v"10.0"
                dst = similar(src)
                @cuda backend=ct blocks=(1,) copy_kernel!(src, dst)
                CUDACore.synchronize()
                @test round_trip(src, dst)
            else
                @info "device is not Blackwell: skipping the packed FP4 kernel" capability
            end
        else
            @info "no CUDA device: skipping the cuTile device tests"
        end
    end

end
