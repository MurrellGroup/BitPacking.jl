# The cuTile extension tests are opt-in, since cuTile pulls in a CUDA toolchain:
#
#     BITPACKING_TEST_CUTILE=true julia --project=test test/runtests.jl
#
# It is installed into a temporary environment rather than being a dependency of
# this project, so the default `Pkg.test()` run is unchanged, and Julia 1.10 —
# below cuTile's own floor of 1.11 — resolves as it always did. The environment
# has to exist before BitPacking is loaded: on 1.11 an extension is only
# triggered for a package loaded from an environment that already knows about
# the trigger.
const CUTILE_REQUESTED = get(ENV, "BITPACKING_TEST_CUTILE", "false") == "true"
const TEST_CUTILE = CUTILE_REQUESTED && VERSION >= v"1.11"

if CUTILE_REQUESTED && !TEST_CUTILE
    @info "skipping the cuTile extension tests: cuTile requires Julia 1.11 or later, running $VERSION"
end

if TEST_CUTILE
    using Pkg
    Pkg.activate(; temp=true)
    Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
    # cuTile is pinned exactly: the codegen tests match emitted Tile IR, which
    # is not something upstream keeps stable across releases. Widen to "1" to
    # track the `[compat]` bound for the weak dependency instead.
    # CUDACore is a dependency of cuTile, so asking for it explicitly (to reach
    # `functional`, `CuArray` and `@cuda` in the device tests) costs nothing.
    Pkg.add([PackageSpec(name="cuTile", version="1.0.0"), PackageSpec(name="Microfloats"),
             PackageSpec(name="CUDACore")])
end

using BitPacking
using BitPacking: SVector
using Random
using Test

primitive type Float4_E2M1FN 8 end
primitive type Float6_E3M2FN 8 end

# Stand-ins for narrow floats, with no arithmetic and no conversions: tests
# build them from bit patterns with `float4` and `float6`.
BitPacking.bitwidth(::Type{Float4_E2M1FN}) = 4
BitPacking.bitwidth(::Type{Float6_E3M2FN}) = 6

float4(x::UInt8) = reinterpret(Float4_E2M1FN, x)
float6(x::UInt8) = reinterpret(Float6_E3M2FN, x)
bits(x::Float4_E2M1FN) = reinterpret(UInt8, x)
bits(x::Float6_E3M2FN) = reinterpret(UInt8, x)
bits(x) = x

@testset "BitPacking.jl" begin
    @testset "raw bits" begin
        # bits above a narrow type's width read as zero, and are ignored on the way in
        @test BitPacking.raw_bits(float4(0xfa)) === 0x0a
        @test BitPacking.raw_bits(BitPacking.from_raw_bits(Float4_E2M1FN, 0xfa)) === 0x0a
        @test bits(BitPacking.from_raw_bits(Float4_E2M1FN, 0xfa)) === 0x0a
        @test BitPacking.raw_bits(true) === 0x01
        @test BitPacking.from_raw_bits(Bool, 0x03) === true
        @test BitPacking.from_raw_bits(Bool, 0x02) === false

        # the input may be wider or narrower than the type
        @test BitPacking.from_raw_bits(UInt8, 0x0102) === 0x02
        @test BitPacking.from_raw_bits(UInt16, 0x01) === 0x0001
        @test BitPacking.from_raw_bits(Int16, 0xffff) === Int16(-1)
        @test BitPacking.raw_bits(Int8(-1)) === 0xff
    end

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
        @test BitPacking.raw_bits(v) === 0x55
        @test reinterpret(UInt8, v) === NVector{UInt8,1}(0x55)
        @test BitPacking.from_raw_bits(NVector{Bool,8}, 0x55) === v

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
        @test BitPacking.raw_bits(f4) === 0x3a

        bools9 = (true, false, true, false, true, false, true, false, true)
        v9 = NVector(bools9)
        @test v9 isa NVector{Bool,9}
        @test v9.data === 0x0155
        @test Tuple(v9) == bools9
        @test v9[1] === true
        @test v9[9] === true
        @test BitPacking.raw_bits(v9) === 0x0155
        @test BitPacking.from_raw_bits(NVector{Bool,9}, 0x0155) === v9
        @test_throws ArgumentError reinterpret(UInt16, v9)

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
        @test BitPacking.raw_bits(n) === 0x0201
        @test reinterpret(UInt16, n) === NVector{UInt16,1}(0x0201)

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

        f4_lanes = (float4(0x0a), float4(0x03))
        f4_from_values = NVector{Float4_E2M1FN,2}(BitPacking.SVector(f4_lanes))
        @test f4_from_values isa NVector{Float4_E2M1FN,2}
        @test f4_from_values.data == 0x3a
        @test bits.(Tuple(f4_from_values)) == (0x0a, 0x03)
        @test NVector{Float4_E2M1FN,2}(f4_lanes) == f4_from_values
        @test NVector{Float4_E2M1FN,2}(f4_lanes...) == f4_from_values
        @test BitPacking.from_raw_bits(NVector{Float4_E2M1FN,2}, 0x3a) == f4_from_values
        # an element type without a conversion from integers takes no integers
        @test_throws MethodError NVector{Float4_E2M1FN,2}(0x0a, 0x03)

        @test BitPacking.StaticArrays.similar_type(typeof(v), Int, BitPacking.StaticArrays.Size(2)) === NVector{Int,2}
        @test_throws ArgumentError NVector{Bool,4}((true, false))
        @test_throws ArgumentError NVector{Bool,9}(0x01)
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
        @test int_values isa Vector{Int}
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
        @test BitPacking.raw_bits.(f4_narrow) == UInt8[0x01, 0x02, 0x03, 0x04]
        @test NarrowArray{Float4_E2M1FN}(float4.(UInt8[0x01, 0x02, 0x03, 0x04])) == f4_narrow
        # across element types, the constructor converts values
        bool_narrow = NarrowArray{Bool}(Bool[1, 0, 1, 1, 0, 0, 1, 0])
        @test NarrowArray{Bool}([1, 0, 1, 1, 0, 0, 1, 0]) == bool_narrow
        @test copy(NarrowArray{UInt8}(bool_narrow)) == UInt8[1, 0, 1, 1, 0, 0, 1, 0]

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
        f4_dense = Float4_E2M1FN.(f4_matrix)
        @test f4_dense isa Matrix{Float4_E2M1FN}
        @test bits.(f4_dense) == UInt8[0x01 0x03; 0x02 0x04]
        bool_matrix = NarrowArray{Bool}(Bool[1 0; 0 1; 1 1; 0 0; 1 0; 0 1; 1 1; 0 0])
        uint_matrix = UInt8.(bool_matrix)
        @test uint_matrix isa Matrix{UInt8}
        @test uint_matrix == UInt8[1 0; 0 1; 1 1; 0 0; 1 0; 0 1; 1 1; 0 0]

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
        # a type that fills its bytes is always a plain view, even of itself
        @test !(reinterpret(UInt8, NarrowArray{UInt8}(UInt8[1, 2])) isa NarrowArray)
        @test reinterpret(Bool, narrow) === narrow
    end

    @testset "NarrowArray setindex!" begin
        values = Bool[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        narrow = NarrowArray{Bool}(copy(values))
        narrow[2] = true
        values[2] = true
        @test copy(narrow) == values
        narrow[16] = 1
        values[16] = true
        @test copy(narrow) == values
        @test_throws BoundsError narrow[17] = true
        @test_throws BoundsError narrow[17]
        @test sort(narrow) == sort(values)
        @test reverse(narrow) == reverse(values)
        dest = similar(narrow)
        copyto!(dest, narrow)
        @test copy(dest) == values
        fill!(dest, false)
        @test copy(dest) == falses(16)

        f6 = NarrowArray{Float6_E3M2FN}(reshape(float6.(UInt8.(1:8)), 4, 2))
        f6[3, 2] = float6(0x3f)
        @test bits.(copy(f6)) == [0x01 0x05; 0x02 0x06; 0x03 0x3f; 0x04 0x08]
        @test collect(reinterpret(UInt8, f6))[:, 1] == UInt8[0x81, 0x30, 0x10]
    end

    @testset "Narrow" begin
        values = Bool[1, 0, 1, 0, 1, 0, 1, 0]

        # Narrow(T).(arr) packs into a NarrowArray{T}
        packed = Narrow(Bool).(values)
        @test packed isa NarrowVector{Bool}
        @test copy(packed) == values
        @test packed == NarrowArray{Bool}(values)

        # the inner expression fuses, narrowing happens at the boundary
        fused = Narrow(Bool).(.!values)
        @test fused isa NarrowVector{Bool}
        @test copy(fused) == .!values

        # reinterpret(Narrow(T), bytes) views packed bytes without copying
        bytes = UInt8[0x55]
        viewed = reinterpret(Narrow(Bool), bytes)
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
        f4_packed = Narrow(Float4_E2M1FN).(f4_values)
        @test f4_packed isa NarrowVector{Float4_E2M1FN}
        @test collect(reinterpret(UInt8, f4_packed)) == UInt8[0x21, 0x43]
        @test bits.(copy(reinterpret(Narrow(Float4_E2M1FN), UInt8[0x21, 0x43]))) == bits.(f4_values)

        # cross-type packing converts values before packing, matching the constructor
        @test Narrow(Bool).([1, 0, 1, 0, 1, 0, 1, 0]) == Narrow(Bool).(Bool[1, 0, 1, 0, 1, 0, 1, 0])

        # matrix destination chunks along the first dimension
        src = repeat(values, 1, 2)
        mat = similar(NarrowArray{Bool}(src))
        mat .= src
        @test copy(mat) == src

        # cross-type destination broadcast converts before packing
        bool_dest = similar(NarrowArray{Bool}(falses(8)))
        bool_dest .= [1, 0, 1, 0, 1, 0, 1, 0]
        @test collect(reinterpret(UInt8, bool_dest)) == UInt8[0x55]

        # Narrow(T).(narr) dispatches on a NarrowArray source
        @test Narrow(Bool).(packed) == packed

        # similar(arr, Narrow(T), dims) allocates narrow storage of logical size
        sim = similar(values, Narrow(Bool), (16,))
        @test sim isa NarrowVector{Bool}
        @test size(sim) == (16,)
        @test parent(sim) isa Vector
        @test similar(values, Narrow(Bool), 16) isa NarrowVector{Bool}
        @test size(similar(values, Narrow(Bool))) == size(values)
        @test size(similar(packed, Narrow(Float4_E2M1FN), (2, 3))) == (2, 3)
        @test_throws ArgumentError similar(values, Narrow(Bool), (4,))
        @test_throws ArgumentError similar(values, Narrow(Bool), ())

        # eltype-less similar stays narrow; dims are logical
        @test similar(packed) isa NarrowVector{Bool}
        @test size(similar(packed)) == size(packed)
        @test size(similar(packed, (16,))) == (16,)
        @test_throws ArgumentError similar(packed, (4,))

        # similar with a plain eltype is dense
        @test similar(packed, UInt8, (3,)) isa Vector{UInt8}

        # print_array handles arrays beyond vectors and matrices
        arr3 = NarrowArray{Bool}(reshape(repeat(values, 4), 8, 2, 2))
        @test contains(sprint(show, MIME("text/plain"), arr3), "NarrowArray")
    end

    include("widths.jl")

    isdefined(Core, :bitsizeof) && include("unused_bits.jl")

    # A variable that a closure captures and that is reassigned is boxed, which
    # costs speed on the CPU and does not compile in GPU code.
    # Generated methods have no lowered source to read.
    @testset "no boxed variables" begin
        has_box(m) = try
            occursin("Core.Box", string(Base.uncompressed_ast(m)))
        catch
            false
        end
        boxed = [m for n in names(BitPacking; all=true) if isdefined(BitPacking, n)
                   for f in (getfield(BitPacking, n),) if f isa Function || f isa Type
                   for m in methods(f) if m.module === BitPacking && has_box(m)]
        @test isempty(boxed)
    end

    TEST_CUTILE && include("cutile.jl")

end
