# SPDX-License-Identifier: MPL-2.0
# Dependency-light smoke test for the checked Julia -> Zig pointwise boundary.

module CheckedZigFFISmoke

using Libdl
using Test

struct ZigBackend end
relu(x) = max.(zero(eltype(x)), x)

include(joinpath(@__DIR__, "..", "..", "src", "backends", "zig_ffi.jl"))

lib_path = get(ENV, "AXIOM_ZIG_LIB", "")
isempty(lib_path) && error("AXIOM_ZIG_LIB is required")
init_zig_backend(lib_path)

@testset "checked Julia-Zig ReLU boundary" begin
    output = backend_relu(ZigBackend(), Float32[-2, -0.0, 2.5, 9])
    @test reinterpret(UInt32, output) == UInt32[0, 0, 0x40200000, 0x41100000]
    @test_throws ErrorException backend_relu(ZigBackend(), Float32[1, NaN32, 3])
end

@testset "checked Julia-Zig matmul boundary" begin
    left = Float32[1 2 3; 4 5 6]
    right = Float32[7 8; 9 10; 11 12]
    @test backend_matmul(ZigBackend(), left, right) == Float32[58 64; 139 154]
    @test_throws ErrorException backend_matmul(
        ZigBackend(),
        reshape(Float32[floatmax(Float32), floatmax(Float32)], 1, 2),
        reshape(Float32[2, 2], 2, 1),
    )
end

end # module CheckedZigFFISmoke
