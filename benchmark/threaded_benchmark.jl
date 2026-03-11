# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Threaded Benchmark Suite
#
# Measures Zig multi-threaded dispatch vs Julia across all operations.
# Tests small arrays (single-thread) and large arrays (multi-thread path).
#
# Run: julia --project=. benchmark/threaded_benchmark.jl

using Axiom
using Statistics
using Printf

const ZIG_LIB = joinpath(@__DIR__, "..", "zig", "zig-out", "lib", "libaxiom_zig.so")
const HAS_ZIG = isfile(ZIG_LIB)

if !HAS_ZIG
    error("Zig backend not found at $ZIG_LIB — build with: cd zig && zig build -Doptimize=ReleaseFast")
end

const julia_be = JuliaBackend()
const zig_be = ZigBackend(ZIG_LIB)
global smart_be = nothing
try
    global smart_be = SmartBackend(zig_path=ZIG_LIB)
catch e
    @warn "SmartBackend creation failed: $e"
end

# ============================================================================
# Timing utility
# ============================================================================

function bench(f, warmup=5, iters=50)
    for _ in 1:warmup; f(); end
    GC.gc()
    times = Float64[]
    for _ in 1:iters
        t = @elapsed f()
        push!(times, t)
    end
    median(times) * 1e6  # return μs
end

# ============================================================================
# Results table
# ============================================================================

struct BenchResult
    op::String
    size::String
    julia_us::Union{Float64, Nothing}
    zig_us::Union{Float64, Nothing}
    smart_us::Union{Float64, Nothing}
end

results = BenchResult[]

function add_result(op, size_str, julia_us, zig_us, smart_us=nothing)
    push!(results, BenchResult(op, size_str, julia_us, zig_us, smart_us))
end

# ============================================================================
# Element-wise activations (both backends)
# ============================================================================

println("=" ^ 70)
println("Axiom.jl Threaded Benchmark Suite")
println("=" ^ 70)
println()

# Ops available on BOTH JuliaBackend and ZigBackend
both_ops = [
    ("relu",    (be, x) -> Axiom.backend_relu(be, x)),
    ("sigmoid", (be, x) -> Axiom.backend_sigmoid(be, x)),
    ("gelu",    (be, x) -> Axiom.backend_gelu(be, x)),
    ("tanh",    (be, x) -> Axiom.backend_tanh(be, x)),
]

println("--- Element-wise activations (Julia + Zig) ---")
for (name, op_fn) in both_ops
    for n in [1_000, 100_000, 1_000_000]
        x = randn(Float32, n)
        size_str = n >= 1_000_000 ? "$(n ÷ 1_000_000)M" : "$(n ÷ 1_000)K"

        j_us = bench(() -> op_fn(julia_be, x))
        z_us = bench(() -> op_fn(zig_be, x))
        s_us = smart_be !== nothing ? bench(() -> op_fn(smart_be, x)) : nothing

        add_result(name, size_str, j_us, z_us, s_us)
    end
end

# Extended activations (now available on both backends)
extended_ops = [
    ("swish",       (be, x) -> Axiom.backend_swish(be, x)),
    ("selu",        (be, x) -> Axiom.backend_selu(be, x)),
    ("mish",        (be, x) -> Axiom.backend_mish(be, x)),
    ("hardswish",   (be, x) -> Axiom.backend_hardswish(be, x)),
    ("hardsigmoid", (be, x) -> Axiom.backend_hardsigmoid(be, x)),
    ("softplus",    (be, x) -> Axiom.backend_softplus(be, x)),
]

println("--- Extended activations (Julia + Zig) ---")
for (name, op_fn) in extended_ops
    for n in [1_000, 100_000, 1_000_000]
        x = randn(Float32, n)
        size_str = n >= 1_000_000 ? "$(n ÷ 1_000_000)M" : "$(n ÷ 1_000)K"

        j_us = bench(() -> op_fn(julia_be, x))
        z_us = bench(() -> op_fn(zig_be, x))
        s_us = smart_be !== nothing ? bench(() -> op_fn(smart_be, x)) : nothing

        add_result(name, size_str, j_us, z_us, s_us)
    end
end

# ============================================================================
# Softmax (batched, batch-threaded via Zig)
# ============================================================================

println("--- Softmax (batched) ---")
for (batch, classes) in [(1, 1000), (32, 1000), (128, 50257)]
    x = randn(Float32, batch, classes)
    size_str = "$(batch)×$(classes)"

    j_us = bench(() -> Axiom.backend_softmax(julia_be, x, 2))
    z_us = bench(() -> Axiom.backend_softmax(zig_be, x, 2))

    add_result("softmax", size_str, j_us, z_us)
end

# ============================================================================
# LayerNorm / RMSNorm (Zig vs Julia, via backend dispatch)
# ============================================================================

println("--- LayerNorm ---")
for (batch, hidden) in [(32, 768), (128, 768), (32, 4096)]
    x = randn(Float32, batch, hidden)
    gamma = ones(Float32, hidden)
    beta = zeros(Float32, hidden)
    size_str = "$(batch)×$(hidden)"

    j_us = bench(() -> Axiom.backend_layernorm(julia_be, x, gamma, beta, (hidden,), Float32(1e-5)))
    z_us = bench(() -> Axiom.backend_layernorm(zig_be, x, gamma, beta, Float32(1e-5)))

    add_result("layernorm", size_str, j_us, z_us)
end

println("--- RMSNorm ---")
for (batch, hidden) in [(32, 768), (128, 768), (32, 4096)]
    x = randn(Float32, batch, hidden)
    weight = ones(Float32, hidden)
    size_str = "$(batch)×$(hidden)"

    j_us = bench(() -> Axiom.backend_rmsnorm(julia_be, x, weight, Float32(1e-5)))
    z_us = bench(() -> Axiom.backend_rmsnorm(zig_be, x, weight, Float32(1e-5)))

    add_result("rmsnorm", size_str, j_us, z_us)
end

# ============================================================================
# BatchNorm
# ============================================================================

println("--- BatchNorm ---")
for (batch, features) in [(32, 128), (64, 256), (128, 512)]
    x = randn(Float32, batch, features)
    gamma = ones(Float32, features)
    beta_bn = zeros(Float32, features)
    running_mean = zeros(Float32, features)
    running_var = ones(Float32, features)
    size_str = "$(batch)×$(features)"

    j_us = bench(() -> Axiom.backend_batchnorm(julia_be, x, gamma, beta_bn, running_mean, running_var, Float32(1e-5), false))
    z_us = bench(() -> Axiom.backend_batchnorm(zig_be, x, gamma, beta_bn, running_mean, running_var, Float32(1e-5), false))

    add_result("batchnorm", size_str, j_us, z_us)
end

# ============================================================================
# Matrix Multiplication
# ============================================================================

println("--- MatMul ---")
for sz in [64, 256, 512, 1024]
    A = randn(Float32, sz, sz)
    B = randn(Float32, sz, sz)
    size_str = "$(sz)×$(sz)"

    j_us = bench(() -> Axiom.backend_matmul(julia_be, A, B))
    z_us = bench(() -> Axiom.backend_matmul(zig_be, A, B))

    add_result("matmul", size_str, j_us, z_us)
end

# ============================================================================
# Print results
# ============================================================================

println()
println("=" ^ 90)
@printf("%-14s %-12s %12s %12s %12s %10s\n",
        "Operation", "Size", "Julia (μs)", "Zig (μs)", "Smart (μs)", "Zig/Julia")
println("-" ^ 90)

for r in results
    j_str = r.julia_us !== nothing ? @sprintf("%10.1f", r.julia_us) : "    N/A   "
    z_str = r.zig_us !== nothing ? @sprintf("%10.1f", r.zig_us) : "    N/A   "
    s_str = r.smart_us !== nothing ? @sprintf("%10.1f", r.smart_us) : "    N/A   "

    if r.julia_us !== nothing && r.zig_us !== nothing
        ratio = r.zig_us / r.julia_us
        ratio_str = @sprintf("%8.2fx", ratio)
    else
        ratio_str = "    N/A "
    end

    @printf("%-14s %-12s %12s %12s %12s %10s\n",
            r.op, r.size, j_str, z_str, s_str, ratio_str)
end

println("=" ^ 90)
println()
println("Notes:")
println("  - Zig/Julia < 1.0 means Zig is faster")
println("  - Thread threshold: 64K elements (4 threads above)")
println("  - SIMD: 8-wide f32 vectors for GELU/sigmoid/tanh")
println("  - N/A = operation not available on that backend")
