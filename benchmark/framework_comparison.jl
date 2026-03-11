# SPDX-License-Identifier: PMPL-1.0-or-later
# External framework benchmark: Axiom.jl vs Flux.jl vs PyTorch
# Usage:
#   1. pip install torch --index-url https://download.pytorch.org/whl/cpu
#   2. python3 benchmark/pytorch_bench.py
#   3. julia --project=benchmark benchmark/framework_comparison.jl
#
# Requires: PyTorch results at /tmp/pytorch-bench-results.json

using Axiom
using Flux
using Statistics
using Printf
using JSON

# ─── Backend init ─────────────────────────────────────────────────────────────
const AXIOM_DIR = dirname(@__DIR__)

zig_lib = joinpath(AXIOM_DIR, "zig", "zig-out", "lib", "libaxiom_zig.so")

# Init SmartBackend
local smart = nothing
try
    Axiom.init_zig_backend(zig_lib)
    global smart = SmartBackend(zig_path=zig_lib)
    println("✓ Axiom SmartBackend (Zig: $(filesize(zig_lib) ÷ 1024)KB)")
catch e
    println("✗ SmartBackend failed: $e — using JuliaBackend")
end

julia_backend = JuliaBackend()

# ─── Timing helper ────────────────────────────────────────────────────────────
function bench(f, warmup=3, iters=50)
    for _ in 1:warmup; f(); end
    times = Float64[]
    for _ in 1:iters
        t = @elapsed f()
        push!(times, t)
    end
    return median(times) * 1e6  # microseconds
end

# ─── Results storage ──────────────────────────────────────────────────────────
struct FrameworkResult
    op::String
    size::String
    axiom_smart_us::Float64
    axiom_julia_us::Float64
    flux_us::Float64
    pytorch_us::Float64
end

results = FrameworkResult[]

# Load PyTorch results
pytorch_data = Dict{String, Float64}()
pytorch_version = "N/A"
if isfile("/tmp/pytorch-bench-results.json")
    pt_json = JSON.parsefile("/tmp/pytorch-bench-results.json")
    pytorch_version = get(pt_json, "version", "N/A")
    for (key, val) in pt_json["results"]
        pytorch_data[key] = val["median_us"]
    end
    println("✓ PyTorch $pytorch_version results loaded ($(length(pytorch_data)) benchmarks)")
else
    println("✗ No PyTorch results — run benchmark/pytorch_bench.py first")
end

function pt_time(key)
    get(pytorch_data, key, NaN)
end

# ─── Benchmarks ───────────────────────────────────────────────────────────────

println("\n", "="^90)
println("  External Framework Benchmark: Axiom.jl vs Flux.jl vs PyTorch (CPU)")
println("="^90, "\n")

# --- MatMul ---
for n in [64, 256, 512, 1024]
    A = randn(Float32, n, n)
    B = randn(Float32, n, n)

    as = smart !== nothing ? bench(() -> Axiom.backend_matmul(smart, A, B)) :
                             bench(() -> Axiom.backend_matmul(julia_backend, A, B))
    aj = bench(() -> Axiom.backend_matmul(julia_backend, A, B))
    fl = bench(() -> A * B)
    local pt = pt_time("matmul_$(n)x$(n)")

    push!(results, FrameworkResult("matmul", "$(n)×$(n)", as, aj, fl, pt))
end

# --- ReLU ---
for n in [1_000, 100_000, 1_000_000]
    x = randn(Float32, n)
    label = n >= 1_000_000 ? "$(n÷1_000_000)M" : "$(n÷1000)K"

    as = smart !== nothing ? bench(() -> Axiom.backend_relu(smart, x)) :
                             bench(() -> Axiom.backend_relu(julia_backend, x))
    aj = bench(() -> Axiom.backend_relu(julia_backend, x))
    fl = bench(() -> Flux.relu.(x))
    local pt = pt_time("relu_$label")

    push!(results, FrameworkResult("relu", label, as, aj, fl, pt))
end

# --- Sigmoid ---
for n in [1_000, 100_000, 1_000_000]
    x = randn(Float32, n)
    label = n >= 1_000_000 ? "$(n÷1_000_000)M" : "$(n÷1000)K"

    as = smart !== nothing ? bench(() -> Axiom.backend_sigmoid(smart, x)) :
                             bench(() -> Axiom.backend_sigmoid(julia_backend, x))
    aj = bench(() -> Axiom.backend_sigmoid(julia_backend, x))
    fl = bench(() -> Flux.sigmoid.(x))
    local pt = pt_time("sigmoid_$label")

    push!(results, FrameworkResult("sigmoid", label, as, aj, fl, pt))
end

# --- GELU ---
for n in [1_000, 100_000, 1_000_000]
    x = randn(Float32, n)
    label = n >= 1_000_000 ? "$(n÷1_000_000)M" : "$(n÷1000)K"

    as = smart !== nothing ? bench(() -> Axiom.backend_gelu(smart, x)) :
                             bench(() -> Axiom.backend_gelu(julia_backend, x))
    aj = bench(() -> Axiom.backend_gelu(julia_backend, x))
    fl = bench(() -> Flux.gelu.(x))
    local pt = pt_time("gelu_$label")

    push!(results, FrameworkResult("gelu", label, as, aj, fl, pt))
end

# --- Softmax ---
for (batch, classes) in [(32, 10), (64, 1000), (128, 50257)]
    x = randn(Float32, batch, classes)

    as = smart !== nothing ? bench(() -> Axiom.backend_softmax(smart, x, 2)) :
                             bench(() -> Axiom.backend_softmax(julia_backend, x, 2))
    aj = bench(() -> Axiom.backend_softmax(julia_backend, x, 2))
    fl = bench(() -> Flux.softmax(x, dims=2))
    local pt = pt_time("softmax_$(batch)x$(classes)")

    push!(results, FrameworkResult("softmax", "$(batch)×$(classes)", as, aj, fl, pt))
end

# --- LayerNorm ---
for (batch, hidden) in [(32, 128), (64, 768), (128, 1024)]
    x = randn(Float32, batch, hidden)
    gamma = ones(Float32, hidden)
    beta = zeros(Float32, hidden)
    eps = Float32(1e-5)
    nshape = (hidden,)

    if smart !== nothing
        as = bench(() -> Axiom.backend_layernorm(smart, x, gamma, beta, nshape, eps))
    else
        as = bench(() -> Axiom.backend_layernorm(julia_backend, x, gamma, beta, nshape, eps))
    end
    aj = bench(() -> Axiom.backend_layernorm(julia_backend, x, gamma, beta, nshape, eps))

    flux_ln = Flux.LayerNorm(hidden)
    x_flux = permutedims(x, (2, 1))
    fl = bench(() -> flux_ln(x_flux))

    local pt = pt_time("layernorm_$(batch)x$(hidden)")

    push!(results, FrameworkResult("layernorm", "$(batch)×$(hidden)", as, aj, fl, pt))
end

# --- RMSNorm ---
for (batch, hidden) in [(32, 128), (64, 768), (128, 1024)]
    x = randn(Float32, batch, hidden)
    weight = ones(Float32, hidden)
    gamma = ones(Float32, hidden)
    beta = zeros(Float32, hidden)
    eps = Float32(1e-5)
    nshape = (hidden,)

    if smart !== nothing
        as = bench(() -> Axiom.backend_rmsnorm(smart, x, weight, Float32(1e-6)))
    else
        as = bench(() -> Axiom.backend_layernorm(julia_backend, x, gamma, beta, nshape, eps))
    end
    aj = bench(() -> Axiom.backend_layernorm(julia_backend, x, gamma, beta, nshape, eps))

    w = ones(Float32, 1, hidden)
    fl = bench(() -> begin
        rms = sqrt.(mean(x .^ 2, dims=2) .+ 1f-6)
        (x ./ rms) .* w
    end)

    local pt = pt_time("rmsnorm_$(batch)x$(hidden)")

    push!(results, FrameworkResult("rmsnorm", "$(batch)×$(hidden)", as, aj, fl, pt))
end

# --- BatchNorm ---
for (batch, features) in [(32, 64), (64, 256), (128, 512)]
    x = randn(Float32, batch, features)
    gamma = ones(Float32, features)
    beta = zeros(Float32, features)
    rmean = zeros(Float32, features)
    rvar = ones(Float32, features)
    eps = Float32(1e-5)

    as = smart !== nothing ? bench(() -> Axiom.backend_batchnorm(smart, x, gamma, beta, rmean, rvar, eps, false)) :
                             bench(() -> Axiom.backend_batchnorm(julia_backend, x, gamma, beta, rmean, rvar, eps, false))
    aj = bench(() -> Axiom.backend_batchnorm(julia_backend, x, gamma, beta, rmean, rvar, eps, false))

    flux_bn = Flux.BatchNorm(features)
    Flux.testmode!(flux_bn)
    x_flux = permutedims(x, (2, 1))
    fl = bench(() -> flux_bn(x_flux))

    local pt = pt_time("batchnorm_$(batch)x$(features)")

    push!(results, FrameworkResult("batchnorm", "$(batch)×$(features)", as, aj, fl, pt))
end

# ─── Print results table ─────────────────────────────────────────────────────
println()
println("┌─────────────┬──────────────┬───────────────┬───────────────┬───────────────┬───────────────┐")
println("│ Operation    │ Size         │ Axiom Smart   │ Axiom Julia   │ Flux.jl (μs)  │ PyTorch (μs)  │")
println("├─────────────┼──────────────┼───────────────┼───────────────┼───────────────┼───────────────┤")

for r in results
    pt_str = isnan(r.pytorch_us) ? "N/A" : @sprintf("%.1f", r.pytorch_us)

    @printf("│ %-11s │ %-12s │ %13.1f │ %13.1f │ %13.1f │ %13s │\n",
        r.op, r.size, r.axiom_smart_us, r.axiom_julia_us, r.flux_us, pt_str)
end

println("└─────────────┴──────────────┴───────────────┴───────────────┴───────────────┴───────────────┘")
println()

# ─── Speedup table ────────────────────────────────────────────────────────────
println("Speedup vs PyTorch (>1.0x = we're faster):")
println("┌─────────────┬──────────────┬───────────────┬───────────────┬───────────────┐")
println("│ Operation    │ Size         │ Axiom Smart   │ Axiom Julia   │ Flux.jl       │")
println("├─────────────┼──────────────┼───────────────┼───────────────┼───────────────┤")

for r in results
    if !isnan(r.pytorch_us) && r.pytorch_us > 0
        as_sp = r.pytorch_us / r.axiom_smart_us
        aj_sp = r.pytorch_us / r.axiom_julia_us
        fl_sp = r.pytorch_us / r.flux_us

        @printf("│ %-11s │ %-12s │ %11.2fx │ %11.2fx │ %11.2fx │\n",
            r.op, r.size, as_sp, aj_sp, fl_sp)
    end
end

println("└─────────────┴──────────────┴───────────────┴───────────────┴───────────────┘")
println()

# ─── Aggregate ────────────────────────────────────────────────────────────────
valid = filter(r -> !isnan(r.pytorch_us) && r.pytorch_us > 0, results)

if !isempty(valid)
    as_ratios = [r.pytorch_us / r.axiom_smart_us for r in valid]
    aj_ratios = [r.pytorch_us / r.axiom_julia_us for r in valid]
    fl_ratios = [r.pytorch_us / r.flux_us for r in valid]

    println("Aggregate vs PyTorch:")
    @printf("  Axiom Smart:  geomean %.2fx, arithmetic mean %.2fx\n",
        exp(mean(log.(as_ratios))), mean(as_ratios))
    @printf("  Axiom Julia:  geomean %.2fx, arithmetic mean %.2fx\n",
        exp(mean(log.(aj_ratios))), mean(aj_ratios))
    @printf("  Flux.jl:      geomean %.2fx, arithmetic mean %.2fx\n",
        exp(mean(log.(fl_ratios))), mean(fl_ratios))
    println()

    as_wins = count(r -> r.pytorch_us / r.axiom_smart_us > 1.0, valid)
    fl_wins = count(r -> r.pytorch_us / r.flux_us > 1.0, valid)
    println("Axiom Smart wins $(as_wins)/$(length(valid)) benchmarks vs PyTorch")
    println("Flux.jl wins $(fl_wins)/$(length(valid)) benchmarks vs PyTorch")
end
