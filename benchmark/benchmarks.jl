# SPDX-License-Identifier: PMPL-1.0-or-later
# Axiom.jl Benchmark Suite
#
# Tracks performance across Julia, Rust, and Zig backends.
# Run with: julia --project=benchmark benchmark/benchmarks.jl
#
# Refs: Issue #13 - Benchmarks and regression baselines

using BenchmarkTools
using Axiom
using Statistics

# Benchmark results storage
const SUITE = BenchmarkGroup()

# ============================================================================
# Backend Configuration
# ============================================================================

# Test all available backends
const BACKENDS_TO_TEST = AbstractBackend[]

push!(BACKENDS_TO_TEST, JuliaBackend())

# Check Rust backend
rust_lib = joinpath(@__DIR__, "..", "rust", "target", "release", "libaxiom_core.so")
if isfile(rust_lib) || isfile(replace(rust_lib, ".so" => ".dylib")) || isfile(replace(rust_lib, ".so" => ".dll"))
    push!(BACKENDS_TO_TEST, RustBackend(rust_lib))
end

println("Testing backends: ", [typeof(b) for b in BACKENDS_TO_TEST])

# ============================================================================
# Matrix Operations
# ============================================================================

SUITE["matmul"] = BenchmarkGroup()

for size in [64, 256, 1024]
    SUITE["matmul"][size] = BenchmarkGroup()

    A = randn(Float32, size, size)
    B = randn(Float32, size, size)

    for backend in BACKENDS_TO_TEST
        backend_name = string(typeof(backend))
        SUITE["matmul"][size][backend_name] = @benchmarkable begin
            Axiom.backend_matmul($backend, $A, $B)
        end
    end
end

# ============================================================================
# Activation Functions
# ============================================================================

SUITE["activations"] = BenchmarkGroup()

sizes = [1000, 10000, 100000]

for size in sizes
    SUITE["activations"][size] = BenchmarkGroup()
    x = randn(Float32, size)

    for backend in BACKENDS_TO_TEST
        backend_name = string(typeof(backend))

        SUITE["activations"][size]["relu_$backend_name"] = @benchmarkable begin
            Axiom.backend_relu($backend, $x)
        end

        SUITE["activations"][size]["gelu_$backend_name"] = @benchmarkable begin
            Axiom.backend_gelu($backend, $x)
        end

        SUITE["activations"][size]["swish_$backend_name"] = @benchmarkable begin
            Axiom.backend_swish($backend, $x)
        end
    end
end

# ============================================================================
# Convolution
# ============================================================================

SUITE["conv2d"] = BenchmarkGroup()

# Common CNN sizes
conv_configs = [
    (batch=1, channels=3, height=224, width=224, filters=64, kernel=7, stride=2, padding=3),
    (batch=32, channels=64, height=56, width=56, filters=128, kernel=3, stride=1, padding=1),
    (batch=32, channels=128, height=28, width=28, filters=256, kernel=3, stride=1, padding=1),
]

for (i, config) in enumerate(conv_configs)
    SUITE["conv2d"]["config_$i"] = BenchmarkGroup()

    input = randn(Float32, config.batch, config.channels, config.height, config.width)
    weight = randn(Float32, config.filters, config.channels, config.kernel, config.kernel)
    bias = randn(Float32, config.filters)

    for backend in BACKENDS_TO_TEST
        backend_name = string(typeof(backend))

        SUITE["conv2d"]["config_$i"][backend_name] = @benchmarkable begin
            Axiom.backend_conv2d($backend, $input, $weight, $bias, ($(config.stride), $(config.stride)), ($(config.padding), $(config.padding)))
        end
    end
end

# ============================================================================
# Normalization
# ============================================================================

SUITE["normalization"] = BenchmarkGroup()

norm_sizes = [(32, 128), (64, 256), (128, 512)]

for (batch, features) in norm_sizes
    SUITE["normalization"]["batchnorm_$(batch)x$(features)"] = BenchmarkGroup()

    x = randn(Float32, batch, features)
    gamma = ones(Float32, features)
    beta = zeros(Float32, features)
    mean = zeros(Float32, features)
    var = ones(Float32, features)
    eps = Float32(1e-5)

    for backend in BACKENDS_TO_TEST
        backend_name = string(typeof(backend))

        SUITE["normalization"]["batchnorm_$(batch)x$(features)"][backend_name] = @benchmarkable begin
            Axiom.backend_batchnorm($backend, $x, $gamma, $beta, $mean, $var, $eps)
        end
    end
end

# ============================================================================
# End-to-End Model Inference
# ============================================================================

SUITE["models"] = BenchmarkGroup()

# Define models at top level
@axiom SmallCNN begin
    input :: Tensor{Float32, (224, 224, 3, 1)}
    
    l1 = input |> Conv2d(3, 32, 3, stride=1, padding=1) |> relu |> MaxPool2d(2)
    l2 = l1 |> Conv2d(32, 64, 3, stride=1, padding=1) |> relu |> MaxPool2d(2)
    l3 = l2 |> Flatten |> Dense(64 * 56 * 56, 10)
    
    output = l3
end

@axiom ResNetBlock begin
    input :: Tensor{Float32, (56, 56, 64, 1)}
    
    l1 = input |> Conv2d(64, 64, 3, stride=1, padding=1) |> BatchNorm(64) |> relu
    l2 = l1 |> Conv2d(64, 64, 3, stride=1, padding=1) |> BatchNorm(64)
    
    output = l2
end

# Small CNN
function create_small_cnn()
    return SmallCNN()
end

# ResNet block
function create_resnet_block()
    return ResNetBlock()
end

SUITE["models"]["small_cnn"] = BenchmarkGroup()
model = create_small_cnn()
input = randn(Float32, 1, 3, 224, 224)

for backend in BACKENDS_TO_TEST
    backend_name = string(typeof(backend))
    model_compiled = compile(model, backend=backend, verify=false)

    SUITE["models"]["small_cnn"][backend_name] = @benchmarkable begin
        Axiom.forward($model_compiled, $input)
    end
end

# ============================================================================
# Memory Allocation Tracking
# ============================================================================

SUITE["allocations"] = BenchmarkGroup()

x = randn(Float32, 1000, 1000)
y = randn(Float32, 1000, 1000)

for backend in BACKENDS_TO_TEST
    backend_name = string(typeof(backend))

    SUITE["allocations"]["matmul_$backend_name"] = @benchmarkable begin
        Axiom.backend_matmul($backend, $x, $y)
    end samples=100 evals=1
end

# ============================================================================
# Run Benchmarks
# ============================================================================

function run_benchmarks()
    println("\n" * "="^80)
    println("Running Axiom.jl Benchmark Suite")
    println("="^80 * "\n")

    # Tune and run
    tune!(SUITE)
    results = run(SUITE, verbose=true)

    # Save results
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    results_file = joinpath(@__DIR__, "results_$timestamp.json")

    BenchmarkTools.save(results_file, results)
    println("\nResults saved to: $results_file")

    # Generate summary
    println("\n" * "="^80)
    println("Benchmark Summary")
    println("="^80 * "\n")

    generate_summary(results)

    return results
end

function generate_summary(results)
    # Compare Julia vs Rust for each operation
    println("Backend Performance Comparison (relative to Julia):\n")

    for (category, group) in results
        if group isa BenchmarkGroup
            for (test_name, test_group) in group
                if test_group isa BenchmarkGroup
                    julia_time = nothing
                    rust_time = nothing

                    for (backend_name, result) in test_group
                        if occursin("JuliaBackend", string(backend_name))
                            julia_time = median(result).time
                        elseif occursin("RustBackend", string(backend_name))
                            rust_time = median(result).time
                        end
                    end

                    if julia_time !== nothing && rust_time !== nothing
                        speedup = julia_time / rust_time
                        println("  $category/$test_name: Rust is $(round(speedup, digits=2))x $(speedup > 1 ? "faster" : "slower")")
                    end
                end
            end
        end
    end

    println()
end

# ============================================================================
# Regression Detection
# ============================================================================

function check_regression(current_results, baseline_file)
    if !isfile(baseline_file)
        @warn "No baseline found at $baseline_file, skipping regression check"
        return true
    end

    baseline_results = BenchmarkTools.load(baseline_file)[1]

    println("\n" * "="^80)
    println("Regression Detection")
    println("="^80 * "\n")

    has_regression = false
    threshold = 1.1  # 10% regression threshold

    for (category, group) in current_results
        if group isa BenchmarkGroup && haskey(baseline_results, category)
            baseline_group = baseline_results[category]

            for (test_name, test_group) in group
                if test_group isa BenchmarkGroup && haskey(baseline_group, test_name)
                    baseline_test = baseline_group[test_name]

                    for (backend_name, result) in test_group
                        if haskey(baseline_test, backend_name)
                            current_time = median(result).time
                            baseline_time = median(baseline_test[backend_name]).time
                            ratio = current_time / baseline_time

                            if ratio > threshold
                                has_regression = true
                                println("⚠️  REGRESSION: $category/$test_name/$backend_name")
                                println("   Current: $(round(current_time / 1e6, digits=2))ms")
                                println("   Baseline: $(round(baseline_time / 1e6, digits=2))ms")
                                println("   Ratio: $(round(ratio, digits=2))x slower")
                                println()
                            end
                        end
                    end
                end
            end
        end
    end

    if !has_regression
        println("✓ No regressions detected\n")
    end

    return !has_regression
end

# ============================================================================
# Main Entry Point
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    results = run_benchmarks()

    # Check for regressions against baseline
    baseline_file = joinpath(@__DIR__, "baseline.json")
    regression_free = check_regression(results, baseline_file)

    if !regression_free
        println("\n⚠️  Performance regressions detected!")
        exit(1)
    end

    println("\n✓ Benchmark suite completed successfully")
end
