# SPDX-License-Identifier: PMPL-1.0-or-later
# PyTorch benchmark for Axiom.jl external comparison
# Outputs JSON for consumption by framework_comparison.jl
#
# Usage: pip install torch --index-url https://download.pytorch.org/whl/cpu
#        python3 benchmark/pytorch_bench.py

import torch
import torch.nn.functional as F
import time
import json
import statistics

def bench(fn, warmup=3, iters=50):
    """Benchmark a function, return median time in microseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        torch.manual_seed(42)
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1e6)
    return {
        "median_us": statistics.median(times),
        "min_us": min(times),
        "max_us": max(times),
    }

results = {}

# --- MatMul ---
for n in [64, 256, 512, 1024]:
    A = torch.randn(n, n, dtype=torch.float32)
    B = torch.randn(n, n, dtype=torch.float32)
    r = bench(lambda: torch.mm(A, B))
    results[f"matmul_{n}x{n}"] = r
    print(f"matmul {n}x{n}: {r['median_us']:.1f} us")

# --- ReLU ---
for n in [1000, 100_000, 1_000_000]:
    x = torch.randn(n, dtype=torch.float32)
    label = f"{n//1_000_000}M" if n >= 1_000_000 else f"{n//1000}K"
    r = bench(lambda: F.relu(x))
    results[f"relu_{label}"] = r
    print(f"relu {label}: {r['median_us']:.1f} us")

# --- Sigmoid ---
for n in [1000, 100_000, 1_000_000]:
    x = torch.randn(n, dtype=torch.float32)
    label = f"{n//1_000_000}M" if n >= 1_000_000 else f"{n//1000}K"
    r = bench(lambda: torch.sigmoid(x))
    results[f"sigmoid_{label}"] = r
    print(f"sigmoid {label}: {r['median_us']:.1f} us")

# --- GELU ---
for n in [1000, 100_000, 1_000_000]:
    x = torch.randn(n, dtype=torch.float32)
    label = f"{n//1_000_000}M" if n >= 1_000_000 else f"{n//1000}K"
    r = bench(lambda: F.gelu(x))
    results[f"gelu_{label}"] = r
    print(f"gelu {label}: {r['median_us']:.1f} us")

# --- Softmax ---
for batch, classes in [(32, 10), (64, 1000), (128, 50257)]:
    x = torch.randn(batch, classes, dtype=torch.float32)
    r = bench(lambda: F.softmax(x, dim=1))
    results[f"softmax_{batch}x{classes}"] = r
    print(f"softmax {batch}x{classes}: {r['median_us']:.1f} us")

# --- LayerNorm ---
for batch, hidden in [(32, 128), (64, 768), (128, 1024)]:
    x = torch.randn(batch, hidden, dtype=torch.float32)
    ln = torch.nn.LayerNorm(hidden, dtype=torch.float32)
    ln.eval()
    r = bench(lambda: ln(x))
    results[f"layernorm_{batch}x{hidden}"] = r
    print(f"layernorm {batch}x{hidden}: {r['median_us']:.1f} us")

# --- RMSNorm ---
try:
    rms_cls = torch.nn.RMSNorm
    has_rmsnorm = True
except AttributeError:
    has_rmsnorm = False

for batch, hidden in [(32, 128), (64, 768), (128, 1024)]:
    x = torch.randn(batch, hidden, dtype=torch.float32)
    if has_rmsnorm:
        rn = torch.nn.RMSNorm(hidden, dtype=torch.float32)
        rn.eval()
        r = bench(lambda: rn(x))
    else:
        weight = torch.ones(hidden, dtype=torch.float32)
        eps = 1e-6
        def rmsnorm_manual():
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
            return (x / rms) * weight
        r = bench(rmsnorm_manual)
    results[f"rmsnorm_{batch}x{hidden}"] = r
    print(f"rmsnorm {batch}x{hidden}: {r['median_us']:.1f} us (native={has_rmsnorm})")

# --- BatchNorm ---
for batch, features in [(32, 64), (64, 256), (128, 512)]:
    x = torch.randn(batch, features, dtype=torch.float32)
    bn = torch.nn.BatchNorm1d(features, dtype=torch.float32)
    bn.eval()
    r = bench(lambda: bn(x))
    results[f"batchnorm_{batch}x{features}"] = r
    print(f"batchnorm {batch}x{features}: {r['median_us']:.1f} us")

# --- Save results ---
output = {
    "framework": "PyTorch",
    "version": torch.__version__,
    "device": "cpu",
    "method": "50 iterations, 3 warmup, median timing",
    "results": results,
}

with open("/tmp/pytorch-bench-results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nPyTorch {torch.__version__} benchmark complete. Results saved to /tmp/pytorch-bench-results.json")
