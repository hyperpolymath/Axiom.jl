# SPDX-License-Identifier: MPL-2.0
# Measure the finiteness pre-pass in checked matmul against the SIMD kernel.
include("probe_common.jl")
using LinearAlgebra

function matmul_unchecked(a, b, c, n)
  ccall(Libdl.dlsym(lib, :axiom_matmul), Cvoid,
        (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Csize_t, Csize_t), a, b, c, n, n, n)
end

function matmul_checked(a, b, c, n)
  status = ccall(Libdl.dlsym(lib, :axiom_matmul_checked), UInt32,
                (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t, Csize_t, Csize_t), a, b, c, n, n, n)
  status == 0 || error("axiom_matmul_checked returned status $status")
end

function bench(fn, reps)
  fn() # Warm up compilation before timing Julia closures or BLAS.
  best = Inf
  for _ in 1:reps
    start = time_ns()
    fn()
    best = min(best, (time_ns() - start) / 1e6)
  end
  return best
end

rng = MersenneTwister(1)
println("  m=k=n | unchecked ms | checked ms | ratio | Julia(BLAS) ms")
for n in (64, 128, 256, 512)
  a, b = Float32.(randn(rng, n, n)), Float32.(randn(rng, n, n))
  # Explicit row-major copies for the C ABI; convert output back before comparing.
  ar, br, cr = vec(permutedims(a)), vec(permutedims(b)), zeros(Float32, n * n)
  reps = n >= 512 ? 5 : 20
  tu = bench(() -> matmul_unchecked(ar, br, cr, n), reps)
  tc = bench(() -> matmul_checked(ar, br, cr, n), reps)
  tn = bench(() -> a * b, reps)
  got = permutedims(reshape(cr, n, n))
  ok = all(close_element.(got, a * b; atol=1e-3, rtol=1e-4))
  @printf("%7d | %12.3f | %10.3f | %5.2f | %.3f (result matches BLAS: %s)\n", n, tu, tc, tc / tu, tn, ok)
end
