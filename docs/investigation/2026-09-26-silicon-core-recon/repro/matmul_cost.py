# SPDX-License-Identifier: MPL-2.0
"""Cost of axiom_matmul_checked vs axiom_matmul (the Julia wrapper always uses
the checked one).  The checked variant runs a scalar O(m*n*k) finiteness
pre-pass (matmulCellFinite) before the tiled SIMD kernel."""
import ctypes, sys, time
import numpy as np

lib = ctypes.CDLL(sys.argv[1])
f32p = ctypes.POINTER(ctypes.c_float)
sz = ctypes.c_size_t
lib.axiom_matmul.argtypes = [f32p, f32p, f32p, sz, sz, sz]
lib.axiom_matmul.restype = None
lib.axiom_matmul_checked.argtypes = [f32p, f32p, f32p, sz, sz, sz]
lib.axiom_matmul_checked.restype = ctypes.c_uint32

rng = np.random.default_rng(1)
print(f"{'m=k=n':>7} | {'unchecked ms':>12} | {'checked ms':>10} | ratio | numpy(BLAS) ms")
for n in (64, 128, 256, 512):
    a = np.ascontiguousarray(rng.standard_normal((n, n)).astype(np.float32))
    b = np.ascontiguousarray(rng.standard_normal((n, n)).astype(np.float32))
    c = np.zeros((n, n), np.float32)
    reps = 5 if n >= 512 else 20
    def bench(fn):
        best = 1e9
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t)
        return best * 1e3
    tu = bench(lambda: lib.axiom_matmul(a.ctypes.data_as(f32p), b.ctypes.data_as(f32p), c.ctypes.data_as(f32p), n, n, n))
    tc = bench(lambda: lib.axiom_matmul_checked(a.ctypes.data_as(f32p), b.ctypes.data_as(f32p), c.ctypes.data_as(f32p), n, n, n))
    tn = bench(lambda: a @ b)
    ok = np.allclose(c, a @ b, atol=1e-3, rtol=1e-4)
    print(f"{n:>7} | {tu:12.3f} | {tc:10.3f} | {tc/tu:5.2f} | {tn:.3f}   (result matches BLAS: {ok})")
