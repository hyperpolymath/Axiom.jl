# SPDX-License-Identifier: MPL-2.0
"""
Layout probe for the Julia -> Zig pooling wrappers.

src/backends/zig_ffi.jl passes Julia's column-major (N,H,W,C) buffer *directly*
to axiom_maxpool2d / axiom_global_avgpool2d (no _to_row_major_vec), while
zig/src/pool.zig indexes row-major NHWC.  This script reproduces exactly what
the Julia wrapper does (column-major bytes in, column-major interpretation of
the output) and compares with the mathematically correct pooling result.

Also probes: padding silently dropped, -Inf window handling, NaN handling.
"""
import ctypes, sys
import numpy as np

lib = ctypes.CDLL(sys.argv[1])
f32p = ctypes.POINTER(ctypes.c_float)
sz = ctypes.c_size_t

lib.axiom_maxpool2d.argtypes = [f32p, f32p, sz, sz, sz, sz, sz, sz, sz, sz]
lib.axiom_maxpool2d.restype = None
lib.axiom_global_avgpool2d.argtypes = [f32p, f32p, sz, sz, sz, sz]
lib.axiom_global_avgpool2d.restype = None

def ref_maxpool(x, k, s):
    N, H, W, C = x.shape
    Ho, Wo = (H - k) // s + 1, (W - k) // s + 1
    y = np.empty((N, Ho, Wo, C), np.float32)
    for n in range(N):
        for c in range(C):
            for i in range(Ho):
                for j in range(Wo):
                    y[n, i, j, c] = x[n, i*s:i*s+k, j*s:j*s+k, c].max()
    return y

def julia_style_maxpool(x, k, s):
    """Mimic backend_maxpool2d(::ZigBackend): pass column-major bytes as-is,
    read output back as column-major (N,Ho,Wo,C)."""
    N, H, W, C = x.shape
    Ho, Wo = (H - k) // s + 1, (W - k) // s + 1
    xin = np.asfortranarray(x)                       # Julia memory order
    out = np.zeros(N*Ho*Wo*C, np.float32)
    lib.axiom_maxpool2d(xin.ctypes.data_as(f32p), out.ctypes.data_as(f32p),
                        N, H, W, C, k, k, s, s)
    return out.reshape((N, Ho, Wo, C), order="F")   # Julia reads back column-major

def julia_style_gap(x):
    N, H, W, C = x.shape
    xin = np.asfortranarray(x)
    out = np.zeros(N*C, np.float32)
    lib.axiom_global_avgpool2d(xin.ctypes.data_as(f32p), out.ctypes.data_as(f32p), N, H, W, C)
    return out.reshape((N, C), order="F")

rng = np.random.default_rng(0)
print("case                      | max|zig-ref| | mismatched elements")
for (N, H, W, C, k, s) in [(1, 4, 4, 1, 2, 2), (1, 4, 4, 2, 2, 2), (2, 4, 4, 1, 2, 2), (2, 6, 6, 3, 2, 2), (1, 5, 3, 1, 2, 1)]:
    x = rng.standard_normal((N, H, W, C)).astype(np.float32)
    ref = ref_maxpool(x, k, s)
    got = julia_style_maxpool(x, k, s)
    bad = int((~np.isclose(ref, got)).sum())
    print(f"maxpool N={N} H={H} W={W} C={C} k={k} s={s} | {np.abs(ref-got).max():10.4f} | {bad}/{ref.size}")

for (N, H, W, C) in [(1, 3, 3, 1), (1, 3, 3, 2), (2, 3, 3, 1), (2, 3, 3, 4)]:
    x = rng.standard_normal((N, H, W, C)).astype(np.float32)
    ref = x.mean(axis=(1, 2))
    got = julia_style_gap(x)
    bad = int((~np.isclose(ref, got)).sum())
    print(f"gap     N={N} H={H} W={W} C={C}         | {np.abs(ref-got).max():10.4f} | {bad}/{ref.size}")

print()
print("Edge-case contracts (row-major single-channel so layout is not a factor):")
x = np.full((1, 2, 2, 1), -np.inf, np.float32)
out = np.zeros(1, np.float32)
lib.axiom_maxpool2d(np.ascontiguousarray(x).ctypes.data_as(f32p), out.ctypes.data_as(f32p), 1, 2, 2, 1, 2, 2, 2, 2)
print(f"  all -Inf window  -> zig={out[0]!r}  julia(maximum)=-Inf   (zig initialises max at -floatmax)")
x = np.array([[1.0, np.nan], [0.5, 0.25]], np.float32).reshape(1, 2, 2, 1)
out = np.zeros(1, np.float32)
lib.axiom_maxpool2d(np.ascontiguousarray(x).ctypes.data_as(f32p), out.ctypes.data_as(f32p), 1, 2, 2, 1, 2, 2, 2, 2)
print(f"  window with NaN  -> zig={out[0]!r}  julia(maximum)=NaN    (zig `val > max` skips NaN)")
