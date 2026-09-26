# SPDX-License-Identifier: MPL-2.0
"""Probe Zig activation kernels (built libaxiom_zig.so) for overflow / cancellation.
Usage: PYTHONPATH=<numpy-env> python3 activation_probe.py zig/zig-out/lib/libaxiom_zig.so
Compares against float32 reference formulas evaluated in float64 then rounded."""
import ctypes, sys, numpy as np
lib = ctypes.CDLL(sys.argv[1])
f32p = ctypes.POINTER(ctypes.c_float)
def call(sym, x):
    x = np.ascontiguousarray(x, dtype=np.float32); y = np.zeros_like(x)
    fn = getattr(lib, sym); fn.restype = None
    fn.argtypes = [f32p, f32p, ctypes.c_size_t]
    fn(x.ctypes.data_as(f32p), y.ctypes.data_as(f32p), x.size); return y
def gelu_ref(x):  # same tanh-approx formula Julia's gelu uses, evaluated in float64
    x = x.astype(np.float64); c = np.sqrt(2/np.pi)
    return (0.5*x*(1+np.tanh(c*(x+0.044715*x**3)))).astype(np.float32)
xs = np.array([1e-8, 1e-5, 1e-3, 5, 9, 10, 10.5, 11, 12, 20, 44, 44.3, 44.5, 45, 50, 100, -50, -100, float('inf'), float('-inf')], dtype=np.float32)
print(f"{'x':>10} | {'zig tanh':>12} {'ref tanh':>12} | {'zig gelu':>12} {'ref gelu':>12} | {'zig sigmoid':>12} {'ref':>12}")
zt, zg, zs = call('axiom_tanh', xs), call('axiom_gelu', xs), call('axiom_sigmoid', xs)
rt = np.tanh(xs.astype(np.float64)).astype(np.float32); rg = gelu_ref(xs)
rs = (1/(1+np.exp(-xs.astype(np.float64)))).astype(np.float32)
for i, x in enumerate(xs):
    flag = ""
    if not np.isfinite(zt[i]) and np.isfinite(rt[i]): flag += " TANH-NaN"
    if not np.isfinite(zg[i]) and np.isfinite(rg[i]): flag += " GELU-NaN"
    if np.isfinite(zt[i]) and rt[i] != 0 and abs(zt[i]-rt[i])/abs(rt[i]) > 1e-4: flag += f" tanh-relerr={abs(zt[i]-rt[i])/abs(rt[i]):.1e}"
    print(f"{x:>10.4g} | {zt[i]:>12.6g} {rt[i]:>12.6g} | {zg[i]:>12.6g} {rg[i]:>12.6g} | {zs[i]:>12.6g} {rs[i]:>12.6g}{flag}")
# in-place variants use the same formula
xt = xs.copy(); fn = lib.axiom_tanh_inplace; fn.restype=None; fn.argtypes=[f32p, ctypes.c_size_t]; fn(xt.ctypes.data_as(f32p), xt.size)
print("\naxiom_tanh_inplace(50) =", xt[list(xs).index(np.float32(50))], " axiom_tanh_inplace(11)=", xt[list(xs).index(np.float32(11))])
# realistic batch: how often does a standard-normal*sigma pre-activation trip GELU?
rng = np.random.default_rng(0)
for sigma in (1, 3, 5, 8):
    pre = (rng.standard_normal(1_000_000)*sigma).astype(np.float32)
    g = call('axiom_gelu', pre); print(f"gelu on N(0,{sigma}²) x1e6: NaN count = {np.isnan(g).sum()}, max|x|={np.abs(pre).max():.1f}")
