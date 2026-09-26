# SPDX-License-Identifier: MPL-2.0
"""axiom_batchnorm uses a fixed [4096]f32 stack scratch (norm.zig:143). Probe num_features > 4096
through the production .so, each case in a subprocess (stack corruption may crash the process).
Usage: PYTHONPATH=<numpy-env> python3 batchnorm_probe.py zig/zig-out/lib/libaxiom_zig.so"""
import sys, subprocess
child = r'''
import ctypes, sys, numpy as np
lib = ctypes.CDLL(sys.argv[1]); F = int(sys.argv[2]); B = 2
f32p = ctypes.POINTER(ctypes.c_float)
rng = np.random.default_rng(0)
x = rng.standard_normal((B, F)).astype(np.float32); y = np.zeros_like(x)
g = np.ones(F, np.float32); b = np.zeros(F, np.float32); m = np.zeros(F, np.float32); v = np.ones(F, np.float32)
fn = lib.axiom_batchnorm; fn.restype = None
fn.argtypes = [f32p, f32p, f32p, f32p, f32p, f32p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float]
fn(x.ctypes.data_as(f32p), y.ctypes.data_as(f32p), g.ctypes.data_as(f32p), b.ctypes.data_as(f32p), m.ctypes.data_as(f32p), v.ctypes.data_as(f32p), B * F, F, 1e-5)
ref = (x - m) / np.sqrt(v + 1e-5)
print(int((~np.isclose(y, ref, atol=1e-4)).sum()), "of", B * F)
'''
for F in (4096, 4097, 4100, 8192, 65536, 1_000_000):
    r = subprocess.run([sys.executable, "-c", child, sys.argv[1], str(F)], capture_output=True, text=True)
    out = r.stdout.strip() if r.returncode == 0 else f"CRASH exit={r.returncode} {r.stderr.strip().splitlines()[-1][:80] if r.stderr.strip() else '(signal)'}"
    print(f"num_features={F:>8}: {out}")
