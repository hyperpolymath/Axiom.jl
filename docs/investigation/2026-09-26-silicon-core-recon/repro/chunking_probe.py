# SPDX-License-Identifier: MPL-2.0
"""Which batch sizes trip the chunking underflow in parallel_layernorm/rmsnorm/softmax?
Runs each case in a subprocess because the ReleaseFast .so may segfault / corrupt memory.
Usage: PYTHONPATH=<numpy-env> python3 chunking_probe.py zig/zig-out/lib/libaxiom_zig.so"""
import sys, subprocess, json
lib = sys.argv[1]
child = r'''
import ctypes, sys, numpy as np
lib = ctypes.CDLL(sys.argv[1]); B, H, op = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
f32p = ctypes.POINTER(ctypes.c_float)
x = np.random.default_rng(0).standard_normal((B, H)).astype(np.float32); y = np.full((B, H), 7.0, np.float32)
if op == "softmax":
    fn = lib.axiom_softmax; fn.argtypes = [f32p, f32p, ctypes.c_size_t, ctypes.c_size_t]; fn.restype = None
    fn(x.ctypes.data_as(f32p), y.ctypes.data_as(f32p), B, H)
    ref = np.exp(x - x.max(1, keepdims=True)); ref /= ref.sum(1, keepdims=True)
elif op == "layernorm":
    g = np.ones(H, np.float32); b = np.zeros(H, np.float32)
    fn = lib.axiom_layernorm; fn.argtypes = [f32p, f32p, f32p, f32p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float]; fn.restype = None
    fn(x.ctypes.data_as(f32p), y.ctypes.data_as(f32p), g.ctypes.data_as(f32p), b.ctypes.data_as(f32p), B, H, 1e-5)
    m = x.mean(1, keepdims=True); v = x.var(1, keepdims=True); ref = (x - m) / np.sqrt(v + 1e-5)
bad = int((~np.isclose(y, ref, atol=1e-4)).sum())
print(bad)
'''
for op in ("softmax", "layernorm"):
    for B in (4, 5, 6, 7, 8, 9, 13, 17):
        H = 2048
        r = subprocess.run([sys.executable, "-c", child, lib, str(B), str(H), op], capture_output=True, text=True)
        status = f"exit={r.returncode}" + (f" mismatches={r.stdout.strip()}" if r.returncode == 0 else f" ({r.stderr.strip().splitlines()[-1][:60] if r.stderr.strip() else 'signal'})")
        print(f"{op:>9} B={B:>2} H={H}: {status}")
