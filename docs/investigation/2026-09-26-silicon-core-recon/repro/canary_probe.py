# SPDX-License-Identifier: MPL-2.0
"""Call the production libaxiom_zig.so axiom_layernorm with B=5,H=2048 (the chunking-underflow case)
and check (a) row correctness and (b) a canary region after the output buffer.
Usage: PYTHONPATH=<numpy-env> python3 canary_probe.py zig/zig-out/lib/libaxiom_zig.so"""
import ctypes, sys, numpy as np
lib = ctypes.CDLL(sys.argv[1])
f32p = ctypes.POINTER(ctypes.c_float)
for (B, H) in [(5, 2048), (5, 4096), (5, 65536)]:
    x = np.random.default_rng(0).standard_normal((B, H)).astype(np.float32)
    canary = 8 * H
    ybuf = np.full(B * H + canary, 12345.0, np.float32)   # output rows followed by canary
    g = np.ones(H, np.float32)
    b = np.zeros(H, np.float32)
    fn = lib.axiom_layernorm
    fn.restype = None
    fn.argtypes = [f32p, f32p, f32p, f32p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float]
    fn(x.ctypes.data_as(f32p), ybuf.ctypes.data_as(f32p), g.ctypes.data_as(f32p), b.ctypes.data_as(f32p), B, H, 1e-5)
    y = ybuf[:B*H].reshape(B, H)
    can = ybuf[B*H:]
    ref = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-5)
    print(f"B={B} H={H}: bad_rows={int((~np.isclose(y, ref, atol=1e-3)).any(1).sum())} canary_overwritten={int((can != 12345.0).sum())}/{canary}")
