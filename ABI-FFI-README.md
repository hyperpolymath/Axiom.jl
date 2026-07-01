# Axiom.jl ABI / FFI Notes

This file documents the current ABI/FFI split for Axiom.jl.

## Idris2 ABI Scaffold (illustrative, not the production ABI proof)

The Idris2 ABI scaffold lives under:

- `src/Abi/Types.idr`
- `src/Abi/Layout.idr`
- `src/Abi/Foreign.idr`

These modules use concrete `axiom_*` symbol names and no unresolved template
placeholders, but they declare a **generic** lifecycle/callback surface
(`axiom_init`, `axiom_process`, `axiom_register_callback`, ...) that does
**not** match the ~36 real Zig kernel exports (`axiom_matmul`, `axiom_relu`,
`axiom_conv2d`, and the rest of `zig/src/axiom.zig`) actually consumed by
`src/backends/zig_ffi.jl`. Treat this scaffold as a
worked illustration of the hyperpolymath ABI-FFI pattern (Idris2 ABI + Zig
FFI), not as a proof binding Axiom's production kernels. The `Verify.*`
functions in `Types.idr` are `putStrLn` stubs, not executed proofs.

**ROADMAP (future work, not yet started):** extend the `.idr` declarations
to cover the real `axiom_*` Zig exports and replace the `Verify` stubs with
genuine size/alignment/signature proofs checked against `zig/src/axiom.zig`
and `ffi/zig/include/axiom.h`.

### Idris2 validation

```bash
idris2 --source-dir src --check src/Abi/Types.idr
idris2 --source-dir src --check src/Abi/Layout.idr
idris2 --source-dir src --check src/Abi/Foreign.idr
```

## Runtime FFI Used in Production Paths

Current production-tested backend FFI path is Julia <-> Zig:

- Julia bridge: `src/backends/zig_ffi.jl`
- Zig C ABI exports: `zig/src/axiom.zig` (compiled to `libaxiom_zig.so` via
  `zig/build.zig`; this is the artifact CI, benchmarks, and
  `AXIOM_ZIG_LIB`/`ZigBackend(...)` actually load -- see `.github/workflows/ci.yml`,
  `benchmark/*.jl`, `README.md`)

This path is covered by CI/readiness checks (backend parity + runtime smoke).

## KNOWN ISSUE: orphan second Zig FFI tree (`ffi/zig/` vs `zig/`)

There are **two** Zig trees in this repository and they are not the same
code:

- **`zig/`** (repo root) -- the real, production Zig backend. `zig/src/axiom.zig`
  exports the ~36 real `axiom_*` kernels (`axiom_matmul`, `axiom_relu`,
  `axiom_conv2d`, activations, norm, pooling, attention, ...) consumed by
  `src/backends/zig_ffi.jl` and built into `libaxiom_zig.so`.
- **`ffi/zig/`** -- a separate, smaller Zig tree (`ffi/zig/src/main.zig`,
  `ffi/zig/build.zig`, `ffi/zig/include/axiom.h`) that exports **zero**
  `axiom_*` kernel symbols (`grep -c "export fn axiom_"` = 0) and instead
  implements a different, generic lifecycle/callback C ABI matching the
  `src/Abi/*.idr` scaffold's naming (`axiom_init`, `axiom_process`,
  `axiom_register_callback`, ...). Its header comment even points at
  `src/abi/Foreign.idr`, a stale/incorrect path (the real directory is
  `src/Abi/`, capital A) -- further evidence this tree has drifted from the
  rest of the repo and is not wired into anything Julia actually loads.

**This is flagged, not fixed here.** `ffi/zig/` is not deleted -- it may be
salvageable as the eventual real implementation backing the `src/Abi/*.idr`
scaffold (see ROADMAP notes above), but as of this writing it is disconnected
from both the production Zig backend and the Idris2 ABI's stated symbol
surface. Anyone relying on "the Zig FFI" should confirm which of the two
trees they mean; for release/readiness purposes only `zig/` is authoritative
(next section).

## Zig FFI Status (production: `zig/`)

`zig/` is the sole production-tested native backend and exports concrete
`axiom_*` symbols (~36, see `TOPOLOGY.md`):

- implementation: `zig/src/axiom.zig`
- build entry: `zig/build.zig`
- Julia-side wiring: `src/backends/zig_ffi.jl`

The separate `ffi/zig/` tree (`ffi/zig/src/main.zig`, `ffi/zig/build.zig`,
`ffi/zig/test/integration_test.zig`, `ffi/zig/include/axiom.h`) is also
concrete (non-template) and internally self-consistent, but -- per the
orphan-tree note above -- exports a different, generic `axiom_*` surface
that is not the production kernel FFI. Its own validation still works
in isolation:

```bash
cd ffi/zig
zig build test
```

## Bidirectionality Status

- Implemented and exercised:
  - host -> runtime calls (`axiom_process`, `axiom_process_array`, etc.)
  - runtime -> host callback bridge (`axiom_register_callback`, `axiom_invoke_callback`)
- Idris side includes concrete callback pointer registration and callback invoke declarations in `src/Abi/Foreign.idr`.
- Still not a full cross-language compatibility matrix across all planned targets, but no longer template-only.

## Practical Guidance

For release/readiness decisions, treat `zig/` + `src/backends/zig_ffi.jl` as
the authoritative production FFI boundary. Treat the Idris2 `src/Abi/*.idr`
files as an illustrative ABI-FFI scaffold that typechecks and uses concrete
Axiom naming, but does **not** prove or specify the production kernel FFI --
see the ROADMAP notes above for what closing that gap would require. Treat
`ffi/zig/` as a flagged, currently-disconnected second Zig tree (see
"KNOWN ISSUE" above) pending a decision on whether to wire it to the
`src/Abi/*.idr` scaffold, fold it into `zig/`, or retire it.
