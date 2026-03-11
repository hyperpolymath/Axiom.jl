# Axiom.jl ABI / FFI Notes

This file documents the current ABI/FFI split for Axiom.jl.

## Canonical ABI Spec (Idris2)

The Idris2 ABI scaffold now lives under:

- `src/Abi/Types.idr`
- `src/Abi/Layout.idr`
- `src/Abi/Foreign.idr`

These modules use concrete `axiom_*` symbol names and no unresolved template placeholders.

### Idris2 validation

```bash
idris2 --source-dir src --check src/Abi/Types.idr
idris2 --source-dir src --check src/Abi/Layout.idr
idris2 --source-dir src --check src/Abi/Foreign.idr
```

## Runtime FFI Used in Production Paths

Current production-tested backend FFI path is Julia <-> Zig:

- Julia bridge: `src/backends/zig_ffi.jl`
- Zig C ABI exports: `ffi/zig/src/main.zig`

This path is covered by CI/readiness checks (backend parity + runtime smoke).

## Zig FFI Status

`ffi/zig/` is now concrete (non-template) and exports concrete `axiom_*` symbols:

- implementation: `ffi/zig/src/main.zig`
- build/test entry: `ffi/zig/build.zig`
- integration coverage: `ffi/zig/test/integration_test.zig`
- C header: `ffi/zig/include/axiom.h`

Validation:

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

For release/readiness decisions, treat the Zig FFI path as authoritative.
Treat Idris2 ABI files as formal scaffold/specification that now typechecks and
uses concrete Axiom naming.
