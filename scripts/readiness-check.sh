#!/usr/bin/env bash
# SPDX-License-Identifier: PMPL-1.0-or-later
#
# Consolidated production-readiness gate for Axiom.jl.
# Runs the must-pass local checks that mirror CI jobs.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

JULIA_BIN="${JULIA_BIN:-julia}"
CARGO_BIN="${CARGO_BIN:-cargo}"

# If 0, skipped critical checks fail the run.
ALLOW_SKIPS="${AXIOM_READINESS_ALLOW_SKIPS:-0}"

# Rust profile:
# - auto: run Rust checks if cargo + rust/Cargo.toml are present
# - 1: require Rust checks
# - 0: skip Rust checks
RUN_RUST="${AXIOM_READINESS_RUN_RUST:-auto}"

RUN_BASELINE="${AXIOM_READINESS_RUN_BASELINE:-1}"
RUN_RUNTIME="${AXIOM_READINESS_RUN_RUNTIME:-1}"
RUN_GPU_FALLBACK="${AXIOM_READINESS_RUN_GPU_FALLBACK:-1}"
RUN_COPROCESSOR="${AXIOM_READINESS_RUN_COPROCESSOR:-1}"
RUN_INTEROP="${AXIOM_READINESS_RUN_INTEROP:-1}"
RUN_CERTIFICATE="${AXIOM_READINESS_RUN_CERTIFICATE:-1}"
RUN_DOC_ALIGNMENT="${AXIOM_READINESS_RUN_DOC_ALIGNMENT:-1}"
RUN_MARKERS="${AXIOM_READINESS_RUN_MARKERS:-1}"

declare -a PASSED=()
declare -a FAILED=()
declare -a SKIPPED=()

log() {
  printf '[readiness] %s\n' "$*"
}

record_pass() {
  PASSED+=("$1")
  log "PASS: $1"
}

record_fail() {
  FAILED+=("$1")
  log "FAIL: $1"
}

record_skip() {
  SKIPPED+=("$1")
  log "SKIP: $1"
}

run_check() {
  local name="$1"
  shift

  log "RUN: $name"
  if "$@"; then
    record_pass "$name"
  else
    record_fail "$name"
  fi
}

check_dependencies() {
  command -v "$JULIA_BIN" >/dev/null 2>&1 || {
    echo "Missing Julia executable: $JULIA_BIN"
    return 1
  }
  command -v rg >/dev/null 2>&1 || {
    echo "Missing ripgrep executable: rg"
    return 1
  }
}

check_markers() {
  local markers
  markers="$(rg -n "TO[D]O|FIXM[E]|TB[D]|OPEN_ITEM|FIX_ITEM|HACK|XXX" src ext test || true)"
  if [ -n "$markers" ]; then
    echo "$markers"
    return 1
  fi
  return 0
}

check_doc_alignment() {
  local status=0

  if ! rg -Fq 'model = from_pytorch("model.pt")' README.adoc; then
    echo "README.adoc is missing the direct checkpoint `from_pytorch(\"model.pt\")` example."
    status=1
  fi

  if ! rg -Fq "application/grpc+json" README.adoc; then
    echo "README.adoc is missing gRPC bridge content-type coverage notes."
    status=1
  fi

  if ! rg -Fq "## Deferred Commitments (Tracked)" ROADMAP.md; then
    echo "ROADMAP.md is missing the deferred commitments section."
    status=1
  fi

  if ! rg -Fq "== Deferred Commitments (Tracked)" ROADMAP.adoc; then
    echo "ROADMAP.adoc is missing the deferred commitments section."
    status=1
  fi

  if ! rg -Fq '`from_pytorch(...)` model import | Baseline shipped' docs/wiki/Roadmap-Commitments.md; then
    echo "docs/wiki/Roadmap-Commitments.md is missing baseline-shipped status for from_pytorch."
    status=1
  fi

  if ! rg -Fq '`to_onnx(...)` export | Baseline shipped' docs/wiki/Roadmap-Commitments.md; then
    echo "docs/wiki/Roadmap-Commitments.md is missing baseline-shipped status for to_onnx."
    status=1
  fi

  if ! rg -Fq "test/ci/coprocessor_strategy.jl" docs/wiki/Developer-Guide.md; then
    echo "docs/wiki/Developer-Guide.md is missing coprocessor strategy test guidance."
    status=1
  fi

  if ! rg -Fq 'model = from_pytorch("model.pt")' docs/wiki/User-Guide.md; then
    echo "docs/wiki/User-Guide.md is missing direct checkpoint import guidance."
    status=1
  fi

  return "$status"
}

resolve_rust_lib_path() {
  case "$(uname -s)" in
    Linux)
      echo "rust/target/release/libaxiom_core.so"
      ;;
    Darwin)
      echo "rust/target/release/libaxiom_core.dylib"
      ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
      echo "rust/target/release/axiom_core.dll"
      ;;
    *)
      if [ -f rust/target/release/libaxiom_core.so ]; then
        echo "rust/target/release/libaxiom_core.so"
      elif [ -f rust/target/release/libaxiom_core.dylib ]; then
        echo "rust/target/release/libaxiom_core.dylib"
      else
        echo "rust/target/release/axiom_core.dll"
      fi
      ;;
  esac
}

should_run_rust_checks() {
  case "$RUN_RUST" in
    1)
      return 0
      ;;
    0)
      return 1
      ;;
    auto)
      command -v "$CARGO_BIN" >/dev/null 2>&1 && [ -f rust/Cargo.toml ]
      ;;
    *)
      echo "Invalid AXIOM_READINESS_RUN_RUST value: $RUN_RUST (expected: auto|0|1)"
      return 2
      ;;
  esac
}

run_rust_checks() {
  "$CARGO_BIN" build --release --manifest-path rust/Cargo.toml

  local lib_path
  lib_path="$(resolve_rust_lib_path)"
  [ -f "$lib_path" ] || {
    echo "Rust shared library not found: $lib_path"
    return 1
  }

  AXIOM_RUST_LIB="$ROOT_DIR/$lib_path" \
    "$JULIA_BIN" --project=. test/ci/backend_parity.jl

  AXIOM_RUST_LIB="$ROOT_DIR/$lib_path" \
  AXIOM_SMOKE_ACCELERATOR=rust \
  AXIOM_SMOKE_ACCELERATOR_REQUIRED=1 \
    "$JULIA_BIN" --project=. test/ci/runtime_smoke.jl
}

print_summary() {
  echo
  log "Summary:"
  log "  Passed: ${#PASSED[@]}"
  log "  Failed: ${#FAILED[@]}"
  log "  Skipped: ${#SKIPPED[@]}"

  if [ "${#FAILED[@]}" -gt 0 ]; then
    printf '[readiness] Failed checks:\n'
    printf '  - %s\n' "${FAILED[@]}"
  fi

  if [ "${#SKIPPED[@]}" -gt 0 ]; then
    printf '[readiness] Skipped checks:\n'
    printf '  - %s\n' "${SKIPPED[@]}"
  fi
}

run() {
  run_check "tool dependencies" check_dependencies

  if [ "$RUN_BASELINE" = "1" ]; then
    run_check \
      "Pkg.instantiate/build/precompile/test" \
      "$JULIA_BIN" --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.build(); Pkg.precompile(); Pkg.test()'
  else
    record_skip "Pkg.instantiate/build/precompile/test (disabled)"
  fi

  if [ "$RUN_RUNTIME" = "1" ]; then
    run_check "runtime smoke (CPU/default)" "$JULIA_BIN" --project=. test/ci/runtime_smoke.jl
  else
    record_skip "runtime smoke (disabled)"
  fi

  if [ "$RUN_GPU_FALLBACK" = "1" ]; then
    run_check "GPU fallback behavior" "$JULIA_BIN" --project=. test/ci/gpu_fallback.jl
  else
    record_skip "GPU fallback behavior (disabled)"
  fi

  if [ "$RUN_COPROCESSOR" = "1" ]; then
    run_check "coprocessor strategy behavior" "$JULIA_BIN" --project=. test/ci/coprocessor_strategy.jl
  else
    record_skip "coprocessor strategy behavior (disabled)"
  fi

  if [ "$RUN_INTEROP" = "1" ]; then
    run_check "interop smoke (from_pytorch/to_onnx)" "$JULIA_BIN" --project=. test/ci/interop_smoke.jl
  else
    record_skip "interop smoke (disabled)"
  fi

  if [ "$RUN_CERTIFICATE" = "1" ]; then
    run_check "certificate integrity" "$JULIA_BIN" --project=. test/ci/certificate_integrity.jl
  else
    record_skip "certificate integrity (disabled)"
  fi

  if [ "$RUN_MARKERS" = "1" ]; then
    run_check "no unresolved markers in src/ext/test" check_markers
  else
    record_skip "marker scan (disabled)"
  fi

  if [ "$RUN_DOC_ALIGNMENT" = "1" ]; then
    run_check "README/wiki roadmap alignment" check_doc_alignment
  else
    record_skip "README/wiki roadmap alignment (disabled)"
  fi

  if should_run_rust_checks; then
    run_check "Rust backend parity + accelerated runtime smoke" run_rust_checks
  else
    case "$RUN_RUST" in
      0)
        record_skip "Rust backend parity + accelerated runtime smoke (disabled)"
        ;;
      auto)
        record_skip "Rust backend parity + accelerated runtime smoke (cargo/rust project unavailable)"
        ;;
      *)
        record_fail "Rust backend parity + accelerated runtime smoke"
        ;;
    esac
  fi

  print_summary

  if [ "${#FAILED[@]}" -gt 0 ]; then
    exit 1
  fi

  if [ "${#SKIPPED[@]}" -gt 0 ] && [ "$ALLOW_SKIPS" != "1" ]; then
    log "Skipped critical checks are not allowed (set AXIOM_READINESS_ALLOW_SKIPS=1 to permit skips)."
    exit 2
  fi

  log "Readiness gate passed."
}

run
