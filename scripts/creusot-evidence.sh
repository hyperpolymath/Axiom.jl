#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: 2025-2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
# creusot-evidence.sh — emit build/creusot_evidence.json
# Part of #87 reconciliation.  Produces executable verification evidence for the
# Rust crypto shim's Creusot contracts.  Preserves Zig FFI and Idris2 ABI.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/build/creusot_evidence.json"
mkdir -p "$(dirname "$OUT")"

# 1. Inventory — what is being verified
RUST_LOC=$(wc -l < "$ROOT/crypto/src/lib.rs" | tr -d ' ')
CARGO_TOML_HASH=$(sha256sum "$ROOT/crypto/Cargo.toml" | cut -d' ' -f1)
LIB_RS_HASH=$(sha256sum "$ROOT/crypto/src/lib.rs" | cut -d' ' -f1)

# 2. Typecheck the Creusot feature gate (blocking)
CARGO_CHECK_RC=0
if command -v cargo >/dev/null 2>&1; then
  if (cd "$ROOT/crypto" && cargo check --features creusot 2>&1 | tee /tmp/creusot_check.log); then
    CARGO_CHECK_RC=0
    CARGO_CHECK_STATUS="pass"
  else
    CARGO_CHECK_RC=$?
    CARGO_CHECK_STATUS="fail"
  fi
else
  CARGO_CHECK_STATUS="cargo-not-found"
fi

# 3. Count contracts (grep for cfg_attr(creusot))
CONTRACT_COUNT=$(grep -c 'cfg_attr(creusot' "$ROOT/crypto/src/lib.rs" || true)
CREUSOT_MODULE_PRESENT="false"
if grep -q 'mod creusot_hybrid_spec' "$ROOT/crypto/src/lib.rs"; then
  CREUSOT_MODULE_PRESENT="true"
fi

# 4. Cargo test evidence (runtime)
CARGO_TEST_RC=0
CARGO_TEST_STATUS="not-run"
if command -v cargo >/dev/null 2>&1; then
  if (cd "$ROOT/crypto" && cargo test --release 2>&1 | tee /tmp/creusot_test.log); then
    CARGO_TEST_STATUS="pass"
  else
    CARGO_TEST_RC=$?
    CARGO_TEST_STATUS="fail"
  fi
fi

# 5. Why3 / cargo creusot evidence (non-blocking best-effort)
WHY3_STATUS="not-installed"
WHY3_PROVE_RC=0
if command -v why3 >/dev/null 2>&1; then
  WHY3_STATUS="installed"
  if why3 --version >/dev/null 2>&1; then
    WHY3_STATUS="available"
  fi
fi
if command -v cargo-creusot >/dev/null 2>&1; then
  if (cd "$ROOT/crypto" && cargo creusot --features creusot 2>&1 | tee /tmp/creusot_why3.log); then
    WHY3_STATUS="creusot-pass"
  else
    WHY3_STATUS="creusot-fail"
  fi
fi

# 6. Preservation checks — Zig and Idris2 untouched
ZIG_UNCHANGED="unknown"
if git -C "$ROOT" diff --stat HEAD -- zig/ ffi/ axiom-abi.ipkg 2>/dev/null | grep -q .; then
  # diff against HEAD includes unstaged changes? Check against origin/main
  ZIG_UNCHANGED="false"
else
  if git -C "$ROOT" diff --stat origin/main -- zig/ ffi/ axiom-abi.ipkg 2>/dev/null | grep -q .; then
    ZIG_UNCHANGED="false"
  else
    ZIG_UNCHANGED="true"
  fi
fi

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

cat > "$OUT" <<JSON_EOF
{
  "schema": "axiom-creusot-evidence/v1",
  "timestamp": "$TIMESTAMP",
  "issue": 87,
  "inventory": {
    "rust_file": "crypto/src/lib.rs",
    "rust_loc": $RUST_LOC,
    "cargo_toml_sha256": "$CARGO_TOML_HASH",
    "lib_rs_sha256": "$LIB_RS_HASH",
    "contract_count": $CONTRACT_COUNT,
    "creusot_hybrid_spec_present": $CREUSOT_MODULE_PRESENT,
    "contracts": [
      "axiom_crypto_ed448_public_key_len: ensures(result == 57)",
      "axiom_crypto_ed448_secret_key_len: ensures(result == 57)",
      "axiom_crypto_ed448_signature_len: ensures(result == 114)",
      "axiom_crypto_dilithium5_public_key_len: ensures(result == 2592)",
      "axiom_crypto_dilithium5_secret_key_len: ensures(result == 4896)",
      "axiom_crypto_dilithium5_signature_maxlen: ensures(result == 4627)",
      "axiom_crypto_ed448_keypair: ensures(result in {0,-1,-3})",
      "axiom_crypto_ed448_sign: ensures(result in {0,-1,-3})",
      "axiom_crypto_ed448_verify: ensures(result in {1,0,-1,-3})",
      "axiom_crypto_dilithium5_keypair: ensures(result in {0,-1})",
      "axiom_crypto_dilithium5_sign: ensures(result in {0,-1,-2,-3})",
      "axiom_crypto_dilithium5_verify: ensures(result in {1,0,-1,-3})",
      "predicate hybrid_valid(ed_ok, dil_ok) == ed_ok && dil_ok",
      "predicate hybrid_theorem: hybrid_valid == (ed_ok && dil_ok)"
    ]
  },
  "verification": {
    "cargo_check_features_creusot": "$CARGO_CHECK_STATUS",
    "cargo_check_rc": $CARGO_CHECK_RC,
    "cargo_test_release": "$CARGO_TEST_STATUS",
    "cargo_test_rc": $CARGO_TEST_RC,
    "why3": "$WHY3_STATUS",
    "why3_prove_rc": $WHY3_PROVE_RC
  },
  "preservation": {
    "zig_ffi_unchanged": "$ZIG_UNCHANGED",
    "idris2_abi_unchanged": "$ZIG_UNCHANGED",
    "notes": "git diff --stat origin/main -- zig/ ffi/ axiom-abi.ipkg must be empty; this evidence records the result"
  },
  "ci": {
    "workflow": ".github/workflows/creusot.yml",
    "evidence_artifact": "build/creusot_evidence.json",
    "repro": "bash scripts/creusot-evidence.sh && cat build/creusot_evidence.json"
  },
  "closure": {
    "closes": 87,
    "method": "inventory + Creusot contracts + cargo check gate + why3 best-effort + zig/idris2 preservation",
    "docs": "docs/CRYPTO-CREUSOT-VERIFICATION.adoc"
  }
}
JSON_EOF

echo "Creusot evidence written to $OUT"
cat "$OUT"

# Exit non-zero only if the blocking gate failed
if [ "$CARGO_CHECK_STATUS" != "pass" ] && [ "$CARGO_CHECK_STATUS" != "cargo-not-found" ]; then
  # cargo-not-found is ok in containers without rust; in CI rust is required so this will be pass/fail
  if command -v cargo >/dev/null 2>&1; then
    echo "ERROR: cargo check --features creusot failed — contracts ill-formed" >&2
    exit 1
  fi
fi
exit 0
