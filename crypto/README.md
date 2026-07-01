<!-- SPDX-License-Identifier: MPL-2.0 -->
# axiom_crypto

Hybrid **Ed448 + Dilithium5 (ML-DSA-87)** signing primitives for Axiom.jl
verification certificates, exposed as a C-ABI `cdylib` for `ccall` from
Julia. This is the estate Trustfile hybrid signature scheme: a classical
signature (Ed448 / EdDSA over Curve448, RFC 8032) plus a post-quantum
signature (Dilithium5 / ML-DSA-87, FIPS 204), both required to pass.

No hand-rolled cryptography: Ed448 goes through the system libcrypto
(OpenSSL 3.x) via the vetted [`openssl`](https://docs.rs/openssl) crate;
Dilithium5 goes through the vetted
[`pqcrypto-dilithium`](https://docs.rs/pqcrypto-dilithium) crate (PQClean
reference implementation).

This mirrors the algorithm choice used by `opsm_ex/native/opsm_pq_nif` (the
Elixir/BEAM reference implementation in `odds-and-sods-package-manager`), but
is a fresh, dependency-free (of that project) Rust `cdylib` — Axiom.jl does
not depend on Elixir/BEAM.

## Build

```sh
cd crypto
cargo build --release
cargo test
```

Or via the repo `Justfile`: `just build-crypto` (from the Axiom.jl root).

Output: `crypto/target/release/libaxiom_crypto.so` (Linux) /
`.dylib` (macOS) / `.dll` (Windows). `crypto/target/` is git-ignored; the
compiled shared library is a build artifact, not source.

## Private key custody — NEVER commit private keys

This crate never reads or writes key material to disk on its own. Keys are
either:

- generated in-memory for tests / examples (`*_keypair` functions), or
- generated and held out-of-band by an HSM or an offline signing process,
  with only the **public** keys embedded in a certificate.

No `.pem`/`.key` file produced by a real signing key should ever enter this
repository. See `ROADMAP.adoc` for the custody story.

## C ABI

Every exported function is `extern "C"`, `#[no_mangle]`. Convention:

- **Fixed-size outputs** (public/secret keys, Ed448 signatures) are written
  into a caller-allocated buffer. The required size is given by a paired
  `axiom_crypto_*_len()` getter — call it first, allocate that many bytes,
  then pass the buffer pointer.
- **Variable/bounded-size outputs** (Dilithium5 signatures) are written into
  a caller-allocated buffer sized by `axiom_crypto_dilithium5_signature_maxlen()`;
  the actual length written is returned through an out-parameter
  `sig_len_out: *mut usize`.
- **Every function returns `i32`.** For keypair/sign functions: `0` = OK,
  negative = error (`AXIOM_CRYPTO_ERR_NULL_PTR = -1`,
  `AXIOM_CRYPTO_ERR_BAD_LENGTH = -2`, `AXIOM_CRYPTO_ERR_CRYPTO_FAILURE = -3`).
  For verify functions: `1` = signature valid, `0` = signature invalid,
  negative = the call itself could not be carried out (distinct from "ran
  and failed" — a bad pointer is not the same claim as "this certificate is
  forged").
- **No alloc+free pairs.** All buffers are caller-owned; there is no
  `axiom_crypto_free`. This keeps the Julia side to plain `ccall` +
  `Vector{UInt8}` with no foreign-pointer finalizer needed.

### Length getters

| Function | Returns |
|---|---|
| `axiom_crypto_ed448_public_key_len() -> usize` | 57 |
| `axiom_crypto_ed448_secret_key_len() -> usize` | 57 |
| `axiom_crypto_ed448_signature_len() -> usize` | 114 |
| `axiom_crypto_dilithium5_public_key_len() -> usize` | 2592 |
| `axiom_crypto_dilithium5_secret_key_len() -> usize` | 4896 |
| `axiom_crypto_dilithium5_signature_maxlen() -> usize` | 4627 |

### Ed448

```c
int32_t axiom_crypto_ed448_keypair(uint8_t *pk_out, uint8_t *sk_out);

int32_t axiom_crypto_ed448_sign(
    const uint8_t *msg_ptr, size_t msg_len,
    const uint8_t *sk_ptr,
    uint8_t *sig_out);

int32_t axiom_crypto_ed448_verify(
    const uint8_t *msg_ptr, size_t msg_len,
    const uint8_t *sig_ptr,
    const uint8_t *pk_ptr);  // returns 1/0/negative, see above
```

### Dilithium5 (ML-DSA-87)

```c
int32_t axiom_crypto_dilithium5_keypair(uint8_t *pk_out, uint8_t *sk_out);

int32_t axiom_crypto_dilithium5_sign(
    const uint8_t *msg_ptr, size_t msg_len,
    const uint8_t *sk_ptr,
    uint8_t *sig_out, size_t *sig_len_out);

int32_t axiom_crypto_dilithium5_verify(
    const uint8_t *msg_ptr, size_t msg_len,
    const uint8_t *sig_ptr, size_t sig_len,
    const uint8_t *pk_ptr);  // returns 1/0/negative, see above
```

## Julia usage

See `src/verification/signing.jl` in the Axiom.jl root — it loads this
library via `Libdl`, exposes `generate_hybrid_keypair()`,
`hybrid_sign(content, keys)`, and `hybrid_verify(content, sig, pubkeys)`,
and degrades gracefully (clear error, not a crash) when the shared library
has not been built.
