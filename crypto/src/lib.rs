// SPDX-License-Identifier: MPL-2.0
//! axiom_crypto — Hybrid Ed448 + Dilithium5 (ML-DSA-87) certificate signing.
//!
//! # Estate Trustfile scheme
//!
//! Axiom.jl certificates use a *hybrid* signature: a classical Ed448 (EdDSA
//! over Curve448, RFC 8032) signature AND a post-quantum Dilithium5
//! (ML-DSA-87, FIPS 204) signature over the same message. Both must verify
//! for the hybrid signature to be considered valid — this gives security
//! against a classical break OR a quantum break, but not against a break of
//! both simultaneously (defence in depth, not OR-security downgrade).
//!
//! Ed448 uses the system libcrypto (OpenSSL 3.x) via the vetted `openssl`
//! crate. Dilithium5 uses the vetted `pqcrypto-dilithium` crate (PQClean
//! reference implementation). No hand-rolled cryptography is implemented
//! here — this crate is a thin C-ABI shim over two audited primitives.
//!
//! # C ABI convention
//!
//! Every exported function follows a **caller-allocates-buffer + written-length**
//! convention:
//!
//! - Fixed-size outputs (public keys, secret keys, Ed448 signatures) are
//!   written into a caller-supplied buffer whose required size is exposed via
//!   `axiom_crypto_*_len()` constant-getter functions. The caller must
//!   allocate at least that many bytes before calling.
//! - Variable-size outputs (Dilithium5 signatures, which PQClean's detached
//!   API bounds by `axiom_crypto_dilithium5_signature_maxlen()`) are written
//!   into a caller-supplied buffer of at least that many bytes; the actual
//!   written length is returned via an out-parameter `*mut usize`.
//! - All functions return an `i32` status code: `0` = success, negative =
//!   failure (see `AXIOM_CRYPTO_ERR_*` constants below). Verify functions
//!   return `1` for a valid signature, `0` for an invalid signature, and a
//!   negative code only for a genuine call error (bad pointer / bad length),
//!   so callers can distinguish "verification ran and failed" from "verification
//!   could not run".
//! - No heap-allocate-and-return-pointer + free-fn pairs are used anywhere in
//!   this ABI: callers own all buffers, so there is no `axiom_crypto_free`
//!   to forget. This keeps the FFI boundary simple for the Julia `ccall`
//!   caller, which cannot easily register a finalizer for a foreign pointer.
//!
//! See `crypto/README.md` for the full ABI reference and worked Julia
//! `ccall` examples.

use openssl::pkey::{Id, PKey, Private, Public};
use openssl::sign::{Signer, Verifier};
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{
    DetachedSignature as PqDetachedSignature, PublicKey as PqPublicKey, SecretKey as PqSecretKey,
};
use std::slice;

// ============================================================================
// Status codes
// ============================================================================

/// Success.
pub const AXIOM_CRYPTO_OK: i32 = 0;
/// A required pointer argument was null.
pub const AXIOM_CRYPTO_ERR_NULL_PTR: i32 = -1;
/// A supplied buffer/slice had the wrong length for the operation.
pub const AXIOM_CRYPTO_ERR_BAD_LENGTH: i32 = -2;
/// The underlying cryptographic library (OpenSSL or PQClean) reported an error
/// (e.g. malformed key bytes) that is not itself a signature-verification failure.
pub const AXIOM_CRYPTO_ERR_CRYPTO_FAILURE: i32 = -3;

// ============================================================================
// Fixed-size length constants (Ed448)
// ============================================================================

/// Ed448 raw public key length in bytes (57).
#[no_mangle]
pub extern "C" fn axiom_crypto_ed448_public_key_len() -> usize {
    57
}

/// Ed448 raw private key length in bytes (57).
#[no_mangle]
pub extern "C" fn axiom_crypto_ed448_secret_key_len() -> usize {
    57
}

/// Ed448 (pure EdDSA, no context) signature length in bytes (114).
#[no_mangle]
pub extern "C" fn axiom_crypto_ed448_signature_len() -> usize {
    114
}

// ============================================================================
// Fixed/bounded-size length constants (Dilithium5 / ML-DSA-87)
// ============================================================================

/// Dilithium5 public key length in bytes (2592).
#[no_mangle]
pub extern "C" fn axiom_crypto_dilithium5_public_key_len() -> usize {
    dilithium5::public_key_bytes()
}

/// Dilithium5 secret key length in bytes (4896).
#[no_mangle]
pub extern "C" fn axiom_crypto_dilithium5_secret_key_len() -> usize {
    dilithium5::secret_key_bytes()
}

/// Upper bound on a Dilithium5 detached-signature length in bytes (4627, per
/// the PQClean reference implementation's `CRYPTO_BYTES` for dilithium5 —
/// note this is larger than the "signature bytes" figure quoted in some FIPS
/// 204 summaries because PQClean's clean backend reserves headroom; the
/// actual written length is always returned via `sig_len_out`). The actual
/// signature length is deterministic per key but the C ABI treats it as
/// bounded-variable-length for forward compatibility; callers must allocate
/// a buffer of at least this many bytes and read back the actual length
/// written by `axiom_crypto_dilithium5_sign`.
#[no_mangle]
pub extern "C" fn axiom_crypto_dilithium5_signature_maxlen() -> usize {
    dilithium5::signature_bytes()
}

// ============================================================================
// Helpers
// ============================================================================

/// SAFETY: converts a caller-supplied `(ptr, len)` pair into a `&[u8]`.
/// The caller must guarantee `ptr` is valid for reads of `len` bytes and
/// that the memory is not mutated concurrently for the duration of the
/// call. Returns `None` if `ptr` is null while `len > 0`.
unsafe fn slice_from_raw<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        return Some(&[]);
    }
    if ptr.is_null() {
        return None;
    }
    // SAFETY: forwarded from the caller's contract documented above; the
    // exported extern "C" fn that calls this helper re-states the same
    // contract at its own call site.
    Some(unsafe { slice::from_raw_parts(ptr, len) })
}

/// SAFETY: converts a caller-supplied `(ptr, len)` pair into a `&mut [u8]`
/// output buffer. Same contract as `slice_from_raw` plus exclusive-write
/// access for the duration of the call.
unsafe fn slice_from_raw_mut<'a>(ptr: *mut u8, len: usize) -> Option<&'a mut [u8]> {
    if len == 0 {
        return Some(&mut []);
    }
    if ptr.is_null() {
        return None;
    }
    // SAFETY: forwarded from the caller's contract documented above.
    Some(unsafe { slice::from_raw_parts_mut(ptr, len) })
}

// ============================================================================
// Ed448 (classical) — via system libcrypto (OpenSSL) through the `openssl` crate
// ============================================================================

/// Generate an Ed448 keypair.
///
/// # Parameters
/// - `pk_out`: buffer of at least `axiom_crypto_ed448_public_key_len()` bytes.
/// - `sk_out`: buffer of at least `axiom_crypto_ed448_secret_key_len()` bytes.
///
/// # Safety
/// SAFETY: `pk_out` and `sk_out` must each be valid, non-null, non-overlapping
/// pointers to writable buffers of at least their respective required
/// lengths (see above); the caller retains ownership of both buffers and no
/// pointer is retained past the call.
#[no_mangle]
pub unsafe extern "C" fn axiom_crypto_ed448_keypair(pk_out: *mut u8, sk_out: *mut u8) -> i32 {
    // SAFETY: contract documented on the exported fn; pointers are only
    // dereferenced for the duration of this call and not stored.
    let pk_buf = match unsafe { slice_from_raw_mut(pk_out, axiom_crypto_ed448_public_key_len()) } {
        Some(b) => b,
        None => return AXIOM_CRYPTO_ERR_NULL_PTR,
    };
    // SAFETY: see above.
    let sk_buf = match unsafe { slice_from_raw_mut(sk_out, axiom_crypto_ed448_secret_key_len()) } {
        Some(b) => b,
        None => return AXIOM_CRYPTO_ERR_NULL_PTR,
    };

    let key = match PKey::generate_ed448() {
        Ok(k) => k,
        Err(_) => return AXIOM_CRYPTO_ERR_CRYPTO_FAILURE,
    };

    let raw_public = match key.raw_public_key() {
        Ok(b) => b,
        Err(_) => return AXIOM_CRYPTO_ERR_CRYPTO_FAILURE,
    };
    let raw_private = match key.raw_private_key() {
        Ok(b) => b,
        Err(_) => return AXIOM_CRYPTO_ERR_CRYPTO_FAILURE,
    };

    if raw_public.len() != pk_buf.len() || raw_private.len() != sk_buf.len() {
        return AXIOM_CRYPTO_ERR_CRYPTO_FAILURE;
    }

    pk_buf.copy_from_slice(&raw_public);
    sk_buf.copy_from_slice(&raw_private);
    AXIOM_CRYPTO_OK
}

/// Sign `msg` with an Ed448 raw secret key.
///
/// # Parameters
/// - `msg_ptr`/`msg_len`: message to sign.
/// - `sk_ptr`: raw Ed448 secret key, `axiom_crypto_ed448_secret_key_len()` bytes.
/// - `sig_out`: buffer of at least `axiom_crypto_ed448_signature_len()` bytes.
///
/// # Safety
/// SAFETY: `msg_ptr` must be valid for reads of `msg_len` bytes (or null iff
/// `msg_len == 0`); `sk_ptr` must be valid for reads of
/// `axiom_crypto_ed448_secret_key_len()` bytes; `sig_out` must be valid for
/// writes of `axiom_crypto_ed448_signature_len()` bytes. No pointer is
/// retained past the call.
#[no_mangle]
pub unsafe extern "C" fn axiom_crypto_ed448_sign(
    msg_ptr: *const u8,
    msg_len: usize,
    sk_ptr: *const u8,
    sig_out: *mut u8,
) -> i32 {
    // SAFETY: contract documented on the exported fn.
    let msg = match unsafe { slice_from_raw(msg_ptr, msg_len) } {
        Some(b) => b,
        None => return AXIOM_CRYPTO_ERR_NULL_PTR,
    };
    // SAFETY: see above.
    let sk_bytes = match unsafe { slice_from_raw(sk_ptr, axiom_crypto_ed448_secret_key_len()) } {
        Some(b) => b,
        None => return AXIOM_CRYPTO_ERR_NULL_PTR,
    };
    // SAFETY: see above.
    let sig_buf = match unsafe { slice_from_raw_mut(sig_out, axiom_crypto_ed448_signature_len()) } {
        Some(b) => b,
        None => return AXIOM_CRYPTO_ERR_NULL_PTR,
    };

    let key: PKey<Private> = match PKey::private_key_from_raw_bytes(sk_bytes, Id::ED448) {
        Ok(k) => k,
        Err(_) => return AXIOM_CRYPTO_ERR_CRYPTO_FAILURE,
    };

    // Ed448/Ed25519 are one-shot-only ("PureEdDSA"): OpenSSL forbids the
    // streaming `update()` API for these key types, hence `new_without_digest`
    // + `sign_oneshot_to_vec` rather than `Signer::new` + `update`.
    let mut signer = match Signer::new_without_digest(&key) {
        Ok(s) => s,
        Err(_) => return AXIOM_CRYPTO_ERR_CRYPTO_FAILURE,
    };
    let sig = match signer.sign_oneshot_to_vec(msg) {
        Ok(s) => s,
        Err(_) => return AXIOM_CRYPTO_ERR_CRYPTO_FAILURE,
    };

    if sig.len() != sig_buf.len() {
        return AXIOM_CRYPTO_ERR_CRYPTO_FAILURE;
    }
    sig_buf.copy_from_slice(&sig);
    AXIOM_CRYPTO_OK
}

/// Verify an Ed448 signature.
///
/// Returns `1` if the signature is valid, `0` if it is invalid, or a
/// negative `AXIOM_CRYPTO_ERR_*` code if the call itself could not be
/// carried out (bad pointer / malformed key bytes).
///
/// # Safety
/// SAFETY: `msg_ptr` must be valid for reads of `msg_len` bytes (or null iff
/// `msg_len == 0`); `sig_ptr` must be valid for reads of
/// `axiom_crypto_ed448_signature_len()` bytes; `pk_ptr` must be valid for
/// reads of `axiom_crypto_ed448_public_key_len()` bytes. No pointer is
/// retained past the call.
#[no_mangle]
pub unsafe extern "C" fn axiom_crypto_ed448_verify(
    msg_ptr: *const u8,
    msg_len: usize,
    sig_ptr: *const u8,
    pk_ptr: *const u8,
) -> i32 {
    // SAFETY: contract documented on the exported fn.
    let msg = match unsafe { slice_from_raw(msg_ptr, msg_len) } {
        Some(b) => b,
        None => return AXIOM_CRYPTO_ERR_NULL_PTR,
    };
    // SAFETY: see above.
    let sig = match unsafe { slice_from_raw(sig_ptr, axiom_crypto_ed448_signature_len()) } {
        Some(b) => b,
        None => return AXIOM_CRYPTO_ERR_NULL_PTR,
    };
    // SAFETY: see above.
    let pk_bytes = match unsafe { slice_from_raw(pk_ptr, axiom_crypto_ed448_public_key_len()) } {
        Some(b) => b,
        None => return AXIOM_CRYPTO_ERR_NULL_PTR,
    };

    let key: PKey<Public> = match PKey::public_key_from_raw_bytes(pk_bytes, Id::ED448) {
        Ok(k) => k,
        Err(_) => return AXIOM_CRYPTO_ERR_CRYPTO_FAILURE,
    };

    let mut verifier = match Verifier::new_without_digest(&key) {
        Ok(v) => v,
        Err(_) => return AXIOM_CRYPTO_ERR_CRYPTO_FAILURE,
    };

    match verifier.verify_oneshot(sig, msg) {
        Ok(true) => 1,
        Ok(false) => 0,
        Err(_) => AXIOM_CRYPTO_ERR_CRYPTO_FAILURE,
    }
}

// ============================================================================
// Dilithium5 / ML-DSA-87 (post-quantum) — via `pqcrypto-dilithium` (PQClean)
// ============================================================================

/// Generate a Dilithium5 keypair.
///
/// # Parameters
/// - `pk_out`: buffer of at least `axiom_crypto_dilithium5_public_key_len()` bytes.
/// - `sk_out`: buffer of at least `axiom_crypto_dilithium5_secret_key_len()` bytes.
///
/// # Safety
/// SAFETY: `pk_out` and `sk_out` must each be valid, non-null, non-overlapping
/// pointers to writable buffers of at least their respective required
/// lengths; the caller retains ownership and no pointer is retained past the
/// call.
#[no_mangle]
pub unsafe extern "C" fn axiom_crypto_dilithium5_keypair(pk_out: *mut u8, sk_out: *mut u8) -> i32 {
    // SAFETY: contract documented on the exported fn.
    let pk_buf =
        match unsafe { slice_from_raw_mut(pk_out, axiom_crypto_dilithium5_public_key_len()) } {
            Some(b) => b,
            None => return AXIOM_CRYPTO_ERR_NULL_PTR,
        };
    // SAFETY: see above.
    let sk_buf =
        match unsafe { slice_from_raw_mut(sk_out, axiom_crypto_dilithium5_secret_key_len()) } {
            Some(b) => b,
            None => return AXIOM_CRYPTO_ERR_NULL_PTR,
        };

    let (pk, sk) = dilithium5::keypair();
    pk_buf.copy_from_slice(pk.as_bytes());
    sk_buf.copy_from_slice(sk.as_bytes());
    AXIOM_CRYPTO_OK
}

/// Sign `msg` with a Dilithium5 secret key, producing a detached signature.
///
/// # Parameters
/// - `msg_ptr`/`msg_len`: message to sign.
/// - `sk_ptr`: Dilithium5 secret key, `axiom_crypto_dilithium5_secret_key_len()` bytes.
/// - `sig_out`: buffer of at least `axiom_crypto_dilithium5_signature_maxlen()` bytes.
/// - `sig_len_out`: out-parameter receiving the actual signature length written.
///
/// # Safety
/// SAFETY: `msg_ptr` must be valid for reads of `msg_len` bytes (or null iff
/// `msg_len == 0`); `sk_ptr` must be valid for reads of
/// `axiom_crypto_dilithium5_secret_key_len()` bytes; `sig_out` must be valid
/// for writes of `axiom_crypto_dilithium5_signature_maxlen()` bytes;
/// `sig_len_out` must be a valid non-null pointer to a writable `usize`. No
/// pointer is retained past the call.
#[no_mangle]
pub unsafe extern "C" fn axiom_crypto_dilithium5_sign(
    msg_ptr: *const u8,
    msg_len: usize,
    sk_ptr: *const u8,
    sig_out: *mut u8,
    sig_len_out: *mut usize,
) -> i32 {
    if sig_len_out.is_null() {
        return AXIOM_CRYPTO_ERR_NULL_PTR;
    }
    // SAFETY: contract documented on the exported fn.
    let msg = match unsafe { slice_from_raw(msg_ptr, msg_len) } {
        Some(b) => b,
        None => return AXIOM_CRYPTO_ERR_NULL_PTR,
    };
    // SAFETY: see above.
    let sk_bytes = match unsafe { slice_from_raw(sk_ptr, axiom_crypto_dilithium5_secret_key_len()) }
    {
        Some(b) => b,
        None => return AXIOM_CRYPTO_ERR_NULL_PTR,
    };
    // SAFETY: see above.
    let sig_buf =
        match unsafe { slice_from_raw_mut(sig_out, axiom_crypto_dilithium5_signature_maxlen()) } {
            Some(b) => b,
            None => return AXIOM_CRYPTO_ERR_NULL_PTR,
        };

    let sk = match dilithium5::SecretKey::from_bytes(sk_bytes) {
        Ok(k) => k,
        Err(_) => return AXIOM_CRYPTO_ERR_CRYPTO_FAILURE,
    };

    let sig = dilithium5::detached_sign(msg, &sk);
    let sig_bytes = sig.as_bytes();
    if sig_bytes.len() > sig_buf.len() {
        return AXIOM_CRYPTO_ERR_BAD_LENGTH;
    }
    sig_buf[..sig_bytes.len()].copy_from_slice(sig_bytes);
    // SAFETY: `sig_len_out` was checked non-null above; it is a valid
    // writable `usize` per the caller contract documented on this fn.
    unsafe {
        *sig_len_out = sig_bytes.len();
    }
    AXIOM_CRYPTO_OK
}

/// Verify a Dilithium5 detached signature.
///
/// Returns `1` if the signature is valid, `0` if it is invalid, or a
/// negative `AXIOM_CRYPTO_ERR_*` code if the call itself could not be
/// carried out (bad pointer / malformed key or signature bytes).
///
/// # Safety
/// SAFETY: `msg_ptr` must be valid for reads of `msg_len` bytes (or null iff
/// `msg_len == 0`); `sig_ptr` must be valid for reads of `sig_len` bytes;
/// `pk_ptr` must be valid for reads of
/// `axiom_crypto_dilithium5_public_key_len()` bytes. No pointer is retained
/// past the call.
#[no_mangle]
pub unsafe extern "C" fn axiom_crypto_dilithium5_verify(
    msg_ptr: *const u8,
    msg_len: usize,
    sig_ptr: *const u8,
    sig_len: usize,
    pk_ptr: *const u8,
) -> i32 {
    // SAFETY: contract documented on the exported fn.
    let msg = match unsafe { slice_from_raw(msg_ptr, msg_len) } {
        Some(b) => b,
        None => return AXIOM_CRYPTO_ERR_NULL_PTR,
    };
    // SAFETY: see above.
    let sig_bytes = match unsafe { slice_from_raw(sig_ptr, sig_len) } {
        Some(b) => b,
        None => return AXIOM_CRYPTO_ERR_NULL_PTR,
    };
    // SAFETY: see above.
    let pk_bytes = match unsafe { slice_from_raw(pk_ptr, axiom_crypto_dilithium5_public_key_len()) }
    {
        Some(b) => b,
        None => return AXIOM_CRYPTO_ERR_NULL_PTR,
    };

    let pk = match dilithium5::PublicKey::from_bytes(pk_bytes) {
        Ok(k) => k,
        Err(_) => return AXIOM_CRYPTO_ERR_CRYPTO_FAILURE,
    };
    let sig = match dilithium5::DetachedSignature::from_bytes(sig_bytes) {
        Ok(s) => s,
        Err(_) => return AXIOM_CRYPTO_ERR_CRYPTO_FAILURE,
    };

    match dilithium5::verify_detached_signature(&sig, msg, &pk) {
        Ok(()) => 1,
        Err(_) => 0,
    }
}

// ============================================================================
// Tests: round-trip (keypair -> sign -> verify ok) and negative
// (wrong-key / tampered -> reject) for both algorithms.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ed448_roundtrip_ok() {
        let mut pk = vec![0u8; axiom_crypto_ed448_public_key_len()];
        let mut sk = vec![0u8; axiom_crypto_ed448_secret_key_len()];
        // SAFETY: buffers sized exactly to the required lengths above; test-only call.
        let rc = unsafe { axiom_crypto_ed448_keypair(pk.as_mut_ptr(), sk.as_mut_ptr()) };
        assert_eq!(rc, AXIOM_CRYPTO_OK);

        let msg = b"axiom certificate content digest (sha3-512 placeholder)";
        let mut sig = vec![0u8; axiom_crypto_ed448_signature_len()];
        // SAFETY: test-only call with correctly sized buffers.
        let rc = unsafe {
            axiom_crypto_ed448_sign(msg.as_ptr(), msg.len(), sk.as_ptr(), sig.as_mut_ptr())
        };
        assert_eq!(rc, AXIOM_CRYPTO_OK);

        // SAFETY: test-only call with correctly sized buffers.
        let ok = unsafe {
            axiom_crypto_ed448_verify(msg.as_ptr(), msg.len(), sig.as_ptr(), pk.as_ptr())
        };
        assert_eq!(ok, 1, "genuine Ed448 signature must verify");
    }

    #[test]
    fn ed448_tampered_message_rejected() {
        let mut pk = vec![0u8; axiom_crypto_ed448_public_key_len()];
        let mut sk = vec![0u8; axiom_crypto_ed448_secret_key_len()];
        // SAFETY: test-only call with correctly sized buffers.
        unsafe { axiom_crypto_ed448_keypair(pk.as_mut_ptr(), sk.as_mut_ptr()) };

        let msg = b"original certificate content";
        let mut sig = vec![0u8; axiom_crypto_ed448_signature_len()];
        // SAFETY: test-only call with correctly sized buffers.
        unsafe { axiom_crypto_ed448_sign(msg.as_ptr(), msg.len(), sk.as_ptr(), sig.as_mut_ptr()) };

        let tampered = b"TAMPERED certificate content";
        // SAFETY: test-only call with correctly sized buffers.
        let ok = unsafe {
            axiom_crypto_ed448_verify(tampered.as_ptr(), tampered.len(), sig.as_ptr(), pk.as_ptr())
        };
        assert_eq!(ok, 0, "tampered message must be rejected");
    }

    #[test]
    fn ed448_wrong_key_rejected() {
        let mut pk_a = vec![0u8; axiom_crypto_ed448_public_key_len()];
        let mut sk_a = vec![0u8; axiom_crypto_ed448_secret_key_len()];
        // SAFETY: test-only call with correctly sized buffers.
        unsafe { axiom_crypto_ed448_keypair(pk_a.as_mut_ptr(), sk_a.as_mut_ptr()) };

        let mut pk_b = vec![0u8; axiom_crypto_ed448_public_key_len()];
        let mut sk_b = vec![0u8; axiom_crypto_ed448_secret_key_len()];
        // SAFETY: test-only call with correctly sized buffers.
        unsafe { axiom_crypto_ed448_keypair(pk_b.as_mut_ptr(), sk_b.as_mut_ptr()) };

        let msg = b"a certificate signed by key A";
        let mut sig = vec![0u8; axiom_crypto_ed448_signature_len()];
        // SAFETY: test-only call with correctly sized buffers.
        unsafe {
            axiom_crypto_ed448_sign(msg.as_ptr(), msg.len(), sk_a.as_ptr(), sig.as_mut_ptr())
        };

        // SAFETY: test-only call with correctly sized buffers.
        let ok = unsafe {
            axiom_crypto_ed448_verify(msg.as_ptr(), msg.len(), sig.as_ptr(), pk_b.as_ptr())
        };
        assert_eq!(
            ok, 0,
            "signature verified under the wrong public key must be rejected"
        );
    }

    #[test]
    fn dilithium5_roundtrip_ok() {
        let mut pk = vec![0u8; axiom_crypto_dilithium5_public_key_len()];
        let mut sk = vec![0u8; axiom_crypto_dilithium5_secret_key_len()];
        // SAFETY: test-only call with correctly sized buffers.
        let rc = unsafe { axiom_crypto_dilithium5_keypair(pk.as_mut_ptr(), sk.as_mut_ptr()) };
        assert_eq!(rc, AXIOM_CRYPTO_OK);

        let msg = b"axiom certificate content digest (sha3-512 placeholder)";
        let mut sig = vec![0u8; axiom_crypto_dilithium5_signature_maxlen()];
        let mut sig_len: usize = 0;
        // SAFETY: test-only call with correctly sized buffers.
        let rc = unsafe {
            axiom_crypto_dilithium5_sign(
                msg.as_ptr(),
                msg.len(),
                sk.as_ptr(),
                sig.as_mut_ptr(),
                &mut sig_len as *mut usize,
            )
        };
        assert_eq!(rc, AXIOM_CRYPTO_OK);
        assert!(sig_len > 0 && sig_len <= sig.len());

        // SAFETY: test-only call with correctly sized buffers.
        let ok = unsafe {
            axiom_crypto_dilithium5_verify(
                msg.as_ptr(),
                msg.len(),
                sig.as_ptr(),
                sig_len,
                pk.as_ptr(),
            )
        };
        assert_eq!(ok, 1, "genuine Dilithium5 signature must verify");
    }

    #[test]
    fn dilithium5_tampered_message_rejected() {
        let mut pk = vec![0u8; axiom_crypto_dilithium5_public_key_len()];
        let mut sk = vec![0u8; axiom_crypto_dilithium5_secret_key_len()];
        // SAFETY: test-only call with correctly sized buffers.
        unsafe { axiom_crypto_dilithium5_keypair(pk.as_mut_ptr(), sk.as_mut_ptr()) };

        let msg = b"original certificate content";
        let mut sig = vec![0u8; axiom_crypto_dilithium5_signature_maxlen()];
        let mut sig_len: usize = 0;
        // SAFETY: test-only call with correctly sized buffers.
        unsafe {
            axiom_crypto_dilithium5_sign(
                msg.as_ptr(),
                msg.len(),
                sk.as_ptr(),
                sig.as_mut_ptr(),
                &mut sig_len as *mut usize,
            )
        };

        let tampered = b"TAMPERED certificate content";
        // SAFETY: test-only call with correctly sized buffers.
        let ok = unsafe {
            axiom_crypto_dilithium5_verify(
                tampered.as_ptr(),
                tampered.len(),
                sig.as_ptr(),
                sig_len,
                pk.as_ptr(),
            )
        };
        assert_eq!(ok, 0, "tampered message must be rejected");
    }

    #[test]
    fn dilithium5_wrong_key_rejected() {
        let mut pk_a = vec![0u8; axiom_crypto_dilithium5_public_key_len()];
        let mut sk_a = vec![0u8; axiom_crypto_dilithium5_secret_key_len()];
        // SAFETY: test-only call with correctly sized buffers.
        unsafe { axiom_crypto_dilithium5_keypair(pk_a.as_mut_ptr(), sk_a.as_mut_ptr()) };

        let mut pk_b = vec![0u8; axiom_crypto_dilithium5_public_key_len()];
        let mut sk_b = vec![0u8; axiom_crypto_dilithium5_secret_key_len()];
        // SAFETY: test-only call with correctly sized buffers.
        unsafe { axiom_crypto_dilithium5_keypair(pk_b.as_mut_ptr(), sk_b.as_mut_ptr()) };

        let msg = b"a certificate signed by key A";
        let mut sig = vec![0u8; axiom_crypto_dilithium5_signature_maxlen()];
        let mut sig_len: usize = 0;
        // SAFETY: test-only call with correctly sized buffers.
        unsafe {
            axiom_crypto_dilithium5_sign(
                msg.as_ptr(),
                msg.len(),
                sk_a.as_ptr(),
                sig.as_mut_ptr(),
                &mut sig_len as *mut usize,
            )
        };

        // SAFETY: test-only call with correctly sized buffers.
        let ok = unsafe {
            axiom_crypto_dilithium5_verify(
                msg.as_ptr(),
                msg.len(),
                sig.as_ptr(),
                sig_len,
                pk_b.as_ptr(),
            )
        };
        assert_eq!(
            ok, 0,
            "signature verified under the wrong public key must be rejected"
        );
    }

    #[test]
    fn null_pointer_calls_return_null_ptr_error_not_ub() {
        let mut sig_len: usize = 0;
        // SAFETY: intentionally passing null with nonzero len to exercise the
        // documented null-pointer rejection path (not a real dereference).
        let rc = unsafe {
            axiom_crypto_dilithium5_sign(
                std::ptr::null(),
                8,
                std::ptr::null(),
                std::ptr::null_mut(),
                &mut sig_len as *mut usize,
            )
        };
        assert_eq!(rc, AXIOM_CRYPTO_ERR_NULL_PTR);
    }
}
