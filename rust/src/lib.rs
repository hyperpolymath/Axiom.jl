//! Axiom Core - High-performance Rust backend for Axiom.jl
//!
//! This crate provides optimized implementations of neural network operations
//! that can be called from Julia via FFI.

pub mod ops;
pub mod tensor;
pub mod ffi;

use std::ffi::CString;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get library version as C string (for FFI)
#[no_mangle]
pub extern "C" fn axiom_version() -> *const libc::c_char {
    let version = CString::new(VERSION).unwrap();
    version.into_raw()
}

/// Initialize the library
#[no_mangle]
pub extern "C" fn axiom_init() {
    env_logger::init();
    log::info!("Axiom Core {} initialized", VERSION);
}
