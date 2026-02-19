//! Axiom Core - High-performance Rust backend for Axiom.jl
//!
//! This crate provides optimized implementations of neural network operations
//! that can be called from Julia via FFI.

pub mod ffi;
pub mod ops;
pub mod tensor;

use std::env;
use std::ffi::CString;
use std::sync::Once;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

static AXIOM_INIT: Once = Once::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ThreadPolicy {
    Auto,
    Physical,
    Logical,
    Balanced,
}

fn parse_thread_policy(raw: Option<String>) -> ThreadPolicy {
    if let Some(value) = raw {
        let normalized = value.trim().to_ascii_lowercase();
        return match normalized.as_str() {
            "auto" | "" => ThreadPolicy::Auto,
            "physical" => ThreadPolicy::Physical,
            "logical" => ThreadPolicy::Logical,
            "balanced" => ThreadPolicy::Balanced,
            _ => {
                log::warn!(
                    "Unknown AXIOM_RUST_THREAD_POLICY='{}'; using 'auto'. Supported: auto|physical|logical|balanced",
                    value
                );
                ThreadPolicy::Auto
            }
        };
    }
    ThreadPolicy::Auto
}

fn parse_positive_env_usize(key: &str) -> Option<usize> {
    match env::var(key) {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return None;
            }
            match trimmed.parse::<usize>() {
                Ok(value) if value > 0 => Some(value),
                _ => {
                    log::warn!(
                        "Ignoring invalid {}='{}'; expected positive integer",
                        key,
                        raw
                    );
                    None
                }
            }
        }
        Err(_) => None,
    }
}

fn normalize_core_counts(logical_raw: usize, physical_raw: usize) -> (usize, usize) {
    let logical = logical_raw.max(1);
    let physical = if physical_raw == 0 {
        logical
    } else {
        physical_raw.min(logical).max(1)
    };
    (logical, physical)
}

fn choose_rayon_threads(
    logical_raw: usize,
    physical_raw: usize,
    explicit: Option<usize>,
    policy: ThreadPolicy,
) -> usize {
    let (logical, physical) = normalize_core_counts(logical_raw, physical_raw);
    if let Some(requested) = explicit {
        return requested.clamp(1, logical);
    }

    match policy {
        // HT-aware default: prefer physical cores when SMT is likely present.
        ThreadPolicy::Auto => {
            if logical >= physical.saturating_mul(2) {
                physical
            } else {
                logical
            }
        }
        ThreadPolicy::Physical => physical,
        ThreadPolicy::Logical => logical,
        ThreadPolicy::Balanced => (logical + physical).div_ceil(2).clamp(1, logical),
    }
}

/// Get library version as C string (for FFI)
#[no_mangle]
pub extern "C" fn axiom_version() -> *const libc::c_char {
    let version = CString::new(VERSION).unwrap();
    version.into_raw()
}

/// Initialize the library
#[no_mangle]
pub extern "C" fn axiom_init() {
    AXIOM_INIT.call_once(|| {
        let _ = env_logger::try_init();

        let logical_raw = num_cpus::get();
        let physical_raw = num_cpus::get_physical();
        let policy = parse_thread_policy(env::var("AXIOM_RUST_THREAD_POLICY").ok());
        let explicit_threads = parse_positive_env_usize("AXIOM_RUST_NUM_THREADS")
            .or_else(|| parse_positive_env_usize("RAYON_NUM_THREADS"));
        let selected_threads =
            choose_rayon_threads(logical_raw, physical_raw, explicit_threads, policy);

        if let Err(err) = rayon::ThreadPoolBuilder::new()
            .num_threads(selected_threads)
            .build_global()
        {
            log::debug!("Rayon thread pool already configured: {}", err);
        }

        let (logical, physical) = normalize_core_counts(logical_raw, physical_raw);
        log::info!(
            "Axiom Core {} initialized (logical_cores={}, physical_cores={}, rayon_threads={}, policy={:?}, explicit_threads={:?})",
            VERSION,
            logical,
            physical,
            selected_threads,
            policy,
            explicit_threads
        );
    });
}

#[cfg(test)]
mod tests {
    use super::{choose_rayon_threads, parse_thread_policy, ThreadPolicy};

    #[test]
    fn auto_policy_prefers_physical_when_smt_is_present() {
        let chosen = choose_rayon_threads(16, 8, None, ThreadPolicy::Auto);
        assert_eq!(chosen, 8);
    }

    #[test]
    fn auto_policy_uses_logical_without_smt_gap() {
        let chosen = choose_rayon_threads(12, 12, None, ThreadPolicy::Auto);
        assert_eq!(chosen, 12);
    }

    #[test]
    fn explicit_threads_override_and_clamp_to_logical() {
        let chosen = choose_rayon_threads(32, 16, Some(96), ThreadPolicy::Physical);
        assert_eq!(chosen, 32);
    }

    #[test]
    fn balanced_policy_splits_between_physical_and_logical() {
        let chosen = choose_rayon_threads(16, 8, None, ThreadPolicy::Balanced);
        assert_eq!(chosen, 12);
    }

    #[test]
    fn invalid_policy_falls_back_to_auto() {
        let policy = parse_thread_policy(Some("unsupported".to_string()));
        assert_eq!(policy, ThreadPolicy::Auto);
    }
}
