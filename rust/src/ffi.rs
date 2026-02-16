//! Foreign Function Interface for Julia
//!
//! These functions are called from Julia via ccall.

use crate::ops::{matmul, activations, conv, pool, norm};
use crate::tensor::{tensor_from_ptr, tensor_to_ptr, Tensor};
use ndarray::{ArrayD, Array2, Array4, IxDyn, ArrayView2, ArrayView4};
use std::ffi::{CStr, CString};
use std::process::{Command, Stdio};
use std::slice;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{env, thread};

// ============================================================================
// Matrix Operations
// ============================================================================

/// Matrix multiplication: C = A @ B
#[no_mangle]
pub unsafe extern "C" fn axiom_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: libc::size_t,
    k: libc::size_t,
    n: libc::size_t,
) {
    let a_slice = slice::from_raw_parts(a_ptr, m * k);
    let b_slice = slice::from_raw_parts(b_ptr, k * n);

    let a = ArrayView2::from_shape((m, k), a_slice)
        .expect("axiom_matmul: A shape mismatch");
    let b = ArrayView2::from_shape((k, n), b_slice)
        .expect("axiom_matmul: B shape mismatch");

    let c = matmul::matmul_parallel(a, b);

    let c_slice = slice::from_raw_parts_mut(c_ptr, m * n);
    c_slice.copy_from_slice(c.as_slice().expect("axiom_matmul: output not contiguous"));
}

// ============================================================================
// Activation Functions
// ============================================================================

/// ReLU activation
#[no_mangle]
pub unsafe extern "C" fn axiom_relu(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec())
        .expect("axiom_relu: input shape mismatch");

    let y = activations::relu(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("axiom_relu: output not contiguous"));
}

/// Sigmoid activation
#[no_mangle]
pub unsafe extern "C" fn axiom_sigmoid(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = activations::sigmoid(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// Softmax activation
#[no_mangle]
pub unsafe extern "C" fn axiom_softmax(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    batch_size: libc::size_t,
    num_classes: libc::size_t,
) {
    let n = batch_size * num_classes;
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[batch_size, num_classes]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = activations::softmax(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// GELU activation
#[no_mangle]
pub unsafe extern "C" fn axiom_gelu(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = activations::gelu(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// Tanh activation
#[no_mangle]
pub unsafe extern "C" fn axiom_tanh(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = activations::tanh_activation(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// Leaky ReLU activation
#[no_mangle]
pub unsafe extern "C" fn axiom_leaky_relu(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
    alpha: f32,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = activations::leaky_relu(&x, alpha);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// ELU activation
#[no_mangle]
pub unsafe extern "C" fn axiom_elu(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
    alpha: f32,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = activations::elu(&x, alpha);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// SELU activation
#[no_mangle]
pub unsafe extern "C" fn axiom_selu(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = activations::selu(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// Swish/SiLU activation
#[no_mangle]
pub unsafe extern "C" fn axiom_swish(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = activations::swish(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// Mish activation
#[no_mangle]
pub unsafe extern "C" fn axiom_mish(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = activations::mish(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// Hard Swish activation
#[no_mangle]
pub unsafe extern "C" fn axiom_hardswish(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = activations::hardswish(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// Hard Sigmoid activation
#[no_mangle]
pub unsafe extern "C" fn axiom_hardsigmoid(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = activations::hardsigmoid(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// Log Softmax activation
#[no_mangle]
pub unsafe extern "C" fn axiom_log_softmax(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    batch_size: libc::size_t,
    num_classes: libc::size_t,
) {
    let n = batch_size * num_classes;
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[batch_size, num_classes]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = activations::log_softmax(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// Softplus activation
#[no_mangle]
pub unsafe extern "C" fn axiom_softplus(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = activations::softplus(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

// ============================================================================
// Convolution
// ============================================================================

/// 2D Convolution
#[no_mangle]
pub unsafe extern "C" fn axiom_conv2d(
    input_ptr: *const f32,
    weight_ptr: *const f32,
    bias_ptr: *const f32,
    output_ptr: *mut f32,
    n: libc::size_t,
    h_in: libc::size_t,
    w_in: libc::size_t,
    c_in: libc::size_t,
    kh: libc::size_t,
    kw: libc::size_t,
    _c_in_check: libc::size_t,
    c_out: libc::size_t,
    stride_h: libc::size_t,
    stride_w: libc::size_t,
    pad_h: libc::size_t,
    pad_w: libc::size_t,
) {
    let input_size = n * h_in * w_in * c_in;
    let weight_size = kh * kw * c_in * c_out;

    let input_slice = slice::from_raw_parts(input_ptr, input_size);
    let weight_slice = slice::from_raw_parts(weight_ptr, weight_size);

    let input = ArrayView4::from_shape((n, h_in, w_in, c_in), input_slice).expect("FFI: view shape mismatch");
    let weight = ArrayView4::from_shape((kh, kw, c_in, c_out), weight_slice).expect("FFI: view shape mismatch");

    let bias = if bias_ptr.is_null() {
        None
    } else {
        Some(slice::from_raw_parts(bias_ptr, c_out))
    };

    let output = conv::conv2d(
        input,
        weight,
        bias,
        (stride_h, stride_w),
        (pad_h, pad_w),
    );

    let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
    let output_size = n * h_out * w_out * c_out;

    let output_slice = slice::from_raw_parts_mut(output_ptr, output_size);
    output_slice.copy_from_slice(output.as_slice().expect("FFI: array not contiguous"));
}

// ============================================================================
// Pooling
// ============================================================================

/// 2D Max Pooling
#[no_mangle]
pub unsafe extern "C" fn axiom_maxpool2d(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    n: libc::size_t,
    h_in: libc::size_t,
    w_in: libc::size_t,
    c: libc::size_t,
    kh: libc::size_t,
    kw: libc::size_t,
    stride_h: libc::size_t,
    stride_w: libc::size_t,
    pad_h: libc::size_t,
    pad_w: libc::size_t,
) {
    let input_size = n * h_in * w_in * c;
    let input_slice = slice::from_raw_parts(input_ptr, input_size);
    let input = ArrayView4::from_shape((n, h_in, w_in, c), input_slice).expect("FFI: view shape mismatch");

    let output = pool::maxpool2d(
        input,
        (kh, kw),
        (stride_h, stride_w),
        (pad_h, pad_w),
    );

    let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
    let output_size = n * h_out * w_out * c;

    let output_slice = slice::from_raw_parts_mut(output_ptr, output_size);
    output_slice.copy_from_slice(output.as_slice().expect("FFI: array not contiguous"));
}

// ============================================================================
// SMT Solver Runner
// ============================================================================

/// Check if containerized SMT execution is enabled
fn use_containerized_smt() -> bool {
    env::var("AXIOM_SMT_RUNNER").unwrap_or_default() == "container"
}

/// Get container image for SMT runner
fn get_smt_container_image() -> String {
    env::var("AXIOM_SMT_CONTAINER_IMAGE")
        .unwrap_or_else(|_| "axiom-smt-runner:latest".to_string())
}

/// Get container runtime (podman, svalinn, docker)
fn get_container_runtime() -> String {
    env::var("AXIOM_CONTAINER_RUNTIME")
        .unwrap_or_else(|_| "podman".to_string())
}

/// Run SMT solver in container
unsafe fn run_smt_containerized(
    solver_kind: &str,
    script_path: &std::path::Path,
    timeout_ms: u64,
) -> Result<String, String> {
    let runtime = get_container_runtime();
    let image = get_smt_container_image();

    let mut cmd = Command::new(&runtime);
    cmd.arg("run")
        .arg("--rm")
        .arg("--read-only")
        .arg("--security-opt=no-new-privileges")
        .arg("--cap-drop=ALL")
        .arg("--network=none")
        .arg("--memory=2g")
        .arg("--cpus=2")
        .arg(format!("-v={}:/tmp/query.smt2:ro", script_path.display()))
        .arg(&image)
        .arg(solver_kind)
        .arg("/tmp/query.smt2")
        .arg(timeout_ms.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    match cmd.spawn() {
        Ok(mut child) => {
            let timeout = Duration::from_millis(timeout_ms);
            let start = Instant::now();
            let mut timed_out = false;

            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) => {
                        if start.elapsed() >= timeout {
                            timed_out = true;
                            let _ = child.kill();
                            let _ = child.wait();
                            break;
                        }
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            }

            if timed_out {
                Ok("timeout".to_string())
            } else {
                match child.wait_with_output() {
                    Ok(out) => {
                        let mut text = String::from_utf8_lossy(&out.stdout).to_string();
                        if !out.stderr.is_empty() {
                            if !text.ends_with('\n') && !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(&String::from_utf8_lossy(&out.stderr));
                        }
                        Ok(text)
                    }
                    Err(e) => Err(format!("Container execution failed: {}", e)),
                }
            }
        }
        Err(e) => Err(format!("Failed to spawn container: {}", e)),
    }
}

/// Run SMT solver directly (without container)
unsafe fn run_smt_direct(
    solver_kind: &str,
    solver_path: &str,
    script_path: &std::path::Path,
    timeout_ms: u64,
) -> Result<String, String> {
    let timeout = Duration::from_millis(timeout_ms);
    let timeout_sec = (timeout_ms / 1000) as u64;

    let mut cmd = Command::new(solver_path);
    match solver_kind {
        "z3" => {
            cmd.arg(format!("-T:{}", timeout_sec))
                .arg(script_path);
        }
        "cvc5" => {
            cmd.arg(format!("--tlimit={}", timeout_ms))
                .arg(script_path);
        }
        "yices" => {
            cmd.arg(format!("--timeout={}", timeout_sec))
                .arg(script_path);
        }
        _ => {
            cmd.arg(script_path);
        }
    }

    let mut child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(child) => child,
        Err(e) => return Err(format!("Failed to spawn solver: {}", e)),
    };

    let start = Instant::now();
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if start.elapsed() >= timeout {
                    timed_out = true;
                    let _ = child.kill();
                    let _ = child.wait();
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(_) => break,
        }
    }

    if timed_out {
        Ok("timeout".to_string())
    } else {
        match child.wait_with_output() {
            Ok(out) => {
                let mut text = String::from_utf8_lossy(&out.stdout).to_string();
                if !out.stderr.is_empty() {
                    if !text.ends_with('\n') && !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&String::from_utf8_lossy(&out.stderr));
                }
                Ok(text)
            }
            Err(e) => Err(format!("Solver execution failed: {}", e)),
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn axiom_smt_run(
    solver_kind: *const libc::c_char,
    solver_path: *const libc::c_char,
    script: *const libc::c_char,
    timeout_ms: libc::c_uint,
) -> *mut libc::c_char {
    if solver_path.is_null() || script.is_null() {
        return CString::new("error: missing solver path or script")
            .expect("FFI: CString conversion failed")
            .into_raw();
    }

    let kind = if solver_kind.is_null() {
        ""
    } else {
        CStr::from_ptr(solver_kind).to_str().unwrap_or("")
    };
    let path = CStr::from_ptr(solver_path).to_str().unwrap_or("");
    let script_str = CStr::from_ptr(script).to_str().unwrap_or("");

    // Write script to temporary file
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let filename = format!("axiom_smt_{}_{}.smt2", std::process::id(), nanos);
    let script_path = env::temp_dir().join(filename);

    if let Err(e) = std::fs::write(&script_path, script_str) {
        return CString::new(format!("error: {}", e)).expect("FFI: CString conversion failed").into_raw();
    }

    // Choose execution method: containerized or direct
    let output = if use_containerized_smt() {
        run_smt_containerized(kind, &script_path, timeout_ms as u64)
    } else {
        run_smt_direct(kind, path, &script_path, timeout_ms as u64)
    };

    // Clean up temporary file
    let _ = std::fs::remove_file(&script_path);

    // Return result
    match output {
        Ok(text) => CString::new(text)
            .unwrap_or_else(|_| CString::new("error").unwrap())
            .into_raw(),
        Err(e) => CString::new(format!("error: {}", e))
            .expect("FFI: CString conversion failed")
            .into_raw(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn axiom_smt_free(ptr: *mut libc::c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}

/// Global Average Pooling
#[no_mangle]
pub unsafe extern "C" fn axiom_global_avgpool2d(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    n: libc::size_t,
    h: libc::size_t,
    w: libc::size_t,
    c: libc::size_t,
) {
    let input_size = n * h * w * c;
    let input_slice = slice::from_raw_parts(input_ptr, input_size);
    let input = ArrayView4::from_shape((n, h, w, c), input_slice).expect("FFI: view shape mismatch");

    let output = pool::global_avgpool2d(input);

    let output_size = n * c;
    let output_slice = slice::from_raw_parts_mut(output_ptr, output_size);
    output_slice.copy_from_slice(output.as_slice().expect("FFI: array not contiguous"));
}

// ============================================================================
// Normalization
// ============================================================================

/// Batch Normalization
#[no_mangle]
pub unsafe extern "C" fn axiom_batchnorm(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    gamma_ptr: *const f32,
    beta_ptr: *const f32,
    running_mean_ptr: *mut f32,
    running_var_ptr: *mut f32,
    n_elements: libc::size_t,
    n_features: libc::size_t,
    eps: f32,
    training: libc::c_int,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n_elements);
    let gamma = slice::from_raw_parts(gamma_ptr, n_features);
    let beta = slice::from_raw_parts(beta_ptr, n_features);
    let running_mean = slice::from_raw_parts_mut(running_mean_ptr, n_features);
    let running_var = slice::from_raw_parts_mut(running_var_ptr, n_features);

    let batch_size = n_elements / n_features;
    let x = ArrayD::from_shape_vec(IxDyn(&[batch_size, n_features]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = norm::batchnorm(
        &x,
        gamma,
        beta,
        running_mean,
        running_var,
        eps,
        0.1,
        training != 0,
    );

    let y_slice = slice::from_raw_parts_mut(y_ptr, n_elements);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// Layer Normalization
#[no_mangle]
pub unsafe extern "C" fn axiom_layernorm(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    gamma_ptr: *const f32,
    beta_ptr: *const f32,
    batch_size: libc::size_t,
    hidden_size: libc::size_t,
    eps: f32,
) {
    let n_elements = batch_size * hidden_size;
    let x_slice = slice::from_raw_parts(x_ptr, n_elements);
    let gamma_slice = slice::from_raw_parts(gamma_ptr, hidden_size);
    let beta_slice = slice::from_raw_parts(beta_ptr, hidden_size);

    let x = ArrayD::from_shape_vec(IxDyn(&[batch_size, hidden_size]), x_slice.to_vec()).expect("FFI: shape mismatch");
    let gamma = ArrayD::from_shape_vec(IxDyn(&[hidden_size]), gamma_slice.to_vec()).expect("FFI: shape mismatch");
    let beta = ArrayD::from_shape_vec(IxDyn(&[hidden_size]), beta_slice.to_vec()).expect("FFI: shape mismatch");

    let y = norm::layernorm(&x, &gamma, &beta, &[hidden_size], eps);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n_elements);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}

/// RMS Normalization
#[no_mangle]
pub unsafe extern "C" fn axiom_rmsnorm(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    weight_ptr: *const f32,
    batch_size: libc::size_t,
    hidden_size: libc::size_t,
    eps: f32,
) {
    let n_elements = batch_size * hidden_size;
    let x_slice = slice::from_raw_parts(x_ptr, n_elements);
    let weight = slice::from_raw_parts(weight_ptr, hidden_size);

    let x = ArrayD::from_shape_vec(IxDyn(&[batch_size, hidden_size]), x_slice.to_vec()).expect("FFI: shape mismatch");

    let y = norm::rmsnorm(&x, weight, eps);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n_elements);
    y_slice.copy_from_slice(y.as_slice().expect("FFI: array not contiguous"));
}
