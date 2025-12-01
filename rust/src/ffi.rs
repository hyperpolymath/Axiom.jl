//! Foreign Function Interface for Julia
//!
//! These functions are called from Julia via ccall.

use crate::ops::{matmul, activations, conv, pool, norm};
use crate::tensor::{tensor_from_ptr, tensor_to_ptr, Tensor};
use ndarray::{ArrayD, Array2, Array4, IxDyn, ArrayView2, ArrayView4};
use std::slice;

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

    let a = ArrayView2::from_shape((m, k), a_slice).unwrap();
    let b = ArrayView2::from_shape((k, n), b_slice).unwrap();

    let c = matmul::matmul_parallel(a, b);

    let c_slice = slice::from_raw_parts_mut(c_ptr, m * n);
    c_slice.copy_from_slice(c.as_slice().unwrap());
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
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).unwrap();

    let y = activations::relu(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().unwrap());
}

/// Sigmoid activation
#[no_mangle]
pub unsafe extern "C" fn axiom_sigmoid(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).unwrap();

    let y = activations::sigmoid(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().unwrap());
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
    let x = ArrayD::from_shape_vec(IxDyn(&[batch_size, num_classes]), x_slice.to_vec()).unwrap();

    let y = activations::softmax(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().unwrap());
}

/// GELU activation
#[no_mangle]
pub unsafe extern "C" fn axiom_gelu(
    x_ptr: *const f32,
    y_ptr: *mut f32,
    n: libc::size_t,
) {
    let x_slice = slice::from_raw_parts(x_ptr, n);
    let x = ArrayD::from_shape_vec(IxDyn(&[n]), x_slice.to_vec()).unwrap();

    let y = activations::gelu(&x);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n);
    y_slice.copy_from_slice(y.as_slice().unwrap());
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

    let input = ArrayView4::from_shape((n, h_in, w_in, c_in), input_slice).unwrap();
    let weight = ArrayView4::from_shape((kh, kw, c_in, c_out), weight_slice).unwrap();

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
    output_slice.copy_from_slice(output.as_slice().unwrap());
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
    let input = ArrayView4::from_shape((n, h_in, w_in, c), input_slice).unwrap();

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
    output_slice.copy_from_slice(output.as_slice().unwrap());
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
    let input = ArrayView4::from_shape((n, h, w, c), input_slice).unwrap();

    let output = pool::global_avgpool2d(input);

    let output_size = n * c;
    let output_slice = slice::from_raw_parts_mut(output_ptr, output_size);
    output_slice.copy_from_slice(output.as_slice().unwrap());
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
    let x = ArrayD::from_shape_vec(IxDyn(&[batch_size, n_features]), x_slice.to_vec()).unwrap();

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
    y_slice.copy_from_slice(y.as_slice().unwrap());
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

    let x = ArrayD::from_shape_vec(IxDyn(&[batch_size, hidden_size]), x_slice.to_vec()).unwrap();
    let gamma = ArrayD::from_shape_vec(IxDyn(&[hidden_size]), gamma_slice.to_vec()).unwrap();
    let beta = ArrayD::from_shape_vec(IxDyn(&[hidden_size]), beta_slice.to_vec()).unwrap();

    let y = norm::layernorm(&x, &gamma, &beta, &[hidden_size], eps);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n_elements);
    y_slice.copy_from_slice(y.as_slice().unwrap());
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

    let x = ArrayD::from_shape_vec(IxDyn(&[batch_size, hidden_size]), x_slice.to_vec()).unwrap();

    let y = norm::rmsnorm(&x, weight, eps);

    let y_slice = slice::from_raw_parts_mut(y_ptr, n_elements);
    y_slice.copy_from_slice(y.as_slice().unwrap());
}
