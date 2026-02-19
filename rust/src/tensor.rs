//! Tensor types and operations

use ndarray::{ArrayD, IxDyn};
use std::sync::Arc;

/// A multi-dimensional tensor
#[derive(Debug, Clone)]
pub struct Tensor<T> {
    data: ArrayD<T>,
    requires_grad: bool,
    _grad: Option<Arc<Tensor<T>>>,
}

impl<T: Clone + num_traits::Zero> Tensor<T> {
    /// Create a new tensor from shape
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::zeros(IxDyn(shape)),
            requires_grad: false,
            _grad: None,
        }
    }
}

impl<T: Clone> Tensor<T> {
    /// Create tensor from ndarray
    pub fn from_array(data: ArrayD<T>) -> Self {
        Self {
            data,
            requires_grad: false,
            _grad: None,
        }
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get raw data pointer
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Get mutable raw data pointer
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Set requires_grad
    pub fn requires_grad_(mut self, requires: bool) -> Self {
        self.requires_grad = requires;
        self
    }
}

impl Tensor<f32> {
    /// Create tensor with random values from normal distribution
    pub fn randn(shape: &[usize]) -> Self {
        use ndarray_rand::rand_distr::StandardNormal;
        use ndarray_rand::RandomExt;

        let data = ArrayD::random(IxDyn(shape), StandardNormal);
        Self::from_array(data)
    }

    /// Create tensor with uniform random values
    pub fn rand(shape: &[usize]) -> Self {
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;

        let data = ArrayD::random(IxDyn(shape), Uniform::new(0.0, 1.0));
        Self::from_array(data)
    }
}

/// Tensor from raw pointer (for FFI)
///
/// # Safety
/// The pointer must be valid and point to `len` elements
pub unsafe fn tensor_from_ptr<T: Clone>(ptr: *const T, shape: &[usize]) -> Tensor<T> {
    let len: usize = shape.iter().product();
    let slice = std::slice::from_raw_parts(ptr, len);
    let data =
        ArrayD::from_shape_vec(IxDyn(shape), slice.to_vec()).expect("tensor: shape mismatch");
    Tensor::from_array(data)
}

/// Copy tensor data to pointer (for FFI)
///
/// # Safety
/// The destination pointer must have enough space
pub unsafe fn tensor_to_ptr<T: Clone>(tensor: &Tensor<T>, dst: *mut T) {
    let src = tensor
        .data
        .as_slice()
        .expect("tensor: array not contiguous");
    std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
}
