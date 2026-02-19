//! Activation functions

use ndarray::ArrayD;

/// ReLU activation: max(0, x)
pub fn relu(x: &ArrayD<f32>) -> ArrayD<f32> {
    x.mapv(|v| v.max(0.0))
}

/// ReLU in-place
pub fn relu_inplace(x: &mut ArrayD<f32>) {
    x.mapv_inplace(|v| v.max(0.0));
}

/// Leaky ReLU: x if x > 0 else alpha * x
pub fn leaky_relu(x: &ArrayD<f32>, alpha: f32) -> ArrayD<f32> {
    x.mapv(|v| if v > 0.0 { v } else { alpha * v })
}

/// ELU: x if x > 0 else alpha * (exp(x) - 1)
pub fn elu(x: &ArrayD<f32>, alpha: f32) -> ArrayD<f32> {
    x.mapv(|v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
}

/// SELU (Scaled ELU)
pub fn selu(x: &ArrayD<f32>) -> ArrayD<f32> {
    const ALPHA: f32 = 1.673_263_2;
    const SCALE: f32 = 1.050_701;

    x.mapv(|v| SCALE * if v > 0.0 { v } else { ALPHA * (v.exp() - 1.0) })
}

/// Sigmoid: 1 / (1 + exp(-x))
pub fn sigmoid(x: &ArrayD<f32>) -> ArrayD<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Sigmoid in-place
pub fn sigmoid_inplace(x: &mut ArrayD<f32>) {
    x.mapv_inplace(|v| 1.0 / (1.0 + (-v).exp()));
}

/// Tanh
pub fn tanh_activation(x: &ArrayD<f32>) -> ArrayD<f32> {
    x.mapv(|v| v.tanh())
}

/// GELU (Gaussian Error Linear Unit)
pub fn gelu(x: &ArrayD<f32>) -> ArrayD<f32> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044715;

    x.mapv(|v| 0.5 * v * (1.0 + (SQRT_2_OVER_PI * (v + COEFF * v.powi(3))).tanh()))
}

/// Softmax along last dimension
pub fn softmax(x: &ArrayD<f32>) -> ArrayD<f32> {
    let shape = x.shape();
    let last_dim = shape.len() - 1;
    let batch_size: usize = shape[..last_dim].iter().product();
    let num_classes = shape[last_dim];

    let mut result = x.clone();

    // Process each batch element
    for batch_idx in 0..batch_size {
        // Find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for class_idx in 0..num_classes {
            let idx = batch_idx * num_classes + class_idx;
            let val = result
                .as_slice()
                .expect("activations: array not contiguous")[idx];
            if val > max_val {
                max_val = val;
            }
        }

        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        let result_slice = result
            .as_slice_mut()
            .expect("activations: array not contiguous");
        for class_idx in 0..num_classes {
            let idx = batch_idx * num_classes + class_idx;
            let exp_val = (result_slice[idx] - max_val).exp();
            result_slice[idx] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for class_idx in 0..num_classes {
            let idx = batch_idx * num_classes + class_idx;
            result_slice[idx] /= sum;
        }
    }

    result
}

/// Log softmax (more numerically stable than log(softmax(x)))
pub fn log_softmax(x: &ArrayD<f32>) -> ArrayD<f32> {
    let shape = x.shape();
    let last_dim = shape.len() - 1;
    let batch_size: usize = shape[..last_dim].iter().product();
    let num_classes = shape[last_dim];

    let mut result = x.clone();

    for batch_idx in 0..batch_size {
        // Find max
        let mut max_val = f32::NEG_INFINITY;
        for class_idx in 0..num_classes {
            let idx = batch_idx * num_classes + class_idx;
            let val = result
                .as_slice()
                .expect("activations: array not contiguous")[idx];
            if val > max_val {
                max_val = val;
            }
        }

        // Compute log(sum(exp(x - max)))
        let mut log_sum_exp = 0.0f32;
        let result_slice = result
            .as_slice()
            .expect("activations: array not contiguous");
        for class_idx in 0..num_classes {
            let idx = batch_idx * num_classes + class_idx;
            log_sum_exp += (result_slice[idx] - max_val).exp();
        }
        log_sum_exp = log_sum_exp.ln() + max_val;

        // Subtract from each element
        let result_slice = result
            .as_slice_mut()
            .expect("activations: array not contiguous");
        for class_idx in 0..num_classes {
            let idx = batch_idx * num_classes + class_idx;
            result_slice[idx] -= log_sum_exp;
        }
    }

    result
}

/// Softplus: log(1 + exp(x))
pub fn softplus(x: &ArrayD<f32>) -> ArrayD<f32> {
    x.mapv(|v| {
        // Numerically stable version
        if v > 20.0 {
            v
        } else if v < -20.0 {
            v.exp()
        } else {
            (1.0 + v.exp()).ln()
        }
    })
}

/// Swish / SiLU: x * sigmoid(x)
pub fn swish(x: &ArrayD<f32>) -> ArrayD<f32> {
    x.mapv(|v| v / (1.0 + (-v).exp()))
}

/// Mish: x * tanh(softplus(x))
pub fn mish(x: &ArrayD<f32>) -> ArrayD<f32> {
    x.mapv(|v| {
        let sp = if v > 20.0 { v } else { (1.0 + v.exp()).ln() };
        v * sp.tanh()
    })
}

/// Hard swish (efficient approximation)
pub fn hardswish(x: &ArrayD<f32>) -> ArrayD<f32> {
    x.mapv(|v| {
        if v <= -3.0 {
            0.0
        } else if v >= 3.0 {
            v
        } else {
            v * (v + 3.0) / 6.0
        }
    })
}

/// Hard sigmoid (efficient approximation)
pub fn hardsigmoid(x: &ArrayD<f32>) -> ArrayD<f32> {
    x.mapv(|v| ((v + 3.0) / 6.0).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_relu() {
        let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![-2.0f32, -1.0, 0.0, 1.0])
            .expect("test: shape mismatch");

        let y = relu(&x);

        assert_abs_diff_eq!(y[[0]], 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(y[[1]], 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(y[[2]], 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(y[[3]], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sigmoid() {
        let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.0f32, 1.0, -1.0])
            .expect("test: shape mismatch");

        let y = sigmoid(&x);

        assert_abs_diff_eq!(y[[0]], 0.5, epsilon = 1e-5);
        assert!(y[[1]] > 0.5);
        assert!(y[[2]] < 0.5);
    }

    #[test]
    fn test_softmax() {
        let x = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 3]),
            vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0],
        )
        .expect("test: shape mismatch");

        let y = softmax(&x);

        // Each row should sum to 1
        let sum1: f32 = y.slice(ndarray::s![0, ..]).sum();
        let sum2: f32 = y.slice(ndarray::s![1, ..]).sum();

        assert_abs_diff_eq!(sum1, 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(sum2, 1.0, epsilon = 1e-5);
    }
}
