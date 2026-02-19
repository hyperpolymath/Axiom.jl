//! Normalization operations

use ndarray::ArrayD;

/// Batch Normalization
///
/// x_norm = (x - mean) / sqrt(var + eps)
/// output = gamma * x_norm + beta
#[allow(clippy::too_many_arguments)]
pub fn batchnorm(
    input: &ArrayD<f32>,
    gamma: &[f32],
    beta: &[f32],
    running_mean: &mut [f32],
    running_var: &mut [f32],
    eps: f32,
    momentum: f32,
    training: bool,
) -> ArrayD<f32> {
    let shape = input.shape();
    let ndim = shape.len();
    let num_features = shape[ndim - 1];

    assert_eq!(gamma.len(), num_features);
    assert_eq!(beta.len(), num_features);

    let mut output = input.clone();

    if training {
        // Compute batch statistics
        let batch_size: usize = shape[..ndim - 1].iter().product();

        for c in 0..num_features {
            // Compute mean
            let mut sum = 0.0f32;
            for i in 0..batch_size {
                let idx = i * num_features + c;
                sum += input.as_slice().expect("norm: array not contiguous")[idx];
            }
            let mean = sum / batch_size as f32;

            // Compute variance
            let mut var_sum = 0.0f32;
            for i in 0..batch_size {
                let idx = i * num_features + c;
                let diff = input.as_slice().expect("norm: array not contiguous")[idx] - mean;
                var_sum += diff * diff;
            }
            let var = var_sum / batch_size as f32;

            // Update running statistics
            running_mean[c] = (1.0 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.0 - momentum) * running_var[c] + momentum * var;

            // Normalize and scale
            let inv_std = 1.0 / (var + eps).sqrt();
            let output_slice = output.as_slice_mut().expect("norm: array not contiguous");
            for i in 0..batch_size {
                let idx = i * num_features + c;
                let normalized = (output_slice[idx] - mean) * inv_std;
                output_slice[idx] = gamma[c] * normalized + beta[c];
            }
        }
    } else {
        // Use running statistics
        let batch_size: usize = shape[..ndim - 1].iter().product();

        for c in 0..num_features {
            let inv_std = 1.0 / (running_var[c] + eps).sqrt();
            let output_slice = output.as_slice_mut().expect("norm: array not contiguous");

            for i in 0..batch_size {
                let idx = i * num_features + c;
                let normalized = (output_slice[idx] - running_mean[c]) * inv_std;
                output_slice[idx] = gamma[c] * normalized + beta[c];
            }
        }
    }

    output
}

/// Layer Normalization
///
/// Normalizes over the last N dimensions
pub fn layernorm(
    input: &ArrayD<f32>,
    gamma: &ArrayD<f32>,
    beta: &ArrayD<f32>,
    normalized_shape: &[usize],
    eps: f32,
) -> ArrayD<f32> {
    let shape = input.shape();
    let ndim = shape.len();
    let norm_ndim = normalized_shape.len();

    // Number of elements to normalize over
    let norm_size: usize = normalized_shape.iter().product();
    let batch_size: usize = shape[..ndim - norm_ndim].iter().product();

    let mut output = input.clone();

    let output_slice = output.as_slice_mut().expect("norm: array not contiguous");
    let input_slice = input.as_slice().expect("norm: array not contiguous");
    let gamma_slice = gamma.as_slice().expect("norm: array not contiguous");
    let beta_slice = beta.as_slice().expect("norm: array not contiguous");

    for batch in 0..batch_size {
        let start = batch * norm_size;
        let end = start + norm_size;

        // Compute mean
        let sum: f32 = input_slice[start..end].iter().sum();
        let mean = sum / norm_size as f32;

        // Compute variance
        let var: f32 = input_slice[start..end]
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / norm_size as f32;

        // Normalize and scale
        let inv_std = 1.0 / (var + eps).sqrt();

        for i in 0..norm_size {
            let idx = start + i;
            let normalized = (output_slice[idx] - mean) * inv_std;
            output_slice[idx] = gamma_slice[i] * normalized + beta_slice[i];
        }
    }

    output
}

/// Instance Normalization
///
/// Normalizes each sample independently over spatial dimensions
pub fn instancenorm(
    input: &ArrayD<f32>, // (N, H, W, C) or (N, ..., C)
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    eps: f32,
) -> ArrayD<f32> {
    let shape = input.shape();
    let ndim = shape.len();
    let batch_size = shape[0];
    let num_channels = shape[ndim - 1];
    let spatial_size: usize = shape[1..ndim - 1].iter().product();

    let mut output = input.clone();

    let output_slice = output.as_slice_mut().expect("norm: array not contiguous");
    let input_slice = input.as_slice().expect("norm: array not contiguous");

    for n in 0..batch_size {
        for c in 0..num_channels {
            // Compute mean over spatial dimensions
            let mut sum = 0.0f32;
            for s in 0..spatial_size {
                let idx = n * spatial_size * num_channels + s * num_channels + c;
                sum += input_slice[idx];
            }
            let mean = sum / spatial_size as f32;

            // Compute variance
            let mut var_sum = 0.0f32;
            for s in 0..spatial_size {
                let idx = n * spatial_size * num_channels + s * num_channels + c;
                let diff = input_slice[idx] - mean;
                var_sum += diff * diff;
            }
            let var = var_sum / spatial_size as f32;

            // Normalize and scale
            let inv_std = 1.0 / (var + eps).sqrt();
            let g = gamma.map_or(1.0, |g| g[c]);
            let b = beta.map_or(0.0, |b| b[c]);

            for s in 0..spatial_size {
                let idx = n * spatial_size * num_channels + s * num_channels + c;
                let normalized = (output_slice[idx] - mean) * inv_std;
                output_slice[idx] = g * normalized + b;
            }
        }
    }

    output
}

/// Group Normalization
///
/// Divides channels into groups and normalizes within each group
pub fn groupnorm(
    input: &ArrayD<f32>, // (N, H, W, C)
    num_groups: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> ArrayD<f32> {
    let shape = input.shape();
    let ndim = shape.len();
    let batch_size = shape[0];
    let num_channels = shape[ndim - 1];
    let spatial_size: usize = shape[1..ndim - 1].iter().product();

    assert_eq!(
        num_channels % num_groups,
        0,
        "num_channels must be divisible by num_groups"
    );
    let channels_per_group = num_channels / num_groups;

    let mut output = input.clone();

    let output_slice = output.as_slice_mut().expect("norm: array not contiguous");
    let input_slice = input.as_slice().expect("norm: array not contiguous");

    for n in 0..batch_size {
        for g in 0..num_groups {
            let c_start = g * channels_per_group;
            let c_end = c_start + channels_per_group;

            // Compute mean over spatial and channel dimensions within group
            let mut sum = 0.0f32;
            let count = spatial_size * channels_per_group;

            for s in 0..spatial_size {
                for c in c_start..c_end {
                    let idx = n * spatial_size * num_channels + s * num_channels + c;
                    sum += input_slice[idx];
                }
            }
            let mean = sum / count as f32;

            // Compute variance
            let mut var_sum = 0.0f32;
            for s in 0..spatial_size {
                for c in c_start..c_end {
                    let idx = n * spatial_size * num_channels + s * num_channels + c;
                    let diff = input_slice[idx] - mean;
                    var_sum += diff * diff;
                }
            }
            let var = var_sum / count as f32;

            // Normalize and scale
            let inv_std = 1.0 / (var + eps).sqrt();

            for s in 0..spatial_size {
                for c in c_start..c_end {
                    let idx = n * spatial_size * num_channels + s * num_channels + c;
                    let normalized = (output_slice[idx] - mean) * inv_std;
                    output_slice[idx] = gamma[c] * normalized + beta[c];
                }
            }
        }
    }

    output
}

/// RMS Normalization (used in LLaMA, etc.)
///
/// x_norm = x / sqrt(mean(x^2) + eps)
/// output = weight * x_norm
pub fn rmsnorm(input: &ArrayD<f32>, weight: &[f32], eps: f32) -> ArrayD<f32> {
    let shape = input.shape();
    let ndim = shape.len();
    let hidden_size = shape[ndim - 1];
    let batch_size: usize = shape[..ndim - 1].iter().product();

    let mut output = input.clone();

    let output_slice = output.as_slice_mut().expect("norm: array not contiguous");
    let input_slice = input.as_slice().expect("norm: array not contiguous");

    for batch in 0..batch_size {
        let start = batch * hidden_size;
        let end = start + hidden_size;

        // Compute RMS
        let rms: f32 =
            input_slice[start..end].iter().map(|&x| x * x).sum::<f32>() / hidden_size as f32;
        let inv_rms = 1.0 / (rms + eps).sqrt();

        // Normalize and scale
        for (i, w) in weight.iter().enumerate().take(hidden_size) {
            let idx = start + i;
            output_slice[idx] = w * input_slice[idx] * inv_rms;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layernorm() {
        let input = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 4]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .expect("test: shape mismatch");

        let gamma = ArrayD::from_elem(ndarray::IxDyn(&[4]), 1.0f32);
        let beta = ArrayD::from_elem(ndarray::IxDyn(&[4]), 0.0f32);

        let output = layernorm(&input, &gamma, &beta, &[4], 1e-5);

        // After layer norm, mean should be ~0, std should be ~1
        let first_row: Vec<f32> = output.slice(ndarray::s![0, ..]).iter().cloned().collect();
        let mean: f32 = first_row.iter().sum::<f32>() / 4.0;
        assert!((mean - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_rmsnorm() {
        let input = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 4]),
            vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        )
        .expect("test: shape mismatch");

        let weight = vec![1.0f32; 4];

        let output = rmsnorm(&input, &weight, 1e-5);

        // Output should be normalized
        assert!(output[[0, 0]].abs() < 1.0);
    }
}
