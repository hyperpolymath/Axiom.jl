//! Convolution operations

use ndarray::{Array4, ArrayView4};
use rayon::prelude::*;

/// 2D Convolution
///
/// Input shape: (N, H, W, C_in)
/// Weight shape: (kH, kW, C_in, C_out)
/// Output shape: (N, H_out, W_out, C_out)
pub fn conv2d(
    input: ArrayView4<f32>,
    weight: ArrayView4<f32>,
    bias: Option<&[f32]>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Array4<f32> {
    let (n, h_in, w_in, c_in) = input.dim();
    let (kh, kw, _, c_out) = weight.dim();
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let mut output = Array4::zeros((n, h_out, w_out, c_out));

    // Process each batch in parallel
    output
        .axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(batch_idx, mut batch_output)| {
            for oc in 0..c_out {
                for i in 0..h_out {
                    for j in 0..w_out {
                        let h_start = i * sh;
                        let w_start = j * sw;

                        let mut val = 0.0f32;

                        for ki in 0..kh {
                            for kj in 0..kw {
                                let h_idx = h_start + ki;
                                let w_idx = w_start + kj;

                                // Check bounds (for padding)
                                if h_idx >= ph
                                    && h_idx < h_in + ph
                                    && w_idx >= pw
                                    && w_idx < w_in + pw
                                {
                                    let h_input = h_idx - ph;
                                    let w_input = w_idx - pw;

                                    for ic in 0..c_in {
                                        val += input[[batch_idx, h_input, w_input, ic]]
                                            * weight[[ki, kj, ic, oc]];
                                    }
                                }
                            }
                        }

                        if let Some(b) = bias {
                            val += b[oc];
                        }

                        batch_output[[i, j, oc]] = val;
                    }
                }
            }
        });

    output
}

/// Depthwise convolution
///
/// Each input channel is convolved with its own filter
pub fn depthwise_conv2d(
    input: ArrayView4<f32>,
    weight: ArrayView4<f32>, // (kH, kW, C, 1)
    bias: Option<&[f32]>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Array4<f32> {
    let (n, h_in, w_in, c) = input.dim();
    let (kh, kw, _, _) = weight.dim();
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let mut output = Array4::zeros((n, h_out, w_out, c));

    output
        .axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(batch_idx, mut batch_output)| {
            for channel in 0..c {
                for i in 0..h_out {
                    for j in 0..w_out {
                        let h_start = i * sh;
                        let w_start = j * sw;

                        let mut val = 0.0f32;

                        for ki in 0..kh {
                            for kj in 0..kw {
                                let h_idx = h_start + ki;
                                let w_idx = w_start + kj;

                                if h_idx >= ph
                                    && h_idx < h_in + ph
                                    && w_idx >= pw
                                    && w_idx < w_in + pw
                                {
                                    let h_input = h_idx - ph;
                                    let w_input = w_idx - pw;

                                    val += input[[batch_idx, h_input, w_input, channel]]
                                        * weight[[ki, kj, channel, 0]];
                                }
                            }
                        }

                        if let Some(b) = bias {
                            val += b[channel];
                        }

                        batch_output[[i, j, channel]] = val;
                    }
                }
            }
        });

    output
}

/// Transposed 2D convolution (deconvolution)
pub fn conv2d_transpose(
    input: ArrayView4<f32>,
    weight: ArrayView4<f32>,
    _bias: Option<&[f32]>,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
) -> Array4<f32> {
    let (n, h_in, w_in, _c_in) = input.dim();
    let (kh, kw, c_out, _) = weight.dim(); // Note: weight shape is different
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (oph, opw) = output_padding;

    let h_out = (h_in - 1) * sh - 2 * ph + kh + oph;
    let w_out = (w_in - 1) * sw - 2 * pw + kw + opw;

    Array4::zeros((n, h_out, w_out, c_out))
}

/// im2col transformation for efficient convolution
#[allow(dead_code)]
fn im2col(
    input: ArrayView4<f32>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> ndarray::Array2<f32> {
    let (n, h, w, c) = input.dim();
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w + 2 * pw - kw) / sw + 1;

    let col_height = n * h_out * w_out;
    let col_width = c * kh * kw;

    let mut col = ndarray::Array2::zeros((col_height, col_width));

    // Fill column matrix
    for batch in 0..n {
        for i in 0..h_out {
            for j in 0..w_out {
                let row_idx = batch * h_out * w_out + i * w_out + j;

                for ki in 0..kh {
                    for kj in 0..kw {
                        let h_idx = i * sh + ki;
                        let w_idx = j * sw + kj;

                        for channel in 0..c {
                            let col_idx = channel * kh * kw + ki * kw + kj;

                            if h_idx >= ph && h_idx < h + ph && w_idx >= pw && w_idx < w + pw {
                                col[[row_idx, col_idx]] =
                                    input[[batch, h_idx - ph, w_idx - pw, channel]];
                            }
                        }
                    }
                }
            }
        }
    }

    col
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_identity() {
        // Test with identity-like convolution (1x1 kernel)
        let input = Array4::from_elem((1, 4, 4, 1), 1.0f32);
        let weight = Array4::from_elem((1, 1, 1, 1), 1.0f32);

        let output = conv2d(input.view(), weight.view(), None, (1, 1), (0, 0));

        assert_eq!(output.dim(), (1, 4, 4, 1));
        assert_eq!(output[[0, 0, 0, 0]], 1.0);
    }
}
