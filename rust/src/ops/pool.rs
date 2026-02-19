//! Pooling operations

use ndarray::{Array4, ArrayView4};
use rayon::prelude::*;

/// 2D Max Pooling
///
/// Input shape: (N, H, W, C)
/// Output shape: (N, H_out, W_out, C)
pub fn maxpool2d(
    input: ArrayView4<f32>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Array4<f32> {
    let (n, h_in, w_in, c) = input.dim();
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let mut output = Array4::from_elem((n, h_out, w_out, c), f32::NEG_INFINITY);

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

                        let mut max_val = f32::NEG_INFINITY;

                        for ki in 0..kh {
                            for kj in 0..kw {
                                let h_idx = h_start + ki;
                                let w_idx = w_start + kj;

                                if h_idx >= ph
                                    && h_idx < h_in + ph
                                    && w_idx >= pw
                                    && w_idx < w_in + pw
                                {
                                    let val = input[[batch_idx, h_idx - ph, w_idx - pw, channel]];
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                        }

                        batch_output[[i, j, channel]] = max_val;
                    }
                }
            }
        });

    output
}

/// 2D Max Pooling with indices (for unpooling)
pub fn maxpool2d_with_indices(
    input: ArrayView4<f32>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> (Array4<f32>, Array4<usize>) {
    let (n, h_in, w_in, c) = input.dim();
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let mut output = Array4::from_elem((n, h_out, w_out, c), f32::NEG_INFINITY);
    let mut indices = Array4::zeros((n, h_out, w_out, c));

    for batch_idx in 0..n {
        for channel in 0..c {
            for i in 0..h_out {
                for j in 0..w_out {
                    let h_start = i * sh;
                    let w_start = j * sw;

                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0usize;

                    for ki in 0..kh {
                        for kj in 0..kw {
                            let h_idx = h_start + ki;
                            let w_idx = w_start + kj;

                            if h_idx >= ph && h_idx < h_in + ph && w_idx >= pw && w_idx < w_in + pw
                            {
                                let linear_idx = (h_idx - ph) * w_in + (w_idx - pw);
                                let val = input[[batch_idx, h_idx - ph, w_idx - pw, channel]];

                                if val > max_val {
                                    max_val = val;
                                    max_idx = linear_idx;
                                }
                            }
                        }
                    }

                    output[[batch_idx, i, j, channel]] = max_val;
                    indices[[batch_idx, i, j, channel]] = max_idx;
                }
            }
        }
    }

    (output, indices)
}

/// 2D Average Pooling
pub fn avgpool2d(
    input: ArrayView4<f32>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    count_include_pad: bool,
) -> Array4<f32> {
    let (n, h_in, w_in, c) = input.dim();
    let (kh, kw) = kernel_size;
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

                        let mut sum = 0.0f32;
                        let mut count = 0usize;

                        for ki in 0..kh {
                            for kj in 0..kw {
                                let h_idx = h_start + ki;
                                let w_idx = w_start + kj;

                                if h_idx >= ph
                                    && h_idx < h_in + ph
                                    && w_idx >= pw
                                    && w_idx < w_in + pw
                                {
                                    sum += input[[batch_idx, h_idx - ph, w_idx - pw, channel]];
                                    count += 1;
                                } else if count_include_pad {
                                    count += 1;
                                }
                            }
                        }

                        let divisor = if count_include_pad { kh * kw } else { count };
                        batch_output[[i, j, channel]] = sum / divisor as f32;
                    }
                }
            }
        });

    output
}

/// Global Average Pooling
pub fn global_avgpool2d(input: ArrayView4<f32>) -> ndarray::Array2<f32> {
    let (n, h, w, c) = input.dim();
    let mut output = ndarray::Array2::zeros((n, c));

    for batch_idx in 0..n {
        for channel in 0..c {
            let mut sum = 0.0f32;
            for i in 0..h {
                for j in 0..w {
                    sum += input[[batch_idx, i, j, channel]];
                }
            }
            output[[batch_idx, channel]] = sum / (h * w) as f32;
        }
    }

    output
}

/// Global Max Pooling
pub fn global_maxpool2d(input: ArrayView4<f32>) -> ndarray::Array2<f32> {
    let (n, h, w, c) = input.dim();
    let mut output = ndarray::Array2::from_elem((n, c), f32::NEG_INFINITY);

    for batch_idx in 0..n {
        for channel in 0..c {
            for i in 0..h {
                for j in 0..w {
                    let val = input[[batch_idx, i, j, channel]];
                    if val > output[[batch_idx, channel]] {
                        output[[batch_idx, channel]] = val;
                    }
                }
            }
        }
    }

    output
}

/// Adaptive Average Pooling
pub fn adaptive_avgpool2d(input: ArrayView4<f32>, output_size: (usize, usize)) -> Array4<f32> {
    let (n, h_in, w_in, c) = input.dim();
    let (h_out, w_out) = output_size;

    let mut output = Array4::zeros((n, h_out, w_out, c));

    for batch_idx in 0..n {
        for channel in 0..c {
            for i in 0..h_out {
                for j in 0..w_out {
                    // Compute window
                    let h_start = (i * h_in) / h_out;
                    let h_end = ((i + 1) * h_in).div_ceil(h_out);
                    let w_start = (j * w_in) / w_out;
                    let w_end = ((j + 1) * w_in).div_ceil(w_out);

                    let mut sum = 0.0f32;
                    let mut count = 0usize;

                    for hi in h_start..h_end {
                        for wi in w_start..w_end {
                            sum += input[[batch_idx, hi, wi, channel]];
                            count += 1;
                        }
                    }

                    output[[batch_idx, i, j, channel]] = sum / count as f32;
                }
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxpool2d() {
        let input =
            Array4::from_shape_vec((1, 4, 4, 1), (0..16).map(|x| x as f32).collect()).unwrap();

        let output = maxpool2d(input.view(), (2, 2), (2, 2), (0, 0));

        assert_eq!(output.dim(), (1, 2, 2, 1));
        assert_eq!(output[[0, 0, 0, 0]], 5.0); // max of [0,1,4,5]
        assert_eq!(output[[0, 1, 1, 0]], 15.0); // max of [10,11,14,15]
    }

    #[test]
    fn test_global_avgpool() {
        let input = Array4::from_elem((2, 4, 4, 3), 1.0f32);
        let output = global_avgpool2d(input.view());

        assert_eq!(output.dim(), (2, 3));
        assert_eq!(output[[0, 0]], 1.0);
    }
}
