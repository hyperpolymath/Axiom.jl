//! Matrix multiplication operations

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Matrix multiplication: C = A @ B
///
/// Uses tiled algorithm for cache efficiency
pub fn matmul(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (_, n) = b.dim();

    let mut c = Array2::zeros((m, n));

    // Tile size for cache efficiency
    const TILE_SIZE: usize = 64;

    // Tiled matrix multiplication
    for i_tile in (0..m).step_by(TILE_SIZE) {
        let i_end = (i_tile + TILE_SIZE).min(m);

        for k_tile in (0..k).step_by(TILE_SIZE) {
            let k_end = (k_tile + TILE_SIZE).min(k);

            for j_tile in (0..n).step_by(TILE_SIZE) {
                let j_end = (j_tile + TILE_SIZE).min(n);

                // Multiply tile
                for i in i_tile..i_end {
                    for kk in k_tile..k_end {
                        let a_ik = a[[i, kk]];
                        for j in j_tile..j_end {
                            c[[i, j]] += a_ik * b[[kk, j]];
                        }
                    }
                }
            }
        }
    }

    c
}

/// Parallel matrix multiplication for large matrices
pub fn matmul_parallel(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (_, n) = b.dim();

    // For small matrices, use sequential version
    if m * n < 10000 {
        return matmul(a, b);
    }

    let mut c = Array2::zeros((m, n));

    // Parallel over rows
    c.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for kk in 0..k {
                let a_ik = a[[i, kk]];
                for j in 0..n {
                    row[j] += a_ik * b[[kk, j]];
                }
            }
        });

    c
}

/// Matrix-vector multiplication: y = A @ x
pub fn matvec(a: ArrayView2<f32>, x: ndarray::ArrayView1<f32>) -> ndarray::Array1<f32> {
    a.dot(&x)
}

/// Batch matrix multiplication
pub fn bmm(a: &[Array2<f32>], b: &[Array2<f32>]) -> Vec<Array2<f32>> {
    assert_eq!(a.len(), b.len(), "Batch sizes must match");

    a.par_iter()
        .zip(b.par_iter())
        .map(|(ai, bi)| matmul(ai.view(), bi.view()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_matmul() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0f32, 6.0], [7.0, 8.0]];

        let c = matmul(a.view(), b.view());

        assert_abs_diff_eq!(c[[0, 0]], 19.0, epsilon = 1e-5);
        assert_abs_diff_eq!(c[[0, 1]], 22.0, epsilon = 1e-5);
        assert_abs_diff_eq!(c[[1, 0]], 43.0, epsilon = 1e-5);
        assert_abs_diff_eq!(c[[1, 1]], 50.0, epsilon = 1e-5);
    }

    #[test]
    fn test_matmul_parallel() {
        let a = Array2::from_elem((100, 100), 1.0f32);
        let b = Array2::from_elem((100, 100), 1.0f32);

        let c = matmul_parallel(a.view(), b.view());

        // Each element should be 100 (sum of 100 ones)
        assert_abs_diff_eq!(c[[0, 0]], 100.0, epsilon = 1e-5);
    }
}
