//! Checked row-major matrix multiplication; no memory is retained by BLAS.
/// Compute op(A)[m,k] * op(B)[k,n] into C[m,n]. All slices must be exact.
pub fn matrix_product(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    k: usize,
    n: usize,
    ta: bool,
    tb: bool,
) {
    assert_eq!(m.checked_mul(k), Some(a.len()), "matrix A shape");
    assert_eq!(k.checked_mul(n), Some(b.len()), "matrix B shape");
    assert_eq!(m.checked_mul(n), Some(c.len()), "matrix C shape");
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c.fill(0.);
        return;
    }
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        extern "C" {
            fn cblas_dgemm(
                order: i32,
                ta: i32,
                tb: i32,
                m: i32,
                n: i32,
                k: i32,
                alpha: f64,
                a: *const f64,
                lda: i32,
                b: *const f64,
                ldb: i32,
                beta: f64,
                c: *mut f64,
                ldc: i32,
            );
        }
        let m = i32::try_from(m).expect("BLAS rows fit i32");
        let k = i32::try_from(k).expect("BLAS inner dimension fits i32");
        let n = i32::try_from(n).expect("BLAS columns fit i32");
        // SAFETY: exact lengths and i32 dimensions were checked above; Rust
        // borrowing prevents output aliasing either input. Leading dimensions
        // describe the actual row-major storage, including transposed operands.
        // The synchronous CBLAS call retains no pointers and beta=0 initializes C.
        unsafe {
            cblas_dgemm(
                101,
                if ta { 112 } else { 111 },
                if tb { 112 } else { 111 },
                m,
                n,
                k,
                1.,
                a.as_ptr(),
                if ta { m } else { k },
                b.as_ptr(),
                if tb { k } else { n },
                0.,
                c.as_mut_ptr(),
                n,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        c.fill(0.);
        for i in 0..m {
            for h in 0..k {
                for j in 0..n {
                    c[i * n + j] += a[if ta { h * m + i } else { i * k + h }]
                        * b[if tb { j * k + h } else { h * n + j }];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangular_transposes_match_scalar_and_reject_wrong_shapes() {
        for ta in [false, true] {
            for tb in [false, true] {
                let (m, k, n) = (3, 5, 2);
                let a: Vec<_> = (0..m * k).map(|i| i as f64 / 7.).collect();
                let b: Vec<_> = (0..k * n).map(|i| (i as f64 - 3.) / 9.).collect();
                let mut c = vec![f64::NAN; m * n];
                matrix_product(&a, &b, &mut c, m, k, n, ta, tb);
                for i in 0..m {
                    for j in 0..n {
                        let expected: f64 = (0..k)
                            .map(|h| {
                                a[if ta { h * m + i } else { i * k + h }]
                                    * b[if tb { j * k + h } else { h * n + j }]
                            })
                            .sum();
                        assert!((c[i * n + j] - expected).abs() < 1e-12);
                    }
                }
            }
        }
        assert!(std::panic::catch_unwind(|| matrix_product(
            &[1.],
            &[1.],
            &mut [0.],
            2,
            1,
            1,
            false,
            false
        ))
        .is_err());
        let mut zero = [1.; 6];
        matrix_product(&[], &[], &mut zero, 2, 0, 3, false, false);
        assert_eq!(zero, [0.; 6]);
    }
}
