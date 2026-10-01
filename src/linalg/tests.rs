// The linear algebra against textbook loops (to the bit, and within tolerances of the true
// results), on every thread count, and against fixed hashes of its bits on seeded inputs: CI runs
// these on Linux, macOS (arm64) and Windows.

use super::blas::{axpy, dot, gemm, gemm_with, gemv, gemv_t};
use super::cholesky::{NotPositiveDefinite, cholesky, cholesky_with_jitter, factor};
use super::eigen::eigen_transposed;
use super::triangular::{
    cholesky_solve, cholesky_solve_multi, solve_lower, solve_lower_multi, solve_lower_transposed,
    solve_lower_transposed_multi,
};

// splitmix64, uniform in [−1, 1): portable, and independent of genoxide's random streams
struct Random(u64);

impl Random {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    fn vector(&mut self, len: usize) -> Vec<f64> {
        (0..len).map(|_| self.next()).collect()
    }
}

// FNV-1a of the bits
fn hash(values: &[f64]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

// M Mᵀ / n + I for a random n × n matrix M: symmetric positive definite, well conditioned
fn spd(n: usize, seed: u64) -> Vec<f64> {
    let m = Random(seed).vector(n * n);
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += m[i * n + k] * m[j * n + k];
            }
            a[i * n + j] = s / n as f64 + if i == j { 1.0 } else { 0.0 };
        }
    }
    a
}

fn symmetric(n: usize, seed: u64) -> Vec<f64> {
    let mut a = Random(seed).vector(n * n);
    for i in 0..n {
        for j in 0..i {
            a[i * n + j] = a[j * n + i];
        }
    }
    a
}

// the textbook left-looking Cholesky, the order every version must keep
fn naive_cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if !(d > 0.0 && d < f64::INFINITY) {
            return None;
        }
        let ljj = d.sqrt();
        l[j * n + j] = ljj;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / ljj;
        }
    }
    Some(l)
}

// the textbook triple loop, in gemm's order
#[allow(clippy::too_many_arguments)]
fn naive_gemm(
    m: usize,
    k: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = if beta == 0.0 {
                0.0
            } else {
                beta * c[i * n + j]
            };
            for p in 0..k {
                s += alpha * a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn transpose(a: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    let mut t = vec![0.0; rows * columns];
    for i in 0..rows {
        for j in 0..columns {
            t[j * rows + i] = a[i * columns + j];
        }
    }
    t
}

// L Lᵀ, from the lower triangle of `l`
fn rebuild(l: &[f64], n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = (0..=i.min(j)).map(|k| l[i * n + k] * l[j * n + k]).sum();
        }
    }
    a
}

fn max_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0, |max, (x, y)| max.max((x - y).abs()))
}

#[cfg(feature = "parallel")]
fn on_threads<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}

#[test]
fn dot_axpy_and_gemv_add_in_order() {
    let mut random = Random(1);
    let (m, n) = (7, 13);
    let a = random.vector(m * n);
    let x = random.vector(n);
    let y0 = random.vector(m);
    assert_eq!(dot(&[], &[]), 0.0);
    let mut s = 0.0;
    for j in 0..n {
        s += a[j] * x[j];
    }
    assert_eq!(dot(&a[..n], &x).to_bits(), s.to_bits());

    let mut y = x.clone();
    axpy(0.3, &a[..n], &mut y);
    for j in 0..n {
        assert_eq!(y[j].to_bits(), (x[j] + 0.3 * a[j]).to_bits());
    }

    for (alpha, beta) in [(1.0, 0.0), (0.7, 0.0), (1.0, 1.0), (-1.3, 0.4)] {
        // gemv is gemm of one column, and its rows are dots for α = 1, β = 0
        let mut y = y0.clone();
        gemv(m, n, alpha, &a, &x, beta, &mut y);
        assert!(same_bits(
            &y,
            &naive_gemm(m, n, 1, alpha, &a, &x, beta, &y0)
        ));
        let mut c = y0.clone();
        gemm(m, n, 1, alpha, &a, &x, beta, &mut c);
        assert!(same_bits(&y, &c));
        if (alpha, beta) == (1.0, 0.0) {
            for i in 0..m {
                assert_eq!(y[i].to_bits(), dot(&a[i * n..(i + 1) * n], &x).to_bits());
            }
        }
        // gemv_t is gemv on the transpose
        let z = random.vector(m);
        let w0 = random.vector(n);
        let mut w = w0.clone();
        gemv_t(m, n, alpha, &a, &z, beta, &mut w);
        let mut expected = w0.clone();
        gemv(n, m, alpha, &transpose(&a, m, n), &z, beta, &mut expected);
        assert!(same_bits(&w, &expected));
    }
}

#[test]
fn zero_beta_ignores_the_output_and_empty_dimensions_work() {
    let a = [1.0, 2.0];
    let mut y = [f64::NAN];
    gemv(1, 2, 1.0, &a, &[3.0, 4.0], 0.0, &mut y);
    assert_eq!(y, [11.0]);
    let mut y = [f64::NAN, f64::NAN];
    gemv_t(1, 2, 1.0, &a, &[2.0], 0.0, &mut y);
    assert_eq!(y, [2.0, 4.0]);
    let mut c = [f64::NAN; 4];
    gemm(2, 1, 2, 1.0, &a, &a, 0.0, &mut c);
    assert_eq!(c, [1.0, 2.0, 2.0, 4.0]);
    // k = 0: C = β C
    let mut c = [1.0, 2.0];
    gemm(1, 0, 2, 1.0, &[], &[], 2.0, &mut c);
    assert_eq!(c, [2.0, 4.0]);
    let mut y = [3.0];
    gemv(1, 0, 1.0, &[], &[], 0.5, &mut y);
    assert_eq!(y, [1.5]);
    gemm(0, 3, 0, 1.0, &[], &[], 0.0, &mut []);
    gemv_t(0, 0, 1.0, &[], &[], 0.0, &mut []);
}

#[test]
fn gemm_has_the_bits_of_the_triple_loop() {
    let mut random = Random(2);
    // shapes around the tiles (4 × 8) and the panel depth (256)
    for (m, k, n) in [
        (1, 1, 1),
        (3, 5, 7),
        (4, 8, 8),
        (5, 9, 9),
        (13, 300, 17),
        (33, 257, 31),
        (64, 513, 3),
        (2, 3, 100),
    ] {
        let a = random.vector(m * k);
        let b = random.vector(k * n);
        let c0 = random.vector(m * n);
        for (alpha, beta) in [(1.0, 0.0), (0.5, 1.0), (-2.0, 0.3)] {
            let expected = naive_gemm(m, k, n, alpha, &a, &b, beta, &c0);
            for parallel in [false, true] {
                let mut c = c0.clone();
                gemm_with(m, k, n, alpha, &a, &b, beta, &mut c, parallel);
                assert!(same_bits(&c, &expected), "{m} × {k} × {n}");
            }
        }
    }
}

#[test]
fn cholesky_has_the_bits_of_the_textbook_loops_with_any_block() {
    for n in [0, 1, 2, 3, 7, 8, 9, 31, 64, 100, 129, 300] {
        let a = spd(n, n as u64);
        let expected = naive_cholesky(&a, n).unwrap();
        for block in [1, 2, 7, 64, 128, 256] {
            for parallel_rows in [0, usize::MAX] {
                let mut l = a.clone();
                factor(&mut l, n, block, parallel_rows).unwrap();
                assert!(same_bits(&l, &expected), "n = {n}, block {block}");
            }
        }
        let mut l = a.clone();
        cholesky(&mut l, n).unwrap();
        assert!(same_bits(&l, &expected));
        assert!(max_difference(&rebuild(&l, n), &a) < 1e-12);
    }
}

#[test]
fn cholesky_reads_only_the_lower_triangle() {
    let n = 40;
    let a = spd(n, 3);
    let mut lower = a.clone();
    for i in 0..n {
        for j in i + 1..n {
            lower[i * n + j] = f64::NAN;
        }
    }
    let (mut l, mut m) = (a.clone(), lower);
    cholesky(&mut l, n).unwrap();
    cholesky(&mut m, n).unwrap();
    assert!(same_bits(&l, &m));
}

#[cfg(feature = "parallel")]
#[test]
fn every_thread_count_gives_the_same_bits() {
    let n = 600;
    let a = spd(n, 4);
    let b = Random(5).vector(n * n);
    let run = || {
        let mut l = a.clone();
        factor(&mut l, n, 64, 0).unwrap();
        let mut c = vec![0.0; n * n];
        gemm_with(n, n, n, 1.0, &a, &b, 0.0, &mut c, true);
        (l, c)
    };
    let (l, c) = on_threads(1, run);
    for threads in [2, 3, 8] {
        let (l2, c2) = on_threads(threads, run);
        assert!(
            same_bits(&l, &l2) && same_bits(&c, &c2),
            "{threads} threads"
        );
    }
}

#[test]
fn a_semidefinite_matrix_factors_with_a_jitter() {
    // M Mᵀ for a 6 × 2 matrix M with repeated rows: rank 2, exact in floating point
    let rows = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [2.0, 1.0],
    ];
    let n = rows.len();
    let a: Vec<f64> = (0..n * n)
        .map(|k| {
            let (x, y) = (rows[k / n], rows[k % n]);
            x[0] * y[0] + x[1] * y[1]
        })
        .collect();
    assert_eq!(
        cholesky(&mut a.clone(), n),
        Err(NotPositiveDefinite { column: 2 })
    );
    let mut l = vec![f64::NAN; n * n];
    let jitter = cholesky_with_jitter(&a, n, 1e-10, 1e-2, &mut l).unwrap();
    assert!((1e-10..=1e-2).contains(&jitter), "{jitter}");
    let mut shifted = a.clone();
    for i in 0..n {
        shifted[i * n + i] += jitter;
    }
    assert!(max_difference(&rebuild(&l, n), &shifted) < 1e-12);
    for i in 0..n {
        assert!(l[i * n + i + 1..(i + 1) * n].iter().all(|&x| x == 0.0));
    }
    // the same jitter every time
    let mut l2 = vec![0.0; n * n];
    assert_eq!(
        cholesky_with_jitter(&a, n, 1e-10, 1e-2, &mut l2),
        Ok(jitter)
    );
    assert!(same_bits(&l, &l2));
    // a positive definite matrix needs none
    let b = spd(10, 6);
    let mut l = vec![0.0; 100];
    assert_eq!(cholesky_with_jitter(&b, 10, 1e-10, 1e-2, &mut l), Ok(0.0));
}

#[test]
fn an_indefinite_matrix_is_an_error() {
    // eigenvalues 3 and −1
    let a = [1.0, 2.0, 2.0, 1.0];
    let mut l = [0.0; 4];
    assert_eq!(
        cholesky_with_jitter(&a, 2, 1e-10, 1e-1, &mut l),
        Err(NotPositiveDefinite { column: 1 })
    );
    assert_eq!(
        cholesky(&mut [-1.0], 1),
        Err(NotPositiveDefinite { column: 0 })
    );
    assert!(cholesky(&mut [f64::NAN], 1).is_err());
    assert!(cholesky(&mut [f64::INFINITY], 1).is_err());
    // a NaN below the diagonal reaches a later pivot
    let mut a = spd(5, 7);
    a[4 * 5 + 1] = f64::NAN;
    assert_eq!(cholesky(&mut a, 5), Err(NotPositiveDefinite { column: 4 }));
}

#[test]
fn solves_satisfy_their_equations() {
    let mut random = Random(8);
    for n in [0, 1, 2, 5, 33, 120] {
        let a = spd(n, 9);
        let mut l = a.clone();
        cholesky(&mut l, n).unwrap();
        let m = 3;
        let b = random.vector(n * m);
        let columns: Vec<Vec<f64>> = (0..m)
            .map(|c| (0..n).map(|i| b[i * m + c]).collect())
            .collect();
        let lt = transpose(&l, n, n);
        type Solve = fn(&[f64], usize, &mut [f64]);
        type SolveMulti = fn(&[f64], usize, &mut [f64], usize);
        // L x = b, Lᵀ x = b and A x = b
        let cases: [(Solve, SolveMulti, &[f64]); 3] = [
            (solve_lower, solve_lower_multi, &l),
            (solve_lower_transposed, solve_lower_transposed_multi, &lt),
            (cholesky_solve, cholesky_solve_multi, &a),
        ];
        for (solve, solve_multi, matrix) in cases {
            let mut x = b.clone();
            solve_multi(&l, n, &mut x, m);
            let mut product = vec![0.0; n * m];
            gemm(n, n, m, 1.0, matrix, &x, 0.0, &mut product);
            assert!(max_difference(&product, &b) < 1e-12);
            // each column of the matrix solve is the vector solve
            for (c, column) in columns.iter().enumerate() {
                let mut y = column.clone();
                solve(&l, n, &mut y);
                let xc: Vec<f64> = (0..n).map(|i| x[i * m + c]).collect();
                assert!(same_bits(&y, &xc));
            }
        }
    }
}

// the same hashes on every platform; a change to them changes the results of seeded runs
#[test]
fn results_have_fixed_bits() {
    let mut hashes = Vec::new();
    for n in [10, 100, 500] {
        let a = spd(n, 10 + n as u64);
        let mut l = a.clone();
        cholesky(&mut l, n).unwrap();
        let b = Random(20 + n as u64).vector(n * n);
        let mut c = Random(30 + n as u64).vector(n * n);
        gemm(n, n, n, 0.75, &a, &b, -0.5, &mut c);
        let mut y = Random(40 + n as u64).vector(n);
        gemv(n, n, 1.25, &b, &l[..n], 0.5, &mut y);
        gemv_t(n, n, -0.5, &b, &a[..n], 1.0, &mut y);
        let mut x = b[..n * 4].to_vec();
        cholesky_solve_multi(&l, n, &mut x, 4);
        hashes.push([hash(&l), hash(&c), hash(&y), hash(&x)]);
    }
    let expected = [
        [
            0x688c_8425_d1a8_1db1,
            0x5089_fc44_00cc_e134,
            0x2299_d50e_4562_0fcd,
            0x9690_0a9b_7e71_c1dc,
        ],
        [
            0xb7ea_1e1b_4a60_c461,
            0x0fcc_e7f6_2eb6_0a86,
            0x0e0d_e595_0634_76c7,
            0x571d_1d21_0f46_0ac3,
        ],
        [
            0x97a0_3fb0_cc4f_5abe,
            0x7fed_5ad3_328e_026b,
            0x2229_127c_75d0_be82,
            0xa0d6_312f_e748_95e3,
        ],
    ];
    assert_eq!(hashes, expected, "{hashes:#x?}");
}

#[test]
fn the_eigendecomposition_has_fixed_bits() {
    let mut hashes = Vec::new();
    for n in [10, 100, 500] {
        let (values, vectors) = eigen_transposed(&symmetric(n, 50 + n as u64), n);
        hashes.push([hash(&values), hash(&vectors)]);
    }
    let expected = [
        [0x3ba9_7259_fa8a_4229, 0xa63d_1b22_82d9_915d],
        [0xfb50_0ce0_681b_f2aa, 0x14db_5d06_7db0_68ae],
        [0x78f0_eff6_90df_0fb7, 0x0a0e_3ef1_bac5_4cab],
    ];
    assert_eq!(hashes, expected, "{hashes:#x?}");
}
