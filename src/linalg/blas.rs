//! Vector and matrix products: `dot`, `axpy`, `gemv`, `gemv_t` and `gemm`.
//!
//! Every output element starts from `β` times its old value (`0` when `β` is 0, without reading
//! the old value, as BLAS does), then adds its products `(α · a) · b` one at a time in ascending
//! order of the summed index. With `α = 1` and `β = 0`, that's the textbook sum
//! `0 + a₀b₀ + a₁b₁ + …`, so `gemm` of one column is `gemv`, and a row of `gemv` is `dot`, to the
//! bit.

// L-BFGS-B (batch A2) and the Gaussian processes (batch B) are the first users
#![allow(dead_code)]

use super::{MR, NR, for_each_chunk, tile_add};

// the rows of `b` packed at a time: the packed panel (KC × NR per column tile) stays in the cache
const KC: usize = 256;
// `gemm` runs on rayon from this many multiplications (m · k · n) on
const PARALLEL_WORK: usize = 1 << 21;

/// `x · y`: `0 + x₀y₀ + x₁y₁ + …` in order.
pub(crate) fn dot(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let mut sum = 0.0;
    for (&a, &b) in x.iter().zip(y) {
        sum += a * b;
    }
    sum
}

/// `y ← α x + y`: `yᵢ + α xᵢ`.
pub(crate) fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    for (y, &x) in y.iter_mut().zip(x) {
        *y += alpha * x;
    }
}

// the start of an output element: β times its old value, or 0 for β = 0
#[inline(always)]
fn start(beta: f64, old: f64) -> f64 {
    if beta == 0.0 { 0.0 } else { beta * old }
}

/// `y ← α A x + β y` for the `m × n` matrix `A`: `yᵢ = β yᵢ + (α aᵢ₀) x₀ + (α aᵢ₁) x₁ + …`.
pub(crate) fn gemv(m: usize, n: usize, alpha: f64, a: &[f64], x: &[f64], beta: f64, y: &mut [f64]) {
    debug_assert!(a.len() == m * n && x.len() == n && y.len() == m);
    if n == 0 {
        for y in y.iter_mut() {
            *y = start(beta, *y);
        }
        return;
    }
    // four rows at a time: four independent sums, each in its own order
    let rows = a.chunks_exact(4 * n);
    let rest = rows.remainder();
    let (outputs, last) = y.as_chunks_mut::<4>();
    for (rows, y) in rows.zip(outputs) {
        let (r0, rows) = rows.split_at(n);
        let (r1, rows) = rows.split_at(n);
        let (r2, r3) = rows.split_at(n);
        let mut s = y.map(|y| start(beta, y));
        for j in 0..n {
            let xj = x[j];
            s[0] += alpha * r0[j] * xj;
            s[1] += alpha * r1[j] * xj;
            s[2] += alpha * r2[j] * xj;
            s[3] += alpha * r3[j] * xj;
        }
        *y = s;
    }
    for (row, y) in rest.chunks_exact(n).zip(last) {
        let mut s = start(beta, *y);
        for (&aij, &xj) in row.iter().zip(x) {
            s += alpha * aij * xj;
        }
        *y = s;
    }
}

/// `y ← α Aᵀ x + β y` for the `m × n` matrix `A`: `yⱼ = β yⱼ + (α a₀ⱼ) x₀ + (α a₁ⱼ) x₁ + …`,
/// the bits of [`gemv`] on the transpose.
pub(crate) fn gemv_t(
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    debug_assert!(a.len() == m * n && x.len() == m && y.len() == n);
    for y in y.iter_mut() {
        *y = start(beta, *y);
    }
    if n == 0 {
        return;
    }
    // row by row: each yⱼ adds its products in the order of the rows
    for (row, &xi) in a.chunks_exact(n).zip(x) {
        for (y, &aij) in y.iter_mut().zip(row) {
            *y += alpha * aij * xi;
        }
    }
}

/// `C ← α A B + β C` for the `m × k` matrix `A`, the `k × n` matrix `B` and the `m × n` matrix
/// `C`: `cᵢⱼ = β cᵢⱼ + (α aᵢ₀) b₀ⱼ + (α aᵢ₁) b₁ⱼ + …`.
///
/// Blocked and packed, with a register tile of the output; on rayon (the `parallel` feature) by
/// independent rows of `C` for large products, with the same bits.
// BLAS's order of the arguments
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm(
    m: usize,
    k: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64],
) {
    let parallel = m.saturating_mul(k).saturating_mul(n) >= PARALLEL_WORK;
    gemm_with(m, k, n, alpha, a, b, beta, c, parallel);
}

// `gemm`, on rayon or not
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_with(
    m: usize,
    k: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64],
    parallel: bool,
) {
    debug_assert!(a.len() == m * k && b.len() == k * n && c.len() == m * n);
    if beta != 1.0 {
        for c in c.iter_mut() {
            *c = start(beta, *c);
        }
    }
    if m == 0 || n == 0 {
        return;
    }
    // the products are added to C a panel of KC rows of B at a time, in order
    let tiles = n.div_ceil(NR);
    let mut packed = vec![0.0; tiles * KC.min(k) * NR];
    let mut p0 = 0;
    while p0 < k {
        let p1 = (p0 + KC).min(k);
        let depth = p1 - p0;
        // B[p0..p1][..] by column tiles of NR, zero padded
        let packed = &mut packed[..tiles * depth * NR];
        for (t, tile) in packed.chunks_exact_mut(depth * NR).enumerate() {
            for (row, packed) in b[p0 * n..p1 * n]
                .chunks_exact(n)
                .zip(tile.as_chunks_mut::<NR>().0)
            {
                let j0 = t * NR;
                let width = (n - j0).min(NR);
                packed[..width].copy_from_slice(&row[j0..j0 + width]);
                packed[width..].fill(0.0);
            }
        }
        let packed = &*packed;
        for_each_chunk(c, MR * n, parallel, |t, rows| {
            let i0 = t * MR;
            let height = rows.len() / n;
            // α A[i0..][p0..p1], packed as depth × MR, zero padded
            let mut panel = [0.0; KC * MR];
            for r in 0..height {
                let row = &a[(i0 + r) * k + p0..(i0 + r) * k + p1];
                for (kk, &x) in row.iter().enumerate() {
                    panel[kk * MR + r] = alpha * x;
                }
            }
            let panel = &panel[..depth * MR];
            for (tj, tile) in packed.chunks_exact(depth * NR).enumerate() {
                let j0 = tj * NR;
                let width = (n - j0).min(NR);
                let mut acc = [[0.0; NR]; MR];
                for (acc, row) in acc.iter_mut().zip(rows.chunks_exact(n)) {
                    acc[..width].copy_from_slice(&row[j0..j0 + width]);
                }
                tile_add(&mut acc, panel, tile);
                for (acc, row) in acc.iter().zip(rows.chunks_exact_mut(n)) {
                    row[j0..j0 + width].copy_from_slice(&acc[..width]);
                }
            }
        });
        p0 = p1;
    }
}
