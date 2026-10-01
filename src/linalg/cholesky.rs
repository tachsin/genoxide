//! The Cholesky factorization `A = L Lᵀ` of a symmetric positive definite matrix, blocked, with a
//! growing jitter for matrices that are positive semidefinite or nearly so.
//!
//! Each element of `L` is computed as in the textbook loops: `lⱼⱼ = √(aⱼⱼ − lⱼ₀² − lⱼ₁² − …)` and
//! `lᵢⱼ = (aᵢⱼ − lᵢ₀lⱼ₀ − lᵢ₁lⱼ₁ − …) / lⱼⱼ`, subtracting the products one at a time in ascending
//! order. The blocked factorization keeps that order whatever its block size and thread count, so
//! it gives the bits of the textbook loops.

// L-BFGS-B (batch A2) and the Gaussian processes (batch B) are the first users
#![allow(dead_code)]

use super::{CHUNK_ROWS, MR, NR, for_each_chunk, tile_sub};
use std::fmt;

// the size of the diagonal blocks, and the largest one the stack buffers hold
const BLOCK: usize = 128;
const MAX_BLOCK: usize = 256;
// rows of the panel below a diagonal block solved at a time
const PANEL_ROWS: usize = 8;
// the panel and the update below a diagonal block run on rayon from this many rows on
const PARALLEL_ROWS: usize = 256;

/// A matrix that isn't positive definite: the first pivot that wasn't positive and finite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NotPositiveDefinite {
    /// The column of the pivot.
    pub(crate) column: usize,
}

impl fmt::Display for NotPositiveDefinite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the matrix isn't positive definite: pivot {} isn't positive",
            self.column
        )
    }
}

impl std::error::Error for NotPositiveDefinite {}

/// Factors the symmetric positive definite `n × n` matrix `a` in place: its lower triangle
/// becomes `L`, with `A = L Lᵀ`, and its strict upper triangle zero. Only the lower triangle of
/// `a` is read.
///
/// # Errors
///
/// [`NotPositiveDefinite`] if a pivot isn't positive and finite (`a` is then partly overwritten).
pub(crate) fn cholesky(a: &mut [f64], n: usize) -> Result<(), NotPositiveDefinite> {
    factor(a, n, BLOCK, PARALLEL_ROWS)
}

/// The Cholesky factor of `a + δ I` into `l` (the lower triangle `L`, the strict upper triangle
/// zero), with the smallest jitter `δ` that factors of `0`, `initial`, `10 · initial`,
/// `100 · initial`, … up to `max`. Returns `δ`. Only the lower triangle of `a` is read.
///
/// For positive semidefinite matrices (a kernel matrix with repeated points) and matrices that
/// rounding has made slightly indefinite. The jitter is absolute: scale it to the matrix, e.g.
/// by its mean diagonal.
///
/// # Errors
///
/// [`NotPositiveDefinite`] of the last attempt if no jitter up to `max` factors `a`.
pub(crate) fn cholesky_with_jitter(
    a: &[f64],
    n: usize,
    initial: f64,
    max: f64,
    l: &mut [f64],
) -> Result<f64, NotPositiveDefinite> {
    debug_assert!(a.len() == n * n && l.len() == n * n);
    debug_assert!(initial > 0.0 && initial <= max);
    let mut jitter = 0.0;
    loop {
        l.copy_from_slice(a);
        for i in 0..n {
            l[i * n + i] += jitter;
        }
        match cholesky(l, n) {
            Ok(()) => return Ok(jitter),
            Err(error) => {
                jitter = if jitter == 0.0 {
                    initial
                } else {
                    jitter * 10.0
                };
                if !(jitter > 0.0 && jitter <= max) {
                    return Err(error);
                }
            }
        }
    }
}

// `cholesky` with diagonal blocks of `block` columns, and the rows below them on rayon from
// `parallel_rows` rows on
pub(super) fn factor(
    w: &mut [f64],
    n: usize,
    block: usize,
    parallel_rows: usize,
) -> Result<(), NotPositiveDefinite> {
    debug_assert_eq!(w.len(), n * n);
    assert!((1..=MAX_BLOCK).contains(&block));
    // L[j][k0..k1] for the rows j below the diagonal block, packed by tiles of NR rows
    let mut packed = vec![0.0; n.div_ceil(NR) * NR * block.min(n)];
    // the diagonal block's columns: columns[j * block + i] = L[k0 + i][k0 + j]
    let mut columns = vec![0.0; block * block];
    let mut k0 = 0;
    while k0 < n {
        let k1 = (k0 + block).min(n);
        let size = k1 - k0;
        factor_diagonal(w, n, k0, k1)?;
        if k1 == n {
            break;
        }
        for j in 0..size {
            for i in j..size {
                columns[j * block + i] = w[(k0 + i) * n + k0 + j];
            }
        }
        let below = &mut w[k1 * n..];
        let rows = n - k1;
        let parallel = rows >= parallel_rows;
        // the panel below the diagonal block: L[i][k0..k1] for i ≥ k1, row by row, each column's
        // products subtracted as the columns before it are done
        let columns = &columns;
        for_each_chunk(below, PANEL_ROWS * n, parallel, |_, chunk| {
            for j in 0..size {
                let pivot = columns[j * block + j];
                let column = &columns[j * block + j + 1..j * block + size];
                for row in chunk.chunks_exact_mut(n) {
                    let row = &mut row[k0 + j..k1];
                    let x = row[0] / pivot;
                    row[0] = x;
                    for (y, &l) in row[1..].iter_mut().zip(column) {
                        *y -= x * l;
                    }
                }
            }
        });
        let packed = &mut packed[..rows.div_ceil(NR) * size * NR];
        for (t, tile) in packed.chunks_exact_mut(size * NR).enumerate() {
            for c in 0..NR {
                let j = t * NR + c;
                for kk in 0..size {
                    tile[kk * NR + c] = if j < rows {
                        below[j * n + k0 + kk]
                    } else {
                        0.0
                    };
                }
            }
        }
        // the update of the rest: W[i][j] −= L[i][k] L[j][k] for k in k0..k1 in order, for
        // k1 ≤ j ≤ i, by tiles of MR rows and NR columns. A tile that crosses the diagonal also
        // updates elements above it, which are never read and zeroed at the end.
        let packed = &*packed;
        for_each_chunk(below, CHUNK_ROWS * n, parallel, |t, chunk| {
            let i0 = t * CHUNK_ROWS;
            let height = chunk.len() / n;
            // L[i][k0..k1] for the chunk's rows, by tiles of MR rows, zero padded
            let mut panel = vec![0.0; height.div_ceil(MR) * size * MR];
            for (r, row) in chunk.chunks_exact(n).enumerate() {
                let tile = &mut panel[r / MR * size * MR..];
                for (kk, &x) in row[k0..k1].iter().enumerate() {
                    tile[kk * MR + r % MR] = x;
                }
            }
            // the column tiles up to the diagonal of the last row, each reused by all the row
            // tiles while it's in the cache
            let tiles = packed.chunks_exact(size * NR).enumerate();
            for (tj, tile) in tiles.take((i0 + height).div_ceil(NR)) {
                let j0 = k1 + tj * NR;
                let width = (n - j0).min(NR);
                let row_tiles = chunk.chunks_mut(MR * n).zip(panel.chunks_exact(size * MR));
                for (rt, (rows, a)) in row_tiles.enumerate() {
                    if tj * NR >= i0 + rt * MR + rows.len() / n {
                        // above the diagonal of every row of the tile
                        continue;
                    }
                    let mut acc = [[0.0; NR]; MR];
                    for (acc, row) in acc.iter_mut().zip(rows.chunks_exact(n)) {
                        acc[..width].copy_from_slice(&row[j0..j0 + width]);
                    }
                    tile_sub(&mut acc, a, tile);
                    for (acc, row) in acc.iter().zip(rows.chunks_exact_mut(n)) {
                        row[j0..j0 + width].copy_from_slice(&acc[..width]);
                    }
                }
            }
        });
        k0 = k1;
    }
    for i in 0..n {
        w[i * n + i + 1..(i + 1) * n].fill(0.0);
    }
    Ok(())
}

// factors the diagonal block of columns k0..k1, whose elements have had the products of the
// columns before k0 subtracted
fn factor_diagonal(
    w: &mut [f64],
    n: usize,
    k0: usize,
    k1: usize,
) -> Result<(), NotPositiveDefinite> {
    // L[i][j] for the rows i of the block below j, contiguous
    let mut column = [0.0; MAX_BLOCK];
    for j in k0..k1 {
        let d = w[j * n + j];
        if !(d > 0.0 && d < f64::INFINITY) {
            return Err(NotPositiveDefinite { column: j });
        }
        let ljj = d.sqrt();
        w[j * n + j] = ljj;
        for i in j + 1..k1 {
            let x = w[i * n + j] / ljj;
            w[i * n + j] = x;
            column[i - k0] = x;
        }
        for i in j + 1..k1 {
            let lij = column[i - k0];
            let row = &mut w[i * n + j + 1..=i * n + i];
            for (y, &l) in row.iter_mut().zip(&column[j + 1 - k0..=i - k0]) {
                *y -= lij * l;
            }
        }
    }
    Ok(())
}
