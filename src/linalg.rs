//! Dense linear algebra on row-major `f64` slices with explicit dimensions, the same bits on every
//! platform and thread count.
//!
//! genoxide's guarantee is that a seed gives the same results to the bit everywhere. IEEE 754
//! rounds each `+`, `−`, `×`, `÷` and `sqrt` exactly, so a computation gives the same bits on every
//! CPU as long as each output element goes through the same operations in the same order. The code
//! here keeps to that:
//!
//! - **One fixed order per element.** Each function's docs give the order of its sums: the
//!   blocking, packing, register tiles and threads only decide which *different* elements are
//!   computed together, never the order of one element's operations. The blocked Cholesky and
//!   `gemm` give the same bits as their textbook loops, which the tests check.
//! - **No fused operations.** No `mul_add`: an FMA rounds once where `a * b + c` rounds twice,
//!   and CPUs without FMA would differ. Rust never contracts `a * b + c` into an FMA by itself.
//! - **Threads only across independent outputs.** Rayon (with the `parallel` feature) splits the
//!   rows of an output, never a sum: no `sum` or `reduce` of floats over rayon's splits, whose
//!   order depends on the thread count.
//! - **No runtime-dispatched SIMD.** The crate has no `unsafe` code, so no `std::arch` kernels
//!   chosen at run time. LLVM vectorizes the element-wise loops (the register tiles) across
//!   independent elements, which keeps every element's order; it never reorders a floating-point
//!   sum without fast-math flags. A build with `target-cpu=native` gives the same bits too.
//! - **Transcendental functions** come from [`math`](crate::math), never from the platform.
//!
//! The tests compare the results with fixed hashes of their bits on seeded inputs, so CI on Linux,
//! macOS (arm64) and Windows checks the portability.
//!
//! Matrices are row-major: element `(i, j)` of an `m × n` matrix is at `i * n + j`.

pub(crate) mod blas;
pub(crate) mod cholesky;
pub(crate) mod eigen;
pub(crate) mod triangular;

#[cfg(test)]
mod tests;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// the register tile of the products: MR rows by NR columns of the output, in registers
const MR: usize = 4;
const NR: usize = 4;
// the rows of an output updated together, a multiple of MR: their packed rows of the left factor
// stay in the cache, and each packed column tile of the right one is reused by all their tiles
const CHUNK_ROWS: usize = 32;

// `f(index, chunk)` for each chunk of `size` elements of `data` (the last one may be shorter), on
// rayon when `parallel` is true and the `parallel` feature is on. The chunks must be independent,
// so the results don't depend on it.
fn for_each_chunk<F>(data: &mut [f64], size: usize, parallel: bool, f: F)
where
    F: Fn(usize, &mut [f64]) + Send + Sync,
{
    #[cfg(feature = "parallel")]
    if parallel {
        data.par_chunks_mut(size)
            .enumerate()
            .for_each(|(index, chunk)| f(index, chunk));
        return;
    }
    let _ = parallel;
    for (index, chunk) in data.chunks_mut(size).enumerate() {
        f(index, chunk);
    }
}

// acc[r][c] += a[k][r] · b[k][c] for k in 0..len in order, from `a` packed as len × MR and `b`
// as len × NR: each element's products added one at a time
#[inline(always)]
fn tile_add(acc: &mut [[f64; NR]; MR], a: &[f64], b: &[f64]) {
    let mut x = *acc;
    for (a, b) in a.as_chunks::<MR>().0.iter().zip(b.as_chunks::<NR>().0) {
        for (row, &ar) in x.iter_mut().zip(a) {
            for (y, &bc) in row.iter_mut().zip(b) {
                *y += ar * bc;
            }
        }
    }
    *acc = x;
}

// `tile_add` subtracting the products
#[inline(always)]
fn tile_sub(acc: &mut [[f64; NR]; MR], a: &[f64], b: &[f64]) {
    let mut x = *acc;
    for (a, b) in a.as_chunks::<MR>().0.iter().zip(b.as_chunks::<NR>().0) {
        for (row, &ar) in x.iter_mut().zip(a) {
            for (y, &bc) in row.iter_mut().zip(b) {
                *y -= ar * bc;
            }
        }
    }
    *acc = x;
}
