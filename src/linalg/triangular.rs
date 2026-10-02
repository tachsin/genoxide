//! Triangular solves with a lower triangular `n × n` matrix `L` (a Cholesky factor): `L x = b`
//! (forward substitution) and `Lᵀ x = b` (backward substitution), for one right-hand side or the
//! `m` columns of an `n × m` matrix, in place. Only the lower triangle of `L` is read.
//!
//! Each column of a matrix right-hand side gets the bits of the single-vector solve.

/// Solves `L x = b` in place: `xᵢ = (bᵢ − lᵢ₀x₀ − lᵢ₁x₁ − … − lᵢ,ᵢ₋₁xᵢ₋₁) / lᵢᵢ`, the products
/// subtracted in ascending order.
pub(crate) fn solve_lower(l: &[f64], n: usize, b: &mut [f64]) {
    debug_assert!(l.len() == n * n && b.len() == n);
    for i in 0..n {
        let row = &l[i * n..i * n + i];
        let (done, rest) = b.split_at_mut(i);
        let mut s = rest[0];
        for (&lik, &xk) in row.iter().zip(&*done) {
            s -= lik * xk;
        }
        rest[0] = s / l[i * n + i];
    }
}

/// Solves `Lᵀ x = b` in place: `xᵢ = (bᵢ − lₙ₋₁,ᵢxₙ₋₁ − … − lᵢ₊₁,ᵢxᵢ₊₁) / lᵢᵢ`, the products
/// subtracted in descending order of the row, along the contiguous rows of `L`.
pub(crate) fn solve_lower_transposed(l: &[f64], n: usize, b: &mut [f64]) {
    debug_assert!(l.len() == n * n && b.len() == n);
    for i in (0..n).rev() {
        let (before, rest) = b.split_at_mut(i);
        let x = rest[0] / l[i * n + i];
        rest[0] = x;
        for (y, &lik) in before.iter_mut().zip(&l[i * n..i * n + i]) {
            *y -= lik * x;
        }
    }
}

/// Solves `L X = B` in place for the `n × m` matrix `B`: [`solve_lower`] on each column.
pub(crate) fn solve_lower_multi(l: &[f64], n: usize, b: &mut [f64], m: usize) {
    debug_assert!(l.len() == n * n && b.len() == n * m);
    if m == 0 {
        return;
    }
    for i in 0..n {
        let (done, rest) = b.split_at_mut(i * m);
        let row = &mut rest[..m];
        for (&lik, xk) in l[i * n..i * n + i].iter().zip(done.chunks_exact(m)) {
            for (y, &x) in row.iter_mut().zip(xk) {
                *y -= lik * x;
            }
        }
        let lii = l[i * n + i];
        for y in row {
            *y /= lii;
        }
    }
}

/// Solves `Lᵀ X = B` in place for the `n × m` matrix `B`: [`solve_lower_transposed`] on each
/// column.
pub(crate) fn solve_lower_transposed_multi(l: &[f64], n: usize, b: &mut [f64], m: usize) {
    debug_assert!(l.len() == n * n && b.len() == n * m);
    if m == 0 {
        return;
    }
    for i in (0..n).rev() {
        let (before, rest) = b.split_at_mut(i * m);
        let row = &mut rest[..m];
        let lii = l[i * n + i];
        for y in row.iter_mut() {
            *y /= lii;
        }
        for (&lik, xk) in l[i * n..i * n + i].iter().zip(before.chunks_exact_mut(m)) {
            for (y, &x) in xk.iter_mut().zip(&*row) {
                *y -= lik * x;
            }
        }
    }
}

/// Solves `A x = b` in place from the Cholesky factor `L` of `A = L Lᵀ`: [`solve_lower`], then
/// [`solve_lower_transposed`].
pub(crate) fn cholesky_solve(l: &[f64], n: usize, b: &mut [f64]) {
    solve_lower(l, n, b);
    solve_lower_transposed(l, n, b);
}

/// Solves `A X = B` in place for the `n × m` matrix `B` from the Cholesky factor `L` of
/// `A = L Lᵀ`: [`cholesky_solve`] on each column.
pub(crate) fn cholesky_solve_multi(l: &[f64], n: usize, b: &mut [f64], m: usize) {
    solve_lower_multi(l, n, b, m);
    solve_lower_transposed_multi(l, n, b, m);
}
