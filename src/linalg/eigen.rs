//! The eigendecomposition of symmetric matrices: a Householder reduction to tridiagonal form and
//! the implicit QL method, as tred2 and tql2 of JAMA (public domain), which Hansen's Java CMA-ES
//! uses too. Only +, −, ×, ÷ and `sqrt`, each element's operations in a fixed order, so the same
//! bits on every platform.
//!
//! JAMA's loops run down the columns of its matrix `V`: they work here on its transpose `w`
//! (`V[r][c]` is `w[c * n + r]`), along contiguous rows, with the same operations in the same
//! order.

/// The eigenvalues and eigenvectors (the columns of a row-major matrix) of the symmetric
/// row-major `n × n` matrix.
// the trust-region subproblem (batch D1) is its first user outside the tests
#[allow(dead_code)]
pub(crate) fn eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let (values, transposed) = eigen_transposed(matrix, n);
    let mut vectors = Vec::new();
    transpose_into(&transposed, n, &mut vectors);
    (values, vectors)
}

/// [`eigen`] with the eigenvectors as the rows of a row-major matrix: CMA-ES's form.
pub(crate) fn eigen_transposed(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    // `V` starts as the symmetric matrix, its own transpose
    debug_assert!((0..n).all(|i| (0..i).all(|j| matrix[i * n + j] == matrix[j * n + i])));
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut w = matrix.to_vec();
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    tridiagonalize(&mut w, &mut d, &mut e, n);
    diagonalize(&mut w, &mut d, &mut e, n);
    (d, w)
}

/// The transpose of the row-major `n × n` matrix into `transposed` (empty for an empty matrix).
pub(crate) fn transpose_into(matrix: &[f64], n: usize, transposed: &mut Vec<f64>) {
    transposed.clear();
    if !matrix.is_empty() {
        for c in 0..n {
            transposed.extend(matrix[c..].iter().step_by(n));
        }
    }
}

// sqrt(a² + b²) without overflow or underflow
fn hypot(a: f64, b: f64) -> f64 {
    let (a, b) = (a.abs(), b.abs());
    let (large, small) = if a > b { (a, b) } else { (b, a) };
    if large == 0.0 {
        0.0
    } else {
        let ratio = small / large;
        large * (1.0 + ratio * ratio).sqrt()
    }
}

// tred2: the Householder reduction of the symmetric `V` to a tridiagonal matrix with diagonal
// `d` and subdiagonal `e[1..]`, and the transformation in `V`, transposed in `w`
fn tridiagonalize(w: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) {
    // the last row of V
    for (j, x) in d.iter_mut().enumerate() {
        *x = w[j * n + n - 1];
    }
    for i in (1..n).rev() {
        let scale: f64 = d[..i].iter().map(|x| x.abs()).sum();
        let mut h = 0.0;
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = w[j * n + i - 1];
                w[j * n + i] = 0.0;
                w[i * n + j] = 0.0;
            }
        } else {
            // the Householder vector
            for x in &mut d[..i] {
                *x /= scale;
                h += *x * *x;
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            e[..i].fill(0.0);
            // the similarity transformation of the remaining columns
            for j in 0..i {
                f = d[j];
                w[i * n + j] = f;
                // column j of V from the diagonal down to row i − 1
                let column = &w[j * n + j..j * n + i];
                g = e[j] + column[0] * f;
                let below = column[1..].iter().zip(&d[j + 1..i]).zip(&mut e[j + 1..i]);
                for ((&x, &dk), ek) in below {
                    g += x * dk;
                    *ek += x * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for (ej, &dj) in e[..i].iter_mut().zip(&d[..i]) {
                *ej /= h;
                f += *ej * dj;
            }
            let hh = f / (h + h);
            for (ej, &dj) in e[..i].iter_mut().zip(&d[..i]) {
                *ej -= hh * dj;
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                let column = &mut w[j * n + j..j * n + i];
                for ((x, &ek), &dk) in column.iter_mut().zip(&e[j..i]).zip(&d[j..i]) {
                    *x -= f * ek + g * dk;
                }
                d[j] = w[j * n + i - 1];
                w[j * n + i] = 0.0;
            }
        }
        d[i] = h;
    }
    // the accumulated transformations
    for i in 0..n - 1 {
        w[i * n + n - 1] = w[i * n + i];
        w[i * n + i] = 1.0;
        let h = d[i + 1];
        // columns 0..=i of V, and column i + 1, down to row i
        let (columns, after) = w.split_at_mut((i + 1) * n);
        let next = &mut after[..=i];
        if h != 0.0 {
            for (dk, &x) in d[..=i].iter_mut().zip(&*next) {
                *dk = x / h;
            }
            for column in columns.chunks_exact_mut(n) {
                let column = &mut column[..=i];
                let g: f64 = next.iter().zip(&*column).map(|(a, b)| a * b).sum();
                for (x, &dk) in column.iter_mut().zip(&d[..=i]) {
                    *x -= g * dk;
                }
            }
        }
        next.fill(0.0);
    }
    for j in 0..n {
        d[j] = w[j * n + n - 1];
        w[j * n + n - 1] = 0.0;
    }
    w[(n - 1) * n + n - 1] = 1.0;
    e[0] = 0.0;
}

// tql2: the eigenvalues (in `d`) and eigenvectors (the columns of `V`, the rows of `w`) of the
// tridiagonal matrix from `tridiagonalize`, by the implicit QL method
fn diagonalize(w: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) {
    e.copy_within(1.., 0);
    e[n - 1] = 0.0;
    let mut f = 0.0;
    let mut largest = 0.0f64;
    for l in 0..n {
        // a small subdiagonal element splits the matrix
        largest = largest.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n - 1 && e[m].abs() > f64::EPSILON * largest {
            m += 1;
        }
        if m > l {
            for _ in 0..100 {
                // the implicit shift
                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = hypot(p, 1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for x in &mut d[l + 2..] {
                    *x -= h;
                }
                f += h;
                // the implicit QL transformation
                p = d[m];
                let (mut c, mut c2, mut c3) = (1.0, 1.0, 1.0);
                let el1 = e[l + 1];
                let (mut s, mut s2) = (0.0, 0.0);
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = hypot(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    // the rotation of columns i and i + 1 of V
                    let (before, after) = w.split_at_mut((i + 1) * n);
                    for (x, y) in before[i * n..].iter_mut().zip(&mut after[..n]) {
                        let h = *y;
                        *y = s * *x + c * h;
                        *x = c * *x - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= f64::EPSILON * largest {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn identity(n: usize) -> Vec<f64> {
        let mut matrix = vec![0.0; n * n];
        for i in 0..n {
            matrix[i * n + i] = 1.0;
        }
        matrix
    }

    fn symmetric(n: usize) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(-100.0..100.0f64, n * n).prop_map(move |mut matrix| {
            for i in 0..n {
                for j in 0..i {
                    matrix[i * n + j] = matrix[j * n + i];
                }
            }
            matrix
        })
    }

    // A = V · diag(values) · Vᵀ, and Vᵀ · V = I
    fn assert_decomposes(matrix: &[f64], n: usize) {
        let (values, vectors) = eigen(matrix, n);
        let scale = matrix.iter().fold(1.0f64, |max, x| max.max(x.abs()));
        for i in 0..n {
            for j in 0..n {
                let rebuilt: f64 = (0..n)
                    .map(|k| vectors[i * n + k] * values[k] * vectors[j * n + k])
                    .sum();
                assert!((rebuilt - matrix[i * n + j]).abs() < 1e-12 * scale * n as f64);
                let dot: f64 = (0..n)
                    .map(|k| vectors[k * n + i] * vectors[k * n + j])
                    .sum();
                let identity = if i == j { 1.0 } else { 0.0 };
                assert!((dot - identity).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn eigen_decomposes_special_matrices() {
        for n in [1, 2, 5, 60] {
            assert_decomposes(&vec![0.0; n * n], n);
            assert_decomposes(&identity(n), n);
            // diagonal with repeated values, and rank one
            let diagonal: Vec<f64> = (0..n * n)
                .map(|k| {
                    if k % (n + 1) == 0 {
                        (k % 3) as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            assert_decomposes(&diagonal, n);
            let y: Vec<f64> = (0..n).map(|i| i as f64 - 2.5).collect();
            let rank_one: Vec<f64> = (0..n * n).map(|k| y[k / n] * y[k % n]).collect();
            assert_decomposes(&rank_one, n);
            // badly conditioned
            let scaled: Vec<f64> = (0..n * n)
                .map(|k| {
                    if k % (n + 1) == 0 {
                        crate::math::pow(10.0, -((k % 15) as f64))
                    } else {
                        1e-9
                    }
                })
                .collect();
            assert_decomposes(&scaled, n);
        }
    }

    proptest! {
        #[test]
        fn eigen_decomposes_symmetric_matrices(
            (n, matrix) in (1usize..30).prop_flat_map(|n| (Just(n), symmetric(n)))
        ) {
            assert_decomposes(&matrix, n);
        }
    }
}
