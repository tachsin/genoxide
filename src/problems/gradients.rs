//! The analytic gradients of the smooth classic functions: `∂f / ∂xᵢ` into `gradient[i]`, with
//! `gradient` as long as `x` and zeroed. Each is the derivative of the formula in its problem's
//! docs, written out by hand, and tested against central differences.

use super::classic::{
    FOXHOLES, HARTMANN_3_A, HARTMANN_3_P, HARTMANN_6_A, HARTMANN_6_P, HARTMANN_C, KOWALIK_A,
    KOWALIK_B_INVERSE, LANGERMANN_A, LANGERMANN_C, MICHALEWICZ_M, SHEKEL_A, SHEKEL_C,
};
use crate::math;
use std::f64::consts::PI;

// Σ xᵢ²: 2xᵢ
pub(super) fn sphere(x: &[f64], gradient: &mut [f64]) {
    for (g, xi) in gradient.iter_mut().zip(x) {
        *g = 2.0 * xi;
    }
}

// Σ i xᵢ² (i from 1): 2 i xᵢ
pub(super) fn axis_parallel_ellipsoid(x: &[f64], gradient: &mut [f64]) {
    for (i, (g, xi)) in gradient.iter_mut().zip(x).enumerate() {
        *g = 2.0 * (i + 1) as f64 * xi;
    }
}

// Σₖ Pₖ² with Pₖ = Σ_{j ≤ k} xⱼ: ∂/∂xᵢ = 2 Σ_{k ≥ i} Pₖ, the sums of the prefixes from i on
pub(super) fn schwefel_1_2(x: &[f64], gradient: &mut [f64]) {
    let mut prefix = 0.0;
    for (g, xi) in gradient.iter_mut().zip(x) {
        prefix += xi;
        *g = prefix;
    }
    let mut suffix = 0.0;
    for g in gradient.iter_mut().rev() {
        suffix += *g;
        *g = 2.0 * suffix;
    }
}

// 10n + Σ (xᵢ² − 10 cos 2πxᵢ): 2xᵢ + 20π sin 2πxᵢ
pub(super) fn rastrigin(x: &[f64], gradient: &mut [f64]) {
    for (g, xi) in gradient.iter_mut().zip(x) {
        *g = 2.0 * xi + 20.0 * PI * math::sin(2.0 * PI * xi);
    }
}

// Σ 100 (xᵢ₊₁ − xᵢ²)² + (xᵢ − 1)²: each term adds −400 xᵢ (xᵢ₊₁ − xᵢ²) + 2 (xᵢ − 1) to gene i
// and 200 (xᵢ₊₁ − xᵢ²) to gene i + 1
pub(super) fn rosenbrock(x: &[f64], gradient: &mut [f64]) {
    for (i, pair) in x.windows(2).enumerate() {
        let (xi, next) = (pair[0], pair[1]);
        let valley = next - xi * xi;
        gradient[i] += -400.0 * xi * valley + 2.0 * (xi - 1.0);
        gradient[i + 1] += 200.0 * valley;
    }
}

// 20 − 20 exp(−0.2 r) + e − exp(C / n), with r = √(Σ xᵢ² / n) and C = Σ cos 2πxᵢ:
// 4 exp(−0.2 r) xᵢ / (n r) + (2π / n) exp(C / n) sin 2πxᵢ. At the origin, where the first term
// has a cone, its derivative is taken as 0, the midpoint of its one-sided ones
pub(super) fn ackley(x: &[f64], gradient: &mut [f64]) {
    let n = x.len() as f64;
    let r = (x.iter().map(|xi| xi * xi).sum::<f64>() / n).sqrt();
    let cosines = x.iter().map(|xi| math::cos(2.0 * PI * xi)).sum::<f64>() / n;
    let radial = if r > 0.0 {
        4.0 * math::exp(-0.2 * r) / (n * r)
    } else {
        0.0
    };
    let wave = 2.0 * PI / n * math::exp(cosines);
    for (g, xi) in gradient.iter_mut().zip(x) {
        *g = radial * xi + wave * math::sin(2.0 * PI * xi);
    }
}

// 1 + Σ xᵢ² / 4000 − Π cos(xᵢ / √i): xᵢ / 2000 + sin(xᵢ / √i) / √i · Π_{j ≠ i} cos(xⱼ / √j), the
// product without i from the products of the cosines before and after it
pub(super) fn griewank(x: &[f64], gradient: &mut [f64]) {
    let scale = |i: usize| ((i + 1) as f64).sqrt();
    // the product of the cosines before gene i, then times the product after it
    let mut before = 1.0;
    for (i, (g, xi)) in gradient.iter_mut().zip(x).enumerate() {
        *g = before;
        before *= math::cos(xi / scale(i));
    }
    let mut after = 1.0;
    for (i, (g, xi)) in gradient.iter_mut().zip(x).enumerate().rev() {
        let others = *g * after;
        *g = xi / 2000.0 + math::sin(xi / scale(i)) / scale(i) * others;
        after *= math::cos(xi / scale(i));
    }
}

// −Σ xᵢ sin √|xᵢ|: −(sin u + (u / 2) cos u) with u = √|xᵢ|, for either sign of xᵢ (0 at 0)
pub(super) fn schwefel_2_26(x: &[f64], gradient: &mut [f64]) {
    for (g, xi) in gradient.iter_mut().zip(x) {
        let u = xi.abs().sqrt();
        let (sin, cos) = math::sin_cos(u);
        *g = -(sin + u / 2.0 * cos);
    }
}

// Levy's function of wᵢ = 1 + (xᵢ − 1) / 4: sin²(πw₁) + Σ_{i < n} (wᵢ − 1)² (1 + 10 sin²(πwᵢ + 1))
// + (wₙ − 1)² (1 + sin²(2πwₙ)), each term's derivative in w times dw/dx = 1/4
pub(super) fn levy(x: &[f64], gradient: &mut [f64]) {
    let n = x.len();
    let w = |xi: f64| 1.0 + (xi - 1.0) / 4.0;
    for (i, (g, &xi)) in gradient.iter_mut().zip(x).enumerate() {
        let wi = w(xi);
        let mut derivative = 0.0;
        if i == 0 {
            // d/dw sin²(πw) = π sin(2πw)
            derivative += PI * math::sin(2.0 * PI * wi);
        }
        if i + 1 < n {
            let (sin, cos) = math::sin_cos(PI * wi + 1.0);
            derivative += 2.0 * (wi - 1.0) * (1.0 + 10.0 * sin * sin)
                + math::powi(wi - 1.0, 2) * 20.0 * PI * sin * cos;
        } else {
            let (sin, cos) = math::sin_cos(2.0 * PI * wi);
            derivative += 2.0 * (wi - 1.0) * (1.0 + sin * sin)
                + math::powi(wi - 1.0, 2) * 4.0 * PI * sin * cos;
        }
        *g = derivative / 4.0;
    }
}

// Σ xᵢ² + W² + W⁴ with W = Σ i xᵢ / 2: 2xᵢ + (2W + 4W³) i / 2
pub(super) fn zakharov(x: &[f64], gradient: &mut [f64]) {
    let weighted: f64 = x
        .iter()
        .enumerate()
        .map(|(i, xi)| 0.5 * (i + 1) as f64 * xi)
        .sum();
    let outer = 2.0 * weighted + 4.0 * math::powi(weighted, 3);
    for (i, (g, xi)) in gradient.iter_mut().zip(x).enumerate() {
        *g = 2.0 * xi + outer * 0.5 * (i + 1) as f64;
    }
}

// ½ Σ (xᵢ⁴ − 16xᵢ² + 5xᵢ): 2xᵢ³ − 16xᵢ + 2.5
pub(super) fn styblinski_tang(x: &[f64], gradient: &mut [f64]) {
    for (g, &xi) in gradient.iter_mut().zip(x) {
        *g = 2.0 * math::powi(xi, 3) - 16.0 * xi + 2.5;
    }
}

// −Σ sin(xᵢ) sⁱ^(2m) with sᵢ = sin(i xᵢ² / π): −cos(xᵢ) s^(2m) − sin(xᵢ) 2m s^(2m − 1)
// cos(i xᵢ² / π) 2i xᵢ / π
pub(super) fn michalewicz(x: &[f64], gradient: &mut [f64]) {
    let m = MICHALEWICZ_M;
    for (i, (g, &xi)) in gradient.iter_mut().zip(x).enumerate() {
        let index = (i + 1) as f64;
        let (sin_x, cos_x) = math::sin_cos(xi);
        let (s, c) = math::sin_cos(index * xi * xi / PI);
        let power = math::powi(s, 2 * m - 1);
        *g = -cos_x * power * s - sin_x * f64::from(2 * m) * power * c * 2.0 * index * xi / PI;
    }
}

// (x₁ − 1)² + Σ_{i ≥ 2} i (2xᵢ² − xᵢ₋₁)²: each term adds −2i u to gene i − 1 and 8i u xᵢ to gene
// i, with u = 2xᵢ² − xᵢ₋₁
pub(super) fn dixon_price(x: &[f64], gradient: &mut [f64]) {
    let Some(&first) = x.first() else {
        return;
    };
    gradient[0] = 2.0 * (first - 1.0);
    for (k, pair) in x.windows(2).enumerate() {
        let weight = (k + 2) as f64;
        let u = 2.0 * pair[1] * pair[1] - pair[0];
        gradient[k] += -2.0 * weight * u;
        gradient[k + 1] += 8.0 * weight * u * pair[1];
    }
}

// Σ (xᵢ − 1)² − Σ xᵢ xᵢ₊₁: 2 (xᵢ − 1) − xᵢ₋₁ − xᵢ₊₁
pub(super) fn trid(x: &[f64], gradient: &mut [f64]) {
    for (i, (g, xi)) in gradient.iter_mut().zip(x).enumerate() {
        let previous = if i > 0 { x[i - 1] } else { 0.0 };
        let next = x.get(i + 1).copied().unwrap_or(0.0);
        *g = 2.0 * (xi - 1.0) - previous - next;
    }
}

// per block of four, (x₁ + 10x₂)² + 5 (x₃ − x₄)² + (x₂ − 2x₃)⁴ + 10 (x₁ − x₄)⁴; genes beyond the
// whole blocks don't count
pub(super) fn powell(x: &[f64], gradient: &mut [f64]) {
    let (blocks, _) = x.as_chunks::<4>();
    for (block, g) in blocks.iter().zip(gradient.as_chunks_mut::<4>().0) {
        let [x1, x2, x3, x4] = *block;
        let a = x1 + 10.0 * x2;
        let b = x3 - x4;
        let c = math::powi(x2 - 2.0 * x3, 3);
        let d = math::powi(x1 - x4, 3);
        *g = [
            2.0 * a + 40.0 * d,
            20.0 * a + 4.0 * c,
            10.0 * b - 8.0 * c,
            -10.0 * b - 40.0 * d,
        ];
    }
}

// (x₁² + x₂ − 11)² + (x₁ + x₂² − 7)²
pub(super) fn himmelblau(x: &[f64], gradient: &mut [f64]) {
    let (x1, x2) = (x[0], x[1]);
    let a = x1 * x1 + x2 - 11.0;
    let b = x1 + x2 * x2 - 7.0;
    gradient[0] = 4.0 * x1 * a + 2.0 * b;
    gradient[1] = 2.0 * a + 4.0 * x2 * b;
}

// u² + 10 (1 − t) cos x₁ + 10, with u = x₂ − b x₁² + c x₁ − 6
pub(super) fn branin(x: &[f64], gradient: &mut [f64]) {
    let (x1, x2) = (x[0], x[1]);
    let b = 5.1 / (4.0 * PI * PI);
    let c = 5.0 / PI;
    let t = 1.0 / (8.0 * PI);
    let u = x2 - b * x1 * x1 + c * x1 - 6.0;
    gradient[0] = 2.0 * u * (c - 2.0 * b * x1) - 10.0 * (1.0 - t) * math::sin(x1);
    gradient[1] = 2.0 * u;
}

// A B, with A = 1 + p² q, p = x₁ + x₂ + 1, q = 19 − 14x₁ + 3x₁² − 14x₂ + 6x₁x₂ + 3x₂², and
// B = 30 + r² s, r = 2x₁ − 3x₂, s = 18 − 32x₁ + 12x₁² + 48x₂ − 36x₁x₂ + 27x₂²
pub(super) fn goldstein_price(x: &[f64], gradient: &mut [f64]) {
    let (x1, x2) = (x[0], x[1]);
    let p = x1 + x2 + 1.0;
    let q = 19.0 - 14.0 * x1 + 3.0 * x1 * x1 - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2 * x2;
    let a = 1.0 + p * p * q;
    // ∂q/∂x₁ = ∂q/∂x₂ = −14 + 6x₁ + 6x₂, and ∂p/∂x₁ = ∂p/∂x₂ = 1
    let da = 2.0 * p * q + p * p * (-14.0 + 6.0 * x1 + 6.0 * x2);
    let r = 2.0 * x1 - 3.0 * x2;
    let s = 18.0 - 32.0 * x1 + 12.0 * x1 * x1 + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2 * x2;
    let b = 30.0 + r * r * s;
    let db1 = 4.0 * r * s + r * r * (-32.0 + 24.0 * x1 - 36.0 * x2);
    let db2 = -6.0 * r * s + r * r * (48.0 - 36.0 * x1 + 54.0 * x2);
    gradient[0] = da * b + a * db1;
    gradient[1] = da * b + a * db2;
}

// (4 − 2.1x₁² + x₁⁴ / 3) x₁² + x₁x₂ + (−4 + 4x₂²) x₂²
pub(super) fn six_hump_camel(x: &[f64], gradient: &mut [f64]) {
    let (x1, x2) = (x[0], x[1]);
    gradient[0] = 8.0 * x1 - 8.4 * math::powi(x1, 3) + 2.0 * math::powi(x1, 5) + x2;
    gradient[1] = x1 - 8.0 * x2 + 16.0 * math::powi(x2, 3);
}

// −Σᵢ cᵢ exp(−Dᵢ), Dᵢ = Σⱼ aᵢⱼ (xⱼ − pᵢⱼ)²: Σᵢ cᵢ exp(−Dᵢ) 2 aᵢⱼ (xⱼ − pᵢⱼ)
fn hartmann<const N: usize>(a: &[[f64; N]; 4], p: &[[f64; N]; 4], x: &[f64], gradient: &mut [f64]) {
    let x = &x[..N];
    for i in 0..4 {
        let distance: f64 = (0..N)
            .map(|j| a[i][j] * math::powi(x[j] - p[i][j], 2))
            .sum();
        let weight = HARTMANN_C[i] * math::exp(-distance);
        for j in 0..N {
            gradient[j] += weight * 2.0 * a[i][j] * (x[j] - p[i][j]);
        }
    }
}

pub(super) fn hartmann_3(x: &[f64], gradient: &mut [f64]) {
    hartmann(&HARTMANN_3_A, &HARTMANN_3_P, x, gradient);
}

pub(super) fn hartmann_6(x: &[f64], gradient: &mut [f64]) {
    hartmann(&HARTMANN_6_A, &HARTMANN_6_P, x, gradient);
}

// −Σᵢ 1 / (Dᵢ + cᵢ), Dᵢ = Σⱼ (xⱼ − aᵢⱼ)²: Σᵢ 2 (xⱼ − aᵢⱼ) / (Dᵢ + cᵢ)²
pub(super) fn shekel(m: usize, x: &[f64], gradient: &mut [f64]) {
    let x = &x[..4];
    for i in 0..m {
        let distance: f64 = (0..4).map(|j| math::powi(x[j] - SHEKEL_A[i][j], 2)).sum();
        let weight = 1.0 / math::powi(distance + SHEKEL_C[i], 2);
        for j in 0..4 {
            gradient[j] += 2.0 * (x[j] - SHEKEL_A[i][j]) * weight;
        }
    }
}

// −cos x₁ cos x₂ E, E = exp(−(x₁ − π)² − (x₂ − π)²)
pub(super) fn easom(x: &[f64], gradient: &mut [f64]) {
    let (x1, x2) = (x[0], x[1]);
    let (sin1, cos1) = math::sin_cos(x1);
    let (sin2, cos2) = math::sin_cos(x2);
    let e = math::exp(-(math::powi(x1 - PI, 2) + math::powi(x2 - PI, 2)));
    gradient[0] = e * (sin1 * cos2 + 2.0 * (x1 - PI) * cos1 * cos2);
    gradient[1] = e * (cos1 * sin2 + 2.0 * (x2 - PI) * cos1 * cos2);
}

// 0.5 + (sin² r − 0.5) / (1 + 0.001 s)², s = x₁² + x₂², r = √s: with d sin²r / dxᵢ =
// (sin 2r / r) xᵢ, which is 2xᵢ at the origin
pub(super) fn schaffer_f6(x: &[f64], gradient: &mut [f64]) {
    let s = x[0] * x[0] + x[1] * x[1];
    let r = s.sqrt();
    let (sin, cos) = math::sin_cos(r);
    let numerator = sin * sin - 0.5;
    let base = 1.0 + 0.001 * s;
    let denominator = base * base;
    let sinc = if r > 0.0 { 2.0 * sin * cos / r } else { 2.0 };
    let factor = sinc / denominator - numerator * 0.004 * base / (denominator * denominator);
    gradient[0] = factor * x[0];
    gradient[1] = factor * x[1];
}

// (1.5 − x₁ + x₁x₂)² + (2.25 − x₁ + x₁x₂²)² + (2.625 − x₁ + x₁x₂³)²
pub(super) fn beale(x: &[f64], gradient: &mut [f64]) {
    let (x1, x2) = (x[0], x[1]);
    let t1 = 1.5 - x1 + x1 * x2;
    let t2 = 2.25 - x1 + x1 * x2 * x2;
    let t3 = 2.625 - x1 + x1 * math::powi(x2, 3);
    gradient[0] = 2.0 * (t1 * (x2 - 1.0) + t2 * (x2 * x2 - 1.0) + t3 * (math::powi(x2, 3) - 1.0));
    gradient[1] = 2.0 * x1 * (t1 + 2.0 * t2 * x2 + 3.0 * t3 * x2 * x2);
}

// (x₁ + 2x₂ − 7)² + (2x₁ + x₂ − 5)²
pub(super) fn booth(x: &[f64], gradient: &mut [f64]) {
    let a = x[0] + 2.0 * x[1] - 7.0;
    let b = 2.0 * x[0] + x[1] - 5.0;
    gradient[0] = 2.0 * a + 4.0 * b;
    gradient[1] = 4.0 * a + 2.0 * b;
}

// 0.26 (x₁² + x₂²) − 0.48 x₁x₂
pub(super) fn matyas(x: &[f64], gradient: &mut [f64]) {
    gradient[0] = 0.52 * x[0] - 0.48 * x[1];
    gradient[1] = 0.52 * x[1] - 0.48 * x[0];
}

// x₁² + 2x₂² − 0.3 cos 3πx₁ − 0.4 cos 4πx₂ + 0.7
pub(super) fn bohachevsky_1(x: &[f64], gradient: &mut [f64]) {
    gradient[0] = 2.0 * x[0] + 0.9 * PI * math::sin(3.0 * PI * x[0]);
    gradient[1] = 4.0 * x[1] + 1.6 * PI * math::sin(4.0 * PI * x[1]);
}

// x₁² + 2x₂² − 0.3 cos 3πx₁ cos 4πx₂ + 0.3
pub(super) fn bohachevsky_2(x: &[f64], gradient: &mut [f64]) {
    let (sin1, cos1) = math::sin_cos(3.0 * PI * x[0]);
    let (sin2, cos2) = math::sin_cos(4.0 * PI * x[1]);
    gradient[0] = 2.0 * x[0] + 0.9 * PI * sin1 * cos2;
    gradient[1] = 4.0 * x[1] + 1.2 * PI * cos1 * sin2;
}

// x₁² + 2x₂² − 0.3 cos(3πx₁ + 4πx₂) + 0.3
pub(super) fn bohachevsky_3(x: &[f64], gradient: &mut [f64]) {
    let sin = math::sin(3.0 * PI * x[0] + 4.0 * PI * x[1]);
    gradient[0] = 2.0 * x[0] + 0.9 * PI * sin;
    gradient[1] = 4.0 * x[1] + 1.2 * PI * sin;
}

// 2x₁² − 1.05x₁⁴ + x₁⁶ / 6 + x₁x₂ + x₂²
pub(super) fn three_hump_camel(x: &[f64], gradient: &mut [f64]) {
    let (x1, x2) = (x[0], x[1]);
    gradient[0] = 4.0 * x1 - 4.2 * math::powi(x1, 3) + math::powi(x1, 5) + x2;
    gradient[1] = x1 + 2.0 * x2;
}

// Σᵢ cᵢ exp(−dᵢ / π) cos(π dᵢ), dᵢ = |x − aᵢ|²: d/dd = −cᵢ exp(−d / π) (cos(πd) / π + π sin(πd)),
// times ∂d/∂xⱼ = 2 (xⱼ − aᵢⱼ)
pub(super) fn langermann(x: &[f64], gradient: &mut [f64]) {
    for (a, c) in LANGERMANN_A.iter().zip(LANGERMANN_C) {
        let distance = math::powi(x[0] - a[0], 2) + math::powi(x[1] - a[1], 2);
        let (sin, cos) = math::sin_cos(PI * distance);
        let slope = -c * math::exp(-distance / PI) * (cos / PI + PI * sin);
        gradient[0] += slope * 2.0 * (x[0] - a[0]);
        gradient[1] += slope * 2.0 * (x[1] - a[1]);
    }
}

// 1 / S, S = 1/500 + Σⱼ 1 / Dⱼ, Dⱼ = j + (x₁ − a₁ⱼ)⁶ + (x₂ − a₂ⱼ)⁶: −S' / S², with
// ∂S/∂xₖ = −Σⱼ 6 (xₖ − aₖⱼ)⁵ / Dⱼ²
pub(super) fn shekel_foxholes(x: &[f64], gradient: &mut [f64]) {
    let mut sum = 1.0 / 500.0;
    let mut slopes = [0.0; 2];
    // in the order of the evaluation: x₁'s centers varying first
    let holes = FOXHOLES
        .iter()
        .flat_map(|a2| FOXHOLES.iter().map(move |a1| (a1, a2)));
    for (j, (a1, a2)) in holes.enumerate() {
        let depth = (j + 1) as f64;
        let (d1, d2) = (x[0] - a1, x[1] - a2);
        let denominator = depth + math::powi(d1, 6) + math::powi(d2, 6);
        sum += 1.0 / denominator;
        let weight = 6.0 / (denominator * denominator);
        slopes[0] -= weight * math::powi(d1, 5);
        slopes[1] -= weight * math::powi(d2, 5);
    }
    let scale = -1.0 / (sum * sum);
    gradient[0] = scale * slopes[0];
    gradient[1] = scale * slopes[1];
}

// Σᵢ (aᵢ − mᵢ)², mᵢ = x₁ Nᵢ / Dᵢ, Nᵢ = bᵢ² + bᵢx₂, Dᵢ = bᵢ² + bᵢx₃ + x₄: −2 (aᵢ − mᵢ) ∂mᵢ/∂x
pub(super) fn kowalik(x: &[f64], gradient: &mut [f64]) {
    let x = &x[..4];
    for (a, b_inverse) in KOWALIK_A.iter().zip(KOWALIK_B_INVERSE) {
        let b = 1.0 / b_inverse;
        let numerator = b * b + b * x[1];
        let denominator = b * b + b * x[2] + x[3];
        let model = x[0] * numerator / denominator;
        let residual = -2.0 * (a - model);
        let ratio = x[0] * numerator / (denominator * denominator);
        gradient[0] += residual * numerator / denominator;
        gradient[1] += residual * x[0] * b / denominator;
        gradient[2] += residual * -ratio * b;
        gradient[3] += residual * -ratio;
    }
}

#[cfg(test)]
mod tests;
