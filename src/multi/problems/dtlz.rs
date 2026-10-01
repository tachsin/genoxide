//! DTLZ1-7: Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). *Scalable Test Problems
//! for Evolutionary Multi-Objective Optimization.* TIK-Report 112, ETH Zürich, section 8
//! (eqs. 20-27); and (2002). Scalable multi-objective optimization test problems. *Proceedings of
//! the 2002 Congress on Evolutionary Computation*: 825-830, section VII.
//!
//! The numbering is the report's, the common one. The paper has seven problems, numbered
//! differently from DTLZ5 on: its DTLZ5 is the report's DTLZ6, its DTLZ6 the report's DTLZ7 and
//! its DTLZ7 the report's DTLZ8; the report's DTLZ5 and DTLZ9 aren't in it. DTLZ1-4 are the same
//! in both, and cite the paper; DTLZ5-7 cite the report.

use super::{MultiProblem, das_dennis, divisions_for, evenly};
use crate::genome::{Real, Reals};
use crate::math;
use crate::multi::MultiFitnessFunction;
use std::f64::consts::PI;

const REFERENCE: &str = "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable \
                         multi-objective optimization test problems. Proceedings of the 2002 \
                         Congress on Evolutionary Computation: 825-830.";
const REFERENCE_URL: &str = "https://doi.org/10.1109/CEC.2002.1007032";

const REPORT: &str = "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). Scalable Test \
                      Problems for Evolutionary Multi-Objective Optimization. TIK-Report 112, \
                      Computer Engineering and Networks Laboratory, ETH Zürich.";
const REPORT_URL: &str = "https://sop.tik.ee.ethz.ch/publicationListFiles/dtlz2001a.pdf";

macro_rules! dtlz {
    ($(#[$doc:meta])* $name:ident, $label:literal, $k:literal, $reference:expr, $url:expr) => {
        $(#[$doc])*
        ///
        #[doc = concat!("With `M` objectives and `n` variables in [0, 1]: the first `M − 1` place a solution on the front, the other `k = n − M + 1` its distance to it; ", stringify!($k), " is the standard `k`.")]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name<const M: usize> {
            variables: usize,
        }

        impl<const M: usize> $name<M> {
            /// The problem with `variables` variables, at least `M`.
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives, or fewer variables than objectives.
            pub fn new(variables: usize) -> Self {
                assert!(M >= 2, "{} needs at least 2 objectives", $label);
                assert!(variables >= M, "{} needs at least as many variables as objectives", $label);
                Self { variables }
            }

            /// The number of variables.
            pub fn variables(&self) -> usize {
                self.variables
            }
        }

        impl<const M: usize> Default for $name<M> {
            #[doc = concat!("The standard problem, with `M + ", stringify!($k), " − 1` variables.")]
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives.
            fn default() -> Self {
                Self::new(M + $k - 1)
            }
        }

        impl<const M: usize> MultiProblem<M> for $name<M> {
            type Representation = Real;

            fn name(&self) -> &'static str {
                $label
            }

            fn representation(&self) -> Real {
                Real::uniform(self.variables, 0.0..=1.0).expect("valid bounds")
            }

            fn reference(&self) -> &'static str {
                $reference
            }

            fn reference_url(&self) -> Option<&'static str> {
                Some($url)
            }

            fn optimal_front(&self, points: usize) -> Option<Vec<[f64; M]>> {
                Self::known_front(points)
            }

            fn ideal_point(&self) -> Option<[f64; M]> {
                Self::ideal()
            }

            fn nadir_point(&self) -> Option<[f64; M]> {
                Self::nadir()
            }
        }
    };
}

dtlz!(
    /// DTLZ1: a linear front, the objectives summing to 1/2, behind 11ᵏ − 1 local fronts.
    ///
    /// `g = 100 (k + Σ ((xᵢ − 0.5)² − cos(20π (xᵢ − 0.5))))` over the last k variables;
    /// `f₁ = ½ x₁ ⋯ x_{M−1} (1 + g)`, …, `f_M = ½ (1 − x₁) (1 + g)`. The optimal solutions have
    /// the last k variables at 0.5.
    Dtlz1,
    "DTLZ1",
    5,
    REFERENCE,
    REFERENCE_URL
);
dtlz!(
    /// DTLZ2: a spherical front, the squared objectives summing to 1.
    ///
    /// `g = Σ (xᵢ − 0.5)²` over the last k variables; `f₁ = (1 + g) cos(x₁π/2) ⋯
    /// cos(x_{M−1}π/2)`, …, `f_M = (1 + g) sin(x₁π/2)`. The optimal solutions have the last k
    /// variables at 0.5.
    Dtlz2,
    "DTLZ2",
    10,
    REFERENCE,
    REFERENCE_URL
);
dtlz!(
    /// DTLZ3: the spherical front of DTLZ2 behind 3ᵏ − 1 local fronts: DTLZ2 with the `g` of
    /// DTLZ1.
    Dtlz3,
    "DTLZ3",
    10,
    REFERENCE,
    REFERENCE_URL
);
dtlz!(
    /// DTLZ4: the spherical front of DTLZ2, with solutions biased towards the `f_M`-`f₁` plane:
    /// DTLZ2 with each `xᵢ` of the angles raised to the power 100.
    Dtlz4,
    "DTLZ4",
    10,
    REFERENCE,
    REFERENCE_URL
);

dtlz!(
    /// DTLZ5 (the report's numbering; not in the 2002 paper): DTLZ2 with its angles mapped so
    /// that the front is a curve, a quarter circle, for 2 and 3 objectives; for 4 or more, the
    /// front isn't a curve, and isn't known.
    ///
    /// `g = Σ (xᵢ − 0.5)²` over the last k variables, `θ₁ = x₁π/2`,
    /// `θᵢ = π (1 + 2g xᵢ) / (4 (1 + g))` for i from 2 to M − 1, and `f₁ = (1 + g) cos θ₁ ⋯
    /// cos θ_{M−1}`, …, `f_M = (1 + g) sin θ₁` (the report's eq. 25, p. 20, with the mapping of
    /// its eq. 10, p. 8). The report's eq. 25 has two typos: it writes `cos(θᵢπ/2)` for `cos θᵢ`,
    /// and doesn't define θ₁. Its eq. 8 (p. 7, the sphere problem that eq. 10 maps) has
    /// `θ₁ = x₁π/2` and `cos θᵢ`, and eq. 10 says θᵢ = π/4 at g = 0, which only `cos θᵢ` fits;
    /// eq. 25's `g(r)` is `g(x_M)`.
    ///
    /// The optimal solutions have the last k variables at 0.5: g = 0 and θ₂ = … = θ_{M−1} = π/4.
    /// With 2 objectives there's no θ₂, and the front is DTLZ2's quarter circle; with 3, it's the
    /// curve `f₁ = f₂ = cos θ₁ / √2`, `f₃ = sin θ₁`, from (1/√2, 1/√2, 0) to (0, 0, 1), which
    /// [`optimal_front`] spreads evenly in θ₁, and so in length. With 4 or more, some solutions
    /// with g > 0 aren't dominated by any point of the curve, as Huband, Hingston, Barone and
    /// While (2006, *IEEE Transactions on Evolutionary Computation* 10(5): 477-506, section
    /// VI-A3) report (checked in their paper, the copy read for WFG; the tests show such a
    /// solution, not the paper's four-objective example): `optimal_front` and
    /// `nadir_point` are `None` then. `ideal_point` is the origin for every M: every objective is
    /// at least 0, and it's 0 at (0, …, 0, 1) for all but the last, and at the curve's other end
    /// for the last, both optimal.
    ///
    /// [`optimal_front`]: MultiProblem::optimal_front
    Dtlz5,
    "DTLZ5",
    10,
    REPORT,
    REPORT_URL
);
dtlz!(
    /// DTLZ6 (the report's numbering; the 2002 paper's DTLZ5): DTLZ5 with `g = Σ xᵢ^0.1` over
    /// the last k variables (the report's eq. 26, p. 21), which makes the same front harder to
    /// reach.
    ///
    /// The optimal solutions have the last k variables at 0, where g = 0. The front and its
    /// ideal and nadir points are [`Dtlz5`]'s: known for 2 and 3 objectives, and only the ideal
    /// point for more. The paper (section VII-E, p. 829) states the same problem, with the same
    /// mapping of the angles.
    Dtlz6,
    "DTLZ6",
    10,
    REPORT,
    REPORT_URL
);
dtlz!(
    /// DTLZ7 (the report's numbering; the 2002 paper's DTLZ6): a front of 2^(M−1) disconnected
    /// regions.
    ///
    /// `fᵢ = xᵢ` for i from 1 to M − 1, `g = 1 + 9 Σ xᵢ / k` over the last k variables,
    /// `h = M − Σᵢ₌₁^{M−1} fᵢ (1 + sin 3πfᵢ) / (1 + g)` and `f_M = (1 + g) h` (the report's
    /// eq. 27, p. 22; the paper's eq. 10, p. 829). The optimal solutions have the last k
    /// variables at 0, g = 1, so the front is `f_M = 2M − Σ φ(fᵢ)` with `φ(f) = f (1 + sin 3πf)`.
    ///
    /// The regions, derived here: f_M falls as each φ(fᵢ) rises, and each fᵢ counts on its own,
    /// so a point is optimal when each fᵢ is optimal for the two objectives (fᵢ, −φ(fᵢ)): when
    /// φ(fᵢ) is above φ at every smaller value. That's fᵢ in [0, a] or (b, c], with
    /// a = 0.2514118360889171 and c = 0.8594008566447239 the first two local maxima of φ, and
    /// b = 0.6316265307000612 where φ climbs back to φ(a) (computed to 40 digits). The front is
    /// the product of these ranges, 2^(M−1) regions, and [`optimal_front`] a product grid over
    /// it, with the same values in each fᵢ, spread over the two ranges in proportion to their
    /// widths. `ideal_point` is (0, …, 0, 2M − (M − 1) φ(c)) and `nadir_point` (c, …, c, 2M).
    ///
    /// [`optimal_front`]: MultiProblem::optimal_front
    Dtlz7,
    "DTLZ7",
    20,
    REPORT,
    REPORT_URL
);

// DTLZ1 and DTLZ3's multimodal distance function
pub(super) fn rastrigin_g(tail: &[f64]) -> f64 {
    100.0
        * (tail.len() as f64
            + tail
                .iter()
                .map(|x| (x - 0.5) * (x - 0.5) - math::cos(20.0 * PI * (x - 0.5)))
                .sum::<f64>())
}

// DTLZ2 and DTLZ4's distance function
fn sphere_g(tail: &[f64]) -> f64 {
    tail.iter().map(|x| (x - 0.5) * (x - 0.5)).sum()
}

// the objectives on a sphere of radius `radius`, with the angles from `x`
fn spherical<const M: usize>(x: &[f64], radius: f64, alpha: f64) -> [f64; M] {
    std::array::from_fn(|m| {
        let mut f = radius;
        let angle = |xi: f64| if alpha == 1.0 { xi } else { math::powf(xi, alpha) } * PI / 2.0;
        for &xi in &x[..M - 1 - m] {
            f *= math::cos(angle(xi));
        }
        if m > 0 {
            f *= math::sin(angle(x[M - 1 - m]));
        }
        f
    })
}

// Das and Dennis's points, as many as `MultiProblem::optimal_front` promises: none for 0, exactly
// `points` for 2 objectives (0 and 1 would otherwise give both ends), and the smallest set of
// at least `points` for more
pub(super) fn simplex_points<const M: usize>(points: usize) -> Vec<[f64; M]> {
    if points == 0 {
        return Vec::new();
    }
    let mut simplex = das_dennis::<M>(divisions_for::<M>(points));
    if M == 2 {
        simplex.truncate(points);
    }
    simplex
}

pub(super) fn spherical_front<const M: usize>(points: usize) -> Vec<[f64; M]> {
    simplex_points::<M>(points)
        .into_iter()
        .map(|p| {
            let norm = p.iter().map(|v| v * v).sum::<f64>().sqrt();
            p.map(|v| v / norm)
        })
        .collect()
}

impl<const M: usize> Dtlz1<M> {
    const NADIR: f64 = 0.5;

    // Das and Dennis's points, halved
    fn front(points: usize) -> Vec<[f64; M]> {
        simplex_points::<M>(points)
            .into_iter()
            .map(|p| p.map(|v| v / 2.0))
            .collect()
    }
}

macro_rules! spherical_front {
    ($($name:ident),*) => {$(
        impl<const M: usize> $name<M> {
            const NADIR: f64 = 1.0;

            // Das and Dennis's points, projected on the sphere
            fn front(points: usize) -> Vec<[f64; M]> {
                spherical_front(points)
            }
        }
    )*};
}

spherical_front!(Dtlz2, Dtlz3, Dtlz4);

// the front, ideal and nadir points of DTLZ1-4, known for every number of objectives
macro_rules! known_front {
    ($($name:ident),*) => {$(
        impl<const M: usize> $name<M> {
            fn known_front(points: usize) -> Option<Vec<[f64; M]>> {
                Some(Self::front(points))
            }

            fn ideal() -> Option<[f64; M]> {
                Some([0.0; M])
            }

            fn nadir() -> Option<[f64; M]> {
                Some([Self::NADIR; M])
            }
        }
    )*};
}

known_front!(Dtlz1, Dtlz2, Dtlz3, Dtlz4);

// the objectives on a sphere of radius `radius`, at the angles `theta` (M − 1 of them)
fn at_angles<const M: usize>(theta: &[f64], radius: f64) -> [f64; M] {
    std::array::from_fn(|m| {
        let mut f = radius;
        for &angle in &theta[..M - 1 - m] {
            f *= math::cos(angle);
        }
        if m > 0 {
            f *= math::sin(theta[M - 1 - m]);
        }
        f
    })
}

// DTLZ5 and DTLZ6's objectives: θ₁ = x₁π/2 and θᵢ = π (1 + 2g xᵢ) / (4 (1 + g)) for the
// others (the report's eqs. 8 and 10), on a sphere of radius 1 + g
fn curve<const M: usize>(x: &[f64], g: f64) -> [f64; M] {
    let mut theta = [0.0; M];
    for (i, angle) in theta.iter_mut().take(M - 1).enumerate() {
        *angle = if i == 0 {
            x[0] * PI / 2.0
        } else {
            PI * (1.0 + 2.0 * g * x[i]) / (4.0 * (1.0 + g))
        };
    }
    at_angles(&theta[..M - 1], 1.0 + g)
}

// DTLZ6's distance function
fn root_g(tail: &[f64]) -> f64 {
    tail.iter().map(|&x| math::powf(x, 0.1)).sum()
}

// the front of DTLZ5 and DTLZ6 for 2 and 3 objectives: the curve at g = 0, `points` points
// evenly spread in θ₁ from 0 to π/2; not known for more
fn curve_front<const M: usize>(points: usize) -> Option<Vec<[f64; M]>> {
    (M <= 3).then(|| {
        (0..points)
            .map(|i| {
                let mut x = [0.0; M];
                x[0] = evenly(i, points);
                curve(&x, 0.0)
            })
            .collect()
    })
}

// the worst of each objective on the curve: at its ends, as f₁ … f_{M−1} fall and f_M rises
// along it
fn curve_nadir<const M: usize>() -> Option<[f64; M]> {
    let (start, end) = (curve::<M>(&[0.0; M], 0.0), curve::<M>(&[1.0; M], 0.0));
    (M <= 3).then(|| std::array::from_fn(|j| start[j].max(end[j])))
}

macro_rules! curve_front {
    ($($name:ident),*) => {$(
        impl<const M: usize> $name<M> {
            fn known_front(points: usize) -> Option<Vec<[f64; M]>> {
                curve_front(points)
            }

            fn ideal() -> Option<[f64; M]> {
                Some([0.0; M])
            }

            fn nadir() -> Option<[f64; M]> {
                curve_nadir()
            }
        }
    )*};
}

curve_front!(Dtlz5, Dtlz6);

// the ranges of each of f₁ … f_{M−1} on DTLZ7's front, [0, A] and (B, C]: A and C are the first
// two local maxima of φ(f) = f (1 + sin 3πf), where 1 + sin 3πf + 3πf cos 3πf = 0, and B > A is
// where φ(B) = φ(A); computed to 40 digits and rounded
pub(super) const DTLZ7_A: f64 = 0.251_411_836_088_917_1;
pub(super) const DTLZ7_B: f64 = 0.631_626_530_700_061_2;
pub(super) const DTLZ7_C: f64 = 0.859_400_856_644_723_9;

// DTLZ7's last objective, (1 + g) h, from the others
fn dtlz7_last<const M: usize>(f: &[f64], g: f64) -> f64 {
    let h = M as f64
        - f.iter()
            .map(|fi| fi / (1.0 + g) * (1.0 + math::sin(3.0 * PI * fi)))
            .sum::<f64>();
    (1.0 + g) * h
}

// `count` values of fᵢ on DTLZ7's front, ascending, shared between its two ranges in proportion
// to their widths: both ends of [0, A], and (B, C] without B, which A dominates
fn dtlz7_values(count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![0.0; count];
    }
    let share = count as f64 * DTLZ7_A / (DTLZ7_A + DTLZ7_C - DTLZ7_B);
    let first = (share.round() as usize).clamp(1, count - 1);
    let second = count - first;
    let low = (0..first).map(|i| DTLZ7_A * evenly(i, first));
    let high = (0..second)
        .map(|j| DTLZ7_C - (DTLZ7_C - DTLZ7_B) * (second - 1 - j) as f64 / second as f64);
    low.chain(high).collect()
}

impl<const M: usize> Dtlz7<M> {
    // a product grid over the ranges, with the fewest values per objective that give at least
    // `points` points: exactly `points` for 2 objectives
    fn known_front(points: usize) -> Option<Vec<[f64; M]>> {
        if points == 0 {
            return Some(Vec::new());
        }
        let dimensions = (M - 1) as u32;
        let mut count = 1usize;
        while count.saturating_pow(dimensions) < points {
            count += 1;
        }
        let values = dtlz7_values(count);
        let front = (0..count.pow(dimensions))
            .map(|index| {
                let mut point = [0.0; M];
                let mut rest = index;
                for j in (0..M - 1).rev() {
                    point[j] = values[rest % count];
                    rest /= count;
                }
                point[M - 1] = dtlz7_last::<M>(&point[..M - 1], 1.0);
                point
            })
            .collect();
        Some(front)
    }

    fn ideal() -> Option<[f64; M]> {
        let mut point = [0.0; M];
        point[M - 1] = dtlz7_last::<M>(&[DTLZ7_C; M][..M - 1], 1.0);
        Some(point)
    }

    fn nadir() -> Option<[f64; M]> {
        let mut point = [DTLZ7_C; M];
        point[M - 1] = dtlz7_last::<M>(&[0.0; M][..M - 1], 1.0);
        Some(point)
    }
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz1<M> {
    type Output = [f64; M];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than `M − 1` genes.
    fn evaluate(&self, x: &Reals) -> [f64; M] {
        let g = rastrigin_g(&x[M - 1..]);
        std::array::from_fn(|m| {
            let mut f = 0.5 * (1.0 + g);
            for xi in &x[..M - 1 - m] {
                f *= xi;
            }
            if m > 0 {
                f *= 1.0 - x[M - 1 - m];
            }
            f
        })
    }
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz2<M> {
    type Output = [f64; M];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than `M − 1` genes.
    fn evaluate(&self, x: &Reals) -> [f64; M] {
        spherical(x, 1.0 + sphere_g(&x[M - 1..]), 1.0)
    }
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz3<M> {
    type Output = [f64; M];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than `M − 1` genes.
    fn evaluate(&self, x: &Reals) -> [f64; M] {
        spherical(x, 1.0 + rastrigin_g(&x[M - 1..]), 1.0)
    }
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz4<M> {
    type Output = [f64; M];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than `M − 1` genes.
    fn evaluate(&self, x: &Reals) -> [f64; M] {
        spherical(x, 1.0 + sphere_g(&x[M - 1..]), 100.0)
    }
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz5<M> {
    type Output = [f64; M];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than `M − 1` genes.
    fn evaluate(&self, x: &Reals) -> [f64; M] {
        curve(x, sphere_g(&x[M - 1..]))
    }
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz6<M> {
    type Output = [f64; M];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than `M − 1` genes.
    fn evaluate(&self, x: &Reals) -> [f64; M] {
        curve(x, root_g(&x[M - 1..]))
    }
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz7<M> {
    type Output = [f64; M];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than `M − 1` genes.
    fn evaluate(&self, x: &Reals) -> [f64; M] {
        let tail = &x[M - 1..];
        let g = 1.0 + 9.0 * tail.iter().sum::<f64>() / tail.len() as f64;
        let mut f = [0.0; M];
        f[..M - 1].copy_from_slice(&x[..M - 1]);
        f[M - 1] = dtlz7_last::<M>(&x[..M - 1], g);
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Representation;

    fn at(values: &[f64]) -> Reals {
        Reals::from(values.to_vec())
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= 1e-12 * e.abs().max(1.0),
                "{actual:?} is not {expected:?}"
            );
        }
    }

    // the formulas of Deb, Thiele, Laumanns and Zitzler, at points chosen so that the values
    // follow by hand
    #[test]
    fn values_match_the_paper() {
        // DTLZ1 (M = 3, k = 5) at (0.5, …, 0.5): g = 100 (5 + 5 (0 − cos 0)) = 0, and
        // f = ½ (x₁x₂, x₁(1 − x₂), 1 − x₁) = (0.125, 0.125, 0.25)
        let problem = Dtlz1::<3>::default();
        assert_close(&problem.evaluate(&at(&[0.5; 7])), &[0.125, 0.125, 0.25]);
        // the distance variables at 1: each term is 0.25 − cos 10π = −0.75, so
        // g = 100 (5 − 3.75) = 125, and f is 126 times the above
        let x = at(&[0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_close(&problem.evaluate(&x), &[15.75, 15.75, 31.5]);
        // DTLZ2-4 (M = 3, k = 10) with the distance variables at 0.5: the corners of the front,
        // e.g. x₁ = 0, x₂ = 1: f = (cos 0 cos π/2, cos 0 sin π/2, sin 0) = (0, 1, 0)
        let mut x = vec![0.5; 12];
        for (angles, corner) in [
            ([0.0, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, 1.0], [0.0, 1.0, 0.0]),
            ([1.0, 0.0], [0.0, 0.0, 1.0]),
        ] {
            x[..2].copy_from_slice(&angles);
            for f in [
                Dtlz2::<3>::default().evaluate(&at(&x)),
                Dtlz3::<3>::default().evaluate(&at(&x)),
                Dtlz4::<3>::default().evaluate(&at(&x)),
            ] {
                assert!(f.iter().zip(corner).all(|(f, c)| (f - c).abs() < 1e-15));
            }
        }
        // the distance variables at 1 and the angles at 0: DTLZ2 g = 10 × 0.25, a radius of
        // 3.5; DTLZ3 g = 100 (10 + 10 (0.25 − cos 10π)) = 250, a radius of 251
        let mut x = vec![1.0; 12];
        x[..2].copy_from_slice(&[0.0, 0.0]);
        assert_close(&Dtlz2::<3>::default().evaluate(&at(&x)), &[3.5, 0.0, 0.0]);
        assert_close(&Dtlz3::<3>::default().evaluate(&at(&x)), &[251.0, 0.0, 0.0]);
        // DTLZ4 at x₁ = x₂ = 0.5 on the front: both angles are 0.5¹⁰⁰ π/2 ≈ 1.2e-30, so
        // f = (cos θ cos θ, cos θ sin θ, sin θ) = (1, θ, θ) to double precision: the bias
        // towards f₁ that the paper describes
        let theta = math::powi(0.5f64, 100) * PI / 2.0;
        let f = Dtlz4::<3>::default().evaluate(&at(&[0.5; 12]));
        assert_eq!(f[0], 1.0);
        assert_close(&f[1..], &[theta, theta]);
        assert!(theta > 1.239e-30 && theta < 1.24e-30);
    }

    #[test]
    fn optimal_solutions_lie_on_the_front() {
        // the distance variables at 0.5 put a solution on the front
        let mut x = vec![0.5; 7];
        x[0] = 0.3;
        x[1] = 0.8;
        let f = Dtlz1::<3>::default().evaluate(&at(&x));
        assert!((f.iter().sum::<f64>() - 0.5).abs() < 1e-12);
        let mut x = vec![0.5; 12];
        x[0] = 0.3;
        x[1] = 0.8;
        for f in [
            Dtlz2::<3>::default().evaluate(&at(&x)),
            Dtlz3::<3>::default().evaluate(&at(&x)),
            Dtlz4::<3>::default().evaluate(&at(&x)),
        ] {
            assert!((f.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let front = Dtlz2::<3>::front(91);
        assert_eq!(front.len(), 91);
        assert!(
            front
                .iter()
                .all(|p| (p.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12)
        );
        let front = Dtlz1::<4>::front(20);
        assert!(front.len() >= 20);
        assert!(
            front
                .iter()
                .all(|p| (p.iter().sum::<f64>() - 0.5).abs() < 1e-12)
        );
        assert_eq!(Dtlz1::<3>::default().representation().bounds().len(), 7);
        assert_eq!(Dtlz2::<5>::default().representation().bounds().len(), 14);
    }

    // DTLZ5-7 (the report's eqs. 25-27), at points chosen so that the values follow by hand
    #[test]
    fn dtlz5_to_7_values_match_the_report() {
        let sqrt_half = 0.5f64.sqrt();
        // DTLZ5 (M = 3, k = 10) with the distance variables at 1: g = 10 × 0.25 = 2.5, so
        // θ₂ = π (1 + 5x₂) / 14, π/4 at x₂ = 0.5; at x₁ = 0, f = 3.5 (cos θ₂, sin θ₂, 0)
        let mut x = vec![1.0; 12];
        x[..2].copy_from_slice(&[0.0, 0.5]);
        let problem = Dtlz5::<3>::default();
        assert_close(
            &problem.evaluate(&at(&x)),
            &[3.5 * sqrt_half, 3.5 * sqrt_half, 0.0],
        );
        // x₂ = 1: θ₂ = 6π/14, the widest angle at this g
        x[1] = 1.0;
        let theta = 3.0 * PI / 7.0;
        assert_close(
            &problem.evaluate(&at(&x)),
            &[3.5 * math::cos(theta), 3.5 * math::sin(theta), 0.0],
        );
        // DTLZ6 with the distance variables at 1: g = 10 × 1^0.1 = 10, θ₂ = π (1 + 20x₂) / 44,
        // π/4 at x₂ = 0.5; at x₁ = 1, θ₁ = π/2 and f = (0, 0, 11)
        x[1] = 0.5;
        let problem = Dtlz6::<3>::default();
        assert_close(
            &problem.evaluate(&at(&x)),
            &[11.0 * sqrt_half, 11.0 * sqrt_half, 0.0],
        );
        x[0] = 1.0;
        let f = problem.evaluate(&at(&x));
        assert!(f[0].abs() < 1e-14 && f[1].abs() < 1e-14);
        assert_close(&f[2..], &[11.0]);
        // the distance variables at 2⁻¹⁰: each xᵢ^0.1 = 0.5, g = 5
        x[2..].fill(math::powi(2.0, -10));
        x[0] = 0.0;
        let theta = PI * (1.0 + 10.0 * 0.5) / 24.0;
        assert_close(
            &problem.evaluate(&at(&x)),
            &[6.0 * math::cos(theta), 6.0 * math::sin(theta), 0.0],
        );
        // DTLZ7 (M = 3, k = 20) with the distance variables at 0: g = 1; at f₁ = f₂ = 0.5,
        // sin 1.5π = −1, so h = 3 and f₃ = 2 × 3 = 6
        let problem = Dtlz7::<3>::default();
        let mut x = vec![0.0; 22];
        x[..2].copy_from_slice(&[0.5, 0.5]);
        assert_close(&problem.evaluate(&at(&x)), &[0.5, 0.5, 6.0]);
        // at f₁ = f₂ = 1/6, sin(π/2) = 1: h = 3 − 2 × (1/6) × 2 / 2 = 8/3, f₃ = 16/3
        x[..2].copy_from_slice(&[1.0 / 6.0, 1.0 / 6.0]);
        assert_close(
            &problem.evaluate(&at(&x)),
            &[1.0 / 6.0, 1.0 / 6.0, 16.0 / 3.0],
        );
        // the distance variables at 1: g = 1 + 9 = 10; at f₁ = f₂ = 0, h = 3 and f₃ = 33
        let mut x = vec![1.0; 22];
        x[..2].copy_from_slice(&[0.0, 0.0]);
        assert_close(&problem.evaluate(&at(&x)), &[0.0, 0.0, 33.0]);
        assert_eq!(Dtlz7::<3>::default().variables(), 22);
        assert_eq!(Dtlz5::<4>::default().variables(), 13);
    }

    #[test]
    fn dtlz5_to_7_optimal_solutions_lie_on_the_front() {
        // DTLZ5 with the distance variables at 0.5, and DTLZ6 at 0: g = 0 and θ₂ = π/4 whatever
        // x₂, on the curve f₁ = f₂ = cos θ₁ / √2, f₃ = sin θ₁
        let mut x5 = vec![0.5; 12];
        let mut x6 = vec![0.0; 12];
        for (x1, x2) in [(0.0, 0.8), (0.3, 0.1), (1.0, 1.0)] {
            x5[..2].copy_from_slice(&[x1, x2]);
            x6[..2].copy_from_slice(&[x1, x2]);
            let theta = x1 * PI / 2.0;
            let expected = [
                math::cos(theta) * 0.5f64.sqrt(),
                math::cos(theta) * 0.5f64.sqrt(),
                math::sin(theta),
            ];
            assert_close(&Dtlz5::<3>::default().evaluate(&at(&x5)), &expected);
            assert_close(&Dtlz6::<3>::default().evaluate(&at(&x6)), &expected);
        }
        // the front: a quarter circle for 2 objectives, the curve for 3, unknown for more
        let front = Dtlz5::<3>::known_front(91).expect("known");
        assert_eq!(front.len(), 91);
        for p in &front {
            assert!((p[0] - p[1]).abs() < 1e-15);
            assert!((p.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let front = Dtlz6::<2>::known_front(5).expect("known");
        assert_eq!(front.len(), 5);
        assert!(
            front
                .iter()
                .all(|p| (p[0] * p[0] + p[1] * p[1] - 1.0).abs() < 1e-12)
        );
        assert!(Dtlz5::<4>::known_front(10).is_none());
        let nadir = Dtlz5::<3>::nadir().expect("known");
        assert_close(&nadir, &[0.5f64.sqrt(), 0.5f64.sqrt(), 1.0]);
        assert_eq!(Dtlz6::<2>::nadir(), Some([1.0, 1.0]));
        assert_eq!(Dtlz5::<4>::nadir(), None);
        assert_eq!(Dtlz6::<5>::ideal(), Some([0.0; 5]));
        // DTLZ7 with the distance variables at 0: f₃ = 6 − φ(f₁) − φ(f₂)
        let phi = |f: f64| f * (1.0 + math::sin(3.0 * PI * f));
        let mut x = vec![0.0; 22];
        x[..2].copy_from_slice(&[0.2, 0.7]);
        let f = Dtlz7::<3>::default().evaluate(&at(&x));
        assert_close(&f, &[0.2, 0.7, 6.0 - phi(0.2) - phi(0.7)]);
        let front = Dtlz7::<3>::known_front(50).expect("known");
        assert_eq!(front.len(), 64);
        let in_ranges = |f: f64| (0.0..=DTLZ7_A).contains(&f) || (f > DTLZ7_B && f <= DTLZ7_C);
        for p in &front {
            assert!(in_ranges(p[0]) && in_ranges(p[1]), "{p:?}");
            assert!((p[2] - (6.0 - phi(p[0]) - phi(p[1]))).abs() < 1e-12);
        }
        assert_eq!(Dtlz7::<2>::known_front(7).expect("known").len(), 7);
        // the ideal and nadir points: f₃ is 6 − 2φ(c) at (c, c) and 6 at the origin
        let ideal = Dtlz7::<3>::ideal().expect("known");
        assert_eq!(ideal[..2], [0.0, 0.0]);
        assert!((ideal[2] - 2.614_008_731_003_155).abs() < 1e-14);
        assert_eq!(Dtlz7::<3>::nadir(), Some([DTLZ7_C, DTLZ7_C, 6.0]));
        assert_eq!(Dtlz7::<5>::nadir().expect("known")[4], 10.0);
    }

    // DTLZ7's ranges: A and C are local maxima of φ(f) = f (1 + sin 3πf), φ(B) = φ(A), and the
    // values where φ beats every smaller value, on a fine grid, are [0, A] and (B, C]
    #[test]
    fn dtlz7_ranges_are_where_phi_rises_to_a_record() {
        let phi = |f: f64| f * (1.0 + math::sin(3.0 * PI * f));
        let slope = |f: f64| 1.0 + math::sin(3.0 * PI * f) + 3.0 * PI * f * math::cos(3.0 * PI * f);
        assert!(slope(DTLZ7_A).abs() < 1e-14 && slope(DTLZ7_C).abs() < 1e-14);
        assert!(phi(DTLZ7_A) > phi(DTLZ7_A - 1e-6) && phi(DTLZ7_A) > phi(DTLZ7_A + 1e-6));
        assert!(phi(DTLZ7_C) > phi(DTLZ7_C - 1e-6) && phi(DTLZ7_C) > phi(DTLZ7_C + 1e-6));
        assert!((phi(DTLZ7_B) - phi(DTLZ7_A)).abs() < 1e-15);
        let steps = 1_000_000;
        let mut best = f64::NEG_INFINITY;
        for i in 0..=steps {
            let f = i as f64 / steps as f64;
            let record = phi(f) > best;
            best = best.max(phi(f));
            let inside = f <= DTLZ7_A || (f > DTLZ7_B && f <= DTLZ7_C);
            let near = [DTLZ7_A, DTLZ7_B, DTLZ7_C]
                .iter()
                .any(|edge| (f - edge).abs() < 2.0 / steps as f64);
            assert!(record == inside || near, "{f}");
        }
    }

    // with 4 objectives, DTLZ5's front isn't the curve: a solution with g > 0 that no point of
    // the curve dominates. A point of the curve is (a cos t, sin t), a being the curve at θ₁ = 0,
    // (1/2, 1/2, 1/√2); it's no worse than f when cos t ≤ fⱼ / aⱼ for each j < 4 and
    // sin t ≤ f₄, which some t in [0, π/2] meets only if min (fⱼ / aⱼ)² + f₄² ≥ 1
    #[test]
    fn dtlz5_front_is_not_a_curve_for_4_objectives() {
        let mut x = vec![0.5; 13];
        x[..3].copy_from_slice(&[0.02, 1.0, 1.0]);
        x[12] = 0.5 + 0.12f64.sqrt();
        let f = Dtlz5::<4>::default().evaluate(&at(&x));
        let a = [0.5, 0.5, 0.5f64.sqrt()];
        let reach = (0..3).map(|j| f[j] / a[j]).fold(f64::INFINITY, f64::min);
        assert!(reach * reach + f[3] * f[3] < 0.9, "{f:?}");
        // with 3 objectives, the same trick fails: every solution is dominated by the curve or
        // on it
        let mut rng = crate::StreamRng::seed_from_u64(1);
        let problem = Dtlz5::<3>::default();
        let a = [0.5f64.sqrt(), 0.5f64.sqrt()];
        for _ in 0..10_000 {
            let genome = problem.representation().random_genome(&mut rng);
            let f = problem.evaluate(&genome);
            let reach = (f[0] / a[0]).min(f[1] / a[1]).min(1.0);
            assert!(reach * reach + f[2] * f[2] >= 1.0 - 1e-12, "{f:?}");
        }
    }
}
