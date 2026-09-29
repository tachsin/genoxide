//! WFG1-WFG9, the Walking Fish Group's suite: Huband, S., Hingston, P., Barone, L. and While, L.
//! (2006). A review of multiobjective test problems and a scalable test problem toolkit. *IEEE
//! Transactions on Evolutionary Computation* 10(5): 477-506, section VIII.
//!
//! Where each part comes from:
//!
//! - **The framework** (`fₘ = D x_M + Sₘ hₘ(x₁…x_{M−1})`, x from the last transition vector with
//!   the degeneracy constants A, z normalized by its upper bounds): section VIII's equation.
//! - **The shapes** (linear, convex, concave, mixed, disconnected): table X.
//! - **The transformations** (the biases b_poly, b_flat and b_param, the shifts s_linear, s_decept
//!   and s_multi, the reductions r_sum and r_nonsep): table XI.
//! - **The problems** (D = 1, Sₘ = 2m, A, the bounds zᵢ ∈ [0, 2i], each problem's shapes and
//!   transition vectors, and the rules on k and l): table XIV; WFG9 is table XIII's example.
//! - **The optimal solutions:** section VIII-B for WFG1-8 and section VIII-A for WFG9.
//! - **The recommended sizes** (l = 20; k = 4 for 2 objectives and 2(M − 1) for more): the README
//!   of the authors' C++ toolkit, and section IX of the paper, which uses k = 4 and l = 20.
//!
//! The paper was checked as published (the copy on the WFG group's website,
//! www.wfg.csse.uwa.edu.au/publications/WFG2006c.pdf, as the Internet Archive keeps it), and so
//! was the earlier version of its toolkit: Huband, S., Barone, L., While, L. and Hingston, P.
//! (2005). A scalable multi-objective test problem toolkit. *EMO 2005*, LNCS 3410: 280-295
//! (the authors' corrected version of 25 May 2005, table 6, which has the same problems with D =
//! 1). The values were checked against the authors' C++ toolkit, version 2006.03.28
//! (`WFG_v2006.03.28.zip`: `Toolkit/ExampleProblems.cpp`, `ExampleTransitions.cpp`,
//! `ExampleShapes.cpp`, `TransFunctions.cpp`, `ShapeFunctions.cpp`, `FrameworkFunctions.cpp`,
//! and the optimal solutions of `main.cpp`), compiled and run for the test values below.

use super::dtlz::{simplex_points, spherical_front};
use super::{MultiProblem, das_dennis, divisions_for, evenly};
use crate::genome::{Real, Reals};
use crate::math;
use crate::multi::MultiFitnessFunction;
use std::f64::consts::PI;

const REFERENCE: &str = "Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of \
                         multiobjective test problems and a scalable test problem toolkit. IEEE \
                         Transactions on Evolutionary Computation 10(5): 477-506.";
const REFERENCE_URL: &str = "https://doi.org/10.1109/TEVC.2005.861417";

// the constant A of b_param in WFG7, WFG8 and WFG9 (table XIV), with B = 0.02 and C = 50
const PARAM_A: f64 = 0.98 / 49.98;

// ---- the transformations of table XI ------------------------------------------------------------

// a value that rounding put just outside [0, 1], back in it: the toolkit's `correct_to_01`
fn unit(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

// polynomial bias: y^α
fn b_poly(y: f64, alpha: f64) -> f64 {
    unit(math::powf(y, alpha))
}

// flat region: the values of y from B to C all map to A
fn b_flat(y: f64, a: f64, b: f64, c: f64) -> f64 {
    let below = (y - b).floor().min(0.0) * a * (b - y) / b;
    let above = (c - y).floor().min(0.0) * (1.0 - a) * (y - c) / (1.0 - c);
    unit(a + below - above)
}

// parameter-dependent bias: y to a power from B to C, set by u, a reduction of other parameters
fn b_param(y: f64, u: f64, a: f64, b: f64, c: f64) -> f64 {
    let v = a - (1.0 - 2.0 * u) * ((0.5 - u).floor() + a).abs();
    unit(math::powf(y, b + (c - b) * v))
}

// linear shift: A maps to 0
fn s_linear(y: f64, a: f64) -> f64 {
    unit((y - a).abs() / ((a - y).floor() + a).abs())
}

// deceptive shift: the global minimum 0 at A, in a basin of width 2B, and two deceptive minima C
// at 0 and 1
fn s_decept(y: f64, a: f64, b: f64, c: f64) -> f64 {
    let low = (y - a + b).floor() * (1.0 - c + (a - b) / b) / (a - b);
    let high = (a + b - y).floor() * (1.0 - c + (1.0 - a - b) / b) / (1.0 - a - b);
    unit(1.0 + ((y - a).abs() - b) * (low + high + 1.0 / b))
}

// multimodal shift: 2A local minima and the global minimum 0 at C; B sets the hills' size
fn s_multi(y: f64, a: f64, b: f64, c: f64) -> f64 {
    let distance = (y - c).abs() / (2.0 * ((c - y).floor() + c));
    let angle = (4.0 * a + 2.0) * PI * (0.5 - distance);
    unit((1.0 + math::cos(angle) + 4.0 * b * distance * distance) / (b + 2.0))
}

// weighted sum reduction, with the weights w(i) of the values y[i]
fn r_sum(y: &[f64], w: impl Fn(usize) -> f64) -> f64 {
    let (mut numerator, mut denominator) = (0.0, 0.0);
    for (i, &value) in y.iter().enumerate() {
        numerator += w(i) * value;
        denominator += w(i);
    }
    unit(numerator / denominator)
}

// the unweighted r_sum: the mean
fn mean(y: &[f64]) -> f64 {
    r_sum(y, |_| 1.0)
}

// non-separable reduction, A the degree of non-separability, a divisor of y's length
fn r_nonsep(y: &[f64], a: usize) -> f64 {
    let n = y.len();
    let mut numerator = 0.0;
    for (j, &value) in y.iter().enumerate() {
        numerator += value;
        // the a − 1 values after it, wrapping around to the first: y[(j + k + 1) % n] for k
        // from 0, in that order
        let (after, before) = (&y[j + 1..], &y[..j]);
        let wrapped = (a - 1).saturating_sub(after.len());
        for &other in &after[..a - 1 - wrapped] {
            numerator += (value - other).abs();
        }
        for &other in &before[..wrapped] {
            numerator += (value - other).abs();
        }
    }
    let half = a.div_ceil(2) as f64;
    let a = a as f64;
    unit(numerator / (n as f64 * half * (1.0 + 2.0 * a - 2.0 * half) / a))
}

// ---- the shapes of table X, of x₁…x_M, m from 1 -------------------------------------------------

fn linear(x: &[f64], m: usize) -> f64 {
    let big_m = x.len();
    let mut h: f64 = x[..big_m - m].iter().product();
    if m > 1 {
        h *= 1.0 - x[big_m - m];
    }
    unit(h)
}

fn convex(x: &[f64], m: usize) -> f64 {
    let big_m = x.len();
    let mut h = 1.0;
    for &xi in &x[..big_m - m] {
        h *= 1.0 - math::cos(xi * PI / 2.0);
    }
    if m > 1 {
        h *= 1.0 - math::sin(x[big_m - m] * PI / 2.0);
    }
    unit(h)
}

fn concave(x: &[f64], m: usize) -> f64 {
    let big_m = x.len();
    let mut h = 1.0;
    for &xi in &x[..big_m - m] {
        h *= math::sin(xi * PI / 2.0);
    }
    if m > 1 {
        h *= math::cos(x[big_m - m] * PI / 2.0);
    }
    unit(h)
}

// mixed convex/concave, of x₁: A convex and concave pieces, convex overall for α > 1
fn mixed(x1: f64, a: f64, alpha: f64) -> f64 {
    let tmp = 2.0 * a * PI;
    unit(math::powf(
        1.0 - x1 - math::cos(tmp * x1 + PI / 2.0) / tmp,
        alpha,
    ))
}

// disconnected, of x₁: A regions
fn disc(x1: f64, a: f64, alpha: f64, beta: f64) -> f64 {
    let cos = math::cos(a * math::powf(x1, beta) * PI);
    unit(1.0 - math::powf(x1, alpha) * cos * cos)
}

// ---- the framework ------------------------------------------------------------------------------

// z normalized: zᵢ / 2i, as the first transition vector's input
fn normalized(z: &[f64], variables: usize) -> Vec<f64> {
    assert_eq!(z.len(), variables, "WFG takes genomes of k + l genes");
    z.iter()
        .enumerate()
        .map(|(i, zi)| zi / (2.0 * (i + 1) as f64))
        .collect()
}

// the last transition vector of every problem: each of the M − 1 groups of k / (M − 1) position
// values reduced to one, then the distance values (from k on) to one; `reduce` gets a group and
// the index of its first value
fn reduced<const M: usize>(y: &[f64], k: usize, reduce: impl Fn(&[f64], usize) -> f64) -> [f64; M] {
    let group = k / (M - 1);
    std::array::from_fn(|i| {
        if i < M - 1 {
            reduce(&y[i * group..(i + 1) * group], i * group)
        } else {
            reduce(&y[k..], k)
        }
    })
}

// the objectives from the last transition vector t: xᵢ = max(t_M, Aᵢ)(tᵢ − 0.5) + 0.5 for i < M
// with A₁ = 1 and A₂…A_{M−1} = 0 if `degenerate` (WFG3) and 1 otherwise, x_M = t_M; then
// fₘ = D x_M + Sₘ hₘ(x) with D = 1 and Sₘ = 2m
fn objectives<const M: usize>(
    t: &[f64; M],
    degenerate: bool,
    shape: impl Fn(&[f64; M], usize) -> f64,
) -> [f64; M] {
    let distance = t[M - 1];
    let x: [f64; M] = std::array::from_fn(|i| {
        if i == M - 1 {
            distance
        } else {
            let a = if degenerate && i > 0 { 0.0 } else { 1.0 };
            distance.max(a) * (t[i] - 0.5) + 0.5
        }
    });
    std::array::from_fn(|m| distance + 2.0 * (m + 1) as f64 * shape(&x, m + 1))
}

// the recommended number of position parameters: 4 for 2 objectives, 2(M − 1) for more
const fn default_position(objectives: usize) -> usize {
    if objectives == 2 {
        4
    } else {
        2 * (objectives - 1)
    }
}

// the sizes of a problem, checked
fn check<const M: usize>(name: &str, position: usize, distance: usize, even: bool) {
    assert!(M >= 2, "{name} needs at least 2 objectives");
    assert!(
        position > 0 && position.is_multiple_of(M - 1),
        "{name} needs a positive multiple of M − 1 = {} position parameters, not {position}",
        M - 1
    );
    assert!(distance > 0, "{name} needs at least 1 distance parameter");
    assert!(
        !even || distance.is_multiple_of(2),
        "{name} needs an even number of distance parameters, not {distance}"
    );
}

// ---- the problems -------------------------------------------------------------------------------

macro_rules! wfg {
    (
        $(#[$doc:meta])* $name:ident, $label:literal, $even:literal, $rule:literal, $panics:literal
    ) => {
        $(#[$doc])*
        ///
        /// With `M` objectives, `k` position parameters and `l` distance parameters: `n = k + l`
        /// variables `zᵢ` in [0, 2i]. `k` is a multiple of `M − 1`,
        #[doc = concat!(
            "and `l` is at least 1",
            $rule,
            "; by default, `k` is 4 for 2 objectives and 2(M − 1) for more, and `l` is 20, the ",
            "authors' recommendation."
        )]
        ///
        /// Checked in the paper as published and in its first version (Huband, Barone, While and
        /// Hingston (2005), *EMO 2005*, LNCS 3410: 280-295, corrected version); the values match
        /// the authors' C++ toolkit, version 2006.03.28, compiled for the tests.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name<const M: usize> {
            position: usize,
            distance: usize,
        }

        impl<const M: usize> $name<M> {
            /// The problem with `position` position parameters (`k`) and `distance` distance
            /// parameters (`l`).
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives, if `position` isn't a positive multiple of `M − 1`,
            #[doc = $panics]
            pub fn new(position: usize, distance: usize) -> Self {
                check::<M>($label, position, distance, $even);
                Self { position, distance }
            }

            /// The number of position parameters, `k`.
            pub fn position(&self) -> usize {
                self.position
            }

            /// The number of distance parameters, `l`.
            pub fn distance(&self) -> usize {
                self.distance
            }

            /// The number of variables, `k + l`.
            pub fn variables(&self) -> usize {
                self.position + self.distance
            }
        }

        impl<const M: usize> Default for $name<M> {
            /// The recommended problem: `k` = 4 for 2 objectives and 2(M − 1) for more, `l` = 20.
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives.
            fn default() -> Self {
                assert!(M >= 2, "{} needs at least 2 objectives", $label);
                Self::new(default_position(M), 20)
            }
        }

        impl<const M: usize> MultiProblem<M> for $name<M> {
            type Representation = Real;

            fn name(&self) -> &'static str {
                $label
            }

            fn representation(&self) -> Real {
                let bounds = (1..=self.variables()).map(|i| 0.0..=2.0 * i as f64);
                Real::new(bounds).expect("valid bounds")
            }

            fn reference(&self) -> &'static str {
                REFERENCE
            }

            fn reference_url(&self) -> Option<&'static str> {
                Some(REFERENCE_URL)
            }

            fn optimal_front(&self, points: usize) -> Option<Vec<[f64; M]>> {
                Self::front(points)
            }

            fn ideal_point(&self) -> Option<[f64; M]> {
                Self::extremes().map(|(ideal, _)| ideal)
            }

            fn nadir_point(&self) -> Option<[f64; M]> {
                Self::extremes().map(|(_, nadir)| nadir)
            }
        }

        impl<const M: usize> MultiFitnessFunction<Reals, M> for $name<M> {
            type Output = [f64; M];

            /// The objective values of `z`.
            ///
            /// # Panics
            ///
            /// If `z` doesn't have `k + l` genes.
            fn evaluate(&self, z: &Reals) -> [f64; M] {
                let y = normalized(z, self.variables());
                Self::shape(&self.underlying(y))
            }
        }
    };
}

wfg!(
    /// WFG1: a front of convex and mixed convex/concave parts, behind a flat region and a strong
    /// polynomial bias.
    ///
    /// The distance parameters are shifted, `s_linear(y, 0.35)`, and given a flat region,
    /// `b_flat(y, 0.8, 0.75, 0.85)`; then every parameter is biased, `b_poly(y, 0.02)`, and
    /// each group is reduced by a sum weighted by 2i (`r_sum`). `h₁…h_{M−1}` are convex and
    /// `h_M` mixed, with A = 5 and α = 1: `1 − x₁ − cos(10πx₁ + π/2) / 10π`. Separable and
    /// unimodal. The optimal solutions have the distance parameters at 0.35 × 2i, and the front
    /// is `fₘ = 2m hₘ(x₁…x_{M−1})` for x in [0, 1]^(M−1), where x₁ = 0.2, 0.4, 0.6 and 0.8 are
    /// the flat spots of `h_M`. The bias, y^0.02, crowds the solutions at x = 1: the paper's
    /// NSGA-II covered little of the front.
    ///
    /// In floating point, `zᵢ / 2i` is never exactly 0.35 for some i (3, 6, 12, 24, 48, 53, …),
    /// and `b_poly(·, 0.02)` turns the last bit into a distance of 0.48 in that parameter
    /// ((10⁻¹⁶)^0.02 ≈ 0.48). With such distance parameters, as the default sizes have (i = 6,
    /// 12 and 24 for k = 4, l = 20), no genome reaches the front: the distance `x_M` stays above
    /// about 0.069 with 2 or 3 objectives, and 0.047 with 5 (k = 8). The authors' toolkit
    /// computes the same, and its README warns that WFG1's bias strains double precision.
    ///
    /// Definition: table XIV of Huband, Hingston, Barone and While (2006), with tables X and
    /// XI; optimal solutions: its section VIII-B.
    Wfg1,
    "WFG1",
    false,
    "",
    "or if `distance` is 0."
);
wfg!(
    /// WFG2: a convex front in disconnected regions, with a non-separable reduction.
    ///
    /// The distance parameters are shifted, `s_linear(y, 0.35)`, then reduced in pairs,
    /// `r_nonsep(·, 2)`; each group is reduced to its mean. `h₁…h_{M−1}` are convex and `h_M`
    /// disconnected, with A = 5 and α = β = 1: `1 − x₁ cos²(5πx₁)`. `f₁…f_{M−1}` are unimodal
    /// and `f_M` multimodal. The optimal solutions have the distance parameters at 0.35 × 2i and
    /// x₁ where `h_M` is below all its values at smaller x₁: six intervals, [0, 0.0416],
    /// (0.1297, 0.2096], (0.3549, 0.4050], (0.5641, 0.6034], (0.7691, 0.8025] and (0.9724, 1]
    /// (found to the last bit from the shape, each ending at a local minimum). For 3 objectives or
    /// more, the front is sampled along evenly spread directions, and those that miss the regions
    /// are left out: at least the points asked for, more directions if need be.
    ///
    /// Definition: table XIV of Huband, Hingston, Barone and While (2006), with tables X and
    /// XI; optimal solutions: its section VIII-B, which notes that the front is disconnected.
    Wfg2,
    "WFG2",
    true,
    " and even",
    "or if `distance` is 0 or odd."
);
wfg!(
    /// WFG3: meant to have a degenerate linear front, a line for any number of objectives; its
    /// front for 3 or more isn't known.
    ///
    /// WFG2's transitions, the linear shape, and degeneracy constants A₂…A_{M−1} = 0, which put
    /// x₂…x_{M−1} at 0.5 when the distance parameters are at their optimum, 0.35 × 2i. Those
    /// solutions make a line, the paper's "one dimensional Pareto optimal front": the segment
    /// from (0, …, 0, 2M) to the point with fₘ = 2m · 0.5^(M−m) for m < M and f_M = 0 (for 3
    /// objectives, from (0, 0, 6) to (1, 2, 0)). They are optimal: every point with the
    /// distance x_M = 0 has `Σ fₘ / 2m = 1`, and any other has `Σ fₘ / 2m = 1 + x_M Σ 1/2m`.
    /// But for 3 objectives or more, solutions off the optimal distance are optimal too, as
    /// Ishibuchi, Masuda and Nojima show (2016, *IEEE Transactions on Evolutionary Computation*
    /// 20(5): 807-813, not yet checked against the letter itself,
    /// [#168](https://github.com/tachsin/genoxide/issues/168)): with 3 objectives, e.g.,
    /// (3, 1, 1), where x_M = 1, which nothing dominates. `optimal_front`, `ideal_point` and
    /// `nadir_point` are then `None`; for 2 objectives, the front is the segment from (0, 4) to
    /// (2, 0).
    ///
    /// Definition: table XIV of Huband, Hingston, Barone and While (2006), with tables X and
    /// XI; optimal solutions: its section VIII-B.
    Wfg3,
    "WFG3",
    true,
    " and even",
    "or if `distance` is 0 or odd."
);
wfg!(
    /// WFG4: a concave front behind many local fronts.
    ///
    /// Every parameter is shifted by `s_multi(y, 30, 10, 0.35)`, which has 60 local minima and
    /// large hills between them; each group is reduced to its mean. Concave shapes: the front is
    /// `Σ (fₘ / 2m)² = 1`, and the optimal solutions have the distance parameters at 0.35 × 2i.
    /// Separable and multimodal.
    ///
    /// Definition: table XIV of Huband, Hingston, Barone and While (2006), with tables X and
    /// XI; optimal solutions: its section VIII-B.
    Wfg4,
    "WFG4",
    false,
    "",
    "or if `distance` is 0."
);
wfg!(
    /// WFG5: a concave front with deceptive parameters.
    ///
    /// Every parameter is shifted by `s_decept(y, 0.35, 0.001, 0.05)`: its optimum at 0.35 is a
    /// narrow basin, and 0 and 1 are wide deceptive minima. Each group is reduced to its mean.
    /// Concave shapes: the front is `Σ (fₘ / 2m)² = 1`, and the optimal solutions have the
    /// distance parameters at 0.35 × 2i. Separable and deceptive.
    ///
    /// Definition: table XIV of Huband, Hingston, Barone and While (2006), with tables X and
    /// XI; optimal solutions: its section VIII-B.
    Wfg5,
    "WFG5",
    false,
    "",
    "or if `distance` is 0."
);
wfg!(
    /// WFG6: a concave front, non-separable.
    ///
    /// The distance parameters are shifted, `s_linear(y, 0.35)`; then each group of position
    /// parameters is reduced by `r_nonsep` with A = k / (M − 1), and the distance parameters by
    /// `r_nonsep` with A = l, which confounds them all. Concave shapes: the front is
    /// `Σ (fₘ / 2m)² = 1`, and the optimal solutions have the distance parameters at 0.35 × 2i.
    ///
    /// Definition: table XIV of Huband, Hingston, Barone and While (2006), with tables X and
    /// XI; optimal solutions: its section VIII-B.
    Wfg6,
    "WFG6",
    false,
    "",
    "or if `distance` is 0."
);
wfg!(
    /// WFG7: a concave front, the position parameters biased by the parameters after them.
    ///
    /// Each position parameter is biased by `b_param(yᵢ, mean(yᵢ₊₁…yₙ), 0.98/49.98, 0.02, 50)`;
    /// then the distance parameters are shifted, `s_linear(y, 0.35)`, and each group reduced to
    /// its mean. Concave shapes: the front is `Σ (fₘ / 2m)² = 1`, and the optimal solutions have
    /// the distance parameters at 0.35 × 2i. Separable and unimodal.
    ///
    /// Definition: table XIV of Huband, Hingston, Barone and While (2006), with tables X and
    /// XI; optimal solutions: its section VIII-B.
    Wfg7,
    "WFG7",
    false,
    "",
    "or if `distance` is 0."
);
wfg!(
    /// WFG8: a concave front, the distance parameters biased by the parameters before them.
    ///
    /// Each distance parameter is biased by `b_param(yᵢ, mean(y₁…yᵢ₋₁), 0.98/49.98, 0.02,
    /// 50)`; then the distance parameters are shifted, `s_linear(y, 0.35)`, and each group
    /// reduced to its mean. Concave shapes: the front is `Σ (fₘ / 2m)² = 1`. Non-separable: the
    /// optimal distance parameters depend on the position, `zᵢ = 2i × 0.35^(1 / (0.02 + 49.98
    /// v(u)))` with u = mean(y₁…yᵢ₋₁) and v b_param's (table XI), from z_{k+1} to zₙ in turn.
    ///
    /// Definition: table XIV of Huband, Hingston, Barone and While (2006), with tables X and
    /// XI; optimal solutions: its section VIII-B.
    Wfg8,
    "WFG8",
    false,
    "",
    "or if `distance` is 0."
);
wfg!(
    /// WFG9: a concave front, multimodal, deceptive and non-separable.
    ///
    /// Each parameter but the last is biased by `b_param(yᵢ, mean(yᵢ₊₁…yₙ), 0.98/49.98, 0.02,
    /// 50)`; then the position parameters are shifted by `s_decept(y, 0.35, 0.001, 0.05)` and
    /// the distance parameters by `s_multi(y, 30, 95, 0.35)`, and reduced as in [`Wfg6`].
    /// Concave shapes: the front is `Σ (fₘ / 2m)² = 1`. The optimal distance parameters depend
    /// on the ones after them only: zₙ = 0.35 × 2n, then `zᵢ = 2i × 0.35^(1 / (0.02 + 1.96 u))`
    /// with u = mean(yᵢ₊₁…yₙ), from z_{n−1} back to z_{k+1}; the same for every position.
    ///
    /// Definition: table XIV of Huband, Hingston, Barone and While (2006), the example of their
    /// table XIII, with tables X and XI; optimal solutions: its section VIII-A.
    Wfg9,
    "WFG9",
    false,
    "",
    "or if `distance` is 0."
);

// ---- the transitions of table XIV ---------------------------------------------------------------

impl<const M: usize> Wfg1<M> {
    fn underlying(&self, mut y: Vec<f64>) -> [f64; M] {
        let k = self.position;
        for yi in &mut y[k..] {
            *yi = b_flat(s_linear(*yi, 0.35), 0.8, 0.75, 0.85);
        }
        for yi in &mut y {
            *yi = b_poly(*yi, 0.02);
        }
        // the weights 2i, i counted from 1 over all the parameters
        reduced(&y, k, |group, start| {
            r_sum(group, |i| 2.0 * (start + i + 1) as f64)
        })
    }

    fn shape(t: &[f64; M]) -> [f64; M] {
        objectives(t, false, |x, m| {
            if m < M {
                convex(x, m)
            } else {
                mixed(x[0], 5.0, 1.0)
            }
        })
    }
}

// WFG2 and WFG3's transitions: the distance parameters shifted and reduced in pairs, then the
// groups reduced to their means
fn wfg2_underlying<const M: usize>(mut y: Vec<f64>, k: usize) -> [f64; M] {
    for yi in &mut y[k..] {
        *yi = s_linear(*yi, 0.35);
    }
    let pairs: Vec<f64> = y[k..].chunks(2).map(|pair| r_nonsep(pair, 2)).collect();
    y.truncate(k);
    y.extend(pairs);
    reduced(&y, k, |group, _| mean(group))
}

impl<const M: usize> Wfg2<M> {
    fn underlying(&self, y: Vec<f64>) -> [f64; M] {
        wfg2_underlying(y, self.position)
    }

    fn shape(t: &[f64; M]) -> [f64; M] {
        objectives(
            t,
            false,
            |x, m| {
                if m < M { convex(x, m) } else { wfg2_disc(x[0]) }
            },
        )
    }
}

impl<const M: usize> Wfg3<M> {
    fn underlying(&self, y: Vec<f64>) -> [f64; M] {
        wfg2_underlying(y, self.position)
    }

    fn shape(t: &[f64; M]) -> [f64; M] {
        objectives(t, true, |x, m| linear(x, m))
    }
}

impl<const M: usize> Wfg4<M> {
    fn underlying(&self, mut y: Vec<f64>) -> [f64; M] {
        for yi in &mut y {
            *yi = s_multi(*yi, 30.0, 10.0, 0.35);
        }
        reduced(&y, self.position, |group, _| mean(group))
    }
}

impl<const M: usize> Wfg5<M> {
    fn underlying(&self, mut y: Vec<f64>) -> [f64; M] {
        for yi in &mut y {
            *yi = s_decept(*yi, 0.35, 0.001, 0.05);
        }
        reduced(&y, self.position, |group, _| mean(group))
    }
}

// WFG6 and WFG9's reduction: r_nonsep over each group, with A its length
fn nonseparable<const M: usize>(y: &[f64], k: usize) -> [f64; M] {
    reduced(y, k, |group, _| r_nonsep(group, group.len()))
}

impl<const M: usize> Wfg6<M> {
    fn underlying(&self, mut y: Vec<f64>) -> [f64; M] {
        let k = self.position;
        for yi in &mut y[k..] {
            *yi = s_linear(*yi, 0.35);
        }
        nonseparable(&y, k)
    }
}

// the means of the values after each: yᵢ₊₁…yₙ for each i, 0 for the last
fn means_after(y: &[f64]) -> Vec<f64> {
    let mut sum = 0.0;
    let mut means = vec![0.0; y.len()];
    for i in (0..y.len().saturating_sub(1)).rev() {
        sum += y[i + 1];
        means[i] = unit(sum / (y.len() - 1 - i) as f64);
    }
    means
}

impl<const M: usize> Wfg7<M> {
    fn underlying(&self, mut y: Vec<f64>) -> [f64; M] {
        let k = self.position;
        let after = means_after(&y);
        for i in 0..k {
            y[i] = b_param(y[i], after[i], PARAM_A, 0.02, 50.0);
        }
        for yi in &mut y[k..] {
            *yi = s_linear(*yi, 0.35);
        }
        reduced(&y, k, |group, _| mean(group))
    }
}

impl<const M: usize> Wfg8<M> {
    fn underlying(&self, mut y: Vec<f64>) -> [f64; M] {
        let k = self.position;
        // the mean of the values before each, of the input
        let mut sum: f64 = y[..k].iter().sum();
        for (i, yi) in y.iter_mut().enumerate().skip(k) {
            let before = unit(sum / i as f64);
            sum += *yi;
            *yi = s_linear(b_param(*yi, before, PARAM_A, 0.02, 50.0), 0.35);
        }
        reduced(&y, k, |group, _| mean(group))
    }
}

impl<const M: usize> Wfg9<M> {
    fn underlying(&self, mut y: Vec<f64>) -> [f64; M] {
        let k = self.position;
        let after = means_after(&y);
        let last = y.len() - 1;
        for i in 0..last {
            y[i] = b_param(y[i], after[i], PARAM_A, 0.02, 50.0);
        }
        for (i, yi) in y.iter_mut().enumerate() {
            *yi = if i < k {
                s_decept(*yi, 0.35, 0.001, 0.05)
            } else {
                s_multi(*yi, 30.0, 95.0, 0.35)
            };
        }
        nonseparable(&y, k)
    }
}

// ---- the fronts ---------------------------------------------------------------------------------

// WFG2's disconnected shape, of x₁: 1 − x₁ cos²(5πx₁)
fn wfg2_disc(x1: f64) -> f64 {
    disc(x1, 5.0, 1.0, 1.0)
}

// the smallest x in [lo, hi], to the last bit, where `f` isn't negative, for f(lo) < 0 <= f(hi)
// and `f` changing sign once
fn bisect(f: impl Fn(f64) -> f64, mut lo: f64, mut hi: f64) -> f64 {
    loop {
        let mid = lo + (hi - lo) / 2.0;
        if mid <= lo || mid >= hi {
            return hi;
        }
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
}

// the values of x₁ on WFG2's front, where 1 − x₁ cos²(5πx₁) is below all its values at smaller
// x₁, as intervals (start, end): the first from 0, 0 included; the others without their start,
// where the shape equals the previous interval's end
fn wfg2_intervals() -> Vec<(f64, f64)> {
    // the local minima, where x cos²(5πx) peaks: cos(5πx) = 10πx sin(5πx), once in each
    // (j/5, j/5 + 1/10); then x₁ = 1, where the shape is 0
    let mut ends: Vec<f64> = (0..5)
        .map(|j| {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            let slope =
                |x: f64| sign * (10.0 * PI * x * math::sin(5.0 * PI * x) - math::cos(5.0 * PI * x));
            let start = j as f64 / 5.0;
            bisect(slope, start, start + 0.1)
        })
        .collect();
    ends.push(1.0);
    let mut intervals = vec![(0.0, ends[0])];
    for j in 1..ends.len() {
        // from where the shape falls to the previous minimum again, after the peak at
        // (2j − 1) / 10, where it is 1
        let previous = wfg2_disc(ends[j - 1]);
        let start = bisect(
            |x| previous - wfg2_disc(x),
            (2 * j - 1) as f64 / 10.0,
            ends[j],
        );
        intervals.push((start, ends[j]));
    }
    intervals
}

// `points` points spread evenly by length over the curve `point(x)` for x over `intervals` in
// turn, from the first's start to the last's end; the others' starts are left out, the points
// there being dominated
fn curve_front(
    intervals: &[(f64, f64)],
    point: impl Fn(f64) -> [f64; 2],
    points: usize,
) -> Vec<[f64; 2]> {
    const STEPS: usize = 2_000;
    // (x, the length of the curve up to x)
    let mut table: Vec<(f64, f64)> = Vec::with_capacity(intervals.len() * (STEPS + 1));
    let mut length = 0.0;
    for &(start, end) in intervals {
        let mut previous = point(start);
        table.push((start, length));
        for step in 1..=STEPS {
            let x = start + (end - start) * step as f64 / STEPS as f64;
            let next = point(x);
            length += (next[0] - previous[0]).hypot(next[1] - previous[1]);
            table.push((x, length));
            previous = next;
        }
    }
    let last = intervals.last().map_or(0.0, |&(_, end)| end);
    let mut index = 0;
    (0..points)
        .map(|i| {
            if i > 0 && i == points - 1 {
                return point(last);
            }
            let target = length * evenly(i, points);
            while table[index].1 < target {
                index += 1;
            }
            let (x1, l1) = table[index];
            if index == 0 || l1 == target {
                return point(x1);
            }
            // within one interval: the entries before and after a gap have the same length
            let (x0, l0) = table[index - 1];
            point(x0 + (x1 - x0) * (target - l0) / (l1 - l0))
        })
        .collect()
}

// 1 − cos(xπ/2) and 1 − sin(xπ/2), accurate near 0
fn one_minus_cos(x: f64) -> f64 {
    let s = math::sin(x * PI / 4.0);
    2.0 * s * s
}

fn one_minus_sin(x: f64) -> f64 {
    let s = math::sin((1.0 - x) * PI / 4.0);
    2.0 * s * s
}

// the underlying parameters x (with x_M = 0) of the point of a front with convex h₁…h_{M−1} in
// the direction `w` (its values positive), or None if the ray misses the front: x_{M−1} down to
// x₂ from the ratios of consecutive objectives, then x₁ by `root`, which gets
// (1 − cos(x₁π/2)) (1 − sin(x₂π/2)) w_M, what h_M(x₁) w_{M−1} must equal (h_{M−1} / h_M =
// w_{M−1} / w_M)
fn convex_position<const M: usize>(
    w: &[f64; M],
    root: impl Fn(&dyn Fn(f64) -> f64) -> Option<f64>,
) -> Option<[f64; M]> {
    let mut x = [0.0; M];
    // hₘ / hₘ₊₁ = cⱼ sⱼ₊₁ / sⱼ for j = M − m, with c = 1 − cos(xπ/2), s = 1 − sin(xπ/2) and
    // s_M = 1: cⱼ / sⱼ = q, and by the half-angle formulas, tan(xⱼπ/4) = r / (1 + r) with
    // r = √(q/2)
    let mut s_next = 1.0;
    for m in 1..M - 1 {
        let q = w[m - 1] / (w[m] * s_next);
        let r = (q / 2.0).sqrt();
        let xj = (4.0 / PI * math::atan(1.0 / (1.0 + 1.0 / r))).min(1.0);
        x[M - m - 1] = xj;
        s_next = one_minus_sin(xj);
    }
    x[0] = root(&|x1: f64| one_minus_cos(x1) * s_next * w[M - 1])?;
    Some(x)
}

// Das and Dennis's points, off the simplex's boundary by a hair, so that every ratio of their
// values is finite: directions from the origin
fn directions<const M: usize>(simplex: Vec<[f64; M]>) -> impl Iterator<Item = [f64; M]> {
    simplex.into_iter().map(|w| w.map(|v| v + 1e-30))
}

// WFG1's h_M, 1 − x − cos(10πx + π/2) / 10π, as d − sin(10πd) / 10π with d = 1 − x, and near
// x = 1 by its series, (10πd)³ / (6 · 10π) (1 − (10πd)² / 20 + (10πd)⁴ / 840): accurate where
// the shape falls to 0, for the directions near the f_M = 0 plane
fn wfg1_mixed(x1: f64) -> f64 {
    let a = 10.0 * PI;
    let d = 1.0 - x1;
    let ad = a * d;
    if ad < 1e-2 {
        let square = ad * ad;
        ad * square / (6.0 * a) * (1.0 - square / 20.0 + square * square / 840.0)
    } else {
        d - math::sin(ad) / a
    }
}

impl<const M: usize> Wfg1<M> {
    fn front(points: usize) -> Option<Vec<[f64; M]>> {
        let point = |x: [f64; M]| Self::shape(&x);
        if M == 2 {
            let curve = |x1: f64| {
                let f = point(std::array::from_fn(|i| if i == 0 { x1 } else { 0.0 }));
                [f[0], f[1]]
            };
            let front = curve_front(&[(0.0, 1.0)], curve, points);
            return Some(front.into_iter().map(resized).collect());
        }
        let front = directions(simplex_points::<M>(points)).map(|w| {
            let x = convex_position(&w, |product| {
                let f = |x1: f64| product(x1) - wfg1_mixed(x1) * w[M - 2];
                Some(bisect(f, 0.0, 1.0))
            });
            point(x.expect("a root"))
        });
        Some(front.collect())
    }

    fn extremes() -> Option<([f64; M], [f64; M])> {
        Some(scaled_extremes())
    }
}

impl<const M: usize> Wfg2<M> {
    fn front(points: usize) -> Option<Vec<[f64; M]>> {
        let intervals = wfg2_intervals();
        let point = |x: [f64; M]| Self::shape(&x);
        if M == 2 {
            let curve = |x1: f64| {
                let f = point(std::array::from_fn(|i| if i == 0 { x1 } else { 0.0 }));
                [f[0], f[1]]
            };
            let front = curve_front(&intervals, curve, points);
            return Some(front.into_iter().map(resized).collect());
        }
        if points == 0 {
            return Some(Vec::new());
        }
        // more directions until enough of them meet the front
        let mut divisions = divisions_for::<M>(points);
        loop {
            let front: Vec<[f64; M]> = directions(das_dennis::<M>(divisions))
                .filter_map(|w| {
                    convex_position(&w, |product| {
                        let f = |x1: f64| product(x1) - wfg2_disc(x1) * w[M - 2];
                        // F increases over the front's intervals: the first whose end isn't
                        // negative has the root, unless the ray passes between two intervals
                        let (index, &(start, end)) = intervals
                            .iter()
                            .enumerate()
                            .find(|(_, (_, end))| f(*end) >= 0.0)?;
                        (index == 0 || f(start) < 0.0).then(|| bisect(f, start, end))
                    })
                })
                .map(point)
                .collect();
            if front.len() >= points {
                return Some(front);
            }
            divisions += 1;
        }
    }

    fn extremes() -> Option<([f64; M], [f64; M])> {
        Some(scaled_extremes())
    }
}

impl<const M: usize> Wfg3<M> {
    // the segment from (0, 4) to (2, 0) for 2 objectives; not known for more
    fn front(points: usize) -> Option<Vec<[f64; M]>> {
        (M == 2).then(|| {
            (0..points)
                .map(|i| {
                    let x1 = evenly(i, points);
                    resized([2.0 * x1, 4.0 * (1.0 - x1)])
                })
                .collect()
        })
    }

    fn extremes() -> Option<([f64; M], [f64; M])> {
        (M == 2).then(scaled_extremes)
    }
}

// the ideal point 0 and the nadir point (2, 4, …, 2M): each hₘ from 0 to 1 on the front
fn scaled_extremes<const M: usize>() -> ([f64; M], [f64; M]) {
    ([0.0; M], std::array::from_fn(|m| 2.0 * (m + 1) as f64))
}

// the values of an array of K values as one of M, when M = K
fn resized<const K: usize, const M: usize>(values: [f64; K]) -> [f64; M] {
    std::array::from_fn(|i| values[i])
}

macro_rules! concave {
    ($($name:ident),*) => {$(
        impl<const M: usize> $name<M> {
            fn shape(t: &[f64; M]) -> [f64; M] {
                objectives(t, false, |x, m| concave(x, m))
            }

            // Das and Dennis's points projected on the unit sphere, fₘ scaled by 2m
            fn front(points: usize) -> Option<Vec<[f64; M]>> {
                let front = spherical_front::<M>(points).into_iter();
                Some(front.map(|p| std::array::from_fn(|m| 2.0 * (m + 1) as f64 * p[m])).collect())
            }

            fn extremes() -> Option<([f64; M], [f64; M])> {
                Some(scaled_extremes())
            }
        }
    )*};
}

concave!(Wfg4, Wfg5, Wfg6, Wfg7, Wfg8, Wfg9);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamRng;
    use crate::genome::Representation;

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= tolerance * e.abs().max(1.0),
                "{actual:?} is not {expected:?}"
            );
        }
    }

    // whether `a` is better than `b` in every objective by more than `slack`
    fn dominates<const M: usize>(a: &[f64; M], b: &[f64; M], slack: f64) -> bool {
        a.iter().zip(b).all(|(a, b)| *a < b - slack)
    }

    // the formulas of table XI, at points where the values follow by hand
    #[test]
    fn transformations_match_table_xi() {
        assert_eq!(b_poly(0.5, 2.0), 0.25);
        assert_eq!(b_poly(0.0, 0.02), 0.0);
        // b_flat: 0 and 1 stay, [B, C] maps to A, and below B, A y / B
        assert_eq!(b_flat(0.0, 0.8, 0.75, 0.85), 0.0);
        assert_eq!(b_flat(0.8, 0.8, 0.75, 0.85), 0.8);
        assert_close(&[b_flat(1.0, 0.8, 0.75, 0.85)], &[1.0], 1e-15);
        assert_close(&[b_flat(0.375, 0.8, 0.75, 0.85)], &[0.4], 1e-15);
        // b_param: u = 0 gives y^B, u = 1 y^C, and u = 1/2 y^(B + (C − B) A) = y here
        let (a, b, c) = (PARAM_A, 0.02, 50.0);
        assert_close(
            &[b_param(0.5, 0.0, a, b, c)],
            &[math::powf(0.5, 0.02)],
            1e-15,
        );
        assert_close(&[b_param(0.9, 1.0, a, b, c)], &[math::powi(0.9, 50)], 1e-13);
        assert_close(&[b_param(0.3, 0.5, a, b, c)], &[0.3], 1e-14);
        // s_linear: A maps to 0, and 0 and 1 to 1
        assert_eq!(s_linear(0.35, 0.35), 0.0);
        assert_eq!(s_linear(0.0, 0.35), 1.0);
        assert_eq!(s_linear(1.0, 0.35), 1.0);
        assert_close(&[s_linear(0.675, 0.35)], &[0.5], 1e-15);
        // s_decept: 0 at A, C at 0 and 1 (the deceptive minima), 1 at the basin's edges
        assert_eq!(s_decept(0.35, 0.35, 0.001, 0.05), 0.0);
        assert_close(&[s_decept(0.0, 0.35, 0.001, 0.05)], &[0.05], 1e-12);
        assert_close(&[s_decept(1.0, 0.35, 0.001, 0.05)], &[0.05], 1e-12);
        assert_close(&[s_decept(0.351, 0.35, 0.001, 0.05)], &[1.0], 1e-12);
        // s_multi: 0 at C, where cos((4A + 2)π / 2) = −1; 1 at 0 and 1, where the distance
        // term is ±1/2 and the cosine 1: (2 + 4B / 4) / (B + 2)
        assert_close(&[s_multi(0.35, 30.0, 10.0, 0.35)], &[0.0], 1e-15);
        assert_close(&[s_multi(0.0, 30.0, 10.0, 0.35)], &[1.0], 1e-15);
        assert_close(&[s_multi(1.0, 30.0, 95.0, 0.35)], &[1.0], 1e-15);
        // r_sum: (1 × 0 + 3 × 1) / 4; r_nonsep with A = 1 is the mean, and with A = 2 on a
        // pair, (a + b + 2|a − b|) / 3
        assert_eq!(r_sum(&[0.0, 1.0], |i| [1.0, 3.0][i]), 0.75);
        assert_eq!(mean(&[0.2, 0.4, 0.9]), 0.5);
        assert_eq!(r_nonsep(&[0.2, 0.4, 0.9], 1), 0.5);
        assert_eq!(r_nonsep(&[0.0, 1.0], 2), 1.0);
        assert_close(&[r_nonsep(&[0.5, 0.5], 2)], &[1.0 / 3.0], 1e-15);
        // A = 3 on (0, 0, 1): 1 + |0 − 0| + |0 − 1| + |0 − 1| + |0 − 0| + |1 − 0| + |1 − 0| = 5,
        // over 3 ⌈3/2⌉ (1 + 6 − 4) / 3 = 6
        assert_close(&[r_nonsep(&[0.0, 0.0, 1.0], 3)], &[5.0 / 6.0], 1e-15);
    }

    // the formulas of table X
    #[test]
    fn shapes_match_table_x() {
        // M = 3 with x₁ = x₂ = 0.5 (x₃, the distance, unused)
        let x = [0.5, 0.5, 0.0];
        let linear: Vec<f64> = (1..=3).map(|m| super::linear(&x, m)).collect();
        assert_eq!(linear, [0.25, 0.25, 0.5]);
        let c = 1.0 - math::cos(PI / 4.0);
        let s = 1.0 - math::sin(PI / 4.0);
        let convex: Vec<f64> = (1..=3).map(|m| super::convex(&x, m)).collect();
        assert_close(&convex, &[c * c, c * s, s], 1e-15);
        // concave: a point of the unit sphere, (1/2, 1/2, √2/2)
        let concave: Vec<f64> = (1..=3).map(|m| super::concave(&x, m)).collect();
        assert_close(&concave, &[0.5, 0.5, 0.5f64.sqrt()], 1e-15);
        // mixed with A = 5, α = 1: 1 − x at x = 0.2k (cos(2kπ + π/2) = 0), 1 at 0, 0 at 1 (to
        // the rounding of cos(10.5π))
        assert_eq!(mixed(0.0, 5.0, 1.0), 1.0);
        assert!(mixed(1.0, 5.0, 1.0) < 1e-16);
        assert_close(&[mixed(0.4, 5.0, 1.0)], &[0.6], 1e-15);
        // disc with A = 5, α = β = 1: 1 − x cos²(5πx): 1 − x at x = 0.2k, 1 at x = 0.1
        assert_eq!(disc(0.0, 5.0, 1.0, 1.0), 1.0);
        assert_close(&[disc(0.4, 5.0, 1.0, 1.0)], &[0.6], 1e-15);
        assert_close(&[disc(0.1, 5.0, 1.0, 1.0)], &[1.0], 1e-15);
        assert_eq!(disc(1.0, 5.0, 1.0, 1.0), 0.0);
    }

    // z with zᵢ = 2i ((a i + b) mod 1): a point anywhere in the bounds, without a list of numbers
    fn spread_point(n: usize, a: f64, b: f64) -> Reals {
        (1..=n)
            .map(|i| 2.0 * i as f64 * ((a * i as f64 + b) % 1.0))
            .collect()
    }

    type Two<'a> = &'a dyn MultiFitnessFunction<Reals, 2, Output = [f64; 2]>;
    type Three<'a> = &'a dyn MultiFitnessFunction<Reals, 3, Output = [f64; 3]>;

    // values of the authors' C++ toolkit, version 2006.03.28 (`Problems::WFG1` … `WFG9` of
    // ExampleProblems.cpp), compiled and run at these points, printed to 17 digits
    #[test]
    fn values_match_the_toolkit() {
        // 2 objectives, k = 2, l = 4, at zᵢ = 2i ((0.37 i + 0.11) mod 1)
        let z = spread_point(6, 0.37, 0.11);
        let two: [(Two<'_>, [f64; 2]); 9] = [
            (
                &Wfg1::<2>::new(2, 4),
                [2.9521691109522714, 0.97445227500261],
            ),
            (
                &Wfg2::<2>::new(2, 4),
                [1.5793517175960388, 3.8576901485363813],
            ),
            (
                &Wfg3::<2>::new(2, 4),
                [1.9138827838827839, 1.9238827838827837],
            ),
            (
                &Wfg4::<2>::new(2, 4),
                [1.4056727977183536, 3.7515601023124354],
            ),
            (
                &Wfg5::<2>::new(2, 4),
                [2.0899502646748944, 3.2324642099371106],
            ),
            (
                &Wfg6::<2>::new(2, 4),
                [2.470388403034529, 2.5745763998994367],
            ),
            (
                &Wfg7::<2>::new(2, 4),
                [1.3020135240884834, 4.037767908965662],
            ),
            (
                &Wfg8::<2>::new(2, 4),
                [2.4877866969669267, 2.7674219485001066],
            ),
            (
                &Wfg9::<2>::new(2, 4),
                [2.624356071590899, 2.9327440749083484],
            ),
        ];
        for (problem, expected) in two {
            assert_close(&problem.evaluate(&z), &expected, 1e-13);
        }
        // 3 objectives, k = 4, l = 4, at zᵢ = 2i ((0.61 i + 0.3) mod 1)
        let z = spread_point(8, 0.61, 0.3);
        let three: [(Three<'_>, [f64; 3]); 9] = [
            (
                &Wfg1::<3>::new(4, 4),
                [2.8015876273655227, 0.8964875000359592, 0.8952188623922328],
            ),
            (
                &Wfg2::<3>::new(4, 4),
                [0.910322694902473, 1.491957405463698, 6.4218866500617064],
            ),
            (
                &Wfg3::<3>::new(4, 4),
                [1.3097324175824177, 2.2075681318681326, 2.365677655677656],
            ),
            (
                &Wfg4::<3>::new(4, 4),
                [0.9932073560555065, 2.201357248150744, 5.260644030908747],
            ),
            (
                &Wfg5::<3>::new(4, 4),
                [1.1483344947075573, 2.455882304884854, 4.7876166780702984],
            ),
            (
                &Wfg6::<3>::new(4, 4),
                [2.396138328269317, 2.448664170763778, 3.1804778687269697],
            ),
            (
                &Wfg7::<3>::new(4, 4),
                [1.1004377505401686, 3.8094611116454455, 2.999443223922662],
            ),
            (
                &Wfg8::<3>::new(4, 4),
                [2.079513802281638, 3.737539809885213, 3.538476875082198],
            ),
            (
                &Wfg9::<3>::new(4, 4),
                [2.4662435170380617, 1.8739095262489003, 3.6775684130105892],
            ),
        ];
        for (problem, expected) in three {
            assert_close(&problem.evaluate(&z), &expected, 1e-13);
        }
    }

    #[derive(Clone, Copy)]
    enum Optimum {
        // the distance parameters at 0.35 × 2i (WFG1-7)
        Fixed,
        // each from the values before it (WFG8)
        Before,
        // each from the values after it (WFG9)
        After,
    }

    // an optimal solution, from normalized position values yᵢ = zᵢ / 2i: the distance parameters
    // of sections VIII-A and VIII-B, as the toolkit's main.cpp computes them
    fn optimal(position: &[f64], distance: usize, optimum: Optimum) -> Reals {
        let k = position.len();
        let n = k + distance;
        let mut y = position.to_vec();
        match optimum {
            Optimum::Fixed => y.resize(n, 0.35),
            Optimum::Before => {
                for i in k..n {
                    let u = y.iter().sum::<f64>() / i as f64;
                    let v = PARAM_A - (1.0 - 2.0 * u) * ((0.5 - u).floor() + PARAM_A).abs();
                    y.push(math::powf(0.35, 1.0 / (0.02 + 49.98 * v)));
                }
            }
            Optimum::After => {
                let mut tail = vec![0.35];
                for count in 1..distance {
                    let u = tail.iter().sum::<f64>() / count as f64;
                    tail.push(math::powf(0.35, 1.0 / (0.02 + 1.96 * u)));
                }
                tail.reverse();
                y.extend(tail);
            }
        }
        y.iter()
            .enumerate()
            .map(|(i, yi)| yi * 2.0 * (i + 1) as f64)
            .collect()
    }

    fn random_position(k: usize, rng: &mut StreamRng) -> Vec<f64> {
        (0..k).map(|_| rng.unit_f64()).collect()
    }

    // Σ (fₘ / 2m)²: 1 on the concave fronts
    fn spherical<const M: usize>(f: &[f64; M]) -> f64 {
        let scaled = f
            .iter()
            .enumerate()
            .map(|(m, v)| v / (2.0 * (m + 1) as f64));
        scaled.map(|v| v * v).sum()
    }

    // Σ fₘ / 2m: 1 on WFG3's front
    fn planar<const M: usize>(f: &[f64; M]) -> f64 {
        f.iter()
            .enumerate()
            .map(|(m, v)| v / (2.0 * (m + 1) as f64))
            .sum()
    }

    // the optimal distance parameters put the underlying distance x_M at 0 (to the rounding that
    // WFG8's power, up to 50, magnifies), on the front: there,
    // the concave problems have Σ (fₘ / 2m)² = 1 and WFG3 Σ fₘ / 2m = 1; WFG1's sizes avoid the
    // indices where zᵢ / 2i can't be 0.35 (see below)
    #[test]
    fn optimal_solutions_lie_on_the_front() {
        fn check<const M: usize>(k: usize, l: usize, rng: &mut StreamRng) {
            macro_rules! on_front {
                ($problem:expr, $optimum:expr, $identity:expr) => {{
                    let problem = $problem;
                    for _ in 0..20 {
                        let z = optimal(&random_position(k, rng), l, $optimum);
                        assert!(problem.representation().validate(&z).is_ok());
                        let t = problem.underlying(normalized(&z, k + l));
                        assert!(t[M - 1] < 1e-14, "{}: {t:?}", problem.name());
                        let identity: Option<fn(&[f64; M]) -> f64> = $identity;
                        if let Some(identity) = identity {
                            let value = identity(&problem.evaluate(&z));
                            assert!((value - 1.0).abs() < 1e-12, "{}: {value}", problem.name());
                        }
                    }
                }};
            }
            on_front!(Wfg1::<M>::new(k, l), Optimum::Fixed, None);
            on_front!(Wfg2::<M>::new(k, l), Optimum::Fixed, None);
            on_front!(Wfg3::<M>::new(k, l), Optimum::Fixed, Some(planar));
            on_front!(Wfg4::<M>::new(k, l), Optimum::Fixed, Some(spherical));
            on_front!(Wfg5::<M>::new(k, l), Optimum::Fixed, Some(spherical));
            on_front!(Wfg6::<M>::new(k, l), Optimum::Fixed, Some(spherical));
            on_front!(Wfg7::<M>::new(k, l), Optimum::Fixed, Some(spherical));
            on_front!(Wfg8::<M>::new(k, l), Optimum::Before, Some(spherical));
            on_front!(Wfg9::<M>::new(k, l), Optimum::After, Some(spherical));
        }
        let mut rng = StreamRng::seed_from_u64(1);
        check::<2>(6, 4, &mut rng);
        check::<2>(3, 2, &mut rng);
        check::<3>(6, 4, &mut rng);
        check::<5>(16, 6, &mut rng);
    }

    // zᵢ / 2i is never exactly 0.35 in floating point for some i, 3, 6, 12, 24, 48, 53, …, and
    // WFG1's bias b_poly(·, 0.02) turns the last bit into a distance of 0.48: with such distance
    // parameters, no genome reaches WFG1's front, as the toolkit's README warns. The default
    // sizes (k = 4, l = 20) have i = 6, 12 and 24: x_M is at least about 0.069
    #[test]
    fn floating_point_keeps_wfg1_off_its_front() {
        let chain = |y: f64| b_poly(b_flat(s_linear(y, 0.35), 0.8, 0.75, 0.85), 0.02);
        for i in [3, 6, 12, 24] {
            let d = 2.0 * i as f64;
            let mut z = 0.35 * d;
            for _ in 0..8 {
                z = z.next_down();
            }
            let best = (0..16)
                .map(|_| {
                    z = z.next_up();
                    chain(z / d)
                })
                .fold(f64::INFINITY, f64::min);
            assert!(best > 0.47 && best < 0.49, "{i}: {best}");
        }
        assert_eq!(chain(0.35), 0.0);
        // the toolkit's optimal solution with the positions at 1: f₁ = x_M + 2, so x_M ≈ 0.069
        let problem = Wfg1::<2>::default();
        let f = problem.evaluate(&optimal(&[1.0; 4], 20, Optimum::Fixed));
        assert!(f[0] - 2.0 > 0.069 && f[0] - 2.0 < 0.07, "{f:?}");
    }

    // points of the fronts against optimal solutions: no optimal solution dominates a point of
    // the front, and (where the whole surface is optimal) no point of the front dominates one
    #[test]
    fn fronts_agree_with_the_optimal_solutions() {
        fn check<const M: usize, P>(problem: &P, points: usize, whole: bool)
        where
            P: MultiProblem<M, Representation = Real>
                + MultiFitnessFunction<Reals, M, Output = [f64; M]>,
        {
            let mut rng = StreamRng::seed_from_u64(2);
            let front = problem.optimal_front(points).expect("known");
            assert!(front.len() >= points);
            for _ in 0..300 {
                let position = random_position(6, &mut rng);
                let f = problem.evaluate(&optimal(&position, 4, Optimum::Fixed));
                for p in &front {
                    let name = problem.name();
                    assert!(!dominates(&f, p, 1e-9), "{name}: {f:?} dominates {p:?}");
                    assert!(!whole || !dominates(p, &f, 1e-9), "{name}: {p:?}, {f:?}");
                }
            }
        }
        check(&Wfg1::<2>::new(6, 4), 200, true);
        check(&Wfg1::<3>::new(6, 4), 200, true);
        check(&Wfg1::<4>::new(6, 4), 200, true);
        check(&Wfg2::<2>::new(6, 4), 200, false);
        check(&Wfg2::<3>::new(6, 4), 200, false);
        check(&Wfg2::<4>::new(6, 4), 200, false);
        check(&Wfg3::<2>::new(6, 4), 50, true);
        // the concave fronts lie on the sphere
        let front = Wfg4::<3>::default().optimal_front(91).expect("known");
        assert_eq!(front.len(), 91);
        assert!(front.iter().all(|p| (spherical(p) - 1.0).abs() < 1e-12));
        // WFG1 with 2 objectives: f₂ = 4 mixed(x₁) where f₁ = 2 (1 − cos(x₁π/2))
        for p in Wfg1::<2>::default().optimal_front(100).expect("known") {
            let x1 = math::acos(1.0 - p[0] / 2.0) * 2.0 / PI;
            assert!((p[1] - 4.0 * mixed(x1, 5.0, 1.0)).abs() < 1e-6, "{p:?}");
        }
        // WFG3 with 2 objectives: the segment f₁ / 2 + f₂ / 4 = 1
        let front = Wfg3::<2>::default().optimal_front(10).expect("known");
        assert!(front.iter().all(|p| (planar(p) - 1.0).abs() < 1e-15));
    }

    // WFG2's front: six intervals of x₁, each ending at a local minimum of the shape h, where
    // cos(5πx) = 10πx sin(5πx), or at 1, and starting (but the first) where h equals the
    // previous end's; h falls throughout each
    #[test]
    fn wfg2_has_six_regions() {
        let intervals = wfg2_intervals();
        assert_eq!(intervals.len(), 6);
        assert_eq!(intervals[0].0, 0.0);
        assert_eq!(intervals[5].1, 1.0);
        for (j, &(start, end)) in intervals.iter().enumerate() {
            assert!(start < end);
            if j < 5 {
                let slope = math::cos(5.0 * PI * end) - 10.0 * PI * end * math::sin(5.0 * PI * end);
                assert!(slope.abs() < 1e-12, "{j}: {slope}");
                assert!(end > j as f64 / 5.0 && end < j as f64 / 5.0 + 0.1);
            }
            if j > 0 {
                let previous = wfg2_disc(intervals[j - 1].1);
                assert!((wfg2_disc(start) - previous).abs() < 1e-15);
                assert!(start > (2 * j - 1) as f64 / 10.0);
            }
            let values: Vec<f64> = (0..=100)
                .map(|step| wfg2_disc(start + (end - start) * step as f64 / 100.0))
                .collect();
            assert!(values.windows(2).all(|pair| pair[1] <= pair[0]), "{j}");
        }
        // the same intervals from a grid: the runs of values below every value before them
        let mut runs: Vec<(f64, f64)> = Vec::new();
        let (mut lowest, mut inside) = (f64::INFINITY, false);
        for step in 0..=100_000 {
            let x = step as f64 / 100_000.0;
            let value = wfg2_disc(x);
            if value < lowest {
                lowest = value;
                match runs.last_mut() {
                    Some(run) if inside => run.1 = x,
                    _ => runs.push((x, x)),
                }
                inside = true;
            } else {
                inside = false;
            }
        }
        assert_eq!(runs.len(), 6);
        for (run, interval) in runs.iter().zip(&intervals) {
            assert!((run.0 - interval.0).abs() < 2e-5 && (run.1 - interval.1).abs() < 2e-5);
        }
    }

    // WFG3 isn't degenerate for 3 objectives: with x₁ = x₂ = 1 and the distance x₃ = 1,
    // f = (1 + 2 · 1 · 1, 1 + 4 · 1 · 0, 1 + 6 · 0) = (3, 1, 1), which no point of the line
    // (t, 2t, 6 − 6t) dominates (f₃ ≤ 1 needs t ≥ 5/6, and then f₂ = 2t > 1), nor any solution
    #[test]
    fn wfg3_is_not_degenerate_for_three_objectives() {
        let problem = Wfg3::<3>::new(2, 2);
        // y = (1, 1, 0.35, 1): the pair of distance values reduces to r_nonsep((0, 1), 2) = 1
        let z = Reals::from(vec![2.0, 4.0, 0.35 * 6.0, 8.0]);
        let f = problem.evaluate(&z);
        assert_close(&f, &[3.0, 1.0, 1.0], 1e-15);
        for i in 0..=1000 {
            let t = i as f64 / 1000.0;
            assert!(!dominates(&[t, 2.0 * t, 6.0 - 6.0 * t], &f, 0.0));
        }
        let mut rng = StreamRng::seed_from_u64(3);
        let real = problem.representation();
        for _ in 0..100_000 {
            let other = problem.evaluate(&real.random_genome(&mut rng));
            assert!(!dominates(&other, &f, 0.0), "{other:?}");
        }
        assert!(problem.optimal_front(10).is_none());
        assert!(problem.ideal_point().is_none() && problem.nadir_point().is_none());
    }

    #[test]
    fn sizes_and_bounds() {
        let problem = Wfg1::<2>::default();
        assert_eq!((problem.position(), problem.distance()), (4, 20));
        assert_eq!(Wfg9::<3>::default().variables(), 24);
        assert_eq!(Wfg4::<5>::default().position(), 8);
        let real = Wfg2::<3>::new(2, 6).representation();
        assert_eq!(real.bounds().len(), 8);
        assert_eq!(real.bounds()[0], 0.0..=2.0);
        assert_eq!(real.bounds()[7], 0.0..=16.0);
        assert_eq!(
            Wfg5::<4>::default().nadir_point(),
            Some([2.0, 4.0, 6.0, 8.0])
        );
    }

    #[test]
    #[should_panic(expected = "WFG1 needs a positive multiple of M − 1 = 2 position parameters")]
    fn position_parameters_are_a_multiple_of_m_minus_1() {
        let _ = Wfg1::<3>::new(3, 20);
    }

    #[test]
    #[should_panic(expected = "WFG2 needs an even number of distance parameters, not 5")]
    fn wfg2_needs_an_even_l() {
        let _ = Wfg2::<2>::new(4, 5);
    }

    #[test]
    #[should_panic(expected = "WFG4 needs at least 1 distance parameter")]
    fn distance_parameters_are_needed() {
        let _ = Wfg4::<2>::new(4, 0);
    }

    #[test]
    #[should_panic(expected = "WFG takes genomes of k + l genes")]
    fn genomes_have_k_plus_l_genes() {
        let _ = Wfg6::<2>::default().evaluate(&Reals::from(vec![0.0; 23]));
    }
}
