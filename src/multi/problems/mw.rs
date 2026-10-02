//! MW1-MW14, Ma and Wang's constrained suite: Ma, Z. and Wang, Y. (2019). Evolutionary
//! constrained multiobjective optimization: test suite construction and performance comparisons.
//! *IEEE Transactions on Evolutionary Computation* 23(6): 972-986, section III (eqs. 12-28,
//! table II, figures 4 and 5).
//!
//! Every problem is `fᵢ = g(x_II) sᵢ(x_I)` (eq. 11): the shape functions sᵢ of the position
//! variables x_I = (x₁, …, x_{m−1}) give the unconstrained front, and one of three distance
//! functions of the other variables, each at least 1, how far a solution is from it:
//!
//! - `g₁ = 1 + Σᵢ₌ₘⁿ (1 − exp(−10 (zᵢ − 0.5 − (i − 1)/(2n))²))` with `zᵢ = xᵢ^(n−m)`
//!   (eq. 12), biased: its optimal zᵢ are above 0.5, where `xᵢ^(n−m)` rarely lands;
//! - `g₂ = 1 + Σᵢ₌ₘⁿ (1.5 + 0.1 zᵢ²/n − 1.5 cos 2πzᵢ)` with
//!   `zᵢ = 1 − exp(−10 (xᵢ − (i − 1)/n)²)` (eq. 13), multimodal, minimal at `xᵢ = (i − 1)/n`;
//! - `g₃ = 1 + Σᵢ₌ₘⁿ 2 (xᵢ + (xᵢ₋₁ − 0.5)² − 1)²` (eq. 14), with linked variables, minimal
//!   at `xᵢ = 1 − (xᵢ₋₁ − 0.5)²`.
//!
//! The constraints come from the paper's "similar functions", curves near the front whose shapes
//! a periodic term `A sin(B l^C)^D` adjusts (eq. 9): `sin(·)^D` is the D-th power of the sine.
//! The paper prints them as `c ≥ 0` or `c ≤ 0`; [`constraints`](MultiProblem::constraints)
//! gives them in the paper's order as `g(x) <= 0`, the first form negated. Where the paper writes
//! `arctan(f₂/f₁)`, genoxide takes `atan2(f₂, f₁)`, the same for f₁ > 0 and π/2 at f₁ = 0.
//!
//! The paper runs n = 15 variables, 2 objectives, and 3 for MW4, MW8 and MW14, and for more
//! objectives n = m − 1 + 13 (its supplement): the defaults here, 15 for the two-objective
//! problems and M + 12 for the others. At least m + 1 variables keep `g₁` able to reach 1.
//!
//! **The fronts.** The paper samples each front (the figures, and more than 1,000 points for its
//! IGD), and gives no formulas; genoxide derives them. For MW4, MW8 and MW14 they're analytic. For
//! the two-objective problems, each value of x₁ gives a ray in the objective space (vertical for
//! MW1-MW3, where f₁ = x₁, and from the origin for the others, where `f = g s(x₁)`), and the
//! distance g moves a solution along it, away from the unconstrained front at g = 1; going further
//! along a ray only makes a solution worse. So the front is made of the first feasible point of
//! each ray, the least g ≥ 1 where every constraint holds. [`optimal_front`] finds it on 20,000
//! rays, evenly spread in x₁ (in angle for the circles), and on the rays of isolated optimal
//! points: the least g from where each constraint's factors change sign, to machine precision. It
//! keeps the non-dominated points, computed once, and spreads the requested number evenly along
//! the front: each piece gets points in proportion to its length and half of the gaps beside it,
//! at least one, so that an isolated point stands for the stretch of front around it. Each
//! problem's docs describe its front, and the tests check it against the curves it's made of and
//! against feasible solutions near the Pareto set. The fronts agree with the paper's figures 4 and
//! 5 and with the sampled fronts that the authors published with their code (read only to
//! compare), but for MW5's, whose docs say how.
//!
//! **The authors' code** (<https://intleo.csu.edu.cn/codes/MW.rar>, their C++ implementation for
//! NSGA-II, with no license) was only read, to compare its formulas with the paper's: they agree,
//! but for one difference in `g₁`, where `i / (2 * n)` divides integers and is always 0, so that
//! its optimal zᵢ are all 0.5. genoxide follows the paper.
//!
//! [`optimal_front`]: MultiProblem::optimal_front

use super::dtlz::simplex_points;
use super::{MultiProblem, Piece, das_dennis, divisions_for, evenly, non_dominated, pieces_front};
use crate::constraint::at_most;
use crate::genome::{Real, Reals};
use crate::math;
use crate::multi::MultiFitnessFunction;
use crate::problems::Constraints;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, SQRT_2};
use std::sync::OnceLock;

const REFERENCE: &str = "Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective \
                         optimization: test suite construction and performance comparisons. IEEE \
                         Transactions on Evolutionary Computation 23(6): 972-986.";
const REFERENCE_URL: &str = "https://doi.org/10.1109/TEVC.2019.2896967";

// the paper's number of variables for two objectives
const VARIABLES: usize = 15;

// ---- the distance functions (eqs. 12-14) ---------------------------------------------------------

// g₁, biased: the variables from the m-th (0-based m − 1) on, zᵢ = xᵢ^(n−m), best at
// 0.5 + (i − 1)/(2n) for the 1-based i
fn g1(x: &[f64], m: usize) -> f64 {
    let n = x.len();
    let power = i32::try_from(n - m).unwrap_or(i32::MAX);
    let terms = (m - 1..n).map(|j| {
        let z = math::powi(x[j], power);
        let d = z - 0.5 - j as f64 / (2.0 * n as f64);
        1.0 - math::exp(-10.0 * d * d)
    });
    1.0 + terms.sum::<f64>()
}

// g₂, multimodal: zᵢ = 1 − exp(−10 (xᵢ − (i − 1)/n)²), 0 at the best xᵢ
fn g2(x: &[f64], m: usize) -> f64 {
    let n = x.len() as f64;
    let terms = (m - 1..x.len()).map(|j| {
        let d = x[j] - j as f64 / n;
        let z = 1.0 - math::exp(-10.0 * d * d);
        1.5 + 0.1 / n * z * z - 1.5 * math::cos(2.0 * PI * z)
    });
    1.0 + terms.sum::<f64>()
}

// g₃, linked: each variable best at 1 − (the previous − 0.5)²
fn g3(x: &[f64], m: usize) -> f64 {
    let terms = (m - 1..x.len()).map(|j| {
        let t = x[j] + (x[j - 1] - 0.5) * (x[j - 1] - 0.5) - 1.0;
        2.0 * t * t
    });
    1.0 + terms.sum::<f64>()
}

// the total violation of constraints g(x) <= 0, as `Constraints::violation` adds it up
fn violation(values: &[f64]) -> f64 {
    values.iter().map(|&g| at_most(g, 0.0)).sum()
}

// the number of variables, checked: at least m + 1
fn checked(label: &str, objectives: usize, variables: usize) -> usize {
    assert!(objectives >= 2, "{label} needs at least 2 objectives");
    assert!(
        variables > objectives,
        "{label} needs more variables than objectives"
    );
    variables
}

// ---- the fronts of the two-objective problems ----------------------------------------------------

// a two-objective problem in the plane of its objectives: the point at the position u (from x₁)
// and the distance g, and the functions whose signs decide whether a point is feasible
trait Plane {
    // the range of u
    const LOW: f64;
    const HIGH: f64;

    // the largest g where a ray's first feasible point may be
    const LAST: f64 = LAST_DISTANCE;

    // the position of the ray at t in [0, 1]: evenly spread by default, and evenly spread in
    // angle for the circles
    fn ray(t: f64) -> f64 {
        Self::LOW + (Self::HIGH - Self::LOW) * t
    }

    // positions where the front has a point that evenly spread rays miss
    fn extra_rays() -> Vec<f64> {
        Vec::new()
    }

    // points of the front that no ray reaches in floating point, such as a limit
    fn extra_points() -> Vec<[f64; 2]> {
        Vec::new()
    }

    fn point(u: f64, g: f64) -> [f64; 2];

    fn feasible(f: [f64; 2]) -> bool;

    // the factors of the constraints: feasibility changes only where one of them changes sign;
    // unused ones are 1
    fn factors(f: [f64; 2]) -> [f64; 8];

    // the front, computed once
    fn front() -> &'static [[f64; 2]];
}

// rays of the front: evenly spread over the positions, and how far along each to look
const RAYS: usize = 20_000;
const LAST_DISTANCE: f64 = 2.0;
const DISTANCE_STEP: f64 = 1e-3;

// the least g >= 1 at which the ray at u is feasible, if any up to LAST_DISTANCE: g = 1, or where
// a factor changes sign, found by bisection to the precision of f64
fn first_feasible<P: Plane>(u: f64) -> Option<f64> {
    let feasible = |g: f64| P::feasible(P::point(u, g));
    if feasible(1.0) {
        return Some(1.0);
    }
    let factors = |g: f64| P::factors(P::point(u, g));
    let steps = ((P::LAST - 1.0) / DISTANCE_STEP).round() as usize;
    let (mut low, mut before) = (1.0, factors(1.0));
    for step in 1..=steps {
        let high = 1.0 + step as f64 * DISTANCE_STEP;
        let after = factors(high);
        let mut best: Option<f64> = None;
        for j in 0..8 {
            if (before[j] <= 0.0) == (after[j] <= 0.0) {
                continue;
            }
            // the root of factor j: its sign at `a` is its sign at `low`
            let (mut a, mut b) = (low, high);
            loop {
                let middle = 0.5 * (a + b);
                if middle <= a || middle >= b {
                    break;
                }
                if (factors(middle)[j] <= 0.0) == (before[j] <= 0.0) {
                    a = middle;
                } else {
                    b = middle;
                }
            }
            let root = [a, b].into_iter().find(|&g| feasible(g));
            if let Some(g) = root {
                best = Some(best.map_or(g, |best: f64| best.min(g)));
            }
        }
        if best.is_none() && feasible(high) {
            // rounding made the roots infeasible (a constraint that is 0 there by a few ulps):
            // the least feasible g, by bisection on feasibility itself
            let (mut a, mut b) = (low, high);
            loop {
                let middle = 0.5 * (a + b);
                if middle <= a || middle >= b {
                    break;
                }
                if feasible(middle) {
                    b = middle;
                } else {
                    a = middle;
                }
            }
            best = Some(b);
        }
        if best.is_some() {
            return best;
        }
        (low, before) = (high, after);
    }
    None
}

// the front of a two-objective problem: the first feasible point of each ray, non-dominated,
// sorted by f₁
fn plane_front<P: Plane>() -> Vec<[f64; 2]> {
    let rays = (0..RAYS).map(|i| P::ray(evenly(i, RAYS)));
    let mut points: Vec<[f64; 2]> = rays
        .chain(P::extra_rays())
        .filter_map(|u| first_feasible::<P>(u).map(|g| P::point(u, g)))
        .collect();
    points.extend(P::extra_points());
    non_dominated(points)
}

// the distance between two points
fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (b[0] - a[0]).hypot(b[1] - a[1])
}

// `points` points of a front sorted by f₁, spread evenly along it: a step longer than 1% of the
// front's extent is a gap between two pieces, and each piece gets points in proportion to its
// length and half of the gaps beside it, at least one; within a piece, they're spread evenly by
// length, and a piece that is a single point repeats it. Each is a point of the front.
pub(super) fn spread_by_length(front: &[[f64; 2]], points: usize) -> Vec<[f64; 2]> {
    if points == 0 || front.is_empty() {
        return Vec::new();
    }
    let extent = distance(front[0], front[front.len() - 1]);
    // the pieces, as ranges of indices, and their lengths with half of the gaps beside them
    let mut pieces: Vec<(usize, usize, f64, f64)> = Vec::new();
    let (mut start, mut length, mut gap) = (0, 0.0, 0.0);
    for i in 1..=front.len() {
        let step = match front.get(i) {
            Some(&point) => distance(front[i - 1], point),
            None => f64::INFINITY,
        };
        if step > 0.01 * extent {
            let after = if step.is_finite() { step } else { 0.0 };
            pieces.push((start, i, length, length + 0.5 * (gap + after)));
            (start, length, gap) = (i, 0.0, after);
        } else {
            length += step;
        }
    }
    // the points of each piece: at least one if there are enough, and the rest by weight
    let mut counts = vec![0usize; pieces.len()];
    if points < pieces.len() {
        for i in 0..points {
            counts[(evenly(i, points) * (pieces.len() - 1) as f64).round() as usize] += 1;
        }
    } else {
        counts.fill(1);
        let total: f64 = pieces.iter().map(|piece| piece.3).sum();
        let rest = points - pieces.len();
        let shares: Vec<f64> = pieces
            .iter()
            .map(|piece| rest as f64 * piece.3 / total)
            .collect();
        for (count, share) in counts.iter_mut().zip(&shares) {
            *count += share.floor() as usize;
        }
        let mut order: Vec<usize> = (0..pieces.len()).collect();
        order.sort_by(|&a, &b| {
            (shares[b] - shares[b].floor()).total_cmp(&(shares[a] - shares[a].floor()))
        });
        let assigned: usize = counts.iter().sum();
        for &piece in order.iter().cycle().take(points - assigned) {
            counts[piece] += 1;
        }
    }
    let mut spread = Vec::with_capacity(points);
    let last = pieces.len() - 1;
    for (index, (&(start, end, length, _), &count)) in pieces.iter().zip(&counts).enumerate() {
        // the length along the piece at each of its points
        let mut along = Vec::with_capacity(end - start);
        let mut sum = 0.0;
        for i in start..end {
            if i > start {
                sum += distance(front[i - 1], front[i]);
            }
            along.push(sum);
        }
        for k in 0..count {
            // a lone point of the first piece is its start, of the last its end, of the others
            // its middle
            let t = match (count, index) {
                (1, 0) => 0.0,
                (1, i) if i == last => 1.0,
                (1, _) => 0.5,
                _ => evenly(k, count),
            };
            let target = t * length;
            let i = along.partition_point(|&a| a < target).min(along.len() - 1);
            let i = if i > 0 && target - along[i - 1] < along[i] - target {
                i - 1
            } else {
                i
            };
            spread.push(front[start + i]);
        }
    }
    spread
}

// the ideal and nadir points of a front sorted by f₁
fn corners(front: &[[f64; 2]]) -> ([f64; 2], [f64; 2]) {
    let (first, last) = (front[0], front[front.len() - 1]);
    ([first[0], last[1]], [last[0], first[1]])
}

// the struct, sizes, fitness and metadata of a two-objective problem, whose `objectives` and
// `values` (the constraints as g <= 0) are its own
macro_rules! two_objectives {
    ($(#[$doc:meta])* $name:ident, $label:literal, $count:literal, $upper:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name {
            variables: usize,
        }

        impl $name {
            /// The number of constraints.
            pub const CONSTRAINTS: usize = $count;

            /// The problem with `variables` variables, at least 3.
            ///
            /// # Panics
            ///
            /// With fewer than 3 variables.
            pub fn new(variables: usize) -> Self {
                Self {
                    variables: checked($label, 2, variables),
                }
            }

            /// The number of variables.
            pub fn variables(&self) -> usize {
                self.variables
            }
        }

        impl Default for $name {
            /// The problem with the paper's 15 variables.
            fn default() -> Self {
                Self::new(VARIABLES)
            }
        }

        impl MultiFitnessFunction<Reals, 2> for $name {
            type Output = ([f64; 2], f64);

            /// The objective values of `x` and its constraint violation, 0 when it's feasible.
            ///
            /// # Panics
            ///
            /// If `x` has fewer than 2 genes.
            fn evaluate(&self, x: &Reals) -> ([f64; 2], f64) {
                let f = self.objectives(x);
                (f, violation(&Self::values(f)))
            }
        }

        impl MultiProblem<2> for $name {
            type Representation = Real;

            fn name(&self) -> &'static str {
                $label
            }

            fn representation(&self) -> Real {
                Real::uniform(self.variables, 0.0..=$upper).expect("valid bounds")
            }

            fn reference(&self) -> &'static str {
                REFERENCE
            }

            fn reference_url(&self) -> Option<&'static str> {
                Some(REFERENCE_URL)
            }

            fn constraint_count(&self) -> usize {
                Self::CONSTRAINTS
            }

            fn constraints(&self, genome: &Reals) -> Constraints {
                Constraints::new(Self::values(self.objectives(genome)).to_vec(), Vec::new())
            }

            fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
                Some(spread_by_length(<Self as Plane>::front(), points))
            }

            fn ideal_point(&self) -> Option<[f64; 2]> {
                Some(corners(<Self as Plane>::front()).0)
            }

            fn nadir_point(&self) -> Option<[f64; 2]> {
                Some(corners(<Self as Plane>::front()).1)
            }
        }
    };
}

// the cached front of a problem implementing `Plane`
macro_rules! cached_front {
    () => {
        fn front() -> &'static [[f64; 2]] {
            static FRONT: OnceLock<Vec<[f64; 2]>> = OnceLock::new();
            FRONT.get_or_init(plane_front::<Self>)
        }
    };
}

// the factors of constraints that are single functions, padded with 1
fn padded<const K: usize>(values: [f64; K]) -> [f64; 8] {
    std::array::from_fn(|j| if j < K { values[j] } else { 1.0 })
}

// the angle term l = √2 f₂ − √2 f₁ of MW1-MW3
fn diagonal(f: [f64; 2]) -> f64 {
    SQRT_2 * f[1] - SQRT_2 * f[0]
}

// ---- MW1 -----------------------------------------------------------------------------------------

two_objectives!(
    /// MW1 (eq. 15): `f₁ = x₁`, `f₂ = g₁ (1 − 0.85 f₁/g₁)`, subject to
    /// `1 − f₁ − f₂ + 0.5 sin(2πl)⁸ ≥ 0` with `l = √2 f₂ − √2 f₁`.
    ///
    /// n variables in [0, 1] (15 by default), the biased distance function g₁. Type II (table
    /// II): the constraint cuts the unconstrained front, the line `f₂ = 1 − 0.85 f₁` for f₁ in
    /// [0, 1], into six pieces, and the front is those pieces: f₁ from 0 to 0.1148, 0.2061 to
    /// 0.2988, 0.4027 to 0.4855, 0.5977 to 0.6733, 0.7918 to 0.8616 and 0.9856 to 1 (derived
    /// here). Ideal point (0, 0.15), nadir point (1, 1). The feasible region is under 0.1‰ of the
    /// space (table II): teeth along the line, where the constraint's sine term lets `f₁ + f₂`
    /// exceed 1.
    Mw1,
    "MW1",
    1,
    1.0
);

impl Mw1 {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        Self::point(x[0], g1(x, 2))
    }

    fn values(f: [f64; 2]) -> [f64; 1] {
        let l = diagonal(f);
        [-(1.0 - f[0] - f[1] + 0.5 * math::powi(math::sin(2.0 * PI * l), 8))]
    }
}

impl Plane for Mw1 {
    const LOW: f64 = 0.0;
    const HIGH: f64 = 1.0;

    fn point(u: f64, g: f64) -> [f64; 2] {
        [u, g * (1.0 - 0.85 * u / g)]
    }

    fn feasible(f: [f64; 2]) -> bool {
        Self::values(f).iter().all(|&c| c <= 0.0)
    }

    fn factors(f: [f64; 2]) -> [f64; 8] {
        padded(Self::values(f))
    }

    cached_front!();
}

// ---- MW2 -----------------------------------------------------------------------------------------

two_objectives!(
    /// MW2 (eq. 16): `f₁ = x₁`, `f₂ = g₂ (1 − f₁/g₂)`, subject to
    /// `1 − f₁ − f₂ + 0.5 sin(3πl)⁸ ≥ 0` with `l = √2 f₂ − √2 f₁`.
    ///
    /// n variables in [0, 1] (15 by default), the multimodal distance function g₂. Type I: the
    /// whole unconstrained front, the line `f₂ = 1 − f₁`, is feasible (there `f₁ + f₂ = 1`), and
    /// is the front; the feasible region beyond it is teeth along the line, under 0.1‰ of the
    /// space. Ideal point (0, 0), nadir point (1, 1).
    Mw2,
    "MW2",
    1,
    1.0
);

impl Mw2 {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        Self::point(x[0], g2(x, 2))
    }

    fn values(f: [f64; 2]) -> [f64; 1] {
        let l = diagonal(f);
        [-(1.0 - f[0] - f[1] + 0.5 * math::powi(math::sin(3.0 * PI * l), 8))]
    }
}

impl Plane for Mw2 {
    const LOW: f64 = 0.0;
    const HIGH: f64 = 1.0;

    fn point(u: f64, g: f64) -> [f64; 2] {
        [u, g * (1.0 - u / g)]
    }

    fn feasible(f: [f64; 2]) -> bool {
        Self::values(f).iter().all(|&c| c <= 0.0)
    }

    fn factors(f: [f64; 2]) -> [f64; 8] {
        padded(Self::values(f))
    }

    cached_front!();
}

// ---- MW3 -----------------------------------------------------------------------------------------

two_objectives!(
    /// MW3 (eq. 17): `f₁ = x₁`, `f₂ = g₃ (1 − f₁/g₃)`, subject to
    /// `1.05 − f₁ − f₂ + 0.45 sin(0.75πl)⁶ ≥ 0` and `0.85 − f₁ − f₂ + 0.3 sin(0.75πl)² ≤ 0`,
    /// with `l = √2 f₂ − √2 f₁`.
    ///
    /// n variables in [0, 1] (15 by default), the linked distance function g₃. Type III: the
    /// second constraint cuts two stretches out of the unconstrained front `f₂ = 1 − f₁`, and the
    /// front runs along its boundary there, above the line. Derived here: the line for f₁ in
    /// [0, 0.1464], [0.3821, 0.6179] and [0.8536, 1], and the second constraint's boundary for
    /// f₁ in (0.1464, 0.3821) and (0.6179, 0.8536), where f₂ lies up to 0.15 above the line.
    /// Ideal point (0, 0), nadir point (1, 1). The first constraint leaves a narrow band in
    /// between, under 0.1‰ of the space.
    Mw3,
    "MW3",
    2,
    1.0
);

impl Mw3 {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        Self::point(x[0], g3(x, 2))
    }

    fn values(f: [f64; 2]) -> [f64; 2] {
        let l = diagonal(f);
        let s = math::sin(0.75 * PI * l);
        [
            -(1.05 - f[0] - f[1] + 0.45 * math::powi(s, 6)),
            0.85 - f[0] - f[1] + 0.3 * s * s,
        ]
    }
}

impl Plane for Mw3 {
    const LOW: f64 = 0.0;
    const HIGH: f64 = 1.0;

    fn point(u: f64, g: f64) -> [f64; 2] {
        [u, g * (1.0 - u / g)]
    }

    fn feasible(f: [f64; 2]) -> bool {
        Self::values(f).iter().all(|&c| c <= 0.0)
    }

    fn factors(f: [f64; 2]) -> [f64; 8] {
        padded(Self::values(f))
    }

    cached_front!();
}

// ---- MW4 -----------------------------------------------------------------------------------------

/// MW4 (eq. 18), with `M` objectives: `f₁ = g₁ Π_{i=1}^{M−1} (1 − xᵢ)`,
/// `f_k = g₁ x_{M−k+1} Π_{i=1}^{M−k} (1 − xᵢ)` for k from 2 to M − 1 and `f_M = g₁ x₁`, subject to
/// `1 + 0.4 sin(2.5πl)⁸ − f₁ − … − f_M ≥ 0` with `l = f_M − f₁ − … − f_{M−1}`.
///
/// n variables in [0, 1] (M + 12 by default: 15 for 3 objectives, the paper's), the biased
/// distance function g₁. Type I: the unconstrained front, the simplex `Σ fᵢ = 1`, is feasible (the
/// sine term is at least 0) and is the front. Ideal point the origin, nadir point (1, …, 1);
/// [`optimal_front`](MultiProblem::optimal_front) gives Das and Dennis's points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mw4<const M: usize> {
    variables: usize,
}

// ---- MW5 -----------------------------------------------------------------------------------------

two_objectives!(
    /// MW5 (eq. 19): `f₁ = g₁ x₁`, `f₂ = g₁ √(1 − (f₁/g₁)²)`, subject to
    /// `(1.7 − 0.2 sin 2l₁)² − f₁² − f₂² ≥ 0`, `(1 + 0.5 sin 6l₂³)² − f₁² − f₂² ≤ 0` and
    /// `(1 − 0.45 sin 6l₂³)² − f₁² − f₂² ≤ 0`, with `l₁ = arctan(f₂/f₁)` and
    /// `l₂ = 0.5π − 2 |l₁ − 0.25π|`.
    ///
    /// n variables in [0, 1] (15 by default), the biased distance function g₁. Type II, with a
    /// discrete front: on the unit circle, the unconstrained front, the second and third
    /// constraints both hold only where `sin 6l₂³ = 0`, at `l₂ = (kπ/6)^(1/3)` for k from 0 to 7,
    /// sixteen points (derived here), each at the end of a narrow tunnel of the feasible region.
    /// Off the circle, the first feasible point of a direction is at the radius
    /// `1 + 0.5 sin 6l₂³` or `1 + 0.45 |sin 6l₂³|`, and near each axis (l₂ below about 0.028) that
    /// point is non-dominated too: the front also has two short curves, from (1, 0) to about
    /// (0.99997, 0.0139) and from (0, 1) to about (0.0139, 0.99997), where `f₁ = cos l₁ (1 + 0.5
    /// sin 48 l₁³)` rises again (derived here; the paper's figure 4(e) and the authors' sampled
    /// front have only points there). Ideal point (0, 0), nadir point (1, 1).
    Mw5,
    "MW5",
    3,
    1.0
);

impl Mw5 {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        Self::point(x[0], g1(x, 2))
    }

    fn values(f: [f64; 2]) -> [f64; 3] {
        let l1 = math::atan2(f[1], f[0]);
        let l2 = 0.5 * PI - 2.0 * (l1 - 0.25 * PI).abs();
        let radius = f[0] * f[0] + f[1] * f[1];
        let wave = math::sin(6.0 * l2 * l2 * l2);
        let outer = 1.7 - 0.2 * math::sin(2.0 * l1);
        [
            -(outer * outer - radius),
            (1.0 + 0.5 * wave) * (1.0 + 0.5 * wave) - radius,
            (1.0 - 0.45 * wave) * (1.0 - 0.45 * wave) - radius,
        ]
    }

    // the directions of the sixteen points, as x₁ = cos l₁: l₂ = (kπ/6)^(1/3) for k from 1 to 7
    // on both sides of l₁ = π/4 (k = 0 are the ends, x₁ = 0 and 1)
    fn touching() -> Vec<f64> {
        (1..=7)
            .flat_map(|k| {
                let l2 = math::cbrt(k as f64 * PI / 6.0);
                let away = 0.5 * (FRAC_PI_2 - l2);
                [math::cos(FRAC_PI_4 - away), math::cos(FRAC_PI_4 + away)]
            })
            .collect()
    }
}

impl Plane for Mw5 {
    const LOW: f64 = 0.0;
    const HIGH: f64 = 1.0;

    fn extra_rays() -> Vec<f64> {
        Self::touching()
    }

    fn ray(t: f64) -> f64 {
        Self::HIGH * math::sin(FRAC_PI_2 * t)
    }

    fn point(u: f64, g: f64) -> [f64; 2] {
        circle(u, g, 1.0)
    }

    fn feasible(f: [f64; 2]) -> bool {
        Self::values(f).iter().all(|&c| c <= 0.0)
    }

    fn factors(f: [f64; 2]) -> [f64; 8] {
        padded(Self::values(f))
    }

    cached_front!();
}

// the point g (u, √(r² − u²)): the circle of radius r at g = 1
fn circle(u: f64, g: f64, radius: f64) -> [f64; 2] {
    [g * u, g * (radius * radius - u * u).max(0.0).sqrt()]
}

// ---- MW6 -----------------------------------------------------------------------------------------

two_objectives!(
    /// MW6 (eq. 20): `f₁ = g₂ x₁`, `f₂ = g₂ √(1.1² − (f₁/g₂)²)`, subject to
    /// `1 − (f₁/(1 + 0.15l))² − (f₂/(1 + 0.75l))² ≥ 0` with `l = cos(6 arctan(f₂/f₁)⁴)¹⁰`.
    ///
    /// n variables in [0, 1.1] (15 by default), the multimodal distance function g₂. Type II:
    /// the constraint allows the unconstrained front, the circle `f₁² + f₂² = 1.21`, only where
    /// l is large enough, in twelve pieces (derived here), eleven short ones near f₂ = 1.1 that
    /// narrow as f₁ falls, and one from (0.9549, 0.5461) to (1.1, 0); the first starts at
    /// f₁ = 0.0163. The paper's text reads the powers as `cos(·)¹⁰` of `6 arctan(·)⁴`, as the
    /// authors' code has them. Ideal point (0.0163, 0), nadir point (1.1, 1.0999).
    Mw6,
    "MW6",
    1,
    1.1
);

impl Mw6 {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        Self::point(x[0], g2(x, 2))
    }

    fn values(f: [f64; 2]) -> [f64; 1] {
        let l = math::powi(math::cos(6.0 * math::powi(math::atan2(f[1], f[0]), 4)), 10);
        let a = f[0] / (1.0 + 0.15 * l);
        let b = f[1] / (1.0 + 0.75 * l);
        [-(1.0 - a * a - b * b)]
    }
}

impl Plane for Mw6 {
    const LOW: f64 = 0.0;
    const HIGH: f64 = 1.1;
    // along a ray, the constraint only falls as g grows: a ray infeasible at g = 1 is infeasible
    const LAST: f64 = 1.0;

    fn ray(t: f64) -> f64 {
        Self::HIGH * math::sin(FRAC_PI_2 * t)
    }

    fn point(u: f64, g: f64) -> [f64; 2] {
        circle(u, g, 1.1)
    }

    fn feasible(f: [f64; 2]) -> bool {
        Self::values(f).iter().all(|&c| c <= 0.0)
    }

    fn factors(f: [f64; 2]) -> [f64; 8] {
        padded(Self::values(f))
    }

    cached_front!();
}

// ---- MW7 -----------------------------------------------------------------------------------------

two_objectives!(
    /// MW7 (eq. 21): `f₁ = g₃ x₁`, `f₂ = g₃ √(1 − (f₁/g₃)²)`, subject to
    /// `(1.2 + 0.4 sin(4l)¹⁶)² − f₁² − f₂² ≥ 0` and `(1.15 − 0.2 sin(4l)⁸)² − f₁² − f₂² ≤ 0`, with
    /// `l = arctan(f₂/f₁)`.
    ///
    /// n variables in [0, 1] (15 by default), the linked distance function g₃. Type III: the
    /// second constraint keeps solutions outside the radius `1.15 − 0.2 sin(4l)⁸`, which falls
    /// below 1 only around l = π/8 and 3π/8. The front, derived here: that boundary from
    /// (0, 1.15) to f₁ = 0.3203, the unit circle to 0.4434, the boundary again from
    /// (0.7202, 0.8963) to (0.8963, 0.7202), the circle to f₁ = 0.9473, and the boundary to
    /// (1.15, 0): in three pieces, symmetric in f₁ and f₂. Ideal point (0, 0), nadir point
    /// (1.15, 1.15).
    Mw7,
    "MW7",
    2,
    1.0
);

impl Mw7 {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        Self::point(x[0], g3(x, 2))
    }

    fn values(f: [f64; 2]) -> [f64; 2] {
        let s = math::sin(4.0 * math::atan2(f[1], f[0]));
        let radius = f[0] * f[0] + f[1] * f[1];
        let outer = 1.2 + 0.4 * math::powi(s, 16);
        let inner = 1.15 - 0.2 * math::powi(s, 8);
        [-(outer * outer - radius), inner * inner - radius]
    }
}

impl Plane for Mw7 {
    const LOW: f64 = 0.0;
    const HIGH: f64 = 1.0;

    fn ray(t: f64) -> f64 {
        Self::HIGH * math::sin(FRAC_PI_2 * t)
    }

    fn point(u: f64, g: f64) -> [f64; 2] {
        circle(u, g, 1.0)
    }

    fn feasible(f: [f64; 2]) -> bool {
        Self::values(f).iter().all(|&c| c <= 0.0)
    }

    fn factors(f: [f64; 2]) -> [f64; 8] {
        padded(Self::values(f))
    }

    cached_front!();
}

// ---- MW8 -----------------------------------------------------------------------------------------

/// MW8 (eq. 22), with `M` objectives: DTLZ2's shape with g₂, `f₁ = g₂ Π_{i=1}^{M−1}
/// cos(0.5πxᵢ)`, `f_k = g₂ sin(0.5π x_{M−k+1}) Π_{i=1}^{M−k} cos(0.5πxᵢ)` and
/// `f_M = g₂ sin(0.5πx₁)`, subject to `(1.25 − 0.5 sin(6l)²)² − f₁² − … − f_M² ≥ 0` with
/// `l = arcsin(f_M / √(f₁² + … + f_M²))`.
///
/// n variables in [0, 1] (M + 12 by default: 15 for 3 objectives, the paper's), the multimodal
/// distance function g₂. Type II: the constraint allows the radius up to `1.25 − 0.5 sin(6l)²`,
/// which reaches the unit sphere, the unconstrained front, only where `sin(6l)² ≤ 1/2`: in four
/// bands of l, [0, π/24], [π/8, 5π/24], [7π/24, 3π/8] and [11π/24, π/2] (derived here). No
/// feasible solution exists in the directions between them, and the front is the unit sphere in
/// the four bands. Ideal point the origin, nadir point (1, …, 1);
/// [`optimal_front`](MultiProblem::optimal_front) gives Das and Dennis's points on the sphere in
/// the bands, with more divisions until there are enough (the arcs themselves for 2 objectives).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mw8<const M: usize> {
    variables: usize,
}

// ---- MW9 -----------------------------------------------------------------------------------------

two_objectives!(
    /// MW9 (eq. 23): `f₁ = g₁ x₁`, `f₂ = g₁ (1 − (f₁/g₁)^0.6)`, subject to
    /// `min(T₁, T₂ T₃) ≤ 0`, with `T₁ = (1 − 0.64 f₁² − f₂)(1 − 0.36 f₁² − f₂)`,
    /// `T₂ = 1.35² − (f₁ + 0.35)² − f₂` and `T₃ = 1.15² − (f₁ + 0.15)² − f₂`.
    ///
    /// n variables in [0, 1] (15 by default), the biased distance function g₁. Type IV: the
    /// unconstrained front, `f₂ = 1 − f₁^0.6`, is infeasible but for its ends, and the front is on
    /// the boundary of the feasible region, above it: `f₂ = 1 − 0.64 f₁²` for f₁ in
    /// [0, 0.5868] and `f₂ = 1.15² − (f₁ + 0.15)²` for f₁ in [0.5868, 1], where the two
    /// curves cross at `f₁ = (√0.522 − 0.3)/0.72` (derived here). Concave, from (0, 1) to
    /// (1, 0); ideal point (0, 0), nadir point (1, 1).
    Mw9,
    "MW9",
    1,
    1.0
);

impl Mw9 {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        Self::point(x[0], g1(x, 2))
    }

    // T₁'s two factors, T₂ and T₃
    fn terms(f: [f64; 2]) -> [f64; 4] {
        let (f1, f2) = (f[0], f[1]);
        [
            1.0 - 0.64 * f1 * f1 - f2,
            1.0 - 0.36 * f1 * f1 - f2,
            1.35 * 1.35 - (f1 + 0.35) * (f1 + 0.35) - f2,
            1.15 * 1.15 - (f1 + 0.15) * (f1 + 0.15) - f2,
        ]
    }

    fn values(f: [f64; 2]) -> [f64; 1] {
        let [a, b, t2, t3] = Self::terms(f);
        [(a * b).min(t2 * t3)]
    }
}

impl Plane for Mw9 {
    const LOW: f64 = 0.0;
    const HIGH: f64 = 1.0;

    fn point(u: f64, g: f64) -> [f64; 2] {
        [g * u, g * (1.0 - math::powf(u, 0.6))]
    }

    fn feasible(f: [f64; 2]) -> bool {
        Self::values(f).iter().all(|&c| c <= 0.0)
    }

    fn factors(f: [f64; 2]) -> [f64; 8] {
        padded(Self::terms(f))
    }

    cached_front!();
}

// ---- MW10 ----------------------------------------------------------------------------------------

two_objectives!(
    /// MW10 (eq. 24): `f₁ = g₂ x₁ⁿ`, `f₂ = g₂ (1 − (f₁/g₂)²)`, subject to
    /// `(2 − 4f₁² − f₂)(2 − 8f₁² − f₂) ≥ 0`, `(2 − 2f₁² − f₂)(2 − 16f₁² − f₂) ≤ 0` and
    /// `(1 − f₁² − f₂)(1.2 − 1.2f₁² − f₂) ≤ 0`.
    ///
    /// n variables in [0, 1] (15 by default), the multimodal distance function g₂, and f₁ biased
    /// by the power n of x₁. Type III, disconnected: the feasible region is two islands between
    /// the parabolas, and the front, derived here, has two pieces, each part boundary and part
    /// unconstrained front `f₂ = 1 − f₁²`: from (0.2325, 1.1350) along the boundary of the second
    /// constraint to (0.2582, 0.9333), then the parabola to (0.3779, 0.8572); and from
    /// (0.5345, 0.8571) along the boundary of the first constraint to (0.5774, 0.6667), then the
    /// parabola to (1, 0). Ideal point (0.2325, 0), nadir point (1, 1.1350).
    Mw10,
    "MW10",
    3,
    1.0
);

impl Mw10 {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        let power = i32::try_from(x.len()).unwrap_or(i32::MAX);
        Self::point(math::powi(x[0], power), g2(x, 2))
    }

    fn terms(f: [f64; 2]) -> [f64; 6] {
        let (square, f2) = (f[0] * f[0], f[1]);
        [
            2.0 - 4.0 * square - f2,
            2.0 - 8.0 * square - f2,
            2.0 - 2.0 * square - f2,
            2.0 - 16.0 * square - f2,
            1.0 - square - f2,
            1.2 - 1.2 * square - f2,
        ]
    }

    fn values(f: [f64; 2]) -> [f64; 3] {
        let [a, b, c, d, e, h] = Self::terms(f);
        [-(a * b), c * d, e * h]
    }
}

impl Plane for Mw10 {
    const LOW: f64 = 0.0;
    const HIGH: f64 = 1.0;

    // u = x₁ⁿ
    fn point(u: f64, g: f64) -> [f64; 2] {
        [g * u, g * (1.0 - u * u)]
    }

    fn feasible(f: [f64; 2]) -> bool {
        Self::values(f).iter().all(|&c| c <= 0.0)
    }

    fn factors(f: [f64; 2]) -> [f64; 8] {
        padded(Self::terms(f))
    }

    cached_front!();
}

// ---- MW11 ----------------------------------------------------------------------------------------

two_objectives!(
    /// MW11 (eq. 25): `f₁ = g₃ x₁`, `f₂ = g₃ √(2 − (f₁/g₃)²)`, subject to
    /// `(3 − f₁² − f₂)(3 − 2f₁² − f₂) ≥ 0`, `(3 − 0.625f₁² − f₂)(3 − 7f₁² − f₂) ≤ 0`,
    /// `(1.62 − 0.18f₁² − f₂)(1.125 − 0.125f₁² − f₂) ≥ 0` and
    /// `(2.07 − 0.23f₁² − f₂)(0.63 − 0.07f₁² − f₂) ≤ 0`.
    ///
    /// n variables in [0, √2] (15 by default), the linked distance function g₃. Type IV: the
    /// front is on the boundaries of the feasible islands, and one isolated point, (1, 1), where
    /// two boundaries touch the unconstrained front `f₁² + f₂² = 2` (feasible only there, as the
    /// paper says). Derived here: two pieces, from (0.3707, 2.0383) to (0.8707, 1.4835) and from
    /// (1.4639, 0.8570) to (2.0662, 0.3311), each along two boundaries, and the point (1, 1),
    /// which dominates a third island's boundary (the authors' sampled front has the same).
    /// (1, 1) needs x₁ = 1 and g₃ = 1 exactly: a solution beside it is infeasible. Ideal point
    /// (0.3707, 0.3311), nadir point (2.0662, 2.0383). The bound √2 is `SQRT_2`, whose square is
    /// just above 2: f₂'s square root takes 0 there.
    Mw11,
    "MW11",
    4,
    SQRT_2
);

impl Mw11 {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        Self::point(x[0], g3(x, 2))
    }

    fn terms(f: [f64; 2]) -> [f64; 8] {
        let (square, f2) = (f[0] * f[0], f[1]);
        [
            3.0 - square - f2,
            3.0 - 2.0 * square - f2,
            3.0 - 0.625 * square - f2,
            3.0 - 7.0 * square - f2,
            1.62 - 0.18 * square - f2,
            1.125 - 0.125 * square - f2,
            2.07 - 0.23 * square - f2,
            0.63 - 0.07 * square - f2,
        ]
    }

    fn values(f: [f64; 2]) -> [f64; 4] {
        let t = Self::terms(f);
        [-(t[0] * t[1]), t[2] * t[3], -(t[4] * t[5]), t[6] * t[7]]
    }
}

impl Plane for Mw11 {
    const LOW: f64 = 0.0;
    const HIGH: f64 = SQRT_2;

    // the isolated point (1, 1)
    fn extra_rays() -> Vec<f64> {
        vec![1.0]
    }

    fn ray(t: f64) -> f64 {
        Self::HIGH * math::sin(FRAC_PI_2 * t)
    }

    fn point(u: f64, g: f64) -> [f64; 2] {
        [g * u, g * (2.0 - u * u).max(0.0).sqrt()]
    }

    fn feasible(f: [f64; 2]) -> bool {
        Self::values(f).iter().all(|&c| c <= 0.0)
    }

    fn factors(f: [f64; 2]) -> [f64; 8] {
        Self::terms(f)
    }

    cached_front!();
}

// ---- MW12 ----------------------------------------------------------------------------------------

two_objectives!(
    /// MW12 (eq. 26): `f₁ = g₁ x₁`, `f₂ = g₁ (0.85 − 0.8 f₁/g₁ − 0.08 |sin(3.2π f₁/g₁)|)`,
    /// subject to `T₁ T₄ ≤ 0` and `T₂ T₃ ≥ 0`, with
    /// `T₁ = 1 − 0.8f₁ − f₂ + 0.08 sin(2π(f₂ − f₁/1.5))`,
    /// `T₂ = 1 − 0.625f₁ − f₂ + 0.08 sin(2π(f₂ − f₁/1.6))`,
    /// `T₃ = 1.4 − 0.875f₁ − f₂ + 0.08 sin(2π(f₂/1.4 − f₁/1.6))` and
    /// `T₄ = 1.8 − 1.125f₁ − f₂ + 0.08 sin(2π(f₂/1.8 − f₁/1.6))`.
    ///
    /// n variables in [0, 1] (15 by default), the biased distance function g₁. Type IV: the
    /// unconstrained front is infeasible, and the front, derived here, is the boundary `T₁ = 0`
    /// of the feasible band above it, one curve from (0, 1) to (1.3164, 0.0039), the first
    /// feasible point of the direction x₁ = 1. At f₁ = 0, `T₁ = T₂ = 1 − f₂ + 0.08 sin 2πf₂`
    /// vanish together at f₂ = 1, so (0, 1) is feasible in exact arithmetic, the limit of the
    /// curve, though rounding makes it infeasible by about 10⁻¹⁷. Ideal point (0, 0.0039), nadir
    /// point (1.3164, 1).
    Mw12,
    "MW12",
    2,
    1.0
);

impl Mw12 {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        Self::point(x[0], g1(x, 2))
    }

    fn terms(f: [f64; 2]) -> [f64; 4] {
        let (f1, f2) = (f[0], f[1]);
        [
            1.0 - 0.8 * f1 - f2 + 0.08 * math::sin(2.0 * PI * (f2 - f1 / 1.5)),
            1.0 - 0.625 * f1 - f2 + 0.08 * math::sin(2.0 * PI * (f2 - f1 / 1.6)),
            1.4 - 0.875 * f1 - f2 + 0.08 * math::sin(2.0 * PI * (f2 / 1.4 - f1 / 1.6)),
            1.8 - 1.125 * f1 - f2 + 0.08 * math::sin(2.0 * PI * (f2 / 1.8 - f1 / 1.6)),
        ]
    }

    fn values(f: [f64; 2]) -> [f64; 2] {
        let [t1, t2, t3, t4] = Self::terms(f);
        [t1 * t4, -(t2 * t3)]
    }
}

impl Plane for Mw12 {
    const LOW: f64 = 0.0;
    const HIGH: f64 = 1.0;

    // the limit of the front at f₁ = 0
    fn extra_points() -> Vec<[f64; 2]> {
        vec![[0.0, 1.0]]
    }

    fn point(u: f64, g: f64) -> [f64; 2] {
        let s = 0.85 - 0.8 * u - 0.08 * math::sin(3.2 * PI * u).abs();
        [g * u, g * s]
    }

    fn feasible(f: [f64; 2]) -> bool {
        Self::values(f).iter().all(|&c| c <= 0.0)
    }

    fn factors(f: [f64; 2]) -> [f64; 8] {
        padded(Self::terms(f))
    }

    cached_front!();
}

// ---- MW13 ----------------------------------------------------------------------------------------

two_objectives!(
    /// MW13 (eq. 27): `f₁ = g₂ x₁`, `f₂ = g₂ (5 − exp(f₁/g₂) − 0.5 |sin(3π f₁/g₂)|)`, subject to
    /// `T₁ T₄ ≤ 0` and `T₂ T₃ ≥ 0`, with `T₁ = 5 − exp(f₁) − 0.5 sin(3πf₁) − f₂`,
    /// `T₂ = 5 − (1 + f₁ + 0.5f₁²) − 0.5 sin(3πf₁) − f₂`,
    /// `T₃ = 5 − (1 + 0.7f₁) − 0.5 sin(3πf₁) − f₂` and `T₄ = 5 − (1 + 0.4f₁) − 0.5 sin(3πf₁) − f₂`.
    ///
    /// n variables in [0, 1.5] (15 by default), the multimodal distance function g₂. Type III,
    /// disconnected: the constraints use the first terms of the Taylor series of eˣ, and the front,
    /// derived here, has three pieces: the unconstrained front from (0, 4) to (0.1943, 3.3024);
    /// the boundary `T₁ = 0` from (0.6283, 3.3024) to (0.6667, 3.0523) and the unconstrained front
    /// to (0.8910, 2.1345); and the boundary from (1.2043, 2.1345) to (1.5, 0.0183), where the
    /// direction x₁ = 1.5 is feasible at g = 1. Ideal point (0, 0.0183), nadir point (1.5, 4).
    Mw13,
    "MW13",
    2,
    1.5
);

impl Mw13 {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        Self::point(x[0], g2(x, 2))
    }

    fn terms(f: [f64; 2]) -> [f64; 4] {
        let (f1, f2) = (f[0], f[1]);
        let wave = 0.5 * math::sin(3.0 * PI * f1);
        [
            5.0 - math::exp(f1) - wave - f2,
            5.0 - (1.0 + f1 + 0.5 * f1 * f1) - wave - f2,
            5.0 - (1.0 + 0.7 * f1) - wave - f2,
            5.0 - (1.0 + 0.4 * f1) - wave - f2,
        ]
    }

    fn values(f: [f64; 2]) -> [f64; 2] {
        let [t1, t2, t3, t4] = Self::terms(f);
        [t1 * t4, -(t2 * t3)]
    }
}

impl Plane for Mw13 {
    const LOW: f64 = 0.0;
    const HIGH: f64 = 1.5;

    fn point(u: f64, g: f64) -> [f64; 2] {
        let s = 5.0 - math::exp(u) - 0.5 * math::sin(3.0 * PI * u).abs();
        [g * u, g * s]
    }

    fn feasible(f: [f64; 2]) -> bool {
        Self::values(f).iter().all(|&c| c <= 0.0)
    }

    fn factors(f: [f64; 2]) -> [f64; 8] {
        padded(Self::terms(f))
    }

    cached_front!();
}

// ---- MW14 ----------------------------------------------------------------------------------------

/// MW14 (eq. 28), with `M` objectives: `fᵢ = xᵢ` for i from 1 to M − 1 and
/// `f_M = g₃/(M − 1) Σᵢ₌₁^{M−1} (6 − exp(fᵢ) − 1.5 sin(1.1π fᵢ²))`, subject to
/// `1/(M − 1) Σᵢ₌₁^{M−1} (6.1 − αᵢ) − f_M ≥ 0` with `αᵢ = 1 + fᵢ + 0.5 fᵢ² + 1.5 sin(1.1π fᵢ²)`.
///
/// n variables in [0, 1.5] (M + 12 by default: 15 for 3 objectives, the paper's), the linked
/// distance function g₃. Type I: the whole unconstrained front is feasible (at g₃ = 1 the
/// constraint is `1/(M − 1) Σ (eᶠⁱ − 1 − fᵢ − 0.5fᵢ² + 0.1) ≥ 0.1`), and its non-dominated
/// part is the front, disconnected by dominance as DTLZ7's is: with
/// `φ(t) = 6 − eᵗ − 1.5 sin(1.1πt²)`, each fᵢ counts on its own, and is optimal where φ falls
/// below its values at every smaller t, fᵢ in [0, a] or (b, 1.5], with a = 0.7313522974897325 the
/// first local minimum of φ and b = 1.3296339087402259 where φ falls back to φ(a) (computed to 40
/// digits). The front is `f_M = 1/(M − 1) Σ φ(fᵢ)` on these 2^(M−1) patches;
/// [`optimal_front`](MultiProblem::optimal_front) is a product grid over them. Ideal point
/// (0, …, 0, φ(1.5)), with φ(1.5) = 0.0229, and nadir point (1.5, …, 1.5, 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mw14<const M: usize> {
    variables: usize,
}

// ---- the problems with any number of objectives ------------------------------------------------

macro_rules! many_objectives {
    ($name:ident, $label:literal, $upper:expr) => {
        impl<const M: usize> $name<M> {
            /// The number of constraints.
            pub const CONSTRAINTS: usize = 1;

            /// The problem with `variables` variables, at least `M + 1`.
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives, or `M` variables or fewer.
            pub fn new(variables: usize) -> Self {
                Self {
                    variables: checked($label, M, variables),
                }
            }

            /// The number of variables.
            pub fn variables(&self) -> usize {
                self.variables
            }
        }

        impl<const M: usize> Default for $name<M> {
            /// The problem with `M + 12` variables: the paper's `n = M − 1 + 13`, 15 for 3
            /// objectives.
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives.
            fn default() -> Self {
                Self::new(M + 12)
            }
        }

        impl<const M: usize> MultiFitnessFunction<Reals, M> for $name<M> {
            type Output = ([f64; M], f64);

            /// The objective values of `x` and its constraint violation, 0 when it's feasible.
            ///
            /// # Panics
            ///
            /// If `x` has fewer than `M` genes.
            fn evaluate(&self, x: &Reals) -> ([f64; M], f64) {
                let f = self.objectives(x);
                (f, violation(&[Self::value(&f)]))
            }
        }

        impl<const M: usize> MultiProblem<M> for $name<M> {
            type Representation = Real;

            fn name(&self) -> &'static str {
                $label
            }

            fn representation(&self) -> Real {
                Real::uniform(self.variables, 0.0..=$upper).expect("valid bounds")
            }

            fn reference(&self) -> &'static str {
                REFERENCE
            }

            fn reference_url(&self) -> Option<&'static str> {
                Some(REFERENCE_URL)
            }

            fn constraint_count(&self) -> usize {
                Self::CONSTRAINTS
            }

            fn constraints(&self, genome: &Reals) -> Constraints {
                Constraints::new(vec![Self::value(&self.objectives(genome))], Vec::new())
            }

            fn optimal_front(&self, points: usize) -> Option<Vec<[f64; M]>> {
                Some(Self::front(points))
            }

            fn ideal_point(&self) -> Option<[f64; M]> {
                Some(Self::ideal())
            }

            fn nadir_point(&self) -> Option<[f64; M]> {
                Some(Self::nadir())
            }
        }
    };
}

many_objectives!(Mw4, "MW4", 1.0);
many_objectives!(Mw8, "MW8", 1.0);
many_objectives!(Mw14, "MW14", 1.5);

impl<const M: usize> Mw4<M> {
    fn objectives(&self, x: &Reals) -> [f64; M] {
        let g = g1(x, M);
        std::array::from_fn(|m| {
            let mut f = g;
            for xi in &x[..M - 1 - m] {
                f *= 1.0 - xi;
            }
            if m > 0 {
                f *= x[M - 1 - m];
            }
            f
        })
    }

    fn value(f: &[f64; M]) -> f64 {
        let l = f[M - 1] - f[..M - 1].iter().sum::<f64>();
        let bound = 1.0 + 0.4 * math::powi(math::sin(2.5 * PI * l), 8);
        -(bound - f.iter().sum::<f64>())
    }

    fn front(points: usize) -> Vec<[f64; M]> {
        simplex_points::<M>(points)
    }

    fn ideal() -> [f64; M] {
        [0.0; M]
    }

    fn nadir() -> [f64; M] {
        [1.0; M]
    }
}

// MW8's four bands of l = arcsin(f_M) on the unit sphere, where sin(6l)² <= 1/2
const MW8_BANDS: [(f64, f64); 4] = [
    (0.0, PI / 24.0),
    (PI / 8.0, 5.0 * PI / 24.0),
    (7.0 * PI / 24.0, 3.0 * PI / 8.0),
    (11.0 * PI / 24.0, FRAC_PI_2),
];

impl<const M: usize> Mw8<M> {
    fn objectives(&self, x: &Reals) -> [f64; M] {
        let g = g2(x, M);
        std::array::from_fn(|m| {
            let mut f = g;
            for &xi in &x[..M - 1 - m] {
                f *= math::cos(0.5 * PI * xi);
            }
            if m > 0 {
                f *= math::sin(0.5 * PI * x[M - 1 - m]);
            }
            f
        })
    }

    fn value(f: &[f64; M]) -> f64 {
        let squares: f64 = f.iter().map(|v| v * v).sum();
        let l = math::asin((f[M - 1] / squares.sqrt()).clamp(-1.0, 1.0));
        let s = math::sin(6.0 * l);
        let radius = 1.25 - 0.5 * s * s;
        -(radius * radius - squares)
    }

    // whether a point of the unit sphere is in one of the bands: sin(6l)² <= 1/2
    fn in_bands(point: &[f64; M]) -> bool {
        let s = math::sin(6.0 * math::asin(point[M - 1].clamp(-1.0, 1.0)));
        s * s <= 0.5 + 1e-12
    }

    fn front(points: usize) -> Vec<[f64; M]> {
        if points == 0 {
            return Vec::new();
        }
        if M == 2 {
            let arcs: Vec<_> = MW8_BANDS
                .iter()
                .map(|&(low, high)| {
                    move |t: f64| {
                        let l = low + (high - low) * t;
                        [math::cos(l), math::sin(l)]
                    }
                })
                .collect();
            let pieces: Vec<Piece<'_>> = arcs
                .iter()
                .map(|curve| Piece {
                    curve,
                    with_end: true,
                })
                .collect();
            return pieces_front(&pieces, points)
                .into_iter()
                .map(|[a, b]| std::array::from_fn(|j| if j == 0 { a } else { b }))
                .collect();
        }
        let mut divisions = divisions_for::<M>(points);
        loop {
            let front: Vec<[f64; M]> = das_dennis::<M>(divisions)
                .into_iter()
                .map(|p| {
                    let norm = p.iter().map(|v| v * v).sum::<f64>().sqrt();
                    p.map(|v| v / norm)
                })
                .filter(Self::in_bands)
                .collect();
            if front.len() >= points {
                return front;
            }
            divisions += 1;
        }
    }

    fn ideal() -> [f64; M] {
        [0.0; M]
    }

    fn nadir() -> [f64; M] {
        [1.0; M]
    }
}

// MW14's ranges of each fᵢ on the front, [0, A] and (B, 1.5]: A is the first local minimum of
// φ(t) = 6 − eᵗ − 1.5 sin(1.1πt²) and B > A where φ(B) = φ(A), computed to 40 digits and rounded
pub(super) const MW14_A: f64 = 0.731_352_297_489_732_5;
pub(super) const MW14_B: f64 = 1.329_633_908_740_225_9;
const MW14_C: f64 = 1.5;

// MW14's shape: 6 − eᵗ − 1.5 sin(1.1πt²)
fn mw14_phi(t: f64) -> f64 {
    6.0 - math::exp(t) - 1.5 * math::sin(1.1 * PI * t * t)
}

// `count` values of fᵢ on MW14's front, ascending, shared between its two ranges in proportion to
// their widths: both ends of [0, A], and (B, 1.5] without B, which A dominates
fn mw14_values(count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![0.0; count];
    }
    let share = count as f64 * MW14_A / (MW14_A + MW14_C - MW14_B);
    let first = (share.round() as usize).clamp(1, count - 1);
    let second = count - first;
    let low = (0..first).map(|i| MW14_A * evenly(i, first));
    let high =
        (0..second).map(|j| MW14_C - (MW14_C - MW14_B) * (second - 1 - j) as f64 / second as f64);
    low.chain(high).collect()
}

impl<const M: usize> Mw14<M> {
    fn objectives(&self, x: &Reals) -> [f64; M] {
        let g = g3(x, M);
        let mut f = [0.0; M];
        f[..M - 1].copy_from_slice(&x[..M - 1]);
        f[M - 1] = g * Self::mean_phi(&f[..M - 1]);
        f
    }

    fn mean_phi(f: &[f64]) -> f64 {
        f.iter().map(|&t| mw14_phi(t)).sum::<f64>() / (M - 1) as f64
    }

    fn value(f: &[f64; M]) -> f64 {
        let alpha = |t: f64| 1.0 + t + 0.5 * t * t + 1.5 * math::sin(1.1 * PI * t * t);
        let bound = f[..M - 1].iter().map(|&t| 6.1 - alpha(t)).sum::<f64>() / (M - 1) as f64;
        -(bound - f[M - 1])
    }

    // a product grid over the ranges, with the fewest values per objective that give at least
    // `points` points: exactly `points` for 2 objectives
    fn front(points: usize) -> Vec<[f64; M]> {
        if points == 0 {
            return Vec::new();
        }
        let dimensions = (M - 1) as u32;
        let mut count = 1usize;
        while count.saturating_pow(dimensions) < points {
            count += 1;
        }
        let values = mw14_values(count);
        (0..count.pow(dimensions))
            .map(|index| {
                let mut point = [0.0; M];
                let mut rest = index;
                for j in (0..M - 1).rev() {
                    point[j] = values[rest % count];
                    rest /= count;
                }
                point[M - 1] = Self::mean_phi(&point[..M - 1]);
                point
            })
            .collect()
    }

    fn ideal() -> [f64; M] {
        let mut point = [0.0; M];
        point[M - 1] = mw14_phi(MW14_C);
        point
    }

    fn nadir() -> [f64; M] {
        let mut point = [MW14_C; M];
        point[M - 1] = mw14_phi(0.0);
        point
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective::Minimize;
    use crate::StreamRng;
    use crate::multi::{Scores, non_dominated_sort};
    use rand::RngExt;

    fn at(values: &[f64]) -> Reals {
        Reals::from(values.to_vec())
    }

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= tolerance * e.abs().max(1.0),
                "{actual:?} is not {expected:?}"
            );
        }
    }

    // a genome of n variables with the given position variables and the distance variables where
    // the distance function is 1 (the paper's optimal values): `which` is 1, 2 or 3
    fn optimal(position: &[f64], n: usize, which: usize) -> Vec<f64> {
        let m = position.len() + 1;
        let mut x = position.to_vec();
        for j in m - 1..n {
            let value = match which {
                1 => math::powf(0.5 + j as f64 / (2.0 * n as f64), 1.0 / (n - m) as f64),
                2 => j as f64 / n as f64,
                _ => 1.0 - (x[j - 1] - 0.5) * (x[j - 1] - 0.5),
            };
            x.push(value);
        }
        x
    }

    // the distance functions of eqs. 12-14: 1 at the paper's optimal values, and values that
    // follow by hand elsewhere
    #[test]
    fn distance_functions_match_the_paper() {
        for which in 1..=3 {
            for m in [2, 3] {
                let x = optimal(&vec![0.3; m - 1], 15, which);
                let g = [g1, g2, g3][which - 1](&x, m);
                assert!((g - 1.0).abs() < 1e-14, "g{which} with m = {m}: {g}");
            }
        }
        // g₁ with n = 3, m = 2 and x₂ = x₃ = 1: zᵢ = 1, so the terms are 1 − exp(−10 dᵢ²) with
        // d₂ = 1 − 0.5 − 1/6 = 1/3 and d₃ = 1 − 0.5 − 2/6 = 1/6
        let expected = 3.0 - math::exp(-10.0 / 9.0) - math::exp(-10.0 / 36.0);
        assert!((g1(&[0.4, 1.0, 1.0], 2) - expected).abs() < 1e-15);
        // g₂ with n = 2, m = 2 and x₂ = 1/2: z = 0, the term 1.5 + 0 − 1.5 = 0
        assert_eq!(g2(&[0.9, 0.5], 2), 1.0);
        // g₃ with every variable at 0.5: each term 2 (0.5 + 0 − 1)² = 0.5, 14 of them for n = 15
        assert_eq!(g3(&[0.5; 15], 2), 8.0);
    }

    // the objectives and constraints of eqs. 15-28 at optimal solutions (g = 1) whose values
    // follow by hand; the constraint values are the paper's, as g <= 0
    #[test]
    fn values_match_the_paper() {
        let n = 15;
        // MW1 at x₁ = 0: f = (0, 1), l = √2, and the constraint 0.5 sin(2π√2)⁸ >= 0 holds
        let (f, violation) = Mw1::default().evaluate(&at(&optimal(&[0.0], n, 1)));
        assert_close(&f, &[0.0, 1.0], 1e-14);
        assert_eq!(violation, 0.0);
        // MW2 at x₁ = 0.3: f = (0.3, 0.7), on the line f₁ + f₂ = 1
        let (f, violation) = Mw2::default().evaluate(&at(&optimal(&[0.3], n, 2)));
        assert_close(&f, &[0.3, 0.7], 1e-14);
        assert_eq!(violation, 0.0);
        // MW3 at x₁ = 0.5: f = (0.5, 0.5), l = 0: 1.05 − 1 >= 0 and 0.85 − 1 <= 0
        let x = at(&optimal(&[0.5], n, 3));
        assert_close(&Mw3::default().evaluate(&x).0, &[0.5, 0.5], 1e-14);
        assert_close(
            Mw3::default().constraints(&x).inequalities(),
            &[-0.05, -0.15],
            1e-14,
        );
        // MW5 at x₁ = 1: f = (1, 0), l₁ = l₂ = 0: 1.7² − 1 >= 0, and 1 − 1 <= 0 twice
        let x = at(&optimal(&[1.0], n, 1));
        assert_close(&Mw5::default().evaluate(&x).0, &[1.0, 0.0], 1e-14);
        assert_close(
            Mw5::default().constraints(&x).inequalities(),
            &[-1.89, 0.0, 0.0],
            1e-14,
        );
        // MW6 at x₁ = 1.1: f = (1.1, 0), l = cos(0)¹⁰ = 1: 1 − (1.1/1.15)² >= 0
        let x = at(&optimal(&[1.1], n, 2));
        assert_close(&Mw6::default().evaluate(&x).0, &[1.1, 0.0], 1e-14);
        let expected = -(1.0 - (1.1f64 / 1.15).powi(2));
        assert_close(
            Mw6::default().constraints(&x).inequalities(),
            &[expected],
            1e-14,
        );
        // MW7 at x₁ = 0: f = (0, 1), l = π/2, sin 2π = 0: the second constraint 1.15² − 1 <= 0
        // fails by 0.3225
        let (f, violation) = Mw7::default().evaluate(&at(&optimal(&[0.0], n, 3)));
        assert_close(&f, &[0.0, 1.0], 1e-14);
        assert!((violation - 0.3225).abs() < 1e-14);
        // MW9 at x₁ = 0: f = (0, 1), T₁ = 0, T₂ = 1.35² − 0.35² − 1 = 0.7, T₃ = 0.3: min(0, 0.21)
        let x = at(&optimal(&[0.0], n, 1));
        assert_close(&Mw9::default().evaluate(&x).0, &[0.0, 1.0], 1e-14);
        assert_eq!(Mw9::default().constraints(&x).inequalities(), [0.0]);
        // MW10 at x₁ = 1: f = (1, 0): (2 − 4)(2 − 8) = 12 >= 0, (2 − 2)(2 − 16) = 0 <= 0 and
        // (1 − 1)(1.2 − 1.2) = 0 <= 0
        let x = at(&optimal(&[1.0], n, 2));
        assert_close(&Mw10::default().evaluate(&x).0, &[1.0, 0.0], 1e-14);
        assert_close(
            Mw10::default().constraints(&x).inequalities(),
            &[-12.0, 0.0, 0.0],
            1e-14,
        );
        // MW11 at x₁ = 1: f = (1, 1), the isolated optimal point: (3 − 1 − 1)(3 − 2 − 1) = 0,
        // (1.375)(−5) = −6.875, (0.44)(1.125 − 0.125 − 1) = 0 and (0.84)(−0.44) = −0.3696
        let x = at(&optimal(&[1.0], n, 3));
        assert_eq!(Mw11::default().evaluate(&x), ([1.0, 1.0], 0.0));
        assert_close(
            Mw11::default().constraints(&x).inequalities(),
            &[0.0, -6.875, 0.0, -0.3696],
            1e-14,
        );
        // MW12 at x₁ = 1: f₂ = 0.85 − 0.8 − 0.08 |sin 3.2π| = 0.05 − 0.08 sin 0.2π
        let x = at(&optimal(&[1.0], n, 1));
        let expected = 0.05 - 0.08 * math::sin(0.2 * PI);
        assert_close(&Mw12::default().evaluate(&x).0, &[1.0, expected], 1e-14);
        // MW13 at x₁ = 0: f = (0, 4), T₁ = T₂ = T₃ = T₄ = 0
        let x = at(&optimal(&[0.0], n, 2));
        assert_eq!(Mw13::default().evaluate(&x), ([0.0, 4.0], 0.0));
        // MW4 (M = 3) at x₁ = x₂ = 0.5: f = (0.25, 0.25, 0.5), l = 0, the constraint 1 − 1 >= 0
        let (f, violation) = Mw4::<3>::default().evaluate(&at(&optimal(&[0.5, 0.5], n, 1)));
        assert_close(&f, &[0.25, 0.25, 0.5], 1e-14);
        assert_eq!(violation, 0.0);
        // MW8 (M = 3): at x₁ = 1/3, l = π/6 and sin 6l = 0, the radius 1.25 holds the sphere; at
        // x₁ = 1/6, l = π/12 and sin 6l = 1, the radius 0.75 doesn't: 1 − 0.75² = 0.4375
        let x = at(&optimal(&[1.0 / 3.0, 0.0], n, 2));
        let (f, violation) = Mw8::<3>::default().evaluate(&x);
        assert_close(&f, &[math::cos(PI / 6.0), 0.0, 0.5], 1e-14);
        assert_eq!(violation, 0.0);
        let x = at(&optimal(&[1.0 / 6.0, 0.0], n, 2));
        let (_, violation) = Mw8::<3>::default().evaluate(&x);
        assert!((violation - 0.4375).abs() < 1e-14);
        // MW14 (M = 3) at x₁ = x₂ = 0: f₃ = φ(0) = 6 − 1 − 0 = 5, and the constraint
        // (6.1 − 1) − 5 = 0.1 >= 0
        let x = at(&optimal(&[0.0, 0.0], n, 3));
        assert_close(
            &Mw14::<3>::default().evaluate(&x).0,
            &[0.0, 0.0, 5.0],
            1e-14,
        );
        assert_close(
            Mw14::<3>::default().constraints(&x).inequalities(),
            &[-0.1],
            1e-14,
        );
    }

    // a two-objective front: feasible, non-dominated, and not dominated (by more than 1e-6, the
    // sampling of the rays at a flat end of a piece) by any feasible solution
    // near the Pareto set of the unconstrained problem (x₁ anywhere, the distance variables near
    // their optimal values); `which` is the problem's distance function
    fn check_plane<P>(problem: P, variables: usize, which: usize)
    where
        P: Plane
            + MultiProblem<2, Representation = Real>
            + MultiFitnessFunction<Reals, 2, Output = ([f64; 2], f64)>,
    {
        let name = problem.name();
        let front = P::front();
        assert!(front.len() > 200, "{name}: {}", front.len());
        let extra = P::extra_points();
        for point in front.iter().filter(|point| !extra.contains(point)) {
            assert!(P::feasible(*point), "{name}: {point:?}");
        }
        let spread = problem.optimal_front(300).expect("known");
        assert_eq!(spread.len(), 300);
        let scores: Vec<Scores<2>> = spread.iter().map(|p| Scores::new(*p)).collect();
        assert_eq!(
            non_dominated_sort(&scores, &[Minimize; 2]).len(),
            1,
            "{name}"
        );
        let bounds = problem.representation();
        let (low, high) = (*bounds.bounds()[0].start(), *bounds.bounds()[0].end());
        let mut rng = StreamRng::seed_from_u64(7);
        let reference = problem.optimal_front(2_000).expect("known");
        let mut found = 0;
        for trial in 0..40_000 {
            let x1 = low + (high - low) * rng.random::<f64>();
            let mut x = optimal(&[x1], variables, which);
            let spread = [0.0, 1e-3, 1e-2, 0.1, 0.3, 1.0][trial % 6];
            for xi in x.iter_mut().skip(1) {
                *xi = (*xi + spread * (rng.random::<f64>() - 0.5)).clamp(low, high);
            }
            let (f, violation) = problem.evaluate(&at(&x));
            if violation > 0.0 {
                continue;
            }
            found += 1;
            for point in &reference {
                let dominated = f[0] < point[0] - 1e-6 && f[1] < point[1] - 1e-6;
                assert!(!dominated, "{name}: {f:?} dominates {point:?}");
            }
        }
        assert!(found >= 50, "{name}: {found} feasible solutions");
    }

    #[test]
    fn two_objective_fronts_are_not_dominated_by_feasible_solutions() {
        check_plane(Mw1::default(), 15, 1);
        check_plane(Mw2::default(), 15, 2);
        check_plane(Mw3::default(), 15, 3);
        check_plane(Mw5::default(), 15, 1);
        check_plane(Mw6::default(), 15, 2);
        check_plane(Mw7::default(), 15, 3);
        check_plane(Mw9::default(), 15, 1);
        check_plane(Mw10::default(), 15, 2);
        check_plane(Mw11::default(), 15, 3);
        check_plane(Mw12::default(), 15, 1);
        check_plane(Mw13::default(), 15, 2);
    }

    // the fronts of the docs: the curves they're made of, their pieces and their extremes
    #[test]
    fn two_objective_fronts_are_the_derived_curves() {
        // MW1: on the line f₂ = 1 − 0.85 f₁ (type II); MW2 on f₂ = 1 − f₁ (type I)
        let on_line = |p: &[f64; 2]| (p[1] - (1.0 - 0.85 * p[0])).abs() < 1e-12;
        assert!(Mw1::front().iter().all(on_line));
        assert!(
            Mw2::front()
                .iter()
                .all(|p| (p[0] + p[1] - 1.0).abs() < 1e-12)
        );
        assert_eq!(Mw2::front().len(), RAYS);
        // MW6: on the circle of radius 1.1 (type II)
        let on_circle = |p: &[f64; 2]| (p[0] * p[0] + p[1] * p[1] - 1.21).abs() < 1e-12;
        assert!(Mw6::front().iter().all(on_circle));
        // MW9: on 1 − 0.64 f₁² and then 1.15² − (f₁ + 0.15)², which cross at (√0.522 − 0.3)/0.72
        let cross = (0.522f64.sqrt() - 0.3) / 0.72;
        assert!((cross - 0.586_799_5).abs() < 1e-7);
        for p in Mw9::front() {
            let curve = if p[0] <= cross {
                1.0 - 0.64 * p[0] * p[0]
            } else {
                1.15 * 1.15 - (p[0] + 0.15) * (p[0] + 0.15)
            };
            assert!((p[1] - curve).abs() < 1e-9, "{p:?}");
        }
        // MW5: the sixteen points of the circle where sin 6l₂³ = 0, and the curves near the axes
        // up to l₁ = 1/72, where f₁ ≈ cos l₁ (1 + 24 l₁³) stops falling; the rest on the circle
        let front = Mw5::front();
        let mut touching: Vec<[f64; 2]> = Mw5::touching()
            .into_iter()
            .map(|u| [u, (1.0 - u * u).sqrt()])
            .collect();
        touching.extend([[0.0, 1.0], [1.0, 0.0]]);
        assert_eq!(touching.len(), 16);
        for point in &touching {
            let nearest = front
                .iter()
                .map(|p| (p[0] - point[0]).hypot(p[1] - point[1]))
                .fold(f64::INFINITY, f64::min);
            assert!(nearest < 1e-12, "{point:?}: {nearest}");
        }
        let widest = front
            .iter()
            .filter(|p| p[1] < 0.1)
            .map(|p| p[1])
            .fold(0.0, f64::max);
        assert!((widest - 1.0 / 72.0).abs() < 1e-4, "{widest}");
        for p in front.iter().filter(|p| p[0] > 0.1 && p[1] > 0.1) {
            assert!((p[0] * p[0] + p[1] * p[1] - 1.0).abs() < 1e-12, "{p:?}");
        }
        // MW11: the isolated point (1, 1), and nothing else between f₁ = 0.88 and 1.46
        assert!(Mw11::front().contains(&[1.0, 1.0]));
        let middle = Mw11::front().iter().filter(|p| p[0] > 0.88 && p[0] < 1.46);
        assert_eq!(middle.count(), 1);
        // MW12: from its limit (0, 1) along T₁ = 0
        assert_eq!(Mw12::front()[0], [0.0, 1.0]);
        for p in &Mw12::front()[1..] {
            assert!(Mw12::terms(*p)[0].abs() < 1e-9, "{p:?}");
        }
        // the ideal and nadir points of the docs
        let corners = [
            (
                Mw1::default().ideal_point(),
                [0.0, 0.15],
                Mw1::default().nadir_point(),
                [1.0, 1.0],
            ),
            (
                Mw6::default().ideal_point(),
                [0.0163, 0.0],
                Mw6::default().nadir_point(),
                [1.1, 1.0999],
            ),
            (
                Mw7::default().ideal_point(),
                [0.0, 0.0],
                Mw7::default().nadir_point(),
                [1.15, 1.15],
            ),
            (
                Mw10::default().ideal_point(),
                [0.2325, 0.0],
                Mw10::default().nadir_point(),
                [1.0, 1.1350],
            ),
            (
                Mw11::default().ideal_point(),
                [0.3707, 0.3311],
                Mw11::default().nadir_point(),
                [2.0662, 2.0383],
            ),
            (
                Mw12::default().ideal_point(),
                [0.0, 0.0039],
                Mw12::default().nadir_point(),
                [1.3164, 1.0],
            ),
            (
                Mw13::default().ideal_point(),
                [0.0, 0.0183],
                Mw13::default().nadir_point(),
                [1.5, 4.0],
            ),
        ];
        for (ideal, expected_ideal, nadir, expected_nadir) in corners {
            assert_close(&ideal.expect("known"), &expected_ideal, 1e-4);
            assert_close(&nadir.expect("known"), &expected_nadir, 1e-4);
        }
    }

    #[test]
    fn fronts_with_any_number_of_objectives() {
        // MW4: the simplex
        let front = Mw4::<3>::default().optimal_front(91).expect("known");
        assert_eq!(front.len(), 91);
        assert!(
            front
                .iter()
                .all(|p| (p.iter().sum::<f64>() - 1.0).abs() < 1e-12)
        );
        // MW8: the unit sphere, in the four bands of arcsin f_M
        let front = Mw8::<3>::default().optimal_front(100).expect("known");
        assert!(front.len() >= 100);
        for p in &front {
            assert!((p.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
            let l = math::asin(p[2]);
            let inside = MW8_BANDS
                .iter()
                .any(|&(a, b)| l >= a - 1e-9 && l <= b + 1e-9);
            assert!(inside, "{p:?}");
        }
        let front = Mw8::<2>::default().optimal_front(40).expect("known");
        assert_eq!(front.len(), 40);
        assert!(front.contains(&[1.0, 0.0]));
        // at the bands' edges, sin(6l)² = 1/2: the radius is 1.25 − 0.25 = 1
        for (a, b) in MW8_BANDS {
            for l in [a, b] {
                let s = math::sin(6.0 * l);
                assert!(s * s <= 0.5 + 1e-12);
            }
        }
        // MW14: f_M the mean of φ over f₁ … f_{M−1}, each in [0, A] or (B, 1.5]
        let front = Mw14::<3>::default().optimal_front(50).expect("known");
        assert_eq!(front.len(), 64);
        let inside = |t: f64| (0.0..=MW14_A).contains(&t) || (t > MW14_B && t <= 1.5);
        for p in &front {
            assert!(inside(p[0]) && inside(p[1]), "{p:?}");
            assert!((p[2] - 0.5 * (mw14_phi(p[0]) + mw14_phi(p[1]))).abs() < 1e-12);
        }
        assert_eq!(
            Mw14::<2>::default().optimal_front(9).expect("known").len(),
            9
        );
        let ideal = Mw14::<3>::default().ideal_point().expect("known");
        assert!((ideal[2] - 0.022_934_929_062_243_2).abs() < 1e-14);
        assert_eq!(
            Mw14::<4>::default().nadir_point(),
            Some([1.5, 1.5, 1.5, 5.0])
        );
    }

    // MW14's ranges: A is a local minimum of φ, φ(B) = φ(A), and the values where φ falls below
    // every smaller value, on a fine grid, are [0, A] and (B, 1.5]
    #[test]
    fn mw14_ranges_are_where_phi_falls_to_a_record() {
        let slope = |t: f64| -math::exp(t) - 3.3 * PI * t * math::cos(1.1 * PI * t * t);
        assert!(slope(MW14_A).abs() < 1e-13);
        assert!((mw14_phi(MW14_B) - mw14_phi(MW14_A)).abs() < 1e-14);
        let steps = 1_500_000;
        let mut best = f64::INFINITY;
        for i in 0..=steps {
            let t = 1.5 * i as f64 / steps as f64;
            let record = mw14_phi(t) < best;
            best = best.min(mw14_phi(t));
            let inside = t <= MW14_A || t > MW14_B;
            let near = [MW14_A, MW14_B].iter().any(|edge| (t - edge).abs() < 3e-6);
            assert!(record == inside || near, "{t}");
        }
    }

    #[test]
    fn sizes_bounds_and_constraint_counts() {
        assert_eq!(Mw1::default().variables(), 15);
        assert_eq!(Mw4::<3>::default().variables(), 15);
        assert_eq!(Mw8::<5>::default().variables(), 17);
        assert_eq!(*Mw6::default().representation().bounds()[3].end(), 1.1);
        assert_eq!(*Mw11::default().representation().bounds()[0].end(), SQRT_2);
        assert_eq!(*Mw13::default().representation().bounds()[0].end(), 1.5);
        assert_eq!(
            *Mw14::<3>::default().representation().bounds()[0].end(),
            1.5
        );
        let counts = [
            Mw1::CONSTRAINTS,
            Mw2::CONSTRAINTS,
            Mw3::CONSTRAINTS,
            Mw4::<3>::CONSTRAINTS,
            Mw5::CONSTRAINTS,
            Mw6::CONSTRAINTS,
            Mw7::CONSTRAINTS,
            Mw8::<3>::CONSTRAINTS,
            Mw9::CONSTRAINTS,
            Mw10::CONSTRAINTS,
            Mw11::CONSTRAINTS,
            Mw12::CONSTRAINTS,
            Mw13::CONSTRAINTS,
            Mw14::<3>::CONSTRAINTS,
        ];
        // table II's NoC column
        assert_eq!(counts, [1, 1, 2, 1, 3, 1, 2, 1, 1, 3, 4, 2, 2, 1]);
        // f₂ stays finite at MW11's upper bound, whose square is just above 2
        let x = at(&[SQRT_2; 15]);
        assert!(Mw11::default().evaluate(&x).0.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "more variables than objectives")]
    fn the_distance_needs_a_variable() {
        let _ = Mw4::<3>::new(3);
    }
}
