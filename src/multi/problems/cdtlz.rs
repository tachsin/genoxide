//! C1-DTLZ1, C1-DTLZ3, C2-DTLZ2, convex C2-DTLZ2, C3-DTLZ1 and C3-DTLZ4: Jain and Deb's
//! constrained DTLZ problems, any number of objectives, each DTLZ problem with one or M
//! constraints.
//!
//! Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using
//! reference-point based nondominated sorting approach, part II: handling constraints and
//! extending to an adaptive approach. *IEEE Transactions on Evolutionary Computation* 18(4):
//! 602-622, section V, eqs. 4-8, checked in the accepted manuscript as the journal published it
//! online ("accepted for publication in a future issue of this journal, but has not been fully
//! edited"), hosted by the second author as KanGAL report 2012010; the final text wasn't
//! compared. The convex DTLZ2 that convex C2-DTLZ2 constrains is part I's (Deb and Jain 2014,
//! *IEEE Transactions on Evolutionary Computation* 18(4): 577-601, section VII-C, checked in its
//! accepted manuscript too): DTLZ2's `fᵢ` raised to the power 4 for i < M, and `f_M` squared.
//!
//! Type 1 (C1) keeps DTLZ's front and puts an infeasible barrier before it; type 2 (C2) makes
//! parts of the front infeasible; type 3 (C3) makes the whole of it infeasible, and the front is
//! on the constraints' boundaries. The paper gives the radius of C1-DTLZ3 and of convex
//! C2-DTLZ2 for 3, 5, 8, 10 and 15 objectives only: for another number, `with_radius` sets it.

use super::ctp::spread;
use super::{
    Dtlz1, Dtlz2, Dtlz3, Dtlz4, DynMultiProblem, MultiProblem, boxed, das_dennis, divisions_for,
    evenly,
};
use crate::constraint::at_most;
use crate::genome::{Real, Reals};
use crate::multi::MultiFitnessFunction;
use crate::problems::Constraints;

const REFERENCE: &str = "Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization \
                         algorithm using reference-point based nondominated sorting approach, \
                         part II: handling constraints and extending to an adaptive approach. \
                         IEEE Transactions on Evolutionary Computation 18(4): 602-622.";
const REFERENCE_URL: &str = "https://doi.org/10.1109/TEVC.2013.2281534";

// the samples of a two-objective front that is cut into pieces, and the gap between pieces
const SAMPLES: usize = 100_000;
const GAP: f64 = 0.01;

// the total violation of constraints g(x) <= 0
fn violation(values: &[f64]) -> f64 {
    values.iter().map(|&g| at_most(g, 0.0)).sum()
}

// the smallest set of Das and Dennis's points, mapped by `point` and kept where `keep` holds,
// with at least `points` points; for 2 objectives, exactly `points`, from `SAMPLES` points spread
// evenly on the simplex
fn filtered_front<const M: usize>(
    points: usize,
    point: impl Fn([f64; M]) -> [f64; M],
    keep: impl Fn(&[f64; M]) -> bool,
) -> Vec<[f64; M]> {
    if points == 0 {
        return Vec::new();
    }
    if M == 2 {
        let dense: Vec<[f64; 2]> = (0..SAMPLES)
            .map(|i| {
                let t = evenly(i, SAMPLES);
                let mut w = [0.0; M];
                w[0] = t;
                w[1] = 1.0 - t;
                point(w)
            })
            .filter(|f| keep(f))
            .map(|f| [f[0], f[1]])
            .collect();
        let mut dense = dense;
        dense.sort_by(|a, b| a[0].total_cmp(&b[0]));
        return spread(&dense, points, GAP)
            .into_iter()
            .map(|p| std::array::from_fn(|j| p[j]))
            .collect();
    }
    let mut divisions = divisions_for::<M>(points);
    loop {
        let front: Vec<[f64; M]> = das_dennis::<M>(divisions)
            .into_iter()
            .map(&point)
            .filter(|f| keep(f))
            .collect();
        if front.len() >= points {
            return front;
        }
        divisions += 1;
    }
}

// ---- type 1 --------------------------------------------------------------------------------------

/// C1-DTLZ1: [`Dtlz1`] subject to `1 − f_M/0.6 − Σᵢ₌₁^{M−1} fᵢ/0.5 ≥ 0` (the paper's eq. 4).
///
/// With `M` objectives and `n` variables in [0, 1], `M + 4` by default (k = 5, as the paper
/// uses). Only a thin wedge of the objective space next to DTLZ1's front is feasible, and the
/// front is DTLZ1's, the objectives summing to 1/2: on it the constraint is `f_M/3 ≥ 0`, all
/// feasible. DTLZ1's 11ᵏ − 1 local fronts lie in the infeasible space, so an algorithm must find
/// the feasible wedge while it converges.
///
/// [`constraints`](MultiProblem::constraints) gives `f_M/0.6 + Σᵢ₌₁^{M−1} fᵢ/0.5 − 1`.
///
/// Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using
/// reference-point based nondominated sorting approach, part II. *IEEE Transactions on
/// Evolutionary Computation* 18(4): 602-622, section V-B, eq. 4.
///
/// What was checked where, for all the constrained DTLZ problems: the definitions (section V,
/// eqs. 4-8, and the radii and sizes around them) in the paper's accepted manuscript, as the
/// journal first published it online, hosted by the second author as KanGAL report 2012010; the
/// final text wasn't compared. The convex DTLZ2 of convex C2-DTLZ2 is part I's (Deb, K. and
/// Jain, H. (2014), *IEEE Transactions on Evolutionary Computation* 18(4): 577-601, section
/// VII-C, checked in its accepted manuscript too). The paper tabulates no fronts: those here
/// are derived from the definitions, as each problem's docs say.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct C1Dtlz1<const M: usize> {
    dtlz: Dtlz1<M>,
}

impl<const M: usize> C1Dtlz1<M> {
    /// The number of constraints.
    pub const CONSTRAINTS: usize = 1;

    /// The problem with `variables` variables, at least `M`.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives, or fewer variables than objectives.
    pub fn new(variables: usize) -> Self {
        Self {
            dtlz: Dtlz1::new(variables),
        }
    }

    /// The number of variables.
    pub fn variables(&self) -> usize {
        self.dtlz.variables()
    }

    fn values(f: &[f64; M]) -> [f64; 1] {
        [f[M - 1] / 0.6 + f[..M - 1].iter().map(|fi| fi / 0.5).sum::<f64>() - 1.0]
    }
}

/// C1-DTLZ3: [`Dtlz3`] subject to `(Σ fᵢ² − 16) (Σ fᵢ² − r²) ≥ 0` (the paper's eq. 5), with
/// r = 9 for 3 objectives, 12.5 for 5 and 8, and 15 for 10 and 15.
///
/// With `M` objectives and `n` variables in [0, 1], `M + 9` by default (k = 10, as the paper
/// uses). The objectives lie on a sphere of radius `1 + g`, and the constraint makes the shell
/// between the radii 4 and r infeasible: a population must cross it, past DTLZ3's local fronts
/// inside it, to reach DTLZ3's front, the unit sphere, which is all feasible.
///
/// [`constraints`](MultiProblem::constraints) gives `−(Σ fᵢ² − 16) (Σ fᵢ² − r²)`.
///
/// Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using
/// reference-point based nondominated sorting approach, part II. *IEEE Transactions on
/// Evolutionary Computation* 18(4): 602-622, section V-B, eq. 5 (see
/// [`C1Dtlz1`]'s notes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C1Dtlz3<const M: usize> {
    dtlz: Dtlz3<M>,
    radius: f64,
}

impl<const M: usize> C1Dtlz3<M> {
    /// The number of constraints.
    pub const CONSTRAINTS: usize = 1;

    /// The radius r of the paper for `M` objectives: 9 for 3, 12.5 for 5 and 8, 15 for 10 and 15,
    /// and `None` for others.
    pub fn paper_radius() -> Option<f64> {
        match M {
            3 => Some(9.0),
            5 | 8 => Some(12.5),
            10 | 15 => Some(15.0),
            _ => None,
        }
    }

    /// The problem with `variables` variables, at least `M`, and the paper's radius.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives, fewer variables than objectives, or a number of objectives
    /// the paper gives no radius for (see [`paper_radius`](Self::paper_radius)).
    pub fn new(variables: usize) -> Self {
        let radius = Self::paper_radius().unwrap_or_else(|| {
            panic!("C1-DTLZ3 has no radius for {M} objectives in the paper: use with_radius")
        });
        Self::with_radius(variables, radius)
    }

    /// The problem with `variables` variables, at least `M`, and the radius `radius`.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives, fewer variables than objectives, or a radius that isn't
    /// finite and above 0.
    pub fn with_radius(variables: usize, radius: f64) -> Self {
        assert!(
            radius.is_finite() && radius > 0.0,
            "C1-DTLZ3 needs a finite radius above 0"
        );
        Self {
            dtlz: Dtlz3::new(variables),
            radius,
        }
    }

    /// The number of variables.
    pub fn variables(&self) -> usize {
        self.dtlz.variables()
    }

    /// The radius r.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    fn values(&self, f: &[f64; M]) -> [f64; 1] {
        let squares: f64 = f.iter().map(|v| v * v).sum();
        [-(squares - 16.0) * (squares - self.radius * self.radius)]
    }
}

// ---- type 2 --------------------------------------------------------------------------------------

/// C2-DTLZ2: [`Dtlz2`], feasible only inside one of M + 1 spheres of radius r: centered at the
/// front's M corners, the unit vectors, and at its middle, (1/√M, …, 1/√M). r is 0.4 for 3
/// objectives and 0.5 for others.
///
/// With `M` objectives and `n` variables in [0, 1], `M + 9` by default (k = 10, as the paper
/// uses). The front is the parts of DTLZ2's, the unit sphere, inside the spheres: M + 1
/// disconnected patches (and one piece for 2 objectives, where the circles cover the whole
/// arc). No other solution is optimal (derived here): a feasible solution outside the unit
/// sphere is inside one of the spheres, and so is the point where the ray from the origin meets
/// the unit sphere, since that projection moves no point farther from a center on the unit
/// sphere, and that point dominates it. [`optimal_front`](MultiProblem::optimal_front) keeps the
/// feasible ones of Das and Dennis's points projected on the sphere.
///
/// The paper prints the constraint as `c = max{maxᵢ [(fᵢ − 1)² + Σ_{j≠i} fⱼ² − r²],
/// Σ (fᵢ − 1/√M)² − r²}` with no inequality; the text and its figure 7 make the inside of the
/// spheres feasible, which only `min{minᵢ [(fᵢ − 1)² + Σ_{j≠i} fⱼ² − r²], Σ (fᵢ − 1/√M)² − r²} ≤ 0`
/// gives, what [`constraints`](MultiProblem::constraints) returns.
///
/// Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using
/// reference-point based nondominated sorting approach, part II. *IEEE Transactions on
/// Evolutionary Computation* 18(4): 602-622, section V-C (see
/// [`C1Dtlz1`]'s notes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C2Dtlz2<const M: usize> {
    dtlz: Dtlz2<M>,
    radius: f64,
}

impl<const M: usize> C2Dtlz2<M> {
    /// The number of constraints.
    pub const CONSTRAINTS: usize = 1;

    /// The radius r of the paper for `M` objectives: 0.4 for 3, 0.5 for others.
    pub fn paper_radius() -> f64 {
        if M == 3 { 0.4 } else { 0.5 }
    }

    /// The problem with `variables` variables, at least `M`, and the paper's radius.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives, or fewer variables than objectives.
    pub fn new(variables: usize) -> Self {
        Self::with_radius(variables, Self::paper_radius())
    }

    /// The problem with `variables` variables, at least `M`, and the radius `radius`.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives, fewer variables than objectives, or a radius that isn't
    /// finite and above 0.
    pub fn with_radius(variables: usize, radius: f64) -> Self {
        assert!(
            radius.is_finite() && radius > 0.0,
            "C2-DTLZ2 needs a finite radius above 0"
        );
        Self {
            dtlz: Dtlz2::new(variables),
            radius,
        }
    }

    /// The number of variables.
    pub fn variables(&self) -> usize {
        self.dtlz.variables()
    }

    /// The radius r.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    fn values(&self, f: &[f64; M]) -> [f64; 1] {
        let r2 = self.radius * self.radius;
        let squares: f64 = f.iter().map(|v| v * v).sum();
        // (fᵢ − 1)² + Σ_{j≠i} fⱼ² = Σ fⱼ² − 2fᵢ + 1
        let corners = f
            .iter()
            .map(|fi| squares - 2.0 * fi + 1.0 - r2)
            .fold(f64::INFINITY, f64::min);
        let middle = 1.0 / (M as f64).sqrt();
        let centre = f
            .iter()
            .map(|fi| (fi - middle) * (fi - middle))
            .sum::<f64>()
            - r2;
        [corners.min(centre)]
    }
}

/// Convex C2-DTLZ2: convex DTLZ2, infeasible inside a cylinder of radius r around the diagonal
/// (1, …, 1): subject to `Σ (fᵢ − λ)² − r² ≥ 0` with λ the mean of the objectives (the paper's
/// eq. 6), and r = 0.225 for 3 and 5 objectives, 0.26 for 8 and 10, and 0.27 for 15.
///
/// Convex DTLZ2 is [`Dtlz2`] with `fᵢ` raised to the power 4 for i < M and `f_M` squared (part
/// I, section VII-C), whose front is `f_M + Σᵢ₌₁^{M−1} √fᵢ = 1`. With `M` objectives and `n`
/// variables in [0, 1], `M + 9` by default (k = 10, as the paper uses). The cylinder cuts a hole
/// in the middle of the front, and the front is the rest of it.
/// [`optimal_front`](MultiProblem::optimal_front) maps Das and Dennis's points s to the front,
/// `fᵢ = sᵢ²` for i < M and `f_M = s_M`, and keeps the feasible ones. That nothing behind the
/// hole becomes optimal is checked by the tests against random solutions, not proven.
///
/// [`constraints`](MultiProblem::constraints) gives `r² − Σ (fᵢ − λ)²`.
///
/// Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using
/// reference-point based nondominated sorting approach, part II. *IEEE Transactions on
/// Evolutionary Computation* 18(4): 602-622, section V-C, eq. 6 (see
/// [`C1Dtlz1`]'s notes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvexC2Dtlz2<const M: usize> {
    dtlz: Dtlz2<M>,
    radius: f64,
}

impl<const M: usize> ConvexC2Dtlz2<M> {
    /// The number of constraints.
    pub const CONSTRAINTS: usize = 1;

    /// The radius r of the paper for `M` objectives: 0.225 for 3 and 5, 0.26 for 8 and 10, 0.27
    /// for 15, and `None` for others.
    pub fn paper_radius() -> Option<f64> {
        match M {
            3 | 5 => Some(0.225),
            8 | 10 => Some(0.26),
            15 => Some(0.27),
            _ => None,
        }
    }

    /// The problem with `variables` variables, at least `M`, and the paper's radius.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives, fewer variables than objectives, or a number of objectives
    /// the paper gives no radius for (see [`paper_radius`](Self::paper_radius)).
    pub fn new(variables: usize) -> Self {
        let radius = Self::paper_radius().unwrap_or_else(|| {
            panic!("convex C2-DTLZ2 has no radius for {M} objectives in the paper: use with_radius")
        });
        Self::with_radius(variables, radius)
    }

    /// The problem with `variables` variables, at least `M`, and the radius `radius`.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives, fewer variables than objectives, or a radius that isn't
    /// finite and above 0.
    pub fn with_radius(variables: usize, radius: f64) -> Self {
        assert!(
            radius.is_finite() && radius > 0.0,
            "convex C2-DTLZ2 needs a finite radius above 0"
        );
        Self {
            dtlz: Dtlz2::new(variables),
            radius,
        }
    }

    /// The number of variables.
    pub fn variables(&self) -> usize {
        self.dtlz.variables()
    }

    /// The radius r.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    // convex DTLZ2's objectives from DTLZ2's
    fn convex(mut f: [f64; M]) -> [f64; M] {
        for fi in &mut f[..M - 1] {
            *fi = (*fi * *fi) * (*fi * *fi);
        }
        f[M - 1] *= f[M - 1];
        f
    }

    fn values(&self, f: &[f64; M]) -> [f64; 1] {
        let mean = f.iter().sum::<f64>() / M as f64;
        let spread: f64 = f.iter().map(|fi| (fi - mean) * (fi - mean)).sum();
        [self.radius * self.radius - spread]
    }
}

// ---- type 3 --------------------------------------------------------------------------------------

/// C3-DTLZ1: [`Dtlz1`] subject to M constraints `Σ_{i≠j} fᵢ + fⱼ/0.5 − 1 ≥ 0`, for j = 1, …, M.
///
/// With `M` objectives and `n` variables in [0, 1], `M + 4` by default (k = 5, as the paper
/// uses). DTLZ1's front, where the objectives sum to 1/2, is infeasible, and the front lies on
/// the constraints' boundaries (derived here): the constraints ask that `S + minⱼ fⱼ ≥ 1`, with S
/// the sum of the objectives, and the front is `{f ≥ 0 : S + minⱼ fⱼ = 1}`, M planes meeting at
/// (1/(M + 1), …): any point below one of them breaks a constraint. It runs from the corners, the
/// unit vectors, to the middle point; each is reached by DTLZ1 with g = 2S − 1.
/// [`optimal_front`](MultiProblem::optimal_front) scales Das and Dennis's points w onto it,
/// `w / (1 + minⱼ wⱼ)`.
///
/// The paper prints the constraints (eq. 7) as `Σ_{i≠j} fⱼ + fᵢ/0.5 − 1 ≥ 0`, i and j swapped:
/// that sum is the same for every j, `2S − 1`, and would make DTLZ1's front feasible, against the
/// text ("the unconstrained Pareto-optimal front is now infeasible") and figure 13.
/// [`constraints`](MultiProblem::constraints) gives `1 − Σ_{i≠j} fᵢ − fⱼ/0.5` for each j.
///
/// Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using
/// reference-point based nondominated sorting approach, part II. *IEEE Transactions on
/// Evolutionary Computation* 18(4): 602-622, section V-D, eq. 7 (see
/// [`C1Dtlz1`]'s notes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct C3Dtlz1<const M: usize> {
    dtlz: Dtlz1<M>,
}

impl<const M: usize> C3Dtlz1<M> {
    /// The number of constraints: one per objective.
    pub const CONSTRAINTS: usize = M;

    /// The problem with `variables` variables, at least `M`.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives, or fewer variables than objectives.
    pub fn new(variables: usize) -> Self {
        Self {
            dtlz: Dtlz1::new(variables),
        }
    }

    /// The number of variables.
    pub fn variables(&self) -> usize {
        self.dtlz.variables()
    }

    fn values(f: &[f64; M]) -> [f64; M] {
        let sum: f64 = f.iter().sum();
        // Σ_{i≠j} fᵢ + fⱼ/0.5 = S + fⱼ
        f.map(|fj| 1.0 - (sum - fj + fj / 0.5))
    }
}

/// C3-DTLZ4: [`Dtlz4`] subject to M constraints `fⱼ²/4 + Σ_{i≠j} fᵢ² − 1 ≥ 0`, for
/// j = 1, …, M.
///
/// With `M` objectives and `n` variables in [0, 1], `M + 4` by default: the paper uses
/// n = M + 4 (k = 5) for this problem, where DTLZ4 has k = 10. DTLZ4's front, the unit sphere,
/// is infeasible, and the front lies on the constraints' boundaries (derived here): each
/// constraint grows with every objective, so the front is
/// `{f ≥ 0 : minⱼ [fⱼ²/4 + Σ_{i≠j} fᵢ²] = 1}`, M ellipsoids meeting at (2/√(4M − 3), …), from
/// the points 2 × the unit vectors. [`optimal_front`](MultiProblem::optimal_front) scales Das and
/// Dennis's points onto it. DTLZ4's bias (each angle's variable to the power 100) crowds
/// solutions towards the axes.
///
/// [`constraints`](MultiProblem::constraints) gives `1 − fⱼ²/4 − Σ_{i≠j} fᵢ²` for each j.
///
/// Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using
/// reference-point based nondominated sorting approach, part II. *IEEE Transactions on
/// Evolutionary Computation* 18(4): 602-622, section V-D, eq. 8 (see
/// [`C1Dtlz1`]'s notes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct C3Dtlz4<const M: usize> {
    dtlz: Dtlz4<M>,
}

impl<const M: usize> C3Dtlz4<M> {
    /// The number of constraints: one per objective.
    pub const CONSTRAINTS: usize = M;

    /// The problem with `variables` variables, at least `M`.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives, or fewer variables than objectives.
    pub fn new(variables: usize) -> Self {
        Self {
            dtlz: Dtlz4::new(variables),
        }
    }

    /// The number of variables.
    pub fn variables(&self) -> usize {
        self.dtlz.variables()
    }

    fn values(f: &[f64; M]) -> [f64; M] {
        let squares: f64 = f.iter().map(|v| v * v).sum();
        // fⱼ²/4 + Σ_{i≠j} fᵢ² = Σ fᵢ² − 3fⱼ²/4
        f.map(|fj| 1.0 - (squares - 0.75 * fj * fj))
    }

    // the least over j of fⱼ²/4 + Σ_{i≠j} fᵢ²: 1 on the front
    fn least(f: &[f64; M]) -> f64 {
        let squares: f64 = f.iter().map(|v| v * v).sum();
        let largest = f.iter().copied().fold(0.0, f64::max);
        squares - 0.75 * largest * largest
    }
}

// ---- the implementations -------------------------------------------------------------------------

// Default, the fitness and the metadata of a constrained DTLZ problem: `$objectives` maps DTLZ's
// objectives, `$values` gives the constraints from the objectives
macro_rules! constrained_dtlz {
    ($name:ident, $label:literal, $k:literal, $count:expr, |$problem:ident, $f:ident| $values:expr, $objectives:expr) => {
        impl<const M: usize> Default for $name<M> {
            #[doc = concat!("The paper's problem, with `M + ", stringify!($k), " − 1` variables.")]
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives, or where [`new`](Self::new) panics.
            fn default() -> Self {
                Self::new(M + $k - 1)
            }
        }

        impl<const M: usize> $name<M> {
            fn objectives(&self, x: &Reals) -> [f64; M] {
                let objectives: fn([f64; M]) -> [f64; M] = $objectives;
                objectives(self.dtlz.evaluate(x))
            }

            fn constraint_values(&self, x: &Reals) -> Vec<f64> {
                let $problem = self;
                let $f = &self.objectives(x);
                $values.to_vec()
            }
        }

        impl<const M: usize> MultiFitnessFunction<Reals, M> for $name<M> {
            type Output = ([f64; M], f64);

            /// The objective values of `x` and its constraint violation, 0 when it's feasible.
            ///
            /// # Panics
            ///
            /// If `x` has fewer than `M − 1` genes.
            fn evaluate(&self, x: &Reals) -> ([f64; M], f64) {
                let $problem = self;
                let $f = &self.objectives(x);
                (*$f, violation(&$values))
            }
        }

        impl<const M: usize> MultiProblem<M> for $name<M> {
            type Representation = Real;

            fn name(&self) -> &'static str {
                $label
            }

            fn representation(&self) -> Real {
                self.dtlz.representation()
            }

            fn reference(&self) -> &'static str {
                REFERENCE
            }

            fn reference_url(&self) -> Option<&'static str> {
                Some(REFERENCE_URL)
            }

            fn constraint_count(&self) -> usize {
                $count
            }

            fn constraints(&self, genome: &Reals) -> Constraints {
                Constraints::new(self.constraint_values(genome), Vec::new())
            }

            fn optimal_front(&self, points: usize) -> Option<Vec<[f64; M]>> {
                Some(self.front(points))
            }

            fn ideal_point(&self) -> Option<[f64; M]> {
                Some([0.0; M])
            }

            fn nadir_point(&self) -> Option<[f64; M]> {
                Some([self.nadir(); M])
            }
        }
    };
}

fn same<const M: usize>(f: [f64; M]) -> [f64; M] {
    f
}

constrained_dtlz!(
    C1Dtlz1,
    "C1-DTLZ1",
    5,
    1,
    |_p, f| C1Dtlz1::<M>::values(f),
    same
);
constrained_dtlz!(C1Dtlz3, "C1-DTLZ3", 10, 1, |p, f| p.values(f), same);
constrained_dtlz!(C2Dtlz2, "C2-DTLZ2", 10, 1, |p, f| p.values(f), same);
constrained_dtlz!(
    ConvexC2Dtlz2,
    "convex C2-DTLZ2",
    10,
    1,
    |p, f| p.values(f),
    ConvexC2Dtlz2::<M>::convex
);
constrained_dtlz!(
    C3Dtlz1,
    "C3-DTLZ1",
    5,
    M,
    |_p, f| C3Dtlz1::<M>::values(f),
    same
);
constrained_dtlz!(
    C3Dtlz4,
    "C3-DTLZ4",
    5,
    M,
    |_p, f| C3Dtlz4::<M>::values(f),
    same
);

// the fronts and their nadir points
impl<const M: usize> C1Dtlz1<M> {
    fn front(&self, points: usize) -> Vec<[f64; M]> {
        self.dtlz.optimal_front(points).expect("known")
    }

    fn nadir(&self) -> f64 {
        0.5
    }
}

impl<const M: usize> C1Dtlz3<M> {
    fn front(&self, points: usize) -> Vec<[f64; M]> {
        self.dtlz.optimal_front(points).expect("known")
    }

    fn nadir(&self) -> f64 {
        1.0
    }
}

impl<const M: usize> C2Dtlz2<M> {
    // Das and Dennis's points projected on the unit sphere, where feasible
    fn front(&self, points: usize) -> Vec<[f64; M]> {
        let project = |w: [f64; M]| {
            let norm = w.iter().map(|v| v * v).sum::<f64>().sqrt();
            w.map(|v| v / norm)
        };
        filtered_front(points, project, |f| self.values(f)[0] <= 0.0)
    }

    fn nadir(&self) -> f64 {
        1.0
    }
}

impl<const M: usize> ConvexC2Dtlz2<M> {
    // Das and Dennis's points s on the front, fᵢ = sᵢ² for i < M and f_M = s_M, where feasible
    fn front(&self, points: usize) -> Vec<[f64; M]> {
        let onto = |s: [f64; M]| {
            let mut f = s.map(|v| v * v);
            f[M - 1] = s[M - 1];
            f
        };
        filtered_front(points, onto, |f| self.values(f)[0] <= 0.0)
    }

    fn nadir(&self) -> f64 {
        1.0
    }
}

impl<const M: usize> C3Dtlz1<M> {
    // Das and Dennis's points w, scaled onto the front: w / (1 + min w)
    fn front(&self, points: usize) -> Vec<[f64; M]> {
        let scale = |w: [f64; M]| {
            let least = w.iter().copied().fold(f64::INFINITY, f64::min);
            w.map(|v| v / (1.0 + least))
        };
        filtered_front(points, scale, |_| true)
    }

    fn nadir(&self) -> f64 {
        1.0
    }
}

impl<const M: usize> C3Dtlz4<M> {
    // Das and Dennis's points w, scaled onto the front: w / √least(w)
    fn front(&self, points: usize) -> Vec<[f64; M]> {
        let scale = |w: [f64; M]| {
            let factor = Self::least(&w).sqrt();
            w.map(|v| v / factor)
        };
        filtered_front(points, scale, |_| true)
    }

    fn nadir(&self) -> f64 {
        2.0
    }
}

/// The constrained DTLZ problems with `M` objectives at their default sizes, for
/// [`all`](super::all): C1-DTLZ3 and convex C2-DTLZ2 only for the numbers of objectives the paper
/// gives their radius for.
pub(super) fn all<const M: usize>() -> Vec<Box<dyn DynMultiProblem<M>>> {
    let mut problems = vec![boxed(C1Dtlz1::<M>::default())];
    if C1Dtlz3::<M>::paper_radius().is_some() {
        problems.push(boxed(C1Dtlz3::<M>::default()));
    }
    problems.push(boxed(C2Dtlz2::<M>::default()));
    if ConvexC2Dtlz2::<M>::paper_radius().is_some() {
        problems.push(boxed(ConvexC2Dtlz2::<M>::default()));
    }
    problems.push(boxed(C3Dtlz1::<M>::default()));
    problems.push(boxed(C3Dtlz4::<M>::default()));
    problems
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamRng;
    use crate::genome::Representation;
    use crate::math;
    use std::f64::consts::PI;

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

    // the formulas of Jain and Deb, at points chosen so that the values follow by hand
    #[test]
    fn values_match_the_paper() {
        // C1-DTLZ1 (M = 3, k = 5) at (0.5, …): DTLZ1's (0.125, 0.125, 0.25), and
        // 0.25/0.6 + 0.25/0.5 − 1 = −1/12, feasible
        let problem = C1Dtlz1::<3>::default();
        let x = at(&[0.5; 7]);
        assert_eq!(problem.evaluate(&x), ([0.125, 0.125, 0.25], 0.0));
        assert_close(problem.constraints(&x).inequalities(), &[-1.0 / 12.0]);
        // the distance variables at 1: DTLZ1's g = 125, 126 times the above, and
        // 31.5/0.6 + 31.5/0.5 − 1 = 114.5
        let x = at(&[0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let (f, violation) = problem.evaluate(&x);
        assert_close(&f, &[15.75, 15.75, 31.5]);
        assert_close(&[violation], &[114.5]);

        // C1-DTLZ3 (M = 3, k = 10, r = 9): on the front, Σ f² = 1, (1 − 16) (1 − 81) = 1200 ≥ 0;
        // five distance variables at 0.6 add 100 (0.01 − cos 2π + 1) = 1 each to g, a radius of
        // 6 inside the band: (36 − 16) (36 − 81) = −900
        let problem = C1Dtlz3::<3>::default();
        assert_eq!(problem.radius(), 9.0);
        let mut genes = vec![0.5; 12];
        genes[..2].copy_from_slice(&[0.3, 0.3]);
        let x = at(&genes);
        let (f, violation) = problem.evaluate(&x);
        assert!((f.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(violation, 0.0);
        assert_close(problem.constraints(&x).inequalities(), &[-1200.0]);
        genes[2..7].fill(0.6);
        let (f, violation) = problem.evaluate(&at(&genes));
        assert!((f.iter().map(|v| v * v).sum::<f64>() - 36.0).abs() < 1e-9);
        assert!((violation - 900.0).abs() < 1e-6);

        // C2-DTLZ2 (M = 3, r = 0.4): at the corner (1, 0, 0), inside its sphere by 0.16; at
        // (1/√2, 1/√2, 0), between the spheres: 1 − √2 + 1 − 0.16 from a corner, and
        // 2 (1/√2 − 1/√3)² + 1/3 − 0.16 from the middle
        let problem = C2Dtlz2::<3>::default();
        assert_eq!(problem.radius(), 0.4);
        let mut genes = vec![0.5; 12];
        genes[..2].copy_from_slice(&[0.0, 0.0]);
        let x = at(&genes);
        assert_close(&problem.evaluate(&x).0, &[1.0, 0.0, 0.0]);
        assert_close(problem.constraints(&x).inequalities(), &[-0.16]);
        genes[1] = 0.5;
        let (half, third) = (0.5f64.sqrt(), (1.0f64 / 3.0).sqrt());
        let corner = 2.0 - 2.0 * half - 0.16;
        let middle = 2.0 * (half - third) * (half - third) + 1.0 / 3.0 - 0.16;
        let x = at(&genes);
        assert_close(
            problem.constraints(&x).inequalities(),
            &[corner.min(middle)],
        );
        assert!((problem.evaluate(&x).1 - middle).abs() < 1e-12);
        // at the middle of the front, inside the middle sphere by r²
        // (1/√3, 1/√3, 1/√3) at x₁ = asin(1/√3) / (π/2), x₂ = 1/2
        let x1 = math::asin(third) / (PI / 2.0);
        genes[..2].copy_from_slice(&[x1, 0.5]);
        let x = at(&genes);
        assert_close(&problem.evaluate(&x).0, &[third, third, third]);
        assert_close(problem.constraints(&x).inequalities(), &[-0.16]);
        assert_eq!(C2Dtlz2::<5>::default().radius(), 0.5);

        // convex C2-DTLZ2 (M = 3, r = 0.225): at x₁ = x₂ = 1/2, DTLZ2's (1/2, 1/2, 1/√2) becomes
        // (1/16, 1/16, 1/2), λ = 5/24, and Σ (fᵢ − λ)² = 2 (7/48)² + (7/24)² = 49/384 ≥ r²
        let problem = ConvexC2Dtlz2::<3>::default();
        assert_eq!(problem.radius(), 0.225);
        genes[..2].copy_from_slice(&[0.5, 0.5]);
        let x = at(&genes);
        assert_close(&problem.evaluate(&x).0, &[0.0625, 0.0625, 0.5]);
        assert_close(
            problem.constraints(&x).inequalities(),
            &[0.225 * 0.225 - 49.0 / 384.0],
        );
        // on the diagonal of the front, fᵢ = (√2 − 1)², infeasible by r²
        let f = [(2f64.sqrt() - 1.0).powi(2); 3];
        assert!((f[2] + f[0].sqrt() + f[1].sqrt() - 1.0).abs() < 1e-15);
        assert_close(&problem.values(&f), &[0.225 * 0.225]);

        // C3-DTLZ1 (M = 3): DTLZ1's front point (0.125, 0.125, 0.25), S = 0.5, breaks all three
        // constraints, 1 − (S + fⱼ); the corner (1, 0, 0), with g = 1 (a distance variable at
        // 0.6), meets them, two exactly
        let problem = C3Dtlz1::<3>::default();
        let x = at(&[0.5; 7]);
        assert_close(
            problem.constraints(&x).inequalities(),
            &[0.375, 0.375, 0.25],
        );
        assert_close(&[problem.evaluate(&x).1], &[1.0]);
        let x = at(&[1.0, 1.0, 0.6, 0.5, 0.5, 0.5, 0.5]);
        let (f, violation) = problem.evaluate(&x);
        assert_close(&f, &[1.0, 0.0, 0.0]);
        assert!(violation < 1e-12);
        assert_close(problem.constraints(&x).inequalities(), &[-1.0, 0.0, 0.0]);

        // C3-DTLZ4 (M = 3, k = 5): DTLZ4's front point (1, 0, 0) breaks the constraints of
        // f₂ and f₃, 1 − 1; the corner (2, 0, 0), with g = 1 (four distance variables at 1),
        // is on the first boundary: 4/4 − 1 = 0, and 4 − 1 for the others
        let problem = C3Dtlz4::<3>::default();
        assert_eq!(problem.variables(), 7);
        let x = at(&[0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5]);
        assert_close(problem.constraints(&x).inequalities(), &[0.75, 0.0, 0.0]);
        let x = at(&[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5]);
        let (f, violation) = problem.evaluate(&x);
        assert_close(&f, &[2.0, 0.0, 0.0]);
        assert_eq!(violation, 0.0);
        assert_close(problem.constraints(&x).inequalities(), &[0.0, -3.0, -3.0]);
    }

    #[test]
    fn sizes_and_radii() {
        assert_eq!(C1Dtlz1::<3>::default().variables(), 7);
        assert_eq!(C1Dtlz3::<5>::default().variables(), 14);
        assert_eq!(C1Dtlz3::<5>::default().radius(), 12.5);
        assert_eq!(C1Dtlz3::<8>::paper_radius(), Some(12.5));
        assert_eq!(C1Dtlz3::<10>::paper_radius(), Some(15.0));
        assert_eq!(C1Dtlz3::<2>::paper_radius(), None);
        assert_eq!(C1Dtlz3::<2>::with_radius(5, 6.0).radius(), 6.0);
        assert_eq!(ConvexC2Dtlz2::<8>::paper_radius(), Some(0.26));
        assert_eq!(ConvexC2Dtlz2::<15>::paper_radius(), Some(0.27));
        assert_eq!(ConvexC2Dtlz2::<4>::paper_radius(), None);
        assert_eq!(C2Dtlz2::<3>::default().variables(), 12);
        assert_eq!(C3Dtlz1::<4>::default().constraint_count(), 4);
        assert_eq!(C3Dtlz4::<5>::default().variables(), 9);
        let names: Vec<_> = all::<2>().iter().map(|p| p.name()).collect();
        assert_eq!(names, ["C1-DTLZ1", "C2-DTLZ2", "C3-DTLZ1", "C3-DTLZ4"]);
        assert_eq!(all::<5>().len(), 6);
    }

    #[test]
    #[should_panic(expected = "no radius for 4 objectives")]
    fn a_radius_the_paper_doesnt_give_panics() {
        let _ = C1Dtlz3::<4>::default();
    }

    // the fronts satisfy the identities derived for them, and are feasible
    #[test]
    fn the_fronts_lie_where_derived() {
        let squares = |f: &[f64]| f.iter().map(|v| v * v).sum::<f64>();
        let front = C1Dtlz1::<3>::default().optimal_front(91).expect("known");
        assert!(
            front
                .iter()
                .all(|f| (f.iter().sum::<f64>() - 0.5).abs() < 1e-12)
        );
        let front = C1Dtlz3::<3>::default().optimal_front(91).expect("known");
        assert!(front.iter().all(|f| (squares(f) - 1.0).abs() < 1e-12));
        let problem = C2Dtlz2::<3>::default();
        let front = problem.optimal_front(91).expect("known");
        assert!(front.len() >= 91);
        for f in &front {
            assert!((squares(f) - 1.0).abs() < 1e-12 && problem.values(f)[0] <= 0.0);
        }
        let problem = ConvexC2Dtlz2::<3>::default();
        let front = problem.optimal_front(91).expect("known");
        for f in &front {
            assert!((f[2] + f[0].sqrt() + f[1].sqrt() - 1.0).abs() < 1e-12);
            assert!(problem.values(f)[0] <= 0.0);
        }
        let front = C3Dtlz1::<3>::default().optimal_front(91).expect("known");
        assert_eq!(front.len(), 91);
        for f in &front {
            let least = f.iter().copied().fold(f64::INFINITY, f64::min);
            assert!((f.iter().sum::<f64>() + least - 1.0).abs() < 1e-12);
            assert!(C3Dtlz1::<3>::values(f).iter().all(|&g| g <= 1e-12));
        }
        let front = C3Dtlz4::<3>::default().optimal_front(91).expect("known");
        for f in &front {
            assert!((C3Dtlz4::<3>::least(f) - 1.0).abs() < 1e-12);
            assert!(C3Dtlz4::<3>::values(f).iter().all(|&g| g <= 1e-12));
        }
        // the middle points: (1/4, …) for C3-DTLZ1, (2/√(M + 3), …) for C3-DTLZ4
        let middle = [0.25; 3];
        assert!(
            C3Dtlz1::<3>::values(&middle)
                .iter()
                .all(|g| g.abs() < 1e-15)
        );
        let middle = 2.0 / 3.0;
        assert!((C3Dtlz4::<3>::least(&[middle; 3]) - 1.0).abs() < 1e-12);
        // two objectives: exactly the points asked for
        assert_eq!(C2Dtlz2::<2>::default().optimal_front(37).unwrap().len(), 37);
        assert_eq!(C3Dtlz4::<2>::default().optimal_front(37).unwrap().len(), 37);
        let problem = C2Dtlz2::<3>::with_radius(12, 0.1);
        let front = problem.optimal_front(20).expect("known");
        assert!(front.len() >= 20 && front.iter().all(|f| problem.values(f)[0] <= 0.0));
    }

    // no feasible solution near the front is better than it: random genomes with their distance
    // variables within `noise` of 0.5, the optimal value, and every feasible one weakly dominated by a point of
    // the front, up to its spacing, and dominating none by more than 1e-9
    fn check_against_random<P>(problem: &P, noise: f64, tolerance: f64)
    where
        P: MultiProblem<3, Representation = Real>
            + MultiFitnessFunction<Reals, 3, Output = ([f64; 3], f64)>,
    {
        let front = problem.optimal_front(5_000).expect("known");
        let mut rng = StreamRng::seed_from_u64(11);
        let real = problem.representation();
        let mut feasible = 0;
        for _ in 0..10_000 {
            let mut genome = real.random_genome(&mut rng).to_vec();
            for gene in &mut genome[2..] {
                *gene = 0.5 + (*gene - 0.5) * 2.0 * noise;
            }
            let (f, violation) = problem.evaluate(&at(&genome));
            if violation > 0.0 {
                continue;
            }
            feasible += 1;
            assert!(
                front
                    .iter()
                    .any(|q| (0..3).all(|j| q[j] <= f[j] + tolerance)),
                "{}: {f:?}",
                problem.name()
            );
            for q in &front {
                assert!(
                    !(0..3).all(|j| f[j] < q[j] - 1e-9),
                    "{}: {f:?} dominates {q:?}",
                    problem.name()
                );
            }
        }
        assert!(feasible > 20, "{}: {feasible}", problem.name());
    }

    #[test]
    fn random_solutions_agree_with_the_fronts() {
        // DTLZ1's and DTLZ3's g is multimodal: 0.002 away from 0.5 adds up to 0.2 to g per
        // variable
        check_against_random(&C1Dtlz1::<3>::default(), 0.001, 0.01);
        check_against_random(&C1Dtlz3::<3>::default(), 0.002, 0.02);
        check_against_random(&C2Dtlz2::<3>::default(), 0.05, 0.02);
        check_against_random(&ConvexC2Dtlz2::<3>::default(), 0.05, 0.02);
        check_against_random(&C3Dtlz1::<3>::default(), 0.01, 0.01);
        check_against_random(&C3Dtlz4::<3>::default(), 0.5, 0.02);
    }
}
