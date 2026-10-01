//! DTLZ8 and DTLZ9: the constrained problems of Deb, Thiele, Laumanns and Zitzler's report, built
//! by its constraint surface approach: objectives that each average or sum their own block of
//! variables, and constraints on the objectives that cut the front out of their space.
//!
//! Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). *Scalable Test Problems for
//! Evolutionary Multi-Objective Optimization.* TIK-Report 112, Computer Engineering and Networks
//! Laboratory, ETH Zürich, sections 8.8 and 8.9, eqs. 28 and 29, read in the report (the 2005 book
//! chapter, *Evolutionary Multiobjective Optimization*, Springer: 105-145, which reprints it,
//! wasn't compared). The report's DTLZ8 is the 2002 conference paper's DTLZ7 and its DTLZ9 isn't
//! in the paper; genoxide follows the report's numbering, as for DTLZ5-7.
//!
//! **The blocks.** Each objective fⱼ takes the j-th of M blocks of variables, the report printing
//! its sum as running from `⌊(j − 1) n/M⌋` to `⌊j n/M⌋`. With 1-based variables, as the report
//! counts them elsewhere, that would start the first block at a variable x₀ that doesn't exist
//! and let neighbouring blocks share a variable: the sums here run from `⌊(j − 1) n/M⌋ + 1` to
//! `⌊j n/M⌋`, which splits the n variables into M blocks of ⌊n/M⌋ or ⌈n/M⌉ each, all of them
//! ⌊n/M⌋ = n/M for the report's n = 10M. No code from the report's authors was found to check
//! this reading against.
//!
//! The report gives no constraint-free fronts: the fronts here are derived from the constraints,
//! as each problem's docs show, and the tests check them against random feasible solutions.

use super::{MultiProblem, evenly};
use crate::constraint::at_most;
use crate::genome::{Real, Reals};
use crate::math;
use crate::multi::MultiFitnessFunction;
use crate::problems::Constraints;
use std::f64::consts::FRAC_PI_2;

const REPORT: &str = "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). Scalable Test \
                      Problems for Evolutionary Multi-Objective Optimization. TIK-Report 112, \
                      Computer Engineering and Networks Laboratory, ETH Zürich.";
const REPORT_URL: &str = "https://sop.tik.ee.ethz.ch/publicationListFiles/dtlz2001a.pdf";

// the total violation of constraints g(x) <= 0
fn violation(values: &[f64]) -> f64 {
    values.iter().map(|&g| at_most(g, 0.0)).sum()
}

// the 0-based range of the variables of the j-th block (0-based j) of n variables in M blocks:
// the report's ⌊(j − 1) n/M⌋ + 1 to ⌊j n/M⌋, 1-based
fn block(j: usize, n: usize, m: usize) -> std::ops::Range<usize> {
    j * n / m..(j + 1) * n / m
}

/// DTLZ8: M objectives, each the mean of its own block of variables, subject to M constraints
/// that leave a front made of a line and a surface.
///
/// ```text
/// minimize   fⱼ = (1/⌊n/M⌋) Σ_{i in the j-th block} xᵢ,   j = 1, …, M
/// subject to f_M + 4 fⱼ − 1 ≥ 0,                         j = 1, …, M − 1
///            2 f_M + min_{i≠j} (fᵢ + fⱼ) − 1 ≥ 0,          i, j = 1, …, M − 1
/// ```
///
/// With `M` objectives, from 3 on (the last constraint takes the least sum of two different
/// objectives other than `f_M`), and `n` variables in [0, 1], `10 M` by default, as the report
/// suggests; the j-th block is `x_{⌊(j−1)n/M⌋+1}` to `x_{⌊jn/M⌋}`. The report prints the sum
/// from `⌊(j − 1) n/M⌋`, which with 1-based variables would start the first block at a variable
/// x₀ that doesn't exist and let neighbouring blocks share one: genoxide reads it as above, M
/// blocks of ⌊n/M⌋ or ⌈n/M⌉ variables, n/M each for the report's n = 10 M. The objectives can take any values in [0, 1] independently, so the problem
/// is its constraints: the report's front is "a combination of a straight line and a
/// hyper-plane", the line where the first M − 1 constraints meet with f₁ = … = f_{M−1}, and the
/// last constraint's plane.
///
/// **The front**, derived here: the line `f₁ = … = f_{M−1} = t`, `f_M = 1 − 4t` for t in
/// [0, 1/6], from (0, …, 0, 1) to (1/6, …, 1/6, 1/3), and from its end the part of the plane
/// `2 f_M + fᵢ + fⱼ = 1` where one objective s (any of the first M − 1) is at most all the
/// others, which are equal, u: `f_M = t` in [0, 1/3], s from `(1 − t)/4` (the first constraints)
/// to `(1 − 2t)/2` (s = u), and `u = 1 − 2t − s`. A point with two of the first M − 1
/// objectives above the least that differ is dominated by the one with both at the lesser,
/// feasible still; for M = 3 the plane part is the triangle from (1/6, 1/6, 1/3) to (1/4, 3/4, 0)
/// and (3/4, 1/4, 0). The ideal point is the origin, and the nadir point (3/4, …, 3/4, 1).
///
/// [`constraints`](MultiProblem::constraints) gives `1 − f_M − 4 fⱼ` for j < M, then
/// `1 − 2 f_M − min_{i≠j} (fᵢ + fⱼ)`.
///
/// The report's NSGA-II and SPEA2 runs (figures 28 and 29) find parts of the front with many
/// solutions on the surfaces next to it, which the front only weakly dominates.
///
/// Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). *Scalable Test Problems for
/// Evolutionary Multi-Objective Optimization.* TIK-Report 112, ETH Zürich, section 8.8, eq. 28.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dtlz8<const M: usize> {
    variables: usize,
}

impl<const M: usize> Dtlz8<M> {
    /// The number of constraints: one per objective.
    pub const CONSTRAINTS: usize = M;

    /// The problem with `variables` variables, at least `M`.
    ///
    /// # Panics
    ///
    /// With fewer than 3 objectives, or fewer variables than objectives.
    pub fn new(variables: usize) -> Self {
        assert!(M >= 3, "DTLZ8 needs at least 3 objectives");
        assert!(
            variables >= M,
            "DTLZ8 needs at least as many variables as objectives"
        );
        Self { variables }
    }

    /// The number of variables.
    pub fn variables(&self) -> usize {
        self.variables
    }

    // the objectives: the mean of each block, divided by ⌊n/M⌋ as the report prints it
    fn objectives(&self, x: &Reals) -> [f64; M] {
        let n = x.len();
        let size = (n / M) as f64;
        std::array::from_fn(|j| x[block(j, n, M)].iter().sum::<f64>() / size)
    }

    // the constraints as g <= 0, from the objectives
    fn values(f: &[f64; M]) -> [f64; M] {
        let last = f[M - 1];
        // the least sum of two different objectives other than f_M: the two least of them
        let (mut least, mut second) = (f64::INFINITY, f64::INFINITY);
        for &fi in &f[..M - 1] {
            if fi < least {
                (least, second) = (fi, least);
            } else if fi < second {
                second = fi;
            }
        }
        std::array::from_fn(|j| {
            if j < M - 1 {
                1.0 - last - 4.0 * f[j]
            } else {
                1.0 - 2.0 * last - (least + second)
            }
        })
    }

    // the front with the resolution k: k points of the line, and the plane part on a triangular
    // grid of k + 1 rows in t, for each objective that can be the least
    fn grid(k: usize) -> Vec<[f64; M]> {
        let mut front = Vec::new();
        for i in 0..k {
            let t = i as f64 / k as f64 / 6.0;
            let mut f = [t; M];
            f[M - 1] = 1.0 - 4.0 * t;
            front.push(f);
        }
        for least in 0..M - 1 {
            for i in 0..=k {
                let t = i as f64 / k as f64 / 3.0;
                let low = (1.0 - t) / 4.0;
                let width = (1.0 - 3.0 * t) / 4.0;
                let steps = k - i;
                for j in 0..=steps {
                    // s = u (the end of a row) is the same point for every choice of the least
                    if least > 0 && j == steps {
                        continue;
                    }
                    let s = if steps == 0 {
                        low
                    } else {
                        low + width * j as f64 / steps as f64
                    };
                    let u = 1.0 - 2.0 * t - s;
                    let mut f = [u; M];
                    f[least] = s;
                    f[M - 1] = t;
                    front.push(f);
                }
            }
        }
        front
    }

    fn front(points: usize) -> Vec<[f64; M]> {
        if points == 0 {
            return Vec::new();
        }
        let mut k = 1;
        loop {
            let front = Self::grid(k);
            if front.len() >= points {
                return front;
            }
            k += 1;
        }
    }
}

/// DTLZ9: M objectives, each the sum of `xᵢ^0.1` over its own block of variables, subject to
/// M − 1 constraints that leave a front that is a curve.
///
/// ```text
/// minimize   fⱼ = Σ_{i in the j-th block} xᵢ^0.1,   j = 1, …, M
/// subject to f_M² + fⱼ² − 1 ≥ 0,                   j = 1, …, M − 1
/// ```
///
/// With `M` objectives, from 2 on, and `n` variables in [0, 1], `10 M` by default, as the report
/// suggests; the j-th block is `x_{⌊(j−1)n/M⌋+1}` to `x_{⌊jn/M⌋}`, read as for [`Dtlz8`].
/// The report prints no mean here, unlike DTLZ8: an objective ranges over [0, ⌈n/M⌉].
///
/// **The front** is the curve `f₁ = … = f_{M−1} = cos θ`, `f_M = sin θ` for θ in [0, π/2], where
/// all the constraints meet, as the report says (derived here: for a given `f_M` in [0, 1], each
/// constraint asks `fⱼ ≥ √(1 − f_M²)` alone). With f_M and any other objective, it's the unit
/// circle's quarter; with two others, the diagonal. Its ideal point is the origin and its nadir
/// point (1, …, 1). The power 0.1 makes small objectives hard to reach: on the front each
/// variable is about `(fⱼ/10)^10`, below 10⁻¹⁰, and a random solution is near the top of its
/// range, the report's "the density of solutions gets thinner towards the Pareto-optimal
/// region".
///
/// [`constraints`](MultiProblem::constraints) gives `1 − f_M² − fⱼ²` for j < M.
///
/// Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). *Scalable Test Problems for
/// Evolutionary Multi-Objective Optimization.* TIK-Report 112, ETH Zürich, section 8.9, eq. 29.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dtlz9<const M: usize> {
    variables: usize,
}

impl<const M: usize> Dtlz9<M> {
    /// The number of constraints: one per objective but the last.
    pub const CONSTRAINTS: usize = M - 1;

    /// The problem with `variables` variables, at least `M`.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives, or fewer variables than objectives.
    pub fn new(variables: usize) -> Self {
        assert!(M >= 2, "DTLZ9 needs at least 2 objectives");
        assert!(
            variables >= M,
            "DTLZ9 needs at least as many variables as objectives"
        );
        Self { variables }
    }

    /// The number of variables.
    pub fn variables(&self) -> usize {
        self.variables
    }

    fn objectives(&self, x: &Reals) -> [f64; M] {
        let n = x.len();
        std::array::from_fn(|j| {
            x[block(j, n, M)]
                .iter()
                .map(|&xi| math::powf(xi, 0.1))
                .sum()
        })
    }

    fn values(f: &[f64; M]) -> Vec<f64> {
        let last = f[M - 1] * f[M - 1];
        f[..M - 1].iter().map(|fj| 1.0 - last - fj * fj).collect()
    }

    // `points` points of the curve, evenly spread in angle
    fn front(points: usize) -> Vec<[f64; M]> {
        (0..points)
            .map(|i| {
                let (sin, cos) = math::sin_cos(FRAC_PI_2 * evenly(i, points));
                let mut f = [cos; M];
                f[M - 1] = sin;
                f
            })
            .collect()
    }
}

// Default, the fitness and the metadata
macro_rules! constraint_surface {
    ($name:ident, $label:literal, $count:expr, $ideal:expr, $nadir:expr) => {
        impl<const M: usize> Default for $name<M> {
            /// The report's problem, with `10 M` variables.
            ///
            /// # Panics
            ///
            /// Where [`new`](Self::new) panics.
            fn default() -> Self {
                Self::new(10 * M)
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
                (f, violation(&Self::values(&f)))
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
                REPORT
            }

            fn reference_url(&self) -> Option<&'static str> {
                Some(REPORT_URL)
            }

            fn constraint_count(&self) -> usize {
                $count
            }

            fn constraints(&self, genome: &Reals) -> Constraints {
                let f = self.objectives(genome);
                Constraints::new(Self::values(&f).to_vec(), Vec::new())
            }

            fn optimal_front(&self, points: usize) -> Option<Vec<[f64; M]>> {
                Some(Self::front(points))
            }

            fn ideal_point(&self) -> Option<[f64; M]> {
                Some($ideal)
            }

            fn nadir_point(&self) -> Option<[f64; M]> {
                Some($nadir)
            }
        }
    };
}

constraint_surface!(Dtlz8, "DTLZ8", M, [0.0; M], {
    let mut nadir = [0.75; M];
    nadir[M - 1] = 1.0;
    nadir
});
constraint_surface!(Dtlz9, "DTLZ9", M - 1, [0.0; M], [1.0; M]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective::Minimize;
    use crate::StreamRng;
    use crate::genome::Representation;
    use crate::multi::{Scores, non_dominated_sort};

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

    #[test]
    fn blocks_split_the_variables() {
        // the report's n = 10M: blocks of 10
        assert_eq!(block(0, 30, 3), 0..10);
        assert_eq!(block(2, 30, 3), 20..30);
        // 7 variables in 3 blocks: ⌊7/3⌋ = 2, ⌊14/3⌋ = 4, 7
        assert_eq!(
            (0..3).map(|j| block(j, 7, 3)).collect::<Vec<_>>(),
            [0..2, 2..4, 4..7]
        );
    }

    // the formulas of the report, at points chosen so that the values follow by hand
    #[test]
    fn values_match_the_report() {
        // DTLZ8, M = 3, n = 30: blocks at 1/8, 1/4 and 5/8 give (1/8, 1/4, 5/8); the constraints
        // 1 − 5/8 − 1/2, 1 − 5/8 − 1 and 1 − 5/4 − 3/8, all feasible
        let problem = Dtlz8::<3>::default();
        assert_eq!(problem.variables(), 30);
        let mut genes = vec![0.125; 30];
        genes[10..20].fill(0.25);
        genes[20..].fill(0.625);
        let x = at(&genes);
        let (f, violation) = problem.evaluate(&x);
        assert_close(&f, &[0.125, 0.25, 0.625]);
        assert_eq!(violation, 0.0);
        assert_close(
            problem.constraints(&x).inequalities(),
            &[-0.125, -0.625, -0.625],
        );
        // a mean, not a sum: (1/16, 1/8, 1/4) breaks all three constraints, by 1 − 1/4 − 1/4,
        // 1 − 1/4 − 1/2 and 1 − 1/2 − 3/16
        genes.iter_mut().for_each(|x| *x /= 2.0);
        genes[20..].fill(0.25);
        let (f, violation) = problem.evaluate(&at(&genes));
        assert_close(&f, &[0.0625, 0.125, 0.25]);
        assert_close(&[violation], &[0.5 + 0.25 + 0.3125]);
        // with 4 objectives, the least two of f₁…f₃ count: (0.3, 0.1, 0.2, 0.3) gives
        // 1 − 0.6 − 0.3 = 0.1
        let f = [0.3, 0.1, 0.2, 0.3];
        assert_close(&Dtlz8::<4>::values(&f)[3..], &[1.0 - 0.6 - 0.3]);

        // DTLZ9, M = 3, n = 30: a block of 10 genes at 1 sums to 10; at 2⁻¹⁰ each gives 0.5
        let problem = Dtlz9::<3>::default();
        let mut genes = vec![1.0; 30];
        genes[..10].fill(0.5f64.powi(10));
        let x = at(&genes);
        let (f, violation) = problem.evaluate(&x);
        assert_close(&f, &[5.0, 10.0, 10.0]);
        assert_eq!(violation, 0.0);
        assert_close(problem.constraints(&x).inequalities(), &[-124.0, -199.0]);
        // all at 0: the origin, breaking both constraints by 1
        let (f, violation) = problem.evaluate(&at(&[0.0; 30]));
        assert_eq!(f, [0.0; 3]);
        assert_eq!(violation, 2.0);
    }

    // a genome of DTLZ8 or DTLZ9 whose blocks give the objectives f: DTLZ8 with each block at its
    // mean, DTLZ9 with each variable at (fⱼ / its block's size)^10
    fn dtlz8_genome<const M: usize>(f: &[f64; M], n: usize) -> Reals {
        let mut genes = vec![0.0; n];
        for (j, &fj) in f.iter().enumerate() {
            genes[block(j, n, M)].fill(fj);
        }
        Reals::from(genes)
    }

    fn dtlz9_genome<const M: usize>(f: &[f64; M], n: usize) -> Reals {
        let mut genes = vec![0.0; n];
        for (j, &fj) in f.iter().enumerate() {
            let range = block(j, n, M);
            let size = range.len() as f64;
            genes[range].fill(math::powi(fj / size, 10));
        }
        Reals::from(genes)
    }

    // the fronts: on the curves derived, feasible, reached by genomes, and mutually non-dominated
    #[test]
    fn the_fronts_lie_where_derived() {
        fn dtlz8<const M: usize>() {
            let problem = Dtlz8::<M>::default();
            let front = problem.optimal_front(300).expect("known");
            assert!(front.len() >= 300);
            let scores: Vec<Scores<M>> = front.iter().map(|f| Scores::new(*f)).collect();
            assert_eq!(non_dominated_sort(&scores, &[Minimize; M]).len(), 1);
            for f in &front {
                let values = Dtlz8::<M>::values(f);
                assert!(values.iter().all(|&g| g <= 1e-12), "{f:?}: {values:?}");
                // on the line where the first M − 1 constraints are tight, or on the last
                // constraint's plane
                let on_line = values[..M - 1].iter().all(|g| g.abs() < 1e-12);
                assert!(on_line || values[M - 1].abs() < 1e-12, "{f:?}");
                let (genome_f, violation) = problem.evaluate(&dtlz8_genome(f, 10 * M));
                assert_close(&genome_f, f);
                assert!(violation < 1e-12);
            }
            let ideal = problem.ideal_point().expect("known");
            let nadir = problem.nadir_point().expect("known");
            for j in 0..M {
                let low = front.iter().map(|f| f[j]).fold(f64::INFINITY, f64::min);
                let high = front.iter().map(|f| f[j]).fold(f64::NEG_INFINITY, f64::max);
                assert_eq!((low, high), (ideal[j], nadir[j]));
            }
        }
        dtlz8::<3>();
        dtlz8::<4>();
        dtlz8::<5>();
        // the line's end and the triangle's corners, for 3 objectives
        let front = Dtlz8::<3>::grid(6);
        for corner in [
            [1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0],
            [0.25, 0.75, 0.0],
            [0.75, 0.25, 0.0],
        ] {
            assert!(
                front
                    .iter()
                    .any(|f| (0..3).all(|j| (f[j] - corner[j]).abs() < 1e-15))
            );
        }
        // k = 1: the line's start, three points of the first row and the junction for f₁ the
        // least, and one more for f₂ the least
        assert_eq!(Dtlz8::<3>::grid(1).len(), 1 + 3 + 1);

        let problem = Dtlz9::<3>::default();
        let front = problem.optimal_front(100).expect("known");
        assert_eq!(front.len(), 100);
        for f in &front {
            assert!((f[0] * f[0] + f[2] * f[2] - 1.0).abs() < 1e-15 && f[0] == f[1]);
            let (genome_f, violation) = problem.evaluate(&dtlz9_genome(f, 30));
            assert!((0..3).all(|j| (genome_f[j] - f[j]).abs() < 1e-12));
            assert!(violation < 1e-12);
        }
        assert_eq!(front[0], [1.0, 1.0, 0.0]);
        assert_eq!(Dtlz9::<2>::default().optimal_front(7).unwrap().len(), 7);
    }

    // no feasible solution is better than the front: random objective vectors (the objectives
    // are independent, and any vector in the box is reached), every feasible one weakly dominated
    // by a point of the front up to its spacing, and dominating none
    #[test]
    fn random_solutions_agree_with_the_fronts() {
        fn check<const M: usize>(
            front: &[[f64; M]],
            values: impl Fn(&[f64; M]) -> Vec<f64>,
            high: f64,
            tolerance: f64,
        ) {
            let mut rng = StreamRng::seed_from_u64(3);
            let real = Real::uniform(M, 0.0..=high).expect("valid bounds");
            let mut feasible = 0;
            for _ in 0..20_000 {
                let genome = real.random_genome(&mut rng);
                let f: [f64; M] = std::array::from_fn(|j| genome[j]);
                if values(&f).iter().any(|&g| g > 0.0) {
                    continue;
                }
                feasible += 1;
                assert!(
                    front
                        .iter()
                        .any(|q| (0..M).all(|j| q[j] <= f[j] + tolerance)),
                    "{f:?}"
                );
                for q in front {
                    assert!(
                        !(0..M).all(|j| f[j] < q[j] - 1e-12),
                        "{f:?} dominates {q:?}"
                    );
                }
            }
            assert!(feasible > 1_000, "{feasible}");
        }
        check(
            &Dtlz8::<3>::front(5_000),
            |f| Dtlz8::<3>::values(f).to_vec(),
            1.0,
            0.01,
        );
        check(
            &Dtlz8::<4>::front(5_000),
            |f| Dtlz8::<4>::values(f).to_vec(),
            1.0,
            0.02,
        );
        check(&Dtlz9::<3>::front(2_000), Dtlz9::<3>::values, 1.5, 0.01);
        check(&Dtlz9::<2>::front(2_000), Dtlz9::<2>::values, 1.5, 0.01);
    }

    #[test]
    fn sizes() {
        assert_eq!(Dtlz8::<5>::default().variables(), 50);
        assert_eq!(Dtlz8::<3>::default().constraint_count(), 3);
        assert_eq!(Dtlz9::<3>::default().constraint_count(), 2);
        assert_eq!(Dtlz9::<2>::new(7).representation().bounds().len(), 7);
    }

    #[test]
    #[should_panic(expected = "at least 3 objectives")]
    fn dtlz8_with_two_objectives_panics() {
        let _ = Dtlz8::<2>::default();
    }
}
