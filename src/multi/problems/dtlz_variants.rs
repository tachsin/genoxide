//! Deb and Jain's variants of DTLZ1 and DTLZ2: a convex front, differently scaled objectives and
//! an inverted front.
//!
//! - **Convex DTLZ2 and scaled DTLZ1 and DTLZ2:** Deb, K. and Jain, H. (2014). An evolutionary
//!   many-objective optimization algorithm using reference-point-based nondominated sorting
//!   approach, part I: solving problems with box constraints. *IEEE Transactions on Evolutionary
//!   Computation* 18(4): 577-601, sections V-C and V-D, eq. 8 and table VIII.
//! - **Inverted DTLZ1:** Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization
//!   algorithm using reference-point based nondominated sorting approach, part II: handling
//!   constraints and extending to an adaptive approach. *IEEE Transactions on Evolutionary
//!   Computation* 18(4): 602-622, section VIII-A, eq. 9.
//!
//! Both were read in the authors' copies of the accepted versions
//! (<https://www.egr.msu.edu/~kdeb/papers/k2012009.pdf> and
//! <https://www.egr.msu.edu/~kdeb/papers/k2012010.pdf>); the typeset journal versions weren't
//! compared. Each problem is the DTLZ problem it's built on, with its objectives transformed after
//! they're computed, and the same variables: M + 4 for DTLZ1 and M + 9 for DTLZ2 by default, as
//! part I (section V) uses them.

use super::MultiProblem;
use super::dtlz::{Dtlz1, Dtlz2, rastrigin_g, simplex_points, spherical_front};
use crate::genome::{Real, Reals};
use crate::multi::MultiFitnessFunction;

const PART_ONE: &str = "Deb, K. and Jain, H. (2014). An evolutionary many-objective optimization \
                        algorithm using reference-point-based nondominated sorting approach, part \
                        I: solving problems with box constraints. IEEE Transactions on \
                        Evolutionary Computation 18(4): 577-601.";
const PART_ONE_URL: &str = "https://doi.org/10.1109/TEVC.2013.2281535";

const PART_TWO: &str = "Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization \
                        algorithm using reference-point based nondominated sorting approach, part \
                        II: handling constraints and extending to an adaptive approach. IEEE \
                        Transactions on Evolutionary Computation 18(4): 602-622.";
const PART_TWO_URL: &str = "https://doi.org/10.1109/TEVC.2013.2281534";

// the sizes and metadata that every variant shares with the DTLZ problem it's built on
macro_rules! variant {
    ($name:ident, $label:literal, $reference:expr, $url:expr) => {
        impl<const M: usize> $name<M> {
            /// The number of variables.
            pub fn variables(&self) -> usize {
                self.variables
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
                Some(self.front(points))
            }

            fn ideal_point(&self) -> Option<[f64; M]> {
                Some([0.0; M])
            }

            fn nadir_point(&self) -> Option<[f64; M]> {
                Some(self.nadir())
            }
        }
    };
}

// the checks of every variant's constructor, as DTLZ's
fn checked<const M: usize>(label: &str, variables: usize) -> usize {
    assert!(M >= 2, "{label} needs at least 2 objectives");
    assert!(
        variables >= M,
        "{label} needs at least as many variables as objectives"
    );
    variables
}

// ---- convex DTLZ2 --------------------------------------------------------------------------------

/// Convex DTLZ2 (Deb and Jain, 2014, part I, section V-D): DTLZ2 with its objectives mapped to
/// make the front convex, `fᵢ ← fᵢ⁴` for i from 1 to M − 1 and `f_M ← f_M²`.
///
/// With `M` objectives and `n` variables in [0, 1] (M + 9 by default, DTLZ2's k = 10). The
/// optimal solutions are DTLZ2's, the last `n − M + 1` variables at 0.5, and the front is
/// `f_M + Σᵢ₌₁^{M−1} √fᵢ = 1` (the paper's eq. 8): almost flat near the edges and steep in
/// between, so that evenly spread reference directions meet it unevenly. Its ideal point is the
/// origin and its nadir point (1, …, 1). [`optimal_front`](MultiProblem::optimal_front) maps
/// DTLZ2's front of Das and Dennis's points, so the points bunch up where the front is flat.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConvexDtlz2<const M: usize> {
    variables: usize,
}

impl<const M: usize> ConvexDtlz2<M> {
    /// The problem with `variables` variables, at least `M`.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives, or fewer variables than objectives.
    pub fn new(variables: usize) -> Self {
        let variables = checked::<M>("Convex DTLZ2", variables);
        Self { variables }
    }

    fn convex(f: [f64; M]) -> [f64; M] {
        std::array::from_fn(|i| {
            let square = f[i] * f[i];
            if i + 1 < M { square * square } else { square }
        })
    }

    fn front(&self, points: usize) -> Vec<[f64; M]> {
        spherical_front::<M>(points)
            .into_iter()
            .map(Self::convex)
            .collect()
    }

    fn nadir(&self) -> [f64; M] {
        [1.0; M]
    }
}

impl<const M: usize> Default for ConvexDtlz2<M> {
    /// The problem with DTLZ2's `M + 9` variables.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives.
    fn default() -> Self {
        Self::new(M + 9)
    }
}

variant!(ConvexDtlz2, "Convex DTLZ2", PART_ONE, PART_ONE_URL);

impl<const M: usize> MultiFitnessFunction<Reals, M> for ConvexDtlz2<M> {
    type Output = [f64; M];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than `M − 1` genes.
    fn evaluate(&self, x: &Reals) -> [f64; M] {
        Self::convex(Dtlz2::<M>::new(self.variables).evaluate(x))
    }
}

// ---- scaled DTLZ1 and DTLZ2 ----------------------------------------------------------------------

// the paper's scaling factors (part I, table VIII) by number of objectives, for DTLZ1 and DTLZ2
const DTLZ1_FACTORS: [(usize, f64); 5] = [(3, 10.0), (5, 10.0), (8, 3.0), (10, 2.0), (15, 1.2)];
const DTLZ2_FACTORS: [(usize, f64); 5] = [(3, 10.0), (5, 10.0), (8, 3.0), (10, 3.0), (15, 2.0)];

// the factor of the table for `objectives`, or of the next larger number of objectives in it, or
// of 15 objectives beyond
fn factor_for(table: &[(usize, f64)], objectives: usize) -> f64 {
    let (_, last) = table[table.len() - 1];
    table
        .iter()
        .find(|(m, _)| *m >= objectives)
        .map_or(last, |(_, factor)| *factor)
}

// the objectives scaled by 1, factor, factor², … in turn
fn scaled<const M: usize>(f: [f64; M], factor: f64) -> [f64; M] {
    let mut scale = 1.0;
    f.map(|value| {
        let value = value * scale;
        scale *= factor;
        value
    })
}

macro_rules! scaled_dtlz {
    ($(#[$doc:meta])* $name:ident, $base:ident, $label:literal, $k:literal, $table:ident,
     $radius:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name<const M: usize> {
            variables: usize,
            factor: f64,
        }

        impl<const M: usize> $name<M> {
            /// The problem with `variables` variables, at least `M`, and the paper's scaling
            /// factor for `M` objectives.
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives, or fewer variables than objectives.
            pub fn new(variables: usize) -> Self {
                let variables = checked::<M>($label, variables);
                Self {
                    variables,
                    factor: Self::paper_factor(),
                }
            }

            /// The problem with the objective i multiplied by `factor^(i − 1)` instead.
            ///
            /// # Panics
            ///
            /// If `factor` isn't finite and positive.
            #[must_use]
            pub fn with_factor(self, factor: f64) -> Self {
                assert!(
                    factor.is_finite() && factor > 0.0,
                    "{}'s scaling factor is finite and positive, not {factor}",
                    $label
                );
                Self { factor, ..self }
            }

            /// The scaling factor: objective i is multiplied by `factor^(i − 1)`.
            pub fn factor(&self) -> f64 {
                self.factor
            }

            /// The scaling factor of the paper's table VIII for `M` objectives (3, 5, 8, 10 and
            /// 15), or for the next larger number of objectives in the table, or for 15 beyond.
            pub fn paper_factor() -> f64 {
                factor_for(&$table, M)
            }

            fn front(&self, points: usize) -> Vec<[f64; M]> {
                $base::<M>::default()
                    .optimal_front(points)
                    .expect("known")
                    .into_iter()
                    .map(|point| scaled(point, self.factor))
                    .collect()
            }

            fn nadir(&self) -> [f64; M] {
                scaled([$radius; M], self.factor)
            }
        }

        impl<const M: usize> Default for $name<M> {
            #[doc = concat!("The problem with `M + ", stringify!($k), " − 1` variables and the paper's scaling factor for `M` objectives.")]
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives.
            fn default() -> Self {
                Self::new(M + $k - 1)
            }
        }

        variant!($name, $label, PART_ONE, PART_ONE_URL);

        impl<const M: usize> MultiFitnessFunction<Reals, M> for $name<M> {
            type Output = [f64; M];

            /// The objective values of `x`.
            ///
            /// # Panics
            ///
            /// If `x` has fewer than `M − 1` genes.
            fn evaluate(&self, x: &Reals) -> [f64; M] {
                scaled($base::<M>::new(self.variables).evaluate(x), self.factor)
            }
        }
    };
}

scaled_dtlz!(
    /// Scaled DTLZ1 (Deb and Jain, 2014, part I, section V-C): DTLZ1 with the objective i
    /// multiplied by `s^(i − 1)`, so that the objectives have ranges of different orders of
    /// magnitude.
    ///
    /// With `M` objectives and `n` variables in [0, 1] (M + 4 by default, DTLZ1's k = 5). The
    /// optimal solutions are DTLZ1's, the last `n − M + 1` variables at 0.5, and the front is the
    /// plane `Σ fᵢ / s^(i − 1) = 1/2`, from the origin to the nadir point
    /// (1/2, s/2, …, s^(M−1)/2).
    ///
    /// The factor s is the paper's for `M` objectives (table VIII): 10 for 3 and 5 objectives, 3
    /// for 8, 2 for 10 and 1.2 for 15. The paper doesn't use other numbers of objectives; for
    /// them, [`new`](Self::new) takes the factor of the next larger number of objectives in the
    /// table (10 for 2 and 4, 3 for 6 and 7, …), and 1.2 beyond 15, and
    /// [`with_factor`](Self::with_factor) sets another. The paper's text says "a factor 10ⁱ"
    /// and its example multiplies f₁, f₂ and f₃ by 10⁰, 10¹ and 10², as its figures 24-29 show,
    /// while the captions of tables VII and VIII read "10ⁱ, i = 1, 2, …, M": genoxide follows the
    /// example and the figures, `s^(i − 1)`.
    ScaledDtlz1,
    Dtlz1,
    "Scaled DTLZ1",
    5,
    DTLZ1_FACTORS,
    0.5
);

scaled_dtlz!(
    /// Scaled DTLZ2 (Deb and Jain, 2014, part I, section V-C): DTLZ2 with the objective i
    /// multiplied by `s^(i − 1)`.
    ///
    /// With `M` objectives and `n` variables in [0, 1] (M + 9 by default, DTLZ2's k = 10). The
    /// optimal solutions are DTLZ2's, the last `n − M + 1` variables at 0.5, and the front is the
    /// ellipsoid `Σ (fᵢ / s^(i − 1))² = 1` in the positive orthant, from the origin to the nadir
    /// point (1, s, …, s^(M−1)).
    ///
    /// The factor s is the paper's for `M` objectives (table VIII): 10 for 3 and 5 objectives, 3
    /// for 8 and 10, and 2 for 15, which differ from scaled DTLZ1's for 10 and 15 objectives. For
    /// other numbers of objectives, [`new`](Self::new) takes the factor of the next larger number
    /// in the table, and 2 beyond 15; [`with_factor`](Self::with_factor) sets another. As for
    /// [`ScaledDtlz1`], the factors are `s^(i − 1)`, as the paper's example and figures have
    /// them.
    ScaledDtlz2,
    Dtlz2,
    "Scaled DTLZ2",
    10,
    DTLZ2_FACTORS,
    1.0
);

// ---- inverted DTLZ1 ------------------------------------------------------------------------------

/// Inverted DTLZ1 (Jain and Deb, 2014, part II, section VIII-A): DTLZ1 with each objective
/// replaced by `fᵢ ← 0.5 (1 + g) − fᵢ` (their eq. 9), g being DTLZ1's distance function.
///
/// With `M` objectives and `n` variables in [0, 1] (M + 4 by default, DTLZ1's k = 5, which the
/// paper uses as "the original formulation"). The optimal solutions are DTLZ1's, the last
/// `n − M + 1` variables at 0.5: the front is the simplex of DTLZ1's front turned upside down,
/// `Σ fᵢ = (M − 1)/2` with each fᵢ in [0, 1/2] (derived here: the paper plots it). Its corners
/// are the points with one objective at 0 and the others at 1/2, and each objective's minimum
/// has one solution, where DTLZ1's front had a corner at each maximum. Evenly spread reference
/// directions then miss most of it: the paper reports that NSGA-III finds 28 well spread points
/// for 91 directions with 3 objectives.
///
/// Its ideal point is the origin and its nadir point (1/2, …, 1/2). With 2 objectives, the
/// inversion changes nothing: the front is DTLZ1's, `f₁ + f₂ = 1/2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InvertedDtlz1<const M: usize> {
    variables: usize,
}

impl<const M: usize> InvertedDtlz1<M> {
    /// The problem with `variables` variables, at least `M`.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives, or fewer variables than objectives.
    pub fn new(variables: usize) -> Self {
        let variables = checked::<M>("Inverted DTLZ1", variables);
        Self { variables }
    }

    // Das and Dennis's points p, turned into 0.5 (1 − pᵢ)
    fn front(&self, points: usize) -> Vec<[f64; M]> {
        simplex_points::<M>(points)
            .into_iter()
            .map(|p| p.map(|v| 0.5 * (1.0 - v)))
            .collect()
    }

    fn nadir(&self) -> [f64; M] {
        [0.5; M]
    }
}

impl<const M: usize> Default for InvertedDtlz1<M> {
    /// The problem with DTLZ1's `M + 4` variables.
    ///
    /// # Panics
    ///
    /// With fewer than 2 objectives.
    fn default() -> Self {
        Self::new(M + 4)
    }
}

variant!(InvertedDtlz1, "Inverted DTLZ1", PART_TWO, PART_TWO_URL);

impl<const M: usize> MultiFitnessFunction<Reals, M> for InvertedDtlz1<M> {
    type Output = [f64; M];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than `M − 1` genes.
    fn evaluate(&self, x: &Reals) -> [f64; M] {
        let top = 0.5 * (1.0 + rastrigin_g(&x[M - 1..]));
        Dtlz1::<M>::new(self.variables).evaluate(x).map(|f| top - f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective::Minimize;
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

    // the transformations of the papers at points of DTLZ1 and DTLZ2 whose values follow by hand
    #[test]
    fn values_match_the_papers() {
        // DTLZ1 (M = 3) at (0.5, …, 0.5) is (0.125, 0.125, 0.25), g = 0: scaled by 10⁰, 10¹, 10²
        // (part I, section V-C), and inverted to 0.5 − f (part II, eq. 9)
        let x = at(&[0.5; 7]);
        assert_close(
            &ScaledDtlz1::<3>::default().evaluate(&x),
            &[0.125, 1.25, 25.0],
        );
        assert_close(
            &InvertedDtlz1::<3>::default().evaluate(&x),
            &[0.375, 0.375, 0.25],
        );
        // with the distance variables at 1, g = 125 and DTLZ1 is 126 times the above; the
        // inversion subtracts from 0.5 × 126 = 63
        let x = at(&[0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_close(
            &InvertedDtlz1::<3>::default().evaluate(&x),
            &[63.0 - 15.75, 63.0 - 15.75, 63.0 - 31.5],
        );
        assert_close(
            &ScaledDtlz1::<3>::default().evaluate(&x),
            &[15.75, 157.5, 3150.0],
        );
        // DTLZ2 (M = 3) at x₁ = x₂ = 0.5 on the front: (1/2, 1/2, 1/√2); convex: the first two to
        // the fourth power, the last squared (part I, section V-D)
        let mut x = vec![0.5; 12];
        let f = ConvexDtlz2::<3>::default().evaluate(&at(&x));
        assert_close(&f, &[0.0625, 0.0625, 0.5]);
        // on the front, f₃ + √f₁ + √f₂ = 1 (eq. 8)
        assert!((f[2] + f[0].sqrt() + f[1].sqrt() - 1.0).abs() < 1e-15);
        assert_close(
            &ScaledDtlz2::<3>::default().evaluate(&at(&x)),
            &[0.5, 5.0, 100.0 * 0.5f64.sqrt()],
        );
        // the distance variables at 1: radius 1 + 10 × 0.25 = 3.5 at the corner x₁ = x₂ = 0
        x[..2].copy_from_slice(&[0.0, 0.0]);
        x[2..].fill(1.0);
        assert_close(
            &ConvexDtlz2::<3>::default().evaluate(&at(&x)),
            &[3.5f64.powi(4), 0.0, 0.0],
        );
        assert_close(
            &ScaledDtlz2::<3>::default().evaluate(&at(&x)),
            &[3.5, 0.0, 0.0],
        );
    }

    // table VIII's factors, and the rule for the numbers of objectives it doesn't have
    #[test]
    fn scaling_factors_are_the_papers() {
        assert_eq!(ScaledDtlz1::<3>::paper_factor(), 10.0);
        assert_eq!(ScaledDtlz1::<5>::paper_factor(), 10.0);
        assert_eq!(ScaledDtlz1::<8>::paper_factor(), 3.0);
        assert_eq!(ScaledDtlz1::<10>::paper_factor(), 2.0);
        assert_eq!(ScaledDtlz1::<15>::paper_factor(), 1.2);
        assert_eq!(ScaledDtlz2::<3>::paper_factor(), 10.0);
        assert_eq!(ScaledDtlz2::<5>::paper_factor(), 10.0);
        assert_eq!(ScaledDtlz2::<8>::paper_factor(), 3.0);
        assert_eq!(ScaledDtlz2::<10>::paper_factor(), 3.0);
        assert_eq!(ScaledDtlz2::<15>::paper_factor(), 2.0);
        // not in the table: the next larger number of objectives, and 15's beyond
        assert_eq!(ScaledDtlz1::<2>::paper_factor(), 10.0);
        assert_eq!(ScaledDtlz1::<4>::paper_factor(), 10.0);
        assert_eq!(ScaledDtlz1::<6>::paper_factor(), 3.0);
        assert_eq!(ScaledDtlz1::<9>::paper_factor(), 2.0);
        assert_eq!(ScaledDtlz2::<9>::paper_factor(), 3.0);
        assert_eq!(ScaledDtlz2::<20>::paper_factor(), 2.0);
        let problem = ScaledDtlz2::<3>::default().with_factor(2.0);
        assert_eq!(problem.factor(), 2.0);
        assert_close(&problem.nadir_point().expect("known"), &[1.0, 2.0, 4.0]);
        assert_close(
            &ScaledDtlz1::<4>::default().nadir_point().expect("known"),
            &[0.5, 5.0, 50.0, 500.0],
        );
    }

    #[test]
    #[should_panic(expected = "finite and positive")]
    fn a_scaling_factor_is_positive() {
        let _ = ScaledDtlz1::<3>::default().with_factor(0.0);
    }

    // the optimal solutions of DTLZ1 and DTLZ2 (the distance variables at 0.5) land on each
    // variant's front, and the fronts satisfy their identities
    #[test]
    fn optimal_solutions_lie_on_the_fronts() {
        let mut x1 = vec![0.5; 7];
        let mut x2 = vec![0.5; 12];
        for (a, b) in [(0.0, 0.3), (0.2, 0.9), (0.7, 0.1), (1.0, 1.0)] {
            x1[..2].copy_from_slice(&[a, b]);
            x2[..2].copy_from_slice(&[a, b]);
            let f = ScaledDtlz1::<3>::default().evaluate(&at(&x1));
            assert!((f[0] + f[1] / 10.0 + f[2] / 100.0 - 0.5).abs() < 1e-12);
            let f = InvertedDtlz1::<3>::default().evaluate(&at(&x1));
            assert!((f.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(f.iter().all(|v| (-1e-15..=0.5 + 1e-15).contains(v)));
            let f = ScaledDtlz2::<3>::default().evaluate(&at(&x2));
            let norm = f[0] * f[0] + f[1] * f[1] / 100.0 + f[2] * f[2] / 10_000.0;
            assert!((norm - 1.0).abs() < 1e-12);
            let f = ConvexDtlz2::<3>::default().evaluate(&at(&x2));
            assert!((f[2] + f[0].sqrt() + f[1].sqrt() - 1.0).abs() < 1e-12);
        }
        check_fronts::<2>();
        check_fronts::<3>();
        check_fronts::<5>();
    }

    // each variant's front, from at least 20 points: mutually non-dominated, and on its identity
    fn check_fronts<const M: usize>() {
        let check = |front: Vec<[f64; M]>, identity: &dyn Fn(&[f64; M]) -> f64| {
            assert!(front.len() >= 20);
            let scores: Vec<Scores<M>> = front.iter().map(|p| Scores::new(*p)).collect();
            assert_eq!(non_dominated_sort(&scores, &[Minimize; M]).len(), 1);
            for p in &front {
                assert!(identity(p).abs() < 1e-12, "{p:?}");
            }
        };
        let scale = |i: usize| 10f64.powi(i as i32);
        check(ConvexDtlz2::<M>::default().front(20), &|p| {
            p[M - 1] + p[..M - 1].iter().map(|v| v.sqrt()).sum::<f64>() - 1.0
        });
        check(
            ScaledDtlz1::<M>::default().with_factor(10.0).front(20),
            &|p| p.iter().enumerate().map(|(i, v)| v / scale(i)).sum::<f64>() - 0.5,
        );
        check(
            ScaledDtlz2::<M>::default().with_factor(10.0).front(20),
            &|p| {
                let norm: f64 = p
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v / scale(i)).powi(2))
                    .sum();
                norm - 1.0
            },
        );
        check(InvertedDtlz1::<M>::default().front(20), &|p| {
            p.iter().sum::<f64>() - 0.5 * (M as f64 - 1.0)
        });
    }

    #[test]
    fn sizes_and_names() {
        assert_eq!(ConvexDtlz2::<3>::default().variables(), 12);
        assert_eq!(ScaledDtlz1::<3>::default().variables(), 7);
        assert_eq!(ScaledDtlz2::<5>::default().variables(), 14);
        assert_eq!(InvertedDtlz1::<4>::default().variables(), 8);
        assert_eq!(
            InvertedDtlz1::<3>::new(10).representation().bounds().len(),
            10
        );
        assert_eq!(ConvexDtlz2::<3>::default().name(), "Convex DTLZ2");
        assert_eq!(
            InvertedDtlz1::<3>::default().reference_url(),
            Some(PART_TWO_URL)
        );
        // the corners of the inverted front: one objective 0, the others 1/2
        let front = InvertedDtlz1::<3>::default()
            .optimal_front(3)
            .expect("known");
        assert!(front.contains(&[0.5, 0.5, 0.0]));
        assert!(front.contains(&[0.0, 0.5, 0.5]));
    }
}
