//! The constrained problems of the CEC 2006 special session, g01 to g06.
//!
//! Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello
//! Coello, C. A. and Deb, K. (2006). *Problem Definitions and Evaluation Criteria for the CEC 2006
//! Special Session on Constrained Real-Parameter Optimization.* Technical report, Nanyang
//! Technological University, Singapore, 18 September 2006. The definitions, bounds, solutions and
//! values are the report's (pages 3-4 and table 4). Each problem's docs name the report's source
//! for it; those sources are still to be read
//! ([#168](https://github.com/tachsin/genoxide/issues/168)).
//!
//! Every problem is minimized. Its fitness is `(f(x), violation)`, the violation being
//! `Σ max(0, gᵢ(x)) + Σ max(0, |hⱼ(x)| − δ)`. The report counts an equality as met when
//! `|hⱼ(x)| ≤ δ`, with δ = [`EQUALITY_TOLERANCE`]; `with_tolerance` changes it. Problems that the
//! sources maximize (g02, g03) are negated, as in the report.

// the report's solutions keep every digit it prints
#![allow(clippy::excessive_precision)]

use super::{Constraints, Optimum, Problem};
use crate::engine::FitnessFunction;
use crate::genome::{Real, Reals};

/// The report's tolerance δ of an equality constraint: `|h(x)| ≤ 0.0001` counts as met.
pub const EQUALITY_TOLERANCE: f64 = 0.0001;

const REPORT: &str = "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, \
                      P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and \
                      Evaluation Criteria for the CEC 2006 Special Session on Constrained \
                      Real-Parameter Optimization. Technical report, Nanyang Technological \
                      University, Singapore.";
const REPORT_URL: &str = "https://github.com/P-N-Suganthan/CEC2006";

// bounds that are valid by construction
fn bounds(ranges: &[(f64, f64)]) -> Real {
    Real::new(ranges.iter().map(|&(low, high)| low..=high)).expect("valid bounds")
}

fn reals(values: &[f64]) -> Reals {
    Reals::from(values.to_vec())
}

// the fitness and metadata shared by the problems; `$tolerance` is the equality tolerance
macro_rules! cec2006 {
    ($name:ident, |$problem:ident| $tolerance:expr) => {
        impl FitnessFunction<Reals> for $name {
            type Output = (f64, f64);

            /// The value of `x` and its constraint violation, 0 when it's feasible.
            ///
            /// # Panics
            ///
            /// If `x` has fewer genes than the problem's dimensions.
            fn evaluate(&self, x: &Reals) -> (f64, f64) {
                let $problem = self;
                (self.value(x), self.constraints(x).violation($tolerance))
            }
        }
    };
}

// ---- g01 -----------------------------------------------------------------------------------------

/// g01: `5 Σᵢ₌₁⁴ xᵢ − 5 Σᵢ₌₁⁴ xᵢ² − Σᵢ₌₅¹³ xᵢ`, a quadratic in 13 dimensions with 9 linear
/// inequalities.
///
/// Bounds x₁…x₉ ∈ [0, 1], x₁₀…x₁₂ ∈ [0, 100], x₁₃ ∈ [0, 1]; minimum −15 at
/// (1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1), with g₁, g₂, g₃, g₇, g₈ and g₉ active.
///
/// The report's eqs. 4-5, after Floudas, C. A. and Pardalos, P. M. (1990). *A Collection of Test
/// Problems for Constrained Global Optimization Algorithms.* LNCS 455, Springer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G01;

impl G01 {
    fn value(&self, x: &Reals) -> f64 {
        let first: f64 = x[..4].iter().sum();
        let squares: f64 = x[..4].iter().map(|xi| xi * xi).sum();
        let rest: f64 = x[4..13].iter().sum();
        5.0 * first - 5.0 * squares - rest
    }
}

cec2006!(G01, |_problem| 0.0);

impl Problem for G01 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G01"
    }

    fn representation(&self) -> Real {
        let mut ranges = vec![(0.0, 1.0); 9];
        ranges.extend([(0.0, 100.0); 3]);
        ranges.push((0.0, 1.0));
        bounds(&ranges)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        let mut solution = vec![1.0; 9];
        solution.extend([3.0, 3.0, 3.0, 1.0]);
        Some(Optimum::proven(-15.0, vec![Reals::from(solution)]))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g₁…g₉ of the report's eq. 5.
    fn constraints(&self, x: &Reals) -> Constraints {
        let x = |i: usize| x[i - 1];
        Constraints::new(
            vec![
                2.0 * x(1) + 2.0 * x(2) + x(10) + x(11) - 10.0,
                2.0 * x(1) + 2.0 * x(3) + x(10) + x(12) - 10.0,
                2.0 * x(2) + 2.0 * x(3) + x(11) + x(12) - 10.0,
                -8.0 * x(1) + x(10),
                -8.0 * x(2) + x(11),
                -8.0 * x(3) + x(12),
                -2.0 * x(4) - x(5) + x(10),
                -2.0 * x(6) - x(7) + x(11),
                -2.0 * x(8) - x(9) + x(12),
            ],
            Vec::new(),
        )
    }
}

// ---- g02 -----------------------------------------------------------------------------------------

// g02's lower bound: the report's bounds are open at 0, and this is the smallest positive value
// whose square doesn't underflow, so the division stays finite at the corner of the box
const G02_LOW: f64 = 1.5e-154;

/// g02: `−|Σ cos⁴ xᵢ − 2 Π cos² xᵢ| / √(Σ i xᵢ²)` in 20 dimensions, subject to `Π xᵢ ≥ 0.75` and
/// `Σ xᵢ ≤ 7.5n`: a maximization, negated, with a rugged landscape.
///
/// Bounds (0, 10]²⁰, closed here at 1.5e-154, the smallest positive value whose square doesn't
/// underflow; best known −0.80361910412559 at the report's x*, with g₁ nearly active. Not proven
/// optimal.
///
/// The report's eqs. 6-7, after Koziel, S. and Michalewicz, Z. (1999). Evolutionary algorithms,
/// homomorphous mappings, and constrained parameter optimization. *Evolutionary Computation*
/// 7(1): 19-44.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G02;

const G02_DIMENSIONS: usize = 20;

impl G02 {
    fn value(&self, x: &Reals) -> f64 {
        let x = &x[..G02_DIMENSIONS];
        let fourth: f64 = x.iter().map(|xi| xi.cos().powi(4)).sum();
        let product: f64 = x.iter().map(|xi| xi.cos().powi(2)).product();
        let weighted: f64 = x
            .iter()
            .enumerate()
            .map(|(i, xi)| (i + 1) as f64 * xi * xi)
            .sum();
        -((fourth - 2.0 * product) / weighted.sqrt()).abs()
    }
}

cec2006!(G02, |_problem| 0.0);

impl Problem for G02 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G02"
    }

    fn representation(&self) -> Real {
        bounds(&[(G02_LOW, 10.0); G02_DIMENSIONS])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::best_known(
            -0.803_619_104_125_59,
            vec![reals(&[
                3.162_460_615_721_85,
                3.128_331_428_129_67,
                3.094_792_129_887_91,
                3.061_450_595_234_69,
                3.027_929_158_855_55,
                2.993_826_067_017_30,
                2.958_668_717_652_85,
                2.921_842_273_124_50,
                0.494_825_114_569_33,
                0.488_357_110_054_90,
                0.482_316_427_118_65,
                0.476_644_750_927_42,
                0.471_295_508_354_93,
                0.466_230_992_641_67,
                0.461_420_049_841_99,
                0.456_836_647_672_17,
                0.452_458_769_032_67,
                0.448_267_622_418_53,
                0.444_247_009_587_60,
                0.440_382_859_563_17,
            ])],
        ))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g₁ = 0.75 − Π xᵢ and g₂ = Σ xᵢ − 7.5n, the report's eq. 7.
    fn constraints(&self, x: &Reals) -> Constraints {
        let x = &x[..G02_DIMENSIONS];
        let product: f64 = x.iter().product();
        let sum: f64 = x.iter().sum();
        Constraints::new(
            vec![0.75 - product, sum - 7.5 * G02_DIMENSIONS as f64],
            Vec::new(),
        )
    }
}

// ---- g03 -----------------------------------------------------------------------------------------

const G03_DIMENSIONS: usize = 10;

/// g03: `−(√n)ⁿ Π xᵢ` in 10 dimensions, subject to `Σ xᵢ² = 1`: a maximization, negated, on a
/// sphere.
///
/// Bounds [0, 1]¹⁰. With the equality met to within δ, the minimum is `−(1 + δ)ⁿᐟ²` at
/// xᵢ = √((1 + δ) / n), derived from the definition (the product is largest with equal genes, on
/// the outer edge of the tolerance): −1.0001⁵ = −1.00050010001 for the report's δ, the report's
/// f(x*). Without the tolerance it would be −1 at xᵢ = 1/√n.
///
/// The report's eqs. 8-9, after Michalewicz, Z., Nazhiyath, G. and Michalewicz, M. (1996). A note
/// on usefulness of geometrical crossover for numerical optimization problems. *Proceedings of the
/// 5th Annual Conference on Evolutionary Programming*: 305-312.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G03 {
    tolerance: f64,
}

impl G03 {
    /// The problem with an equality tolerance δ of `tolerance`, instead of the report's
    /// [`EQUALITY_TOLERANCE`].
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// The equality tolerance δ.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    fn value(&self, x: &Reals) -> f64 {
        let n = G03_DIMENSIONS as f64;
        let product: f64 = x[..G03_DIMENSIONS].iter().product();
        -n.sqrt().powi(G03_DIMENSIONS as i32) * product
    }
}

impl Default for G03 {
    /// The problem with the report's tolerance, [`EQUALITY_TOLERANCE`].
    fn default() -> Self {
        Self::with_tolerance(EQUALITY_TOLERANCE)
    }
}

cec2006!(G03, |problem| problem.tolerance);

impl Problem for G03 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G03"
    }

    fn representation(&self) -> Real {
        bounds(&[(0.0, 1.0); G03_DIMENSIONS])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        let radius = 1.0 + self.tolerance.max(0.0);
        let n = G03_DIMENSIONS as f64;
        Some(Optimum::proven(
            -radius.powf(n / 2.0),
            vec![Reals::from(vec![(radius / n).sqrt(); G03_DIMENSIONS])],
        ))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// h₁ = Σ xᵢ² − 1, the report's eq. 9.
    fn constraints(&self, x: &Reals) -> Constraints {
        let squares: f64 = x[..G03_DIMENSIONS].iter().map(|xi| xi * xi).sum();
        Constraints::new(Vec::new(), vec![squares - 1.0])
    }
}

// ---- g04 -----------------------------------------------------------------------------------------

/// g04: `5.3578547x₃² + 0.8356891x₁x₅ + 37.293239x₁ − 40792.141`, a quadratic in 5 dimensions
/// with 6 nonlinear inequalities (three quantities, each between two limits).
///
/// Bounds x₁ ∈ [78, 102], x₂ ∈ [33, 45], x₃…x₅ ∈ [27, 45]; minimum −30665.53867178332 at
/// (78, 33, 29.9952560256815985, 45, 36.7758129057882073), with g₁ and g₆ active.
///
/// The report's eqs. 10-11, after Himmelblau, D. M. (1972). *Applied Nonlinear Programming.*
/// McGraw-Hill. Some engineering papers use a variant with 0.00026 for g₁'s 0.0006262; this is the
/// report's form.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G04;

impl G04 {
    fn value(&self, x: &Reals) -> f64 {
        let (x1, x3, x5) = (x[0], x[2], x[4]);
        5.357_854_7 * x3 * x3 + 0.835_689_1 * x1 * x5 + 37.293_239 * x1 - 40_792.141
    }
}

cec2006!(G04, |_problem| 0.0);

impl Problem for G04 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G04"
    }

    fn representation(&self) -> Real {
        bounds(&[
            (78.0, 102.0),
            (33.0, 45.0),
            (27.0, 45.0),
            (27.0, 45.0),
            (27.0, 45.0),
        ])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(
            -30_665.538_671_783_32,
            vec![reals(&[
                78.0,
                33.0,
                29.995_256_025_681_598_5,
                45.0,
                36.775_812_905_788_207_3,
            ])],
        ))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g₁…g₆ of the report's eq. 11.
    fn constraints(&self, x: &Reals) -> Constraints {
        let (x1, x2, x3, x4, x5) = (x[0], x[1], x[2], x[3], x[4]);
        let u = 85.334_407 + 0.005_685_8 * x2 * x5 + 0.000_626_2 * x1 * x4 - 0.002_205_3 * x3 * x5;
        let v = 80.512_49 + 0.007_131_7 * x2 * x5 + 0.002_995_5 * x1 * x2 + 0.002_181_3 * x3 * x3;
        let w = 9.300_961 + 0.004_702_6 * x3 * x5 + 0.001_254_7 * x1 * x3 + 0.001_908_5 * x3 * x4;
        Constraints::new(
            vec![u - 92.0, -u, v - 110.0, -v + 90.0, w - 25.0, -w + 20.0],
            Vec::new(),
        )
    }
}

// ---- g05 -----------------------------------------------------------------------------------------

/// g05: `3x₁ + 0.000001x₁³ + 2x₂ + (0.000002/3)x₂³`, a cubic in 4 dimensions with 2 linear
/// inequalities and 3 nonlinear equalities.
///
/// Bounds x₁, x₂ ∈ [0, 1200], x₃, x₄ ∈ [−0.55, 0.55]; best known 5126.4967140071 at the report's
/// x*, which meets the equalities within the tolerance δ only. Not proven optimal.
///
/// The report's eqs. 12-13, after Hock, W. and Schittkowski, K. (1981). *Test Examples for
/// Nonlinear Programming Codes.* Lecture Notes in Economics and Mathematical Systems 187,
/// Springer; the best known solution is Koziel and Michalewicz's (1999).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G05 {
    tolerance: f64,
}

impl G05 {
    /// The problem with an equality tolerance δ of `tolerance`, instead of the report's
    /// [`EQUALITY_TOLERANCE`]. Its best known solution is the report's, for the report's δ only.
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// The equality tolerance δ.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    fn value(&self, x: &Reals) -> f64 {
        let (x1, x2) = (x[0], x[1]);
        3.0 * x1 + 0.000_001 * x1.powi(3) + 2.0 * x2 + (0.000_002 / 3.0) * x2.powi(3)
    }
}

impl Default for G05 {
    /// The problem with the report's tolerance, [`EQUALITY_TOLERANCE`].
    fn default() -> Self {
        Self::with_tolerance(EQUALITY_TOLERANCE)
    }
}

cec2006!(G05, |problem| problem.tolerance);

impl Problem for G05 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G05"
    }

    fn representation(&self) -> Real {
        bounds(&[(0.0, 1200.0), (0.0, 1200.0), (-0.55, 0.55), (-0.55, 0.55)])
    }

    /// The report's best known solution, for the report's tolerance; `None` for another.
    fn optimum(&self) -> Option<Optimum<Reals>> {
        (self.tolerance == EQUALITY_TOLERANCE).then(|| {
            Optimum::best_known(
                5_126.496_714_007_1,
                vec![reals(&[
                    679.945_148_297_028_709,
                    1_026.066_976_000_046_91,
                    0.118_876_369_094_410_433,
                    -0.396_233_485_215_178_26,
                ])],
            )
        })
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g₁, g₂, then h₃, h₄, h₅ of the report's eq. 13.
    fn constraints(&self, x: &Reals) -> Constraints {
        let (x1, x2, x3, x4) = (x[0], x[1], x[2], x[3]);
        Constraints::new(
            vec![-x4 + x3 - 0.55, -x3 + x4 - 0.55],
            vec![
                1000.0 * (-x3 - 0.25).sin() + 1000.0 * (-x4 - 0.25).sin() + 894.8 - x1,
                1000.0 * (x3 - 0.25).sin() + 1000.0 * (x3 - x4 - 0.25).sin() + 894.8 - x2,
                1000.0 * (x4 - 0.25).sin() + 1000.0 * (x4 - x3 - 0.25).sin() + 1294.8,
            ],
        )
    }
}

// ---- g06 -----------------------------------------------------------------------------------------

/// g06: `(x₁ − 10)³ + (x₂ − 20)³` in 2 dimensions, inside a thin crescent between two circles.
///
/// Bounds x₁ ∈ [13, 100], x₂ ∈ [0, 100]; minimum −6961.81387558015 at
/// (14.095, 0.8429607892154795668), where both constraints are active: x₁ = 14.095 exactly, where
/// the two circles meet.
///
/// The report's eqs. 14-15, after Floudas and Pardalos (1990).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G06;

impl G06 {
    fn value(&self, x: &Reals) -> f64 {
        (x[0] - 10.0).powi(3) + (x[1] - 20.0).powi(3)
    }
}

cec2006!(G06, |_problem| 0.0);

impl Problem for G06 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G06"
    }

    fn representation(&self) -> Real {
        bounds(&[(13.0, 100.0), (0.0, 100.0)])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(
            -6_961.813_875_580_15,
            vec![reals(&[14.095, 0.842_960_789_215_479_566_8])],
        ))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g₁ = −(x₁ − 5)² − (x₂ − 5)² + 100 and g₂ = (x₁ − 6)² + (x₂ − 5)² − 82.81, the report's
    /// eq. 15.
    fn constraints(&self, x: &Reals) -> Constraints {
        let (x1, x2) = (x[0], x[1]);
        Constraints::new(
            vec![
                -(x1 - 5.0).powi(2) - (x2 - 5.0).powi(2) + 100.0,
                (x1 - 6.0).powi(2) + (x2 - 5.0).powi(2) - 82.81,
            ],
            Vec::new(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Representation;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} is not {expected}"
        );
    }

    // each solution of the optimum is in the bounds, has the optimum's value to `tolerance`
    // (relative), and is feasible to within `slack`
    fn check_optimum<P>(problem: &P, tolerance: f64, slack: f64)
    where
        P: Problem<Representation = Real, Output = (f64, f64)>,
    {
        let optimum = problem.optimum().expect("known");
        for solution in optimum.solutions() {
            problem
                .representation()
                .validate(solution)
                .expect("in bounds");
            let (value, violation) = problem.evaluate(solution);
            assert_close(value, optimum.value(), tolerance);
            assert!(
                violation <= slack,
                "{}: violation {violation}",
                problem.name()
            );
        }
    }

    // the constraints that the report lists as active are within `slack` of 0
    fn assert_active(constraints: &Constraints, active: &[usize], slack: f64) {
        for &i in active {
            let value = constraints.inequalities()[i - 1];
            assert!(value.abs() <= slack, "g{i} = {value}");
        }
    }

    #[test]
    fn g01() {
        check_optimum(&G01, 1e-15, 0.0);
        let x = G01.optimum().expect("known").solutions()[0].clone();
        assert_active(&G01.constraints(&x), &[1, 2, 3, 7, 8, 9], 0.0);
        // at the origin: f = 0, and every constraint met (g₁…g₃ = −10, the others 0)
        let origin = Reals::from(vec![0.0; 13]);
        assert_eq!(G01.evaluate(&origin), (0.0, 0.0));
        assert_eq!(G01.constraints(&origin).len(), 9);
        // x₁₀ = 100 alone violates g₁, g₂ (by 90 each), g₄ and g₇ (by 100 each)
        let mut x = vec![0.0; 13];
        x[9] = 100.0;
        assert_eq!(G01.evaluate(&Reals::from(x)), (-100.0, 380.0));
        assert_eq!(G01.representation().bounds()[10], 0.0..=100.0);
    }

    #[test]
    fn g02() {
        check_optimum(&G02, 1e-12, 1e-12);
        let x = G02.optimum().expect("known").solutions()[0].clone();
        // g₁ is close to active
        assert!(G02.constraints(&x).inequalities()[0].abs() < 1e-10);
        // at xᵢ = π/2, every cosine is 0 (to rounding): f ≈ 0, and Π xᵢ ≥ 0.75
        let (value, violation) = G02.evaluate(&Reals::from(vec![std::f64::consts::FRAC_PI_2; 20]));
        assert!(value.abs() < 1e-12 && violation == 0.0);
        // at xᵢ = π: cos⁴ = cos² = 1, so |20 − 2| / √(π² · 210)
        let (value, _) = G02.evaluate(&Reals::from(vec![std::f64::consts::PI; 20]));
        assert_close(value, -18.0 / (std::f64::consts::PI * 210f64.sqrt()), 1e-12);
        // the corner of the box stays finite
        let (value, violation) = G02.evaluate(&Reals::from(vec![G02_LOW; 20]));
        assert!(value.is_finite() && violation > 0.0);
    }

    #[test]
    fn g03() {
        check_optimum(&G03::default(), 1e-15, 1e-12);
        // the report's f(x*), −1.00050010001000
        let optimum = G03::default().optimum().expect("known").value();
        assert_close(optimum, -1.000_500_100_010_00, 1e-14);
        // the report's x*, to its digits
        let report = [
            0.316_243_576_472_830_69,
            0.316_243_577_414_338_339,
            0.316_243_578_012_345_927,
            0.316_243_575_664_017_895,
            0.316_243_578_205_526_066,
            0.316_243_577_388_550_69,
            0.316_243_575_472_949_512,
            0.316_243_577_164_883_938,
            0.316_243_578_155_920_302,
            0.316_243_576_147_374_916,
        ];
        let (value, violation) = G03::default().evaluate(&reals(&report));
        assert_close(value, optimum, 1e-9);
        assert!(violation < 1e-9);
        // all genes 1/√10: the exact optimum without the tolerance, −1
        let unit = Reals::from(vec![0.1f64.sqrt(); 10]);
        let (value, violation) = G03::default().evaluate(&unit);
        assert_close(value, -1.0, 1e-14);
        assert_eq!(violation, 0.0);
        // the origin misses the equality by 1, less the tolerance
        let (_, violation) = G03::default().evaluate(&Reals::from(vec![0.0; 10]));
        assert_close(violation, 1.0 - EQUALITY_TOLERANCE, 1e-15);
        // without tolerance, the optimum is −1
        let exact = G03::with_tolerance(0.0);
        assert_eq!(exact.optimum().expect("known").value(), -1.0);
        assert_eq!(exact.tolerance(), 0.0);
    }

    #[test]
    fn g04() {
        check_optimum(&G04, 1e-13, 1e-9);
        let x = G04.optimum().expect("known").solutions()[0].clone();
        assert_active(&G04.constraints(&x), &[1, 6], 1e-9);
        // at the lower corner (78, 33, 27, 27, 27):
        // 5.3578547 · 729 + 0.8356891 · 2106 + 37.293239 · 78 − 40792.141
        let corner = reals(&[78.0, 33.0, 27.0, 27.0, 27.0]);
        let expected = 5.357_854_7 * 729.0 + 0.835_689_1 * 2106.0 + 37.293_239 * 78.0 - 40_792.141;
        assert_close(G04.evaluate(&corner).0, expected, 1e-15);
        assert_eq!(G04.constraints(&corner).len(), 6);
    }

    #[test]
    fn g05() {
        check_optimum(&G05::default(), 1e-13, 1e-9);
        let x = G05::default().optimum().expect("known").solutions()[0].clone();
        let constraints = G05::default().constraints(&x);
        // the equalities are met within δ, not exactly
        for h in constraints.equalities() {
            assert!(h.abs() <= EQUALITY_TOLERANCE * (1.0 + 1e-9), "{h}");
        }
        // at x₃ = x₄ = 0: h₅ = 2000 sin(−0.25) + 1294.8, and x₁, x₂ make h₃, h₄ zero
        let a = 2000.0 * (-0.25f64).sin() + 894.8;
        let (value, violation) = G05::default().evaluate(&reals(&[a, a, 0.0, 0.0]));
        assert_close(
            value,
            5.0 * a + 0.000_001 * a.powi(3) + (0.000_002 / 3.0) * a.powi(3),
            1e-15,
        );
        assert_close(violation, a + 400.0 - EQUALITY_TOLERANCE, 1e-12);
        assert!(G05::with_tolerance(1e-6).optimum().is_none());
    }

    #[test]
    fn g06() {
        check_optimum(&G06, 1e-14, 1e-9);
        let x = G06.optimum().expect("known").solutions()[0].clone();
        assert_active(&G06.constraints(&x), &[1, 2], 1e-9);
        // x₁ = 14.095 solves (x₁ − 5)² − (x₁ − 6)² = 100 − 82.81
        assert_close(2.0 * 14.095 - 11.0, 17.19, 1e-15);
        // (15, 5) is on the first circle and inside the second: f = 125 − 3375, feasible
        assert_eq!(G06.evaluate(&reals(&[15.0, 5.0])), (-3250.0, 0.0));
        // (13, 0) is inside the first circle: g₁ = −64 − 25 + 100 = 11, and f = 27 − 8000
        assert_eq!(G06.evaluate(&reals(&[13.0, 0.0])), (-7973.0, 11.0));
    }
}
