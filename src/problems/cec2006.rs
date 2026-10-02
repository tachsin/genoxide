//! The constrained problems of the CEC 2006 special session, g01 to g24.
//!
//! Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello
//! Coello, C. A. and Deb, K. (2006). *Problem Definitions and Evaluation Criteria for the CEC 2006
//! Special Session on Constrained Real-Parameter Optimization.* Technical report, Nanyang
//! Technological University, Singapore, 18 September 2006. The definitions, bounds, solutions and
//! values are the report's (pages 3-15 and tables 1, 2 and 4). Each problem's docs name the report's source
//! for it; those sources are still to be read
//! ([#168](https://github.com/tachsin/genoxide/issues/168)).
//!
//! Every problem is minimized. Its fitness is `(f(x), violation)`, the violation being
//! `Σ max(0, gᵢ(x)) + Σ max(0, |hⱼ(x)| − δ)`. The report counts an equality as met when
//! `|hⱼ(x)| ≤ δ`, with δ = [`EQUALITY_TOLERANCE`]; `with_tolerance` changes it. Problems that the
//! sources maximize (g02, g03, g08, g12, g18) are negated, as in the report.
//!
//! | Problem | n | Constraints | Optimum |
//! |---|---|---|---|
//! | [`G01`] | 13 | 9 linear inequalities | −15 |
//! | [`G02`] | 20 | 2 inequalities | −0.80361910412559, best known |
//! | [`G03`] | 10 | 1 equality | −1.00050010001 (δ = 10⁻⁴) |
//! | [`G04`] | 5 | 6 inequalities | −30665.53867178332 |
//! | [`G05`] | 4 | 2 linear inequalities, 3 equalities | 5126.4967140071, best known |
//! | [`G06`] | 2 | 2 inequalities | −6961.81387558015 |
//! | [`G07`] | 10 | 3 linear and 5 nonlinear inequalities | 24.30620906818 |
//! | [`G08`] | 2 | 2 inequalities | −0.0958250414180359 |
//! | [`G09`] | 7 | 4 inequalities | 680.630057374402 |
//! | [`G10`] | 8 | 3 linear and 3 nonlinear inequalities | 7049.24802052867 |
//! | [`G11`] | 2 | 1 equality | 0.7499 (δ = 10⁻⁴) |
//! | [`G12`] | 3 | 1 inequality: 729 disjoint spheres | −1 |
//! | [`G13`] | 5 | 3 equalities | 0.053941514041898, best known |
//! | [`G14`] | 10 | 3 linear equalities | −47.7648884594915, best known |
//! | [`G15`] | 3 | 2 equalities | 961.715022289961, best known |
//! | [`G16`] | 5 | 38 inequalities | −1.90515525853479, best known |
//! | [`G17`] | 6 | 4 equalities | 8853.5338748065, best known |
//! | [`G18`] | 9 | 13 inequalities | −0.866025403784439, best known |
//! | [`G19`] | 15 | 5 inequalities | 32.6555929502463, best known |
//! | [`G20`] | 24 | 6 inequalities, 14 equalities | none feasible; 0.2049794002, infeasible |
//! | [`G21`] | 7 | 1 inequality, 5 equalities | 193.724510070035, best known |
//! | [`G22`] | 22 | 1 inequality, 19 equalities | 236.430975504001, best known |
//! | [`G23`] | 9 | 2 inequalities, 4 equalities | −400.0551, best known |
//! | [`G24`] | 2 | 2 inequalities | −5.50801327159536 |

// the report's solutions keep every digit it prints
#![expect(clippy::excessive_precision)]

use super::{Constraints, Optimum, Problem};
use crate::engine::{Extras, FitnessFunction, Provided};
use crate::genome::{Real, Reals};
use crate::math;

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
    // a problem with inequalities only, which gives their values
    ($name:ident, inequalities = $count:literal) => {
        impl FitnessFunction<Reals> for $name {
            type Output = (f64, f64);

            /// The value of `x` and its constraint violation, 0 when it's feasible.
            ///
            /// # Panics
            ///
            /// If `x` has fewer genes than the problem's dimensions.
            fn evaluate(&self, x: &Reals) -> (f64, f64) {
                (self.value(x), self.constraints(x).violation(0.0))
            }

            /// The values of its inequality constraints `gᵢ(x) ≤ 0`, in the order of
            /// [`constraints`](Problem::constraints).
            fn provides(&self) -> Provided {
                Provided::NOTHING.with_inequalities($count)
            }

            /// As [`evaluate`](FitnessFunction::evaluate), with the constraints' values if
            /// they're wanted.
            ///
            /// # Panics
            ///
            /// If `x` has fewer genes than the problem's dimensions, or the buffer for the
            /// constraints' values has another length than their number.
            fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> (f64, f64) {
                let constraints = self.constraints(x);
                if let Some(g) = extras.inequalities() {
                    g.copy_from_slice(constraints.inequalities());
                }
                (self.value(x), constraints.violation(0.0))
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

cec2006!(G01, inequalities = 9);

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
        let fourth: f64 = x.iter().map(|xi| math::powi(math::cos(*xi), 4)).sum();
        let product: f64 = x.iter().map(|xi| math::powi(math::cos(*xi), 2)).product();
        let weighted: f64 = x
            .iter()
            .enumerate()
            .map(|(i, xi)| (i + 1) as f64 * xi * xi)
            .sum();
        -((fourth - 2.0 * product) / weighted.sqrt()).abs()
    }
}

cec2006!(G02, inequalities = 2);

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
        -math::powi(n.sqrt(), G03_DIMENSIONS as i32) * product
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
            -math::powf(radius, n / 2.0),
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

cec2006!(G04, inequalities = 6);

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
        3.0 * x1 + 0.000_001 * math::powi(x1, 3) + 2.0 * x2 + (0.000_002 / 3.0) * math::powi(x2, 3)
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
                1000.0 * math::sin(-x3 - 0.25) + 1000.0 * math::sin(-x4 - 0.25) + 894.8 - x1,
                1000.0 * math::sin(x3 - 0.25) + 1000.0 * math::sin(x3 - x4 - 0.25) + 894.8 - x2,
                1000.0 * math::sin(x4 - 0.25) + 1000.0 * math::sin(x4 - x3 - 0.25) + 1294.8,
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
        math::powi(x[0] - 10.0, 3) + math::powi(x[1] - 20.0, 3)
    }
}

cec2006!(G06, inequalities = 2);

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
                -math::powi(x1 - 5.0, 2) - math::powi(x2 - 5.0, 2) + 100.0,
                math::powi(x1 - 6.0, 2) + math::powi(x2 - 5.0, 2) - 82.81,
            ],
            Vec::new(),
        )
    }
}

// ---- g07 -----------------------------------------------------------------------------------------

/// g07: a quadratic in 10 dimensions with 3 linear and 5 nonlinear inequalities,
/// `x₁² + x₂² + x₁x₂ − 14x₁ − 16x₂ + (x₃ − 10)² + 4(x₄ − 5)² + (x₅ − 3)² + 2(x₆ − 1)² + 5x₇² +
/// 7(x₈ − 11)² + 2(x₉ − 10)² + (x₁₀ − 7)² + 45`.
///
/// Bounds [−10, 10]¹⁰; minimum 24.30620906818 at the report's x*, with g₁…g₆ active. The
/// objective and all the constraints are convex (derived from the definition), so the minimum is
/// global. The report's x* exceeds g₁ by 6·10⁻¹⁴, from rounding, as the report says.
///
/// The report's eqs. 16-17 (p. 5), after Hock and Schittkowski (1981).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G07;

impl G07 {
    fn value(&self, x: &Reals) -> f64 {
        let x = |i: usize| x[i - 1];
        let square = |v: f64| v * v;
        square(x(1)) + square(x(2)) + x(1) * x(2) - 14.0 * x(1) - 16.0 * x(2)
            + square(x(3) - 10.0)
            + 4.0 * square(x(4) - 5.0)
            + square(x(5) - 3.0)
            + 2.0 * square(x(6) - 1.0)
            + 5.0 * square(x(7))
            + 7.0 * square(x(8) - 11.0)
            + 2.0 * square(x(9) - 10.0)
            + square(x(10) - 7.0)
            + 45.0
    }
}

cec2006!(G07, inequalities = 8);

impl Problem for G07 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G07"
    }

    fn representation(&self) -> Real {
        bounds(&[(-10.0, 10.0); 10])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(
            24.306_209_068_18,
            vec![reals(&[
                2.171_996_341_426_92,
                2.363_683_041_603_4,
                8.773_925_739_131_57,
                5.095_984_437_451_73,
                0.990_654_756_560_493,
                1.430_573_928_534_63,
                1.321_644_153_643_06,
                9.828_725_765_244_95,
                8.280_091_588_735_6,
                8.375_926_647_734_7,
            ])],
        ))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g₁…g₈ of the report's eq. 17.
    fn constraints(&self, x: &Reals) -> Constraints {
        let x = |i: usize| x[i - 1];
        let square = |v: f64| v * v;
        Constraints::new(
            vec![
                -105.0 + 4.0 * x(1) + 5.0 * x(2) - 3.0 * x(7) + 9.0 * x(8),
                10.0 * x(1) - 8.0 * x(2) - 17.0 * x(7) + 2.0 * x(8),
                -8.0 * x(1) + 2.0 * x(2) + 5.0 * x(9) - 2.0 * x(10) - 12.0,
                3.0 * square(x(1) - 2.0) + 4.0 * square(x(2) - 3.0) + 2.0 * square(x(3))
                    - 7.0 * x(4)
                    - 120.0,
                5.0 * square(x(1)) + 8.0 * x(2) + square(x(3) - 6.0) - 2.0 * x(4) - 40.0,
                square(x(1)) + 2.0 * square(x(2) - 2.0) - 2.0 * x(1) * x(2) + 14.0 * x(5)
                    - 6.0 * x(6),
                0.5 * square(x(1) - 8.0) + 2.0 * square(x(2) - 4.0) + 3.0 * square(x(5))
                    - x(6)
                    - 30.0,
                -3.0 * x(1) + 6.0 * x(2) + 12.0 * square(x(9) - 8.0) - 7.0 * x(10),
            ],
            Vec::new(),
        )
    }
}

// ---- g08 -----------------------------------------------------------------------------------------

/// g08: `−sin³(2πx₁) sin(2πx₂) / (x₁³(x₁ + x₂))` in 2 dimensions, with 2 nonlinear inequalities: a
/// maximization, negated.
///
/// Bounds [0, 10]²; minimum −0.0958250414180359 at the report's x*, where no constraint is active.
/// The value is 0/0 at x₁ = 0, the lower bound, and so the fitness is invalid there.
///
/// The report's eqs. 18-19 (p. 5), after Koziel and Michalewicz (1999).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G08;

impl G08 {
    fn value(&self, x: &Reals) -> f64 {
        let (x1, x2) = (x[0], x[1]);
        let tau = std::f64::consts::TAU;
        -math::powi(math::sin(tau * x1), 3) * math::sin(tau * x2) / (math::powi(x1, 3) * (x1 + x2))
    }
}

cec2006!(G08, inequalities = 2);

impl Problem for G08 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G08"
    }

    fn representation(&self) -> Real {
        bounds(&[(0.0, 10.0); 2])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(
            -0.095_825_041_418_035_9,
            vec![reals(&[1.227_971_352_607_525_99, 4.245_373_366_122_748_85])],
        ))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g₁ = x₁² − x₂ + 1 and g₂ = 1 − x₁ + (x₂ − 4)², the report's eq. 19.
    fn constraints(&self, x: &Reals) -> Constraints {
        let (x1, x2) = (x[0], x[1]);
        Constraints::new(
            vec![x1 * x1 - x2 + 1.0, 1.0 - x1 + (x2 - 4.0) * (x2 - 4.0)],
            Vec::new(),
        )
    }
}

// ---- g09 -----------------------------------------------------------------------------------------

/// g09: `(x₁ − 10)² + 5(x₂ − 12)² + x₃⁴ + 3(x₄ − 11)² + 10x₅⁶ + 7x₆² + x₇⁴ − 4x₆x₇ − 10x₆ − 8x₇`,
/// a polynomial in 7 dimensions with 4 nonlinear inequalities.
///
/// Bounds [−10, 10]⁷; minimum 680.630057374402 at the report's x*, with g₁ and g₄ active. The
/// report's x* exceeds g₁ by 4·10⁻¹⁶, from rounding.
///
/// The report's eqs. 20-21 (pp. 5-6), after Hock and Schittkowski (1981).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G09;

impl G09 {
    fn value(&self, x: &Reals) -> f64 {
        let x = |i: usize| x[i - 1];
        let square = |v: f64| v * v;
        square(x(1) - 10.0)
            + 5.0 * square(x(2) - 12.0)
            + math::powi(x(3), 4)
            + 3.0 * square(x(4) - 11.0)
            + 10.0 * math::powi(x(5), 6)
            + 7.0 * square(x(6))
            + math::powi(x(7), 4)
            - 4.0 * x(6) * x(7)
            - 10.0 * x(6)
            - 8.0 * x(7)
    }
}

cec2006!(G09, inequalities = 4);

impl Problem for G09 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G09"
    }

    fn representation(&self) -> Real {
        bounds(&[(-10.0, 10.0); 7])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(
            680.630_057_374_402,
            vec![reals(&[
                2.330_499_351_474_051_74,
                1.951_372_368_471_145_92,
                -0.477_541_399_510_615_805,
                4.365_726_249_236_258_74,
                -0.624_486_959_100_388_983,
                1.038_130_994_109_621_73,
                1.594_226_678_067_151_9,
            ])],
        ))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g₁…g₄ of the report's eq. 21.
    fn constraints(&self, x: &Reals) -> Constraints {
        let x = |i: usize| x[i - 1];
        let square = |v: f64| v * v;
        Constraints::new(
            vec![
                -127.0
                    + 2.0 * square(x(1))
                    + 3.0 * math::powi(x(2), 4)
                    + x(3)
                    + 4.0 * square(x(4))
                    + 5.0 * x(5),
                -282.0 + 7.0 * x(1) + 3.0 * x(2) + 10.0 * square(x(3)) + x(4) - x(5),
                -196.0 + 23.0 * x(1) + square(x(2)) + 6.0 * square(x(6)) - 8.0 * x(7),
                4.0 * square(x(1)) + square(x(2)) - 3.0 * x(1) * x(2)
                    + 2.0 * square(x(3))
                    + 5.0 * x(6)
                    - 11.0 * x(7),
            ],
            Vec::new(),
        )
    }
}

// ---- g10 -----------------------------------------------------------------------------------------

/// g10: `x₁ + x₂ + x₃`, linear in 8 dimensions, with 3 linear and 3 nonlinear (bilinear)
/// inequalities.
///
/// Bounds x₁ ∈ [100, 10000], x₂, x₃ ∈ [1000, 10000], x₄…x₈ ∈ [10, 1000]; minimum
/// 7049.24802052867 at the report's x*, where all six constraints are active (to 10⁻¹⁰): the
/// report's table 3 counts six, its text names only g₁, g₂ and g₃.
///
/// The report's eq. 22 and the constraints below it (p. 6), after Hock and Schittkowski (1981).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G10;

impl G10 {
    fn value(&self, x: &Reals) -> f64 {
        x[0] + x[1] + x[2]
    }
}

cec2006!(G10, inequalities = 6);

impl Problem for G10 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G10"
    }

    fn representation(&self) -> Real {
        let mut ranges = vec![(100.0, 10_000.0), (1000.0, 10_000.0), (1000.0, 10_000.0)];
        ranges.extend([(10.0, 1000.0); 5]);
        bounds(&ranges)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(
            7_049.248_020_528_67,
            vec![reals(&[
                579.306_685_017_979_589,
                1_359.970_678_079_356_05,
                5_109.970_657_431_333_17,
                182.017_699_630_615_34,
                295.601_173_702_746_792,
                217.982_300_369_384_632,
                286.416_525_927_868_52,
                395.601_173_702_746_735,
            ])],
        ))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g₁…g₆ of the report's p. 6.
    fn constraints(&self, x: &Reals) -> Constraints {
        let x = |i: usize| x[i - 1];
        Constraints::new(
            vec![
                -1.0 + 0.0025 * (x(4) + x(6)),
                -1.0 + 0.0025 * (x(5) + x(7) - x(4)),
                -1.0 + 0.01 * (x(8) - x(5)),
                -x(1) * x(6) + 833.332_52 * x(4) + 100.0 * x(1) - 83_333.333,
                -x(2) * x(7) + 1250.0 * x(5) + x(2) * x(4) - 1250.0 * x(4),
                -x(3) * x(8) + 1_250_000.0 + x(3) * x(5) - 2500.0 * x(5),
            ],
            Vec::new(),
        )
    }
}

// ---- g11 -----------------------------------------------------------------------------------------

/// g11: `x₁² + (x₂ − 1)²` in 2 dimensions, subject to `x₂ = x₁²`.
///
/// Bounds [−1, 1]². With the equality met to within δ, the minimum is `3/4 − δ` at
/// x = (±√(1/2 − δ), 1/2), derived from the definition (on the upper edge of the tolerance,
/// x₂ = x₁² + δ, the value `u + (1 − u − δ)²` of u = x₁² is least at u = 1/2 − δ): 0.7499 for the
/// report's δ, the report's f(x*). Without the tolerance it would be 3/4 at
/// (±1/√2, 1/2).
///
/// The report's eqs. 23-24 (p. 6), after Koziel and Michalewicz (1999).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G11 {
    tolerance: f64,
}

impl G11 {
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
        let (x1, x2) = (x[0], x[1]);
        x1 * x1 + (x2 - 1.0) * (x2 - 1.0)
    }
}

impl Default for G11 {
    /// The problem with the report's tolerance, [`EQUALITY_TOLERANCE`].
    fn default() -> Self {
        Self::with_tolerance(EQUALITY_TOLERANCE)
    }
}

cec2006!(G11, |problem| problem.tolerance);

impl Problem for G11 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G11"
    }

    fn representation(&self) -> Real {
        bounds(&[(-1.0, 1.0); 2])
    }

    /// The minimum for the tolerance δ: `u + (1 − x₂)²` at x₁ = ±√u, with u = max(0, 1/2 − δ)
    /// and x₂ = min(1, u + δ); `3/4 − δ` for δ < 1/2.
    fn optimum(&self) -> Option<Optimum<Reals>> {
        let tolerance = self.tolerance.max(0.0);
        let u = (0.5 - tolerance).max(0.0);
        let x2 = (u + tolerance).min(1.0);
        let x1 = u.sqrt();
        let solutions = if x1 == 0.0 {
            vec![reals(&[0.0, x2])]
        } else {
            vec![reals(&[-x1, x2]), reals(&[x1, x2])]
        };
        Some(Optimum::proven(u + (1.0 - x2) * (1.0 - x2), solutions))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// h = x₂ − x₁², the report's eq. 24.
    fn constraints(&self, x: &Reals) -> Constraints {
        let (x1, x2) = (x[0], x[1]);
        Constraints::new(Vec::new(), vec![x2 - x1 * x1])
    }
}

// ---- g12 -----------------------------------------------------------------------------------------

/// g12: `−(100 − (x₁ − 5)² − (x₂ − 5)² − (x₃ − 5)²)/100` in 3 dimensions, feasible inside any of
/// 9³ = 729 disjoint spheres: a maximization, negated.
///
/// The one constraint is met when some sphere of radius 0.25 centered on (p, q, r), with
/// p, q, r ∈ {1, …, 9}, contains x: `g = min over p, q, r of (x₁ − p)² + (x₂ − q)² + (x₃ − r)²
/// − 0.0625 ≤ 0`, computed from the nearest center, since the sum is separable. Bounds [0, 10]³;
/// minimum −1 at (5, 5, 5), the center of a sphere, where no constraint is active.
///
/// The report's eq. 25 and the constraint below it (p. 6), after Koziel and Michalewicz (1999).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G12;

impl G12 {
    fn value(&self, x: &Reals) -> f64 {
        let squares: f64 = x[..3].iter().map(|xi| (xi - 5.0) * (xi - 5.0)).sum();
        -(100.0 - squares) / 100.0
    }
}

cec2006!(G12, inequalities = 1);

impl Problem for G12 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G12"
    }

    fn representation(&self) -> Real {
        bounds(&[(0.0, 10.0); 3])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(-1.0, vec![reals(&[5.0, 5.0, 5.0])]))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g, the squared distance to the nearest of the 729 centers, less 0.0625.
    fn constraints(&self, x: &Reals) -> Constraints {
        let squares: f64 = x[..3]
            .iter()
            .map(|&xi| {
                let center = xi.round().clamp(1.0, 9.0);
                (xi - center) * (xi - center)
            })
            .sum();
        Constraints::new(vec![squares - 0.0625], Vec::new())
    }
}

// ---- g13 -----------------------------------------------------------------------------------------

/// g13: `exp(x₁x₂x₃x₄x₅)` in 5 dimensions, with 3 nonlinear equalities.
///
/// Bounds x₁, x₂ ∈ [−2.3, 2.3], x₃…x₅ ∈ [−3.2, 3.2]; best known 0.053941514041898 at the report's
/// x*, which meets the equalities within the tolerance δ only, as g05's. Not proven optimal.
///
/// The report's eqs. 26-27 (pp. 6-7), after Hock and Schittkowski (1981).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G13 {
    tolerance: f64,
}

impl G13 {
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
        math::exp(x[..5].iter().product())
    }
}

impl Default for G13 {
    /// The problem with the report's tolerance, [`EQUALITY_TOLERANCE`].
    fn default() -> Self {
        Self::with_tolerance(EQUALITY_TOLERANCE)
    }
}

cec2006!(G13, |problem| problem.tolerance);

impl Problem for G13 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G13"
    }

    fn representation(&self) -> Real {
        bounds(&[
            (-2.3, 2.3),
            (-2.3, 2.3),
            (-3.2, 3.2),
            (-3.2, 3.2),
            (-3.2, 3.2),
        ])
    }

    /// The report's best known solution, for the report's tolerance; `None` for another.
    fn optimum(&self) -> Option<Optimum<Reals>> {
        (self.tolerance == EQUALITY_TOLERANCE).then(|| {
            Optimum::best_known(
                0.053_941_514_041_898,
                vec![reals(&[
                    -1.717_142_240_03,
                    1.595_721_240_494_68,
                    1.827_250_240_627_1,
                    -0.763_659_881_912_867,
                    -0.763_659_867_364_98,
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

    /// h₁, h₂, h₃ of the report's eq. 27.
    fn constraints(&self, x: &Reals) -> Constraints {
        let x = |i: usize| x[i - 1];
        let squares: f64 = (1..=5).map(|i| x(i) * x(i)).sum();
        Constraints::new(
            Vec::new(),
            vec![
                squares - 10.0,
                x(2) * x(3) - 5.0 * x(4) * x(5),
                math::powi(x(1), 3) + math::powi(x(2), 3) + 1.0,
            ],
        )
    }
}

// ---- g14 -----------------------------------------------------------------------------------------

// g14's lower bound: the report's bounds are open at 0 (the logarithm), and this is the smallest
// positive normal number, for which every logarithm stays finite
const G14_LOW: f64 = f64::MIN_POSITIVE;

// the report's c₁…c₁₀ of g14 (p. 7)
const G14_C: [f64; 10] = [
    -6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179,
];

/// g14: `Σ xᵢ (cᵢ + ln(xᵢ / Σⱼ xⱼ))` in 10 dimensions, with 3 linear equalities.
///
/// Bounds (0, 10]¹⁰, closed here at `f64::MIN_POSITIVE` (2.2·10⁻³⁰⁸), the smallest positive
/// normal number; the value is 0 · ln 0, undefined, and so the fitness invalid at 0. Best known
/// −47.7648884594915 at the report's x*, which meets the equalities within the tolerance δ only.
/// Not proven optimal.
///
/// The report's eqs. 28-29 (p. 7), after Himmelblau, D. M. (1972). *Applied Nonlinear
/// Programming.* McGraw-Hill.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G14 {
    tolerance: f64,
}

impl G14 {
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
        let x = &x[..10];
        let sum: f64 = x.iter().sum();
        x.iter()
            .zip(G14_C)
            .map(|(&xi, c)| xi * (c + math::ln(xi / sum)))
            .sum()
    }
}

impl Default for G14 {
    /// The problem with the report's tolerance, [`EQUALITY_TOLERANCE`].
    fn default() -> Self {
        Self::with_tolerance(EQUALITY_TOLERANCE)
    }
}

cec2006!(G14, |problem| problem.tolerance);

impl Problem for G14 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G14"
    }

    fn representation(&self) -> Real {
        bounds(&[(G14_LOW, 10.0); 10])
    }

    /// The report's best known solution, for the report's tolerance; `None` for another.
    fn optimum(&self) -> Option<Optimum<Reals>> {
        (self.tolerance == EQUALITY_TOLERANCE).then(|| {
            Optimum::best_known(
                -47.764_888_459_491_5,
                vec![reals(&[
                    0.040_668_411_321_628_2,
                    0.147_721_240_492_452,
                    0.783_205_732_104_114,
                    0.001_414_339_318_890_84,
                    0.485_293_636_780_388,
                    0.000_693_183_051_556_082,
                    0.027_405_204_068_776_6,
                    0.017_950_966_021_481_8,
                    0.037_326_818_685_971_7,
                    0.096_884_460_433_684_5,
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

    /// h₁, h₂, h₃ of the report's eq. 29.
    fn constraints(&self, x: &Reals) -> Constraints {
        let x = |i: usize| x[i - 1];
        Constraints::new(
            Vec::new(),
            vec![
                x(1) + 2.0 * x(2) + 2.0 * x(3) + x(6) + x(10) - 2.0,
                x(4) + 2.0 * x(5) + x(6) + x(7) - 1.0,
                x(3) + x(7) + x(8) + 2.0 * x(9) + x(10) - 1.0,
            ],
        )
    }
}

// ---- g15 -----------------------------------------------------------------------------------------

/// g15: `1000 − x₁² − 2x₂² − x₃² − x₁x₂ − x₁x₃`, a quadratic in 3 dimensions, subject to a sphere
/// `x₁² + x₂² + x₃² = 25` and a plane `8x₁ + 14x₂ + 7x₃ = 56`.
///
/// Bounds [0, 10]³; best known 961.715022289961 at the report's x*, which meets the equalities
/// within the tolerance δ only. Not proven optimal.
///
/// The report's eqs. 30-31 (p. 7), after Himmelblau (1972).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G15 {
    tolerance: f64,
}

impl G15 {
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
        let (x1, x2, x3) = (x[0], x[1], x[2]);
        1000.0 - x1 * x1 - 2.0 * x2 * x2 - x3 * x3 - x1 * x2 - x1 * x3
    }
}

impl Default for G15 {
    /// The problem with the report's tolerance, [`EQUALITY_TOLERANCE`].
    fn default() -> Self {
        Self::with_tolerance(EQUALITY_TOLERANCE)
    }
}

cec2006!(G15, |problem| problem.tolerance);

impl Problem for G15 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G15"
    }

    fn representation(&self) -> Real {
        bounds(&[(0.0, 10.0); 3])
    }

    /// The report's best known solution, for the report's tolerance; `None` for another.
    fn optimum(&self) -> Option<Optimum<Reals>> {
        (self.tolerance == EQUALITY_TOLERANCE).then(|| {
            Optimum::best_known(
                961.715_022_289_961,
                vec![reals(&[
                    3.512_128_126_117_951_33,
                    0.216_987_510_429_556_135,
                    3.552_178_549_291_799_21,
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

    /// h₁ = x₁² + x₂² + x₃² − 25 and h₂ = 8x₁ + 14x₂ + 7x₃ − 56, the report's eq. 31.
    fn constraints(&self, x: &Reals) -> Constraints {
        let (x1, x2, x3) = (x[0], x[1], x[2]);
        Constraints::new(
            Vec::new(),
            vec![
                x1 * x1 + x2 * x2 + x3 * x3 - 25.0,
                8.0 * x1 + 14.0 * x2 + 7.0 * x3 - 56.0,
            ],
        )
    }
}

// ---- g16 -----------------------------------------------------------------------------------------

// the lower and upper limits of g16's y₁…y₁₇, the report's g₅…g₃₈ (eq. 33)
const G16_LIMITS: [(f64, f64); 17] = [
    (213.1, 405.23),
    (17.505, 1053.6667),
    (11.275, 35.03),
    (214.228, 665.585),
    (7.458, 584.463),
    (0.961, 265.916),
    (1.612, 7.046),
    (0.146, 0.222),
    (107.99, 273.366),
    (922.693, 1286.105),
    (926.832, 1444.046),
    (18.766, 537.141),
    (1072.163, 3247.039),
    (8961.448, 26844.086),
    (0.063, 0.386),
    (71084.33, 140000.0),
    (2802713.0, 12146108.0),
];

/// g16: a nonlinear function of 5 variables through a chain of 17 intermediate quantities y₁…y₁₇
/// and c₁…c₁₇, with 38 inequalities, most of them bounds on the yᵢ.
///
/// `f = 0.000117y₁₄ + 0.1365 + 0.00002358y₁₃ + 0.000001502y₁₆ + 0.0321y₁₂ + 0.004324y₅ +
/// 0.0001c₁₅/c₁₆ + 37.48y₂/c₁₂ − 0.0000005843y₁₇`, with the yᵢ and cᵢ of the report's eq. 34.
/// Bounds x₁ ∈ [704.4148, 906.3855], x₂ ∈ [68.6, 288.88], x₃ ∈ [0, 134.75],
/// x₄ ∈ [193, 287.0966], x₅ ∈ [25, 84.1988]; best known −1.90515525853479 at the report's x*,
/// where g₂, g₃, g₄, g₅ and g₃₆ are active (the report's table 3 counts four) and x₂ is at its lower
/// bound. Not proven optimal.
///
/// The report's eqs. 32-34 (pp. 7-10), after Himmelblau (1972).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G16;

// g16's intermediate quantities: y₁…y₁₇ and c₁…c₁₇ at y[0..17] and c[0..17]
struct G16Chain {
    y: [f64; 17],
    c: [f64; 17],
}

impl G16 {
    // the report's eq. 34, in its order
    fn chain(x: &Reals) -> G16Chain {
        let (x1, x2, x3, x4, x5) = (x[0], x[1], x[2], x[3], x[4]);
        let mut y = [0.0; 17];
        let mut c = [0.0; 17];
        y[0] = x2 + x3 + 41.6;
        c[0] = 0.024 * x4 - 4.62;
        y[1] = 12.5 / c[0] + 12.0;
        c[1] = 0.000_353_5 * x1 * x1 + 0.5311 * x1 + 0.087_05 * y[1] * x1;
        c[2] = 0.052 * x1 + 78.0 + 0.002_377 * y[1] * x1;
        y[2] = c[1] / c[2];
        y[3] = 19.0 * y[2];
        c[3] = 0.047_82 * (x1 - y[2])
            + 0.1956 * (x1 - y[2]) * (x1 - y[2]) / x2
            + 0.6376 * y[3]
            + 1.594 * y[2];
        c[4] = 100.0 * x2;
        c[5] = x1 - y[2] - y[3];
        c[6] = 0.950 - c[3] / c[4];
        y[4] = c[5] * c[6];
        y[5] = x1 - y[4] - y[3] - y[2];
        c[7] = (y[4] + y[3]) * 0.995;
        y[6] = c[7] / y[0];
        y[7] = c[7] / 3798.0;
        c[8] = y[6] - 0.0663 * y[6] / y[7] - 0.3153;
        y[8] = 96.82 / c[8] + 0.321 * y[0];
        y[9] = 1.29 * y[4] + 1.258 * y[3] + 2.29 * y[2] + 1.71 * y[5];
        y[10] = 1.71 * x1 - 0.452 * y[3] + 0.580 * y[2];
        c[9] = 12.3 / 752.3;
        c[10] = (1.75 * y[1]) * (0.995 * x1);
        c[11] = 0.995 * y[9] + 1998.0;
        y[11] = c[9] * x1 + c[10] / c[11];
        y[12] = c[11] - 1.75 * y[1];
        y[13] = 3623.0 + 64.4 * x2 + 58.4 * x3 + 146_312.0 / (y[8] + x5);
        c[12] = 0.995 * y[9] + 60.8 * x2 + 48.0 * x4 - 0.1121 * y[13] - 5095.0;
        y[14] = y[12] / c[12];
        y[15] = 148_000.0 - 331_000.0 * y[14] + 40.0 * y[12] - 61.0 * y[14] * y[12];
        c[13] = 2324.0 * y[9] - 28_740_000.0 * y[1];
        y[16] = 14_130_000.0 - 1328.0 * y[9] - 531.0 * y[10] + c[13] / c[11];
        c[14] = y[12] / y[14] - y[12] / 0.52;
        c[15] = 1.104 - 0.72 * y[14];
        c[16] = y[8] + x5;
        G16Chain { y, c }
    }

    fn value(&self, x: &Reals) -> f64 {
        let G16Chain { y, c } = Self::chain(x);
        0.000_117 * y[13]
            + 0.1365
            + 0.000_023_58 * y[12]
            + 0.000_001_502 * y[15]
            + 0.0321 * y[11]
            + 0.004_324 * y[4]
            + 0.0001 * c[14] / c[15]
            + 37.48 * y[1] / c[11]
            - 0.000_000_584_3 * y[16]
    }
}

cec2006!(G16, inequalities = 38);

impl Problem for G16 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G16"
    }

    fn representation(&self) -> Real {
        bounds(&[
            (704.4148, 906.3855),
            (68.6, 288.88),
            (0.0, 134.75),
            (193.0, 287.0966),
            (25.0, 84.1988),
        ])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::best_known(
            -1.905_155_258_534_79,
            vec![reals(&[
                705.174_537_070_090_537,
                // 68.6 in double precision
                68.599_999_999_999_994_3,
                102.899_999_999_999_991,
                282.324_931_593_660_324,
                37.584_116_425_805_483_2,
            ])],
        ))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g₁…g₃₈ of the report's eq. 33: four, then a lower and an upper limit of each of
    /// y₁…y₁₇.
    fn constraints(&self, x: &Reals) -> Constraints {
        let G16Chain { y, c } = Self::chain(x);
        let (x2, x3) = (x[1], x[2]);
        let mut inequalities = vec![
            0.28 / 0.72 * y[4] - y[3],
            x3 - 1.5 * x2,
            3496.0 * y[1] / c[11] - 21.0,
            110.6 + y[0] - 62_212.0 / c[16],
        ];
        for (yi, (low, high)) in y.iter().zip(G16_LIMITS) {
            inequalities.extend([low - yi, yi - high]);
        }
        Constraints::new(inequalities, Vec::new())
    }
}

// ---- g17 -----------------------------------------------------------------------------------------

/// g17: `f₁(x₁) + f₂(x₂)`, piecewise linear in 6 dimensions, with 4 nonlinear equalities.
///
/// f₁ is 30x₁ below x₁ = 300 and 31x₁ from there; f₂ is 28x₂ below x₂ = 100, 29x₂ below 200 and
/// 30x₂ from there. The report defines f₁ for x₁ < 400 and f₂ for x₂ < 1000, and its bounds
/// include 400 and 1000: here the last pieces include them.
///
/// Bounds x₁ ∈ [0, 400], x₂ ∈ [0, 1000], x₃, x₄ ∈ [340, 420], x₅ ∈ [−1000, 1000],
/// x₆ ∈ [0, 0.5236]; best known 8853.5338748065, reached by the report's x* with x₁ lowered to
/// 201.78446249355, where h₁ reaches the tolerance δ. Not proven optimal. The report's x* itself
/// evaluates to 8853.53401643571; the report prints 8853.53967480648, which is the value of its
/// organizers' code, where f₁ and f₂ take the right-hand sides of h₁ and h₂ instead of x₁ and x₂.
/// Later papers give 8853.5338748065, which the lowered x₁ reaches: its source is unverified.
///
/// The report's eqs. 35-36 (p. 10), after Himmelblau (1972).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G17 {
    tolerance: f64,
}

impl G17 {
    /// The problem with an equality tolerance δ of `tolerance`, instead of the report's
    /// [`EQUALITY_TOLERANCE`]. Its best known solution is for the report's δ only.
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// The equality tolerance δ.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    fn value(&self, x: &Reals) -> f64 {
        let (x1, x2) = (x[0], x[1]);
        let f1 = if x1 < 300.0 { 30.0 * x1 } else { 31.0 * x1 };
        let f2 = if x2 < 100.0 {
            28.0 * x2
        } else if x2 < 200.0 {
            29.0 * x2
        } else {
            30.0 * x2
        };
        f1 + f2
    }
}

impl Default for G17 {
    /// The problem with the report's tolerance, [`EQUALITY_TOLERANCE`].
    fn default() -> Self {
        Self::with_tolerance(EQUALITY_TOLERANCE)
    }
}

cec2006!(G17, |problem| problem.tolerance);

impl Problem for G17 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G17"
    }

    // 0.5236 is the report's bound, which is close to π/6
    #[expect(clippy::approx_constant)]
    fn representation(&self) -> Real {
        bounds(&[
            (0.0, 400.0),
            (0.0, 1000.0),
            (340.0, 420.0),
            (340.0, 420.0),
            (-1000.0, 1000.0),
            (0.0, 0.5236),
        ])
    }

    /// The best known solution, for the report's tolerance; `None` for another.
    fn optimum(&self) -> Option<Optimum<Reals>> {
        (self.tolerance == EQUALITY_TOLERANCE).then(|| {
            Optimum::best_known(
                8_853.533_874_806_5,
                vec![reals(&[
                    201.784_462_493_55,
                    99.999_999_999_999_900_5,
                    383.071_034_852_773_266,
                    420.0,
                    -10.907_658_451_429_265_2,
                    0.073_148_231_208_428_712_8,
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

    /// h₁…h₄ of the report's eq. 36.
    fn constraints(&self, x: &Reals) -> Constraints {
        let (x1, x2, x3, x4, x5, x6) = (x[0], x[1], x[2], x[3], x[4], x[5]);
        let product = x3 * x4 / 131.078;
        let (third, fourth) = (0.907_98 * x3 * x3 / 131.078, 0.907_98 * x4 * x4 / 131.078);
        let (sin, cos) = math::sin_cos(1.475_88);
        Constraints::new(
            Vec::new(),
            vec![
                -x1 + 300.0 - product * math::cos(1.484_77 - x6) + third * cos,
                -x2 - product * math::cos(1.484_77 + x6) + fourth * cos,
                -x5 - product * math::sin(1.484_77 + x6) + fourth * sin,
                200.0 - product * math::sin(1.484_77 - x6) + third * sin,
            ],
        )
    }
}

// ---- g18 -----------------------------------------------------------------------------------------

/// g18: `−0.5(x₁x₄ − x₂x₃ + x₃x₉ − x₅x₉ + x₅x₈ − x₆x₇)`, a quadratic in 9 dimensions with 13
/// nonlinear inequalities: a maximization, negated.
///
/// Bounds x₁…x₈ ∈ [−10, 10], x₉ ∈ [0, 20]; best known −0.866025403784439 (−√3/2 to its digits) at
/// the report's x*, with g₁, g₃, g₄, g₆, g₇ and g₉ active. Not proven optimal.
///
/// The report's eqs. 37-38 (p. 11), after Himmelblau (1972).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G18;

impl G18 {
    fn value(&self, x: &Reals) -> f64 {
        let x = |i: usize| x[i - 1];
        -0.5 * (x(1) * x(4) - x(2) * x(3) + x(3) * x(9) - x(5) * x(9) + x(5) * x(8) - x(6) * x(7))
    }
}

cec2006!(G18, inequalities = 13);

impl Problem for G18 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G18"
    }

    fn representation(&self) -> Real {
        let mut ranges = vec![(-10.0, 10.0); 8];
        ranges.push((0.0, 20.0));
        bounds(&ranges)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::best_known(
            -0.866_025_403_784_439,
            vec![reals(&[
                -0.657_776_192_427_943_163,
                -0.153_418_773_482_438_542,
                0.323_413_871_675_240_938,
                -0.946_257_611_651_304_398,
                -0.657_776_194_376_798_906,
                -0.753_213_434_632_691_414,
                0.323_413_874_123_576_972,
                -0.346_462_947_962_331_735,
                0.599_794_662_852_175_42,
            ])],
        ))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g₁…g₁₃ of the report's eq. 38.
    fn constraints(&self, x: &Reals) -> Constraints {
        let x = |i: usize| x[i - 1];
        let square = |v: f64| v * v;
        Constraints::new(
            vec![
                square(x(3)) + square(x(4)) - 1.0,
                square(x(9)) - 1.0,
                square(x(5)) + square(x(6)) - 1.0,
                square(x(1)) + square(x(2) - x(9)) - 1.0,
                square(x(1) - x(5)) + square(x(2) - x(6)) - 1.0,
                square(x(1) - x(7)) + square(x(2) - x(8)) - 1.0,
                square(x(3) - x(5)) + square(x(4) - x(6)) - 1.0,
                square(x(3) - x(7)) + square(x(4) - x(8)) - 1.0,
                square(x(7)) + square(x(8) - x(9)) - 1.0,
                x(2) * x(3) - x(1) * x(4),
                -x(3) * x(9),
                x(5) * x(9),
                x(6) * x(7) - x(5) * x(8),
            ],
            Vec::new(),
        )
    }
}

// ---- g19 -----------------------------------------------------------------------------------------

// the report's table 1 (p. 12): e₁…e₅, the symmetric c₁₁…c₅₅ and d₁…d₅
const G19_E: [f64; 5] = [-15.0, -27.0, -36.0, -18.0, -12.0];
const G19_C: [[f64; 5]; 5] = [
    [30.0, -20.0, -10.0, 32.0, -10.0],
    [-20.0, 39.0, -6.0, -31.0, 32.0],
    [-10.0, -6.0, 10.0, -6.0, -10.0],
    [32.0, -31.0, -6.0, 39.0, -20.0],
    [-10.0, 32.0, -10.0, -20.0, 30.0],
];
const G19_D: [f64; 5] = [4.0, 8.0, 10.0, 6.0, 2.0];
// a₁₁…a₁₀,₅ of table 1, a row per i
const G19_A: [[f64; 5]; 10] = [
    [-16.0, 2.0, 0.0, 1.0, 0.0],
    [0.0, -2.0, 0.0, 0.4, 2.0],
    [-3.5, 0.0, 2.0, 0.0, 0.0],
    [0.0, -2.0, 0.0, -4.0, -1.0],
    [0.0, -9.0, -2.0, 1.0, -2.8],
    [2.0, 0.0, -4.0, 0.0, 0.0],
    [-1.0, -1.0, -1.0, -1.0, -1.0],
    [-1.0, -2.0, -3.0, -2.0, -1.0],
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
];
// b₁…b₁₀, printed after eq. 40 (p. 11)
const G19_B: [f64; 10] = [-40.0, -2.0, -0.25, -4.0, -4.0, -1.0, -40.0, -60.0, 5.0, 1.0];

/// g19: `Σⱼ Σᵢ cᵢⱼ x₁₀₊ᵢ x₁₀₊ⱼ + 2 Σⱼ dⱼ x₁₀₊ⱼ³ − Σᵢ bᵢ xᵢ`, a cubic in 15 dimensions with 5
/// nonlinear inequalities `−2 Σᵢ cᵢⱼ x₁₀₊ᵢ − 3dⱼ x₁₀₊ⱼ² − eⱼ + Σᵢ aᵢⱼ xᵢ ≤ 0`.
///
/// i and j run over 1…5, except in `Σᵢ bᵢ xᵢ` and `Σᵢ aᵢⱼ xᵢ`, over 1…10; a, b, c, d and e are the
/// report's table 1 and the vector b printed with it. Bounds [0, 10]¹⁵; best known
/// 32.6555929502463 at the report's x*, where all five constraints are active (to 10⁻¹⁴; the
/// report's table 3 counts none). Not proven optimal.
///
/// The report's eqs. 39-40 and table 1 (pp. 11-12), after Himmelblau (1972).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G19;

impl G19 {
    fn value(&self, x: &Reals) -> f64 {
        let y = &x[10..15];
        let mut quadratic = 0.0;
        for (row, &yi) in G19_C.iter().zip(y) {
            for (&c, &yj) in row.iter().zip(y) {
                quadratic += c * yi * yj;
            }
        }
        let cubic: f64 = G19_D.iter().zip(y).map(|(d, yj)| d * yj * yj * yj).sum();
        let linear: f64 = G19_B.iter().zip(&x[..10]).map(|(b, xi)| b * xi).sum();
        quadratic + 2.0 * cubic - linear
    }
}

cec2006!(G19, inequalities = 5);

impl Problem for G19 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G19"
    }

    fn representation(&self) -> Real {
        bounds(&[(0.0, 10.0); 15])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::best_known(
            32.655_592_950_246_3,
            vec![reals(&[
                1.669_913_413_262_913_44e-17,
                3.953_782_292_824_565_09e-16,
                3.945_990_451_432_337_84,
                1.060_365_974_797_212_11e-16,
                3.283_177_345_845_416_1,
                9.999_999_999_999_998_22,
                1.128_294_146_716_053_33e-17,
                1.202_619_459_979_470_9e-17,
                2.507_062_760_007_696_97e-15,
                2.246_241_229_879_706_77e-15,
                0.370_764_847_417_013_987,
                0.278_456_024_942_955_571,
                0.523_838_487_672_241_171,
                0.388_620_152_510_322_781,
                0.298_156_764_974_678_579,
            ])],
        ))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g₁…g₅ of the report's eq. 40.
    fn constraints(&self, x: &Reals) -> Constraints {
        let y = &x[10..15];
        let inequalities = (0..5)
            .map(|j| {
                let quadratic: f64 = G19_C.iter().zip(y).map(|(row, yi)| row[j] * yi).sum();
                let linear: f64 = G19_A
                    .iter()
                    .zip(&x[..10])
                    .map(|(row, xi)| row[j] * xi)
                    .sum();
                -2.0 * quadratic - 3.0 * G19_D[j] * y[j] * y[j] - G19_E[j] + linear
            })
            .collect();
        Constraints::new(inequalities, Vec::new())
    }
}

// ---- g20 -----------------------------------------------------------------------------------------

// the report's table 2 (p. 13): a₁…a₁₂ and b₁…b₁₂, which a₁₃…a₂₄ and b₁₃…b₂₄ repeat, c₁…c₁₂,
// d₁…d₁₂ and e₁…e₆
const G20_A: [f64; 12] = [
    0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09,
];
const G20_B: [f64; 12] = [
    44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507, 46.07, 60.097,
];
const G20_C: [f64; 12] = [
    123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7, 0.85, 0.64,
];
const G20_D: [f64; 12] = [
    31.244, 36.12, 34.784, 92.7, 82.7, 91.6, 56.708, 82.7, 80.8, 64.517, 49.4, 49.1,
];
const G20_E: [f64; 6] = [0.1, 0.3, 0.4, 0.3, 0.6, 0.3];

/// g20: `Σ aᵢxᵢ`, linear in 24 dimensions, with 6 nonlinear inequalities, 12 nonlinear and 2
/// linear equalities, and no feasible solution.
///
/// The inequalities are `(xᵢ + xᵢ₊₁₂) / (Σⱼ xⱼ + eᵢ) ≤ 0` for i = 1, 2, 3 and
/// `(xᵢ₊₃ + xᵢ₊₁₅) / (Σⱼ xⱼ + eᵢ) ≤ 0` for i = 4, 5, 6; the equalities
/// `x₁₂₊ᵢ / (b₁₂₊ᵢ Σⱼ₌₁₃²⁴ xⱼ/bⱼ) − cᵢxᵢ / (40bᵢ Σⱼ₌₁¹² xⱼ/bⱼ) = 0` for i = 1…12,
/// `Σ xᵢ − 1 = 0` and `Σᵢ₌₁¹² xᵢ/dᵢ + k Σᵢ₌₁₃²⁴ xᵢ/bᵢ − 1.671 = 0`, with
/// k = 0.7302 · 530 · 14.7/40; a, b, c, d and e are the report's table 2. The equalities are
/// undefined, and so the fitness invalid, where x₁…x₁₂ or x₁₃…x₂₄ are all 0.
///
/// Bounds [0, 10]²⁴. The report finds no feasible solution, and there is none (derived for
/// genoxide, not in the report): the inequalities hold only where x₁, x₂, x₃, x₇, x₈, x₉ and
/// x₁₃, x₁₄, x₁₅, x₁₉, x₂₀, x₂₁ are 0; h₁₄ then needs `Σᵢ₌₁₃²⁴ xᵢ/bᵢ ≥ 0.0115`, and h₁…h₁₂ make
/// `Σᵢ₌₁₃²⁴ xᵢ` at least 109 times that, 1.26, where h₁₃ allows 1. The best known value is the
/// report's 0.2049794002 (its table 4), at its x*, which is infeasible: g₁ = 0.1438, while the
/// equalities are met within the tolerance δ. Not proven optimal.
///
/// The report's eqs. 41-42 and table 2 (pp. 12-13), after Himmelblau (1972).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G20 {
    tolerance: f64,
}

impl G20 {
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
        let first: f64 = G20_A.iter().zip(&x[..12]).map(|(a, xi)| a * xi).sum();
        let second: f64 = G20_A.iter().zip(&x[12..24]).map(|(a, xi)| a * xi).sum();
        first + second
    }
}

impl Default for G20 {
    /// The problem with the report's tolerance, [`EQUALITY_TOLERANCE`].
    fn default() -> Self {
        Self::with_tolerance(EQUALITY_TOLERANCE)
    }
}

cec2006!(G20, |problem| problem.tolerance);

impl Problem for G20 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G20"
    }

    fn representation(&self) -> Real {
        bounds(&[(0.0, 10.0); 24])
    }

    /// The report's best known solution, which is infeasible, for the report's tolerance; `None`
    /// for another.
    fn optimum(&self) -> Option<Optimum<Reals>> {
        (self.tolerance == EQUALITY_TOLERANCE).then(|| {
            Optimum::best_known(
                0.204_979_400_2,
                vec![reals(&[
                    1.285_823_434_985_280_86e-18,
                    4.834_603_025_261_306_64e-34,
                    0.0,
                    0.0,
                    6.304_599_296_607_818_51e-18,
                    7.571_925_262_011_450_68e-34,
                    5.033_506_983_728_404_37e-34,
                    9.282_680_796_166_180_64e-34,
                    0.0,
                    1.767_233_845_255_473_59e-17,
                    3.556_861_018_229_657_01e-34,
                    2.994_138_500_834_713_46e-34,
                    0.158_143_376_337_580_827,
                    2.296_017_741_616_998_33e-19,
                    1.061_069_386_110_429_47e-18,
                    1.319_683_443_195_063_91e-18,
                    0.530_902_525_044_209_539,
                    0.0,
                    2.891_483_102_577_735_35e-18,
                    3.348_921_261_806_661_59e-18,
                    0.0,
                    0.310_999_974_151_577_319,
                    5.412_446_663_178_335_61e-5,
                    4.849_931_652_469_595_53e-16,
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

    /// g₁…g₆, then h₁…h₁₄ of the report's eq. 42.
    fn constraints(&self, x: &Reals) -> Constraints {
        let (first, second) = (&x[..12], &x[12..24]);
        let total: f64 = x[..24].iter().sum();
        let mut inequalities: Vec<f64> = (0..3)
            .map(|i| (first[i] + second[i]) / (total + G20_E[i]))
            .collect();
        inequalities.extend((3..6).map(|i| (first[i + 3] + second[i + 3]) / (total + G20_E[i])));
        let over_b = |part: &[f64]| -> f64 { part.iter().zip(G20_B).map(|(xi, b)| xi / b).sum() };
        let (first_over_b, second_over_b) = (over_b(first), over_b(second));
        let mut equalities: Vec<f64> = (0..12)
            .map(|i| {
                second[i] / (G20_B[i] * second_over_b)
                    - G20_C[i] * first[i] / (40.0 * G20_B[i] * first_over_b)
            })
            .collect();
        let over_d: f64 = first.iter().zip(G20_D).map(|(xi, d)| xi / d).sum();
        let k = 0.7302 * 530.0 * (14.7 / 40.0);
        equalities.extend([total - 1.0, over_d + k * second_over_b - 1.671]);
        Constraints::new(inequalities, equalities)
    }
}

// ---- g21 -----------------------------------------------------------------------------------------

/// g21: `x₁`, linear in 7 dimensions, with 1 nonlinear inequality and 5 nonlinear equalities.
///
/// Bounds x₁ ∈ [0, 1000], x₂, x₃ ∈ [0, 40], x₄ ∈ [100, 300], x₅ ∈ [6.3, 6.7], x₆ ∈ [5.9, 6.4],
/// x₇ ∈ [4.5, 6.25]; best known 193.724510070035 at the report's x*, where g₁ is active and the
/// equalities are met within the tolerance δ only, each |hⱼ| at δ. Not proven optimal.
///
/// The equalities leave x₄ free (derived for genoxide): h₃, h₄, h₅ give x₅, x₆, x₇, and h₁ and h₂
/// factor as (x₄ − 300)(x₃ − 25(x₅ − x₆)) and (100 − x₄)(x₂ − 155.365 + 25x₇). With g₁ active, f
/// is a function of x₄, least at x₄ = 100, where x₂ = 0 and x₃ = 25 ln 2 meet every equality
/// exactly, at 35 (25 ln 2)^0.6 = 193.788; the tolerance takes the best known 0.064 lower. The
/// other end, x₄ = 299.53 with x₂ at its bound 40, is a local minimum near 325.
///
/// The report's eqs. 43-44 (p. 13), after Epperly, T. *Global optimization test problems with
/// solutions* (the report's reference 6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G21 {
    tolerance: f64,
}

impl G21 {
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
        x[0]
    }
}

impl Default for G21 {
    /// The problem with the report's tolerance, [`EQUALITY_TOLERANCE`].
    fn default() -> Self {
        Self::with_tolerance(EQUALITY_TOLERANCE)
    }
}

cec2006!(G21, |problem| problem.tolerance);

impl Problem for G21 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G21"
    }

    fn representation(&self) -> Real {
        bounds(&[
            (0.0, 1000.0),
            (0.0, 40.0),
            (0.0, 40.0),
            (100.0, 300.0),
            (6.3, 6.7),
            (5.9, 6.4),
            (4.5, 6.25),
        ])
    }

    /// The report's best known solution, for the report's tolerance; `None` for another.
    fn optimum(&self) -> Option<Optimum<Reals>> {
        (self.tolerance == EQUALITY_TOLERANCE).then(|| {
            Optimum::best_known(
                193.724_510_070_035,
                vec![reals(&[
                    193.724_510_070_034_967,
                    5.569_441_315_533_684_33e-27,
                    17.319_188_729_408_491_4,
                    100.047_897_801_386_839,
                    6.684_451_853_623_778_92,
                    5.991_684_284_442_648_33,
                    6.214_516_488_860_704_51,
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

    /// g₁, then h₁…h₅ of the report's eq. 44.
    fn constraints(&self, x: &Reals) -> Constraints {
        let (x1, x2, x3, x4, x5, x6, x7) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
        Constraints::new(
            vec![-x1 + 35.0 * math::powf(x2, 0.6) + 35.0 * math::powf(x3, 0.6)],
            vec![
                -300.0 * x3 + 7500.0 * x5 - 7500.0 * x6 - 25.0 * x4 * x5 + 25.0 * x4 * x6 + x3 * x4,
                100.0 * x2 + 155.365 * x4 + 2500.0 * x7 - x2 * x4 - 25.0 * x4 * x7 - 15_536.5,
                -x5 + math::ln(-x4 + 900.0),
                -x6 + math::ln(x4 + 300.0),
                -x7 + math::ln(-2.0 * x4 + 700.0),
            ],
        )
    }
}

// ---- g22 -----------------------------------------------------------------------------------------

/// g22: `x₁`, linear in 22 dimensions, with 1 nonlinear inequality, 8 linear and 11 nonlinear
/// equalities.
///
/// Bounds x₁ ∈ [0, 20000], x₂, x₃, x₄ ∈ [0, 10⁶], x₅, x₆, x₇ ∈ [0, 4·10⁷], x₈ ∈ [100, 299.99],
/// x₉ ∈ [100, 399.99], x₁₀ ∈ [100.01, 300], x₁₁ ∈ [100, 400], x₁₂ ∈ [100, 600],
/// x₁₃, x₁₄, x₁₅ ∈ [0, 500], x₁₆ ∈ [0.01, 300], x₁₇ ∈ [0.01, 400], x₁₈…x₂₂ ∈ [−4.7, 6.25]; best
/// known 236.430975504001 at the report's x*, which meets the equalities within the tolerance δ
/// only, and where g₁ is nearly active (−2.2·10⁻⁷). Not proven optimal.
///
/// The report's best known isn't the best there is (derived for genoxide, not in the report). The
/// equalities leave x₁, x₈ and x₉ free: h₁…h₁₁ make x₅…x₇, x₁₀…x₁₂, x₁₆ and x₁₇ linear in x₈ and
/// x₉ (x₁₀ = 430 − x₈, x₁₁ = 440 − x₉ + x₈), h₁₂…h₁₆ give x₁₈…x₂₂ as their logarithms,
/// h₁₇…h₁₉ then x₁₃…x₁₅, and h₇…h₉ x₂…x₄. With g₁ active, f is a function of x₈ and x₉, and at
/// x₈ = 130, x₉ = 170, where x₁₀ and x₁₁ reach their upper bounds, it is 236.370313314566, with
/// every equality met exactly: 0.0607 below the report's value, which `optimum` keeps.
///
/// The report's eqs. 45-46 (pp. 13-14), after Epperly (the report's reference 6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G22 {
    tolerance: f64,
}

impl G22 {
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
        x[0]
    }
}

impl Default for G22 {
    /// The problem with the report's tolerance, [`EQUALITY_TOLERANCE`].
    fn default() -> Self {
        Self::with_tolerance(EQUALITY_TOLERANCE)
    }
}

cec2006!(G22, |problem| problem.tolerance);

impl Problem for G22 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G22"
    }

    fn representation(&self) -> Real {
        let mut ranges = vec![(0.0, 20_000.0)];
        ranges.extend([(0.0, 1e6); 3]);
        ranges.extend([(0.0, 4e7); 3]);
        ranges.extend([
            (100.0, 299.99),
            (100.0, 399.99),
            (100.01, 300.0),
            (100.0, 400.0),
            (100.0, 600.0),
        ]);
        ranges.extend([(0.0, 500.0); 3]);
        ranges.extend([(0.01, 300.0), (0.01, 400.0)]);
        ranges.extend([(-4.7, 6.25); 5]);
        bounds(&ranges)
    }

    /// The report's best known solution, for the report's tolerance; `None` for another.
    fn optimum(&self) -> Option<Optimum<Reals>> {
        (self.tolerance == EQUALITY_TOLERANCE).then(|| {
            Optimum::best_known(
                236.430_975_504_001,
                vec![reals(&[
                    236.430_975_504_001_054,
                    135.828_471_517_324_63,
                    204.818_152_544_824_585,
                    6_446.546_540_594_364_16,
                    3_007_540.839_402_155_95,
                    4_074_188.657_713_419_29,
                    32_918_270.502_895_288_2,
                    130.075_408_394_314_167,
                    170.817_294_970_528_621,
                    299.924_591_605_478_554,
                    399.258_113_423_595_205,
                    330.817_294_971_142_758,
                    184.518_312_308_970_65,
                    248.646_702_396_474_24,
                    127.658_546_694_545_862,
                    269.182_627_528_746_707,
                    160.000_016_724_090_955,
                    5.297_882_881_026_805_71,
                    5.135_297_359_039_457_28,
                    5.595_315_264_440_688_27,
                    5.434_444_793_144_534_99,
                    5.075_174_535_358_343_95,
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

    /// g₁, then h₁…h₁₉ of the report's eq. 46.
    fn constraints(&self, x: &Reals) -> Constraints {
        let x = |i: usize| x[i - 1];
        Constraints::new(
            vec![-x(1) + math::powf(x(2), 0.6) + math::powf(x(3), 0.6) + math::powf(x(4), 0.6)],
            vec![
                x(5) - 100_000.0 * x(8) + 1e7,
                x(6) + 100_000.0 * x(8) - 100_000.0 * x(9),
                x(7) + 100_000.0 * x(9) - 5e7,
                x(5) + 100_000.0 * x(10) - 3.3e7,
                x(6) + 100_000.0 * x(11) - 4.4e7,
                x(7) + 100_000.0 * x(12) - 6.6e7,
                x(5) - 120.0 * x(2) * x(13),
                x(6) - 80.0 * x(3) * x(14),
                x(7) - 40.0 * x(4) * x(15),
                x(8) - x(11) + x(16),
                x(9) - x(12) + x(17),
                -x(18) + math::ln(x(10) - 100.0),
                -x(19) + math::ln(-x(8) + 300.0),
                -x(20) + math::ln(x(16)),
                -x(21) + math::ln(-x(9) + 400.0),
                -x(22) + math::ln(x(17)),
                -x(8) - x(10) + x(13) * x(18) - x(13) * x(19) + 400.0,
                x(8) - x(9) - x(11) + x(14) * x(20) - x(14) * x(21) + 400.0,
                x(9) - x(12) - 4.605_17 * x(15) + x(15) * x(22) + 100.0,
            ],
        )
    }
}

// ---- g23 -----------------------------------------------------------------------------------------

/// g23: `−9x₅ − 15x₈ + 6x₁ + 16x₂ + 10(x₆ + x₇)`, linear in 9 dimensions, with 2 nonlinear
/// inequalities, 3 linear equalities and 1 nonlinear equality.
///
/// Bounds x₁, x₂, x₆ ∈ [0, 300], x₃, x₅, x₇ ∈ [0, 100], x₄, x₈ ∈ [0, 200], x₉ ∈ [0.01, 0.03];
/// best known −400.055099999999584 at the report's x*, which meets the equalities within the
/// tolerance δ only, and where g₂ is active. The report prints x* with 8 numbers for 9 variables,
/// a comma missing in its last one, "2000.0100000100000100008": x₈ = 200 and
/// x₉ = 0.0100000100000100008, which evaluate to its value. Not proven optimal.
///
/// The report's eqs. 47-48 (pp. 14-15), after Xia, Q. *Global optimization test problems* (the
/// report's reference 10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G23 {
    tolerance: f64,
}

impl G23 {
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
        let x = |i: usize| x[i - 1];
        -9.0 * x(5) - 15.0 * x(8) + 6.0 * x(1) + 16.0 * x(2) + 10.0 * (x(6) + x(7))
    }
}

impl Default for G23 {
    /// The problem with the report's tolerance, [`EQUALITY_TOLERANCE`].
    fn default() -> Self {
        Self::with_tolerance(EQUALITY_TOLERANCE)
    }
}

cec2006!(G23, |problem| problem.tolerance);

impl Problem for G23 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G23"
    }

    fn representation(&self) -> Real {
        bounds(&[
            (0.0, 300.0),
            (0.0, 300.0),
            (0.0, 100.0),
            (0.0, 200.0),
            (0.0, 100.0),
            (0.0, 300.0),
            (0.0, 100.0),
            (0.0, 200.0),
            (0.01, 0.03),
        ])
    }

    /// The report's best known solution, for the report's tolerance; `None` for another.
    fn optimum(&self) -> Option<Optimum<Reals>> {
        (self.tolerance == EQUALITY_TOLERANCE).then(|| {
            Optimum::best_known(
                -400.055_099_999_999_584,
                vec![reals(&[
                    0.005_100_000_000_002_594_65,
                    99.994_700_000_000_051_4,
                    9.019_201_629_960_458_97e-18,
                    99.999_900_000_000_053_5,
                    0.000_100_000_000_027_086_086,
                    2.757_006_833_895_845_42e-14,
                    99.999_999_999_999_957_4,
                    200.0,
                    0.010_000_010_000_010_000_8,
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

    /// g₁, g₂, then h₁…h₄ of the report's eq. 48.
    fn constraints(&self, x: &Reals) -> Constraints {
        let x = |i: usize| x[i - 1];
        Constraints::new(
            vec![
                x(9) * x(3) + 0.02 * x(6) - 0.025 * x(5),
                x(9) * x(4) + 0.02 * x(7) - 0.015 * x(8),
            ],
            vec![
                x(1) + x(2) - x(3) - x(4),
                0.03 * x(1) + 0.01 * x(2) - x(9) * (x(3) + x(4)),
                x(3) + x(6) - x(5),
                x(4) + x(7) - x(8),
            ],
        )
    }
}

// ---- g24 -----------------------------------------------------------------------------------------

/// g24: `−x₁ − x₂` in 2 dimensions, subject to `x₂ ≤ 2x₁⁴ − 8x₁³ + 8x₁² + 2` and
/// `x₂ ≤ 4x₁⁴ − 32x₁³ + 88x₁² − 96x₁ + 36`.
///
/// Bounds x₁ ∈ [0, 3], x₂ ∈ [0, 4]; minimum −5.50801327159536 at the report's x*, where both
/// constraints are active. The report prints x* as "2.329520197477623.17849307411774", a comma
/// missing: x = (2.32952019747762, 3.17849307411774).
///
/// The second bound on x₂ is 4((x₁ − 1)(x₁ − 3))², 0 at x₁ = 1: the report's "two disconnected
/// sub-regions" of the feasible region meet at (1, 0). The first is 2(x₁(x₁ − 2))² + 2. At each
/// x₁ the best x₂ is the least of the two bounds and 4, and so the minimum is on a function of x₁
/// alone, where the two bounds cross, at a root of x₁⁴ − 12x₁³ + 40x₁² − 48x₁ + 17.
///
/// The report's eqs. 49-50 (p. 15), after Floudas, C. A., Pardalos, P. M., Adjiman, C. S.,
/// Esposito, W. R., Gümüş, Z. H., Harding, S. T., Klepeis, J. L., Meyer, C. A. and Schweiger, C. A.
/// (1999). *Handbook of Test Problems in Local and Global Optimization.* Kluwer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct G24;

impl G24 {
    fn value(&self, x: &Reals) -> f64 {
        -x[0] - x[1]
    }
}

cec2006!(G24, inequalities = 2);

impl Problem for G24 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "G24"
    }

    fn representation(&self) -> Real {
        bounds(&[(0.0, 3.0), (0.0, 4.0)])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(
            -5.508_013_271_595_36,
            vec![reals(&[2.329_520_197_477_62, 3.178_493_074_117_74])],
        ))
    }

    fn reference(&self) -> &'static str {
        REPORT
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REPORT_URL)
    }

    /// g₁ and g₂ of the report's eq. 50.
    fn constraints(&self, x: &Reals) -> Constraints {
        let (x1, x2) = (x[0], x[1]);
        let (square, cube, fourth) = (x1 * x1, x1 * x1 * x1, x1 * x1 * x1 * x1);
        Constraints::new(
            vec![
                -2.0 * fourth + 8.0 * cube - 8.0 * square + x2 - 2.0,
                -4.0 * fourth + 32.0 * cube - 88.0 * square + 96.0 * x1 + x2 - 36.0,
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
        let a = 2000.0 * math::sin(-0.25f64) + 894.8;
        let (value, violation) = G05::default().evaluate(&reals(&[a, a, 0.0, 0.0]));
        assert_close(
            value,
            5.0 * a + 0.000_001 * math::powi(a, 3) + (0.000_002 / 3.0) * math::powi(a, 3),
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

    #[test]
    fn g07() {
        check_optimum(&G07, 1e-13, 1e-12);
        let x = G07.optimum().expect("known").solutions()[0].clone();
        assert_active(&G07.constraints(&x), &[1, 2, 3, 4, 5, 6], 1e-9);
        // at the origin: f = 100 + 4·25 + 9 + 2 + 7·121 + 2·100 + 49 + 45 = 1352, and g₆ = 2·4,
        // g₇ = 0.5·64 + 2·16 − 30 = 34 and g₈ = 12·64 = 768 are violated (g₂ = 0 is met)
        let origin = Reals::from(vec![0.0; 10]);
        assert_eq!(G07.evaluate(&origin), (1352.0, 810.0));
        assert_eq!(G07.constraints(&origin).len(), 8);
        assert_eq!(G07.representation().bounds()[9], -10.0..=10.0);
    }

    #[test]
    fn g08() {
        check_optimum(&G08, 1e-14, 0.0);
        // at (1/4, 17/4): both sines are 1, so f = −1 / ((1/4)³ · 9/2) = −128/9; g₂ = 1 − 1/4 +
        // 1/16 is violated, g₁ = 1/16 − 17/4 + 1 is met
        let (value, violation) = G08.evaluate(&reals(&[0.25, 4.25]));
        assert_close(value, -128.0 / 9.0, 1e-12);
        assert_eq!(violation, 0.8125);
        // at (1, 4): sin(2π) = 0 to rounding, g₁ = −2 and g₂ = 0
        let (value, violation) = G08.evaluate(&reals(&[1.0, 4.0]));
        assert!(value.abs() < 1e-40 && violation == 0.0);
        // 0/0 at x₁ = 0
        assert!(G08.evaluate(&reals(&[0.0, 5.0])).0.is_nan());
    }

    #[test]
    fn g09() {
        check_optimum(&G09, 1e-14, 1e-12);
        let x = G09.optimum().expect("known").solutions()[0].clone();
        assert_active(&G09.constraints(&x), &[1, 4], 1e-9);
        // at the origin: f = 100 + 5·144 + 3·121 = 1183, feasible (g₄ = 0)
        assert_eq!(G09.evaluate(&Reals::from(vec![0.0; 7])), (1183.0, 0.0));
    }

    #[test]
    fn g10() {
        check_optimum(&G10, 1e-14, 0.0);
        let x = G10.optimum().expect("known").solutions()[0].clone();
        // all six are active, not only the three the report's text names
        assert_active(&G10.constraints(&x), &[1, 2, 3, 4, 5, 6], 1e-9);
        // at the lower corner (100, 1000, 1000, 10, …, 10): f = 2100, g₅ = 0, and g₆ =
        // −10000 + 1250000 + 10000 − 25000 is violated
        let corner = reals(&[100.0, 1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
        assert_eq!(G10.evaluate(&corner), (2100.0, 1_225_000.0));
        assert_eq!(G10.representation().bounds()[1], 1000.0..=10_000.0);
    }

    #[test]
    fn g11() {
        check_optimum(&G11::default(), 1e-15, 0.0);
        // the report's f(x*), 0.7499 = 3/4 − δ
        let optimum = G11::default().optimum().expect("known");
        assert_close(optimum.value(), 0.7499, 1e-15);
        assert_eq!(optimum.solutions().len(), 2);
        // the report's x*
        let report = reals(&[-0.707_036_070_037_170_616, 0.500_000_004_333_606_807]);
        let (value, violation) = G11::default().evaluate(&report);
        assert_close(value, 0.7499, 1e-9);
        assert_eq!(violation, 0.0);
        // on the parabola at x₁ = 0: f = 1, feasible; at (0, 1): f = 0, h = 1
        assert_eq!(G11::default().evaluate(&reals(&[0.0, 0.0])), (1.0, 0.0));
        let (value, violation) = G11::default().evaluate(&reals(&[0.0, 1.0]));
        assert_eq!(value, 0.0);
        assert_close(violation, 1.0 - EQUALITY_TOLERANCE, 1e-15);
        // without the tolerance: 3/4 at (±1/√2, 1/2)
        let exact = G11::with_tolerance(0.0).optimum().expect("known");
        assert_eq!(exact.value(), 0.75);
        assert_close(
            exact.solutions()[1][0],
            std::f64::consts::FRAC_1_SQRT_2,
            1e-15,
        );
        // with δ ≥ 1/2 the best is x₁ = 0, x₂ = min(1, δ): (1 − 0.6)² for δ = 0.6, 0 for δ = 1
        let wide = G11::with_tolerance(0.6).optimum().expect("known");
        assert_close(wide.value(), 0.16, 1e-15);
        assert_eq!(wide.solutions(), [reals(&[0.0, 0.6])]);
        assert_eq!(
            G11::with_tolerance(1.0).optimum().expect("known").value(),
            0.0
        );
        check_optimum(&G11::with_tolerance(0.6), 1e-15, 0.0);
    }

    #[test]
    fn g12() {
        check_optimum(&G12, 0.0, 0.0);
        // the center of the sphere at (1, 1, 1): f = −(100 − 3·16)/100, feasible
        assert_eq!(G12.evaluate(&reals(&[1.0, 1.0, 1.0])), (-0.52, 0.0));
        // halfway between centers: 3·0.5² from the nearest, less 0.0625; f = −(100 − 3·3.5²)/100
        let (value, violation) = G12.evaluate(&reals(&[1.5, 1.5, 1.5]));
        assert_close(value, -0.6325, 1e-15);
        assert_eq!(violation, 0.6875);
        // the corners are 1 from the nearest center in each gene: 3 − 0.0625; f = −(100 − 75)/100
        for corner in [0.0, 10.0] {
            let (value, violation) = G12.evaluate(&Reals::from(vec![corner; 3]));
            assert_eq!((value, violation), (-0.25, 2.9375));
        }
        // the nearest center gives the minimum over all 729
        let mut rng = crate::StreamRng::seed_from_u64(3);
        let real = G12.representation();
        for _ in 0..200 {
            let x = real.random_genome(&mut rng);
            let mut least = f64::INFINITY;
            for p in 1..=9 {
                for q in 1..=9 {
                    for r in 1..=9 {
                        let (p, q, r) = (f64::from(p), f64::from(q), f64::from(r));
                        let d = (x[0] - p).powi(2) + (x[1] - q).powi(2) + (x[2] - r).powi(2);
                        least = least.min(d - 0.0625);
                    }
                }
            }
            assert_close(G12.constraints(&x).inequalities()[0], least, 1e-12);
        }
    }

    #[test]
    fn g13() {
        check_optimum(&G13::default(), 1e-14, 1e-12);
        let x = G13::default().optimum().expect("known").solutions()[0].clone();
        for h in G13::default().constraints(&x).equalities() {
            assert!(h.abs() <= EQUALITY_TOLERANCE * (1.0 + 1e-9), "{h}");
        }
        // at the origin: f = e⁰ = 1, and h = (−10, 0, 1)
        let (value, violation) = G13::default().evaluate(&Reals::from(vec![0.0; 5]));
        assert_eq!(value, 1.0);
        assert_close(violation, 11.0 - 2.0 * EQUALITY_TOLERANCE, 1e-15);
        assert!(G13::with_tolerance(0.0).optimum().is_none());
    }

    #[test]
    fn g14() {
        check_optimum(&G14::default(), 1e-14, 1e-12);
        // all genes 0.1: Σ x = 1, so f = 0.1 Σ c + ln 0.1, and h = (−1.3, −0.5, −0.4)
        let tenths = Reals::from(vec![0.1; 10]);
        let (value, violation) = G14::default().evaluate(&tenths);
        assert_close(value, 0.1 * -186.577 + math::ln(0.1), 1e-14);
        assert_close(violation, 2.2 - 3.0 * EQUALITY_TOLERANCE, 1e-14);
        // f is homogeneous: doubling x doubles it
        let x = G14::default().optimum().expect("known").solutions()[0].clone();
        let doubled: Reals = x.iter().map(|xi| 2.0 * xi).collect();
        assert_close(
            G14::default().evaluate(&doubled).0,
            2.0 * G14::default().evaluate(&x).0,
            1e-14,
        );
        // 0 · ln 0 at 0; the lower bound is finite
        let mut zero = vec![0.1; 10];
        zero[0] = 0.0;
        assert!(G14::default().evaluate(&Reals::from(zero)).0.is_nan());
        let (value, _) = G14::default().evaluate(&Reals::from(vec![G14_LOW; 10]));
        assert!(value.is_finite());
        assert!(G14::with_tolerance(0.0).optimum().is_none());
    }

    #[test]
    fn g15() {
        check_optimum(&G15::default(), 1e-14, 1e-12);
        // (5, 0, 0) is on the sphere, h₂ = 40 − 56; f = 1000 − 25
        let (value, violation) = G15::default().evaluate(&reals(&[5.0, 0.0, 0.0]));
        assert_eq!(value, 975.0);
        assert_close(violation, 16.0 - EQUALITY_TOLERANCE, 1e-15);
        // (0, 4, 0) is on the plane, h₁ = 16 − 25; f = 1000 − 32
        let (value, violation) = G15::default().evaluate(&reals(&[0.0, 4.0, 0.0]));
        assert_eq!(value, 968.0);
        assert_close(violation, 9.0 - EQUALITY_TOLERANCE, 1e-15);
        assert!(G15::with_tolerance(0.0).optimum().is_none());
    }

    #[test]
    fn g16() {
        check_optimum(&G16, 1e-14, 0.0);
        let x = G16.optimum().expect("known").solutions()[0].clone();
        // the report's x₂*, 68.5999999999999943, is the lower bound 68.6 in double precision
        assert_eq!(x[1], 68.6);
        let constraints = G16.constraints(&x);
        assert_eq!(constraints.len(), 38);
        assert_active(&constraints, &[2, 3, 4, 5, 36], 1e-8);
        // at (800, 100, 50, 200, 50): y₁ = 100 + 50 + 41.6 and y₂ = 12.5/(0.024·200 − 4.62) + 12
        let point = reals(&[800.0, 100.0, 50.0, 200.0, 50.0]);
        let constraints = G16.constraints(&point);
        let g = constraints.inequalities();
        let (y1, y2) = (191.6, 12.5 / 0.18 + 12.0);
        assert_eq!(g[1], 50.0 - 150.0);
        for (i, expected) in [
            (4, 213.1 - y1),
            (5, y1 - 405.23),
            (6, 17.505 - y2),
            (7, y2 - 1053.6667),
        ] {
            assert_close(g[i], expected, 1e-12);
        }
    }

    #[test]
    fn g17() {
        check_optimum(&G17::default(), 1e-15, 0.0);
        let problem = G17::default();
        // the report's x*, which is feasible and evaluates to 8853.53401643571
        let report = reals(&[
            201.784_467_214_523_659,
            99.999_999_999_999_900_5,
            383.071_034_852_773_266,
            420.0,
            -10.907_658_451_429_265_2,
            0.073_148_231_208_428_712_8,
        ]);
        let (value, violation) = problem.evaluate(&report);
        assert_close(value, 8_853.534_016_435_71, 1e-13);
        assert_eq!(violation, 0.0);
        // the report's f(x*) is 30(x₁ + h₁) + 28(x₂ + h₂): the right-hand sides of h₁ and h₂
        let constraints = problem.constraints(&report);
        let h = constraints.equalities();
        assert_close(
            30.0 * (report[0] + h[0]) + 28.0 * (report[1] + h[1]),
            8_853.539_674_806_48,
            1e-13,
        );
        // the pieces, the last including the upper bounds
        let x = |x1: f64, x2: f64| reals(&[x1, x2, 380.0, 380.0, 0.0, 0.0]);
        assert_eq!(
            problem.evaluate(&x(299.0, 99.0)).0,
            30.0 * 299.0 + 28.0 * 99.0
        );
        assert_eq!(
            problem.evaluate(&x(300.0, 100.0)).0,
            31.0 * 300.0 + 29.0 * 100.0
        );
        assert_eq!(problem.evaluate(&x(0.0, 200.0)).0, 30.0 * 200.0);
        assert_eq!(
            problem.evaluate(&x(400.0, 1000.0)).0,
            31.0 * 400.0 + 30.0 * 1000.0
        );
        // with x₃ = x₄ and x₆ = 0, h₁ + x₁ − 300 = h₂ + x₂ and h₃ + x₅ = h₄ − 200
        let point = reals(&[50.0, 60.0, 380.0, 380.0, 70.0, 0.0]);
        let constraints = problem.constraints(&point);
        let h = constraints.equalities();
        assert_close(h[0] + 50.0 - 300.0, h[1] + 60.0, 1e-12);
        assert_close(h[2] + 70.0, h[3] - 200.0, 1e-12);
        assert!(G17::with_tolerance(0.0).optimum().is_none());
    }

    #[test]
    fn g18() {
        check_optimum(&G18, 1e-14, 0.0);
        let x = G18.optimum().expect("known").solutions()[0].clone();
        assert_active(&G18.constraints(&x), &[1, 3, 4, 6, 7, 9], 1e-9);
        // the best known value is −√3/2 to its digits
        assert_close(
            G18.optimum().expect("known").value(),
            -(3f64.sqrt()) / 2.0,
            1e-15,
        );
        // at the origin: f = 0, g₁…g₉ = −1 and g₁₀…g₁₃ = 0
        let mut origin = vec![0.0; 9];
        assert_eq!(G18.evaluate(&Reals::from(origin.clone())), (0.0, 0.0));
        // x₉ = 2 alone: g₂ = 4 − 1, g₄ = (0 − 2)² − 1 and g₉ = (0 − 2)² − 1
        origin[8] = 2.0;
        assert_eq!(G18.evaluate(&Reals::from(origin)), (0.0, 9.0));
        assert_eq!(G18.representation().bounds()[8], 0.0..=20.0);
    }
    #[test]
    fn g19() {
        check_optimum(&G19, 1e-14, 1e-13);
        let x = G19.optimum().expect("known").solutions()[0].clone();
        // all five are active, where the report's table 3 counts none
        assert_active(&G19.constraints(&x), &[1, 2, 3, 4, 5], 1e-13);
        // table 1's c is symmetric
        for (i, row) in G19_C.iter().enumerate() {
            for (j, &c) in row.iter().enumerate() {
                assert_eq!(c, G19_C[j][i]);
            }
        }
        // at the origin: f = 0, and gⱼ = −eⱼ, violated by 15 + 27 + 36 + 18 + 12
        assert_eq!(G19.evaluate(&Reals::from(vec![0.0; 15])), (0.0, 108.0));
        // x₁₁…x₁₅ = 1 alone: f = Σ cᵢⱼ + 2 Σ dⱼ = 50 + 60, and gⱼ = −2 Σᵢ cᵢⱼ − 3dⱼ − eⱼ, with
        // column sums 22, 14, −22, 14, 22 of c: (−41, −25, 50, −28, −38)
        let mut ones = vec![0.0; 15];
        ones[10..].fill(1.0);
        let ones = Reals::from(ones);
        assert_eq!(G19.evaluate(&ones), (110.0, 50.0));
        assert_eq!(
            G19.constraints(&ones).inequalities(),
            [-41.0, -25.0, 50.0, -28.0, -38.0]
        );
        // x₁…x₁₀ = 10 alone: f = −10 Σ bᵢ = 1452.5, and gⱼ = −eⱼ + 10 Σᵢ aᵢⱼ, with column sums
        // −17.5, −11, −4, 0.4, 2.2 of a: g₄ = 22 and g₅ = 34 are violated
        let mut tens = vec![10.0; 15];
        tens[10..].fill(0.0);
        let (value, violation) = G19.evaluate(&Reals::from(tens));
        assert_close(value, 1452.5, 1e-15);
        assert_close(violation, 56.0, 1e-14);
    }

    #[test]
    fn g20() {
        let problem = G20::default();
        let optimum = problem.optimum().expect("known");
        assert!(!optimum.is_proven());
        let x = optimum.solutions()[0].clone();
        problem.representation().validate(&x).expect("in bounds");
        // the report's x* evaluates to its table 4's value, which it truncates
        let (value, violation) = problem.evaluate(&x);
        assert_close(value, 0.204_979_400_2, 1e-9);
        assert_close(value, 0.204_979_400_285_636, 1e-13);
        // and is infeasible: g₁ = (x₁ + x₁₃)/(Σ x + e₁) = 0.1438, and the equalities are met
        // within the tolerance
        let constraints = problem.constraints(&x);
        assert_eq!(constraints.len(), 20);
        let g1 = constraints.inequalities()[0];
        assert_close(g1, (x[0] + x[12]) / (x.iter().sum::<f64>() + 0.1), 1e-15);
        assert!((0.1437..0.1438).contains(&g1), "{g1}");
        assert!(violation - g1 < 1e-12, "{violation}");
        for h in constraints.equalities() {
            assert!(h.abs() <= EQUALITY_TOLERANCE * (1.0 + 1e-9), "{h}");
        }
        // no solution is feasible. With the inequalities met, the 12 variables they bound are 0;
        // h₁…h₁₂ give xᵢ₊₁₂ in proportion to cᵢxᵢ, and Σᵢ₌₁₃²⁴ xᵢ is least, for a given
        // Σᵢ₌₁₃²⁴ xᵢ/bᵢ, with only x₅ and x₁₀ of x₁…x₁₂. Here that point, scaled to meet h₁₄: every
        // other constraint is met, but Σ x exceeds 1 by 0.29
        let (u5, u10) = (22.3 / 67.0, 44.7 / 67.0);
        let liquid = 1e-6;
        let mut point = vec![0.0; 24];
        point[4] = G20_B[4] * u5 * liquid;
        point[9] = G20_B[9] * u10 * liquid;
        let over_d = point[4] / G20_D[4] + point[9] / G20_D[9];
        let vapor = (1.671 - over_d) / (0.7302 * 530.0 * (14.7 / 40.0));
        point[16] = G20_B[4] * G20_C[4] * u5 / 40.0 * vapor;
        point[21] = G20_B[9] * G20_C[9] * u10 / 40.0 * vapor;
        let constraints = problem.constraints(&Reals::from(point));
        assert!(constraints.inequalities().iter().all(|&g| g == 0.0));
        let h = constraints.equalities();
        for (i, h) in h.iter().enumerate().filter(|&(i, _)| i != 12) {
            assert!(h.abs() < 1e-12, "h{}: {h}", i + 1);
        }
        assert!((0.287..0.288).contains(&h[12]), "{}", h[12]);
        // undefined where x₁…x₁₂ or x₁₃…x₂₄ are all 0
        let (value, violation) = problem.evaluate(&Reals::from(vec![0.0; 24]));
        assert_eq!(value, 0.0);
        assert!(violation.is_nan());
        // all genes 1/24: Σ x = 1 and f = Σ aᵢ / 24
        let even = Reals::from(vec![1.0 / 24.0; 24]);
        assert_close(
            problem.evaluate(&even).0,
            2.0 * G20_A.iter().sum::<f64>() / 24.0,
            1e-15,
        );
        assert_close(problem.constraints(&even).equalities()[12], 0.0, 1e-15);
        assert!(G20::with_tolerance(0.0).optimum().is_none());
    }

    #[test]
    fn g21() {
        let problem = G21::default();
        // the report's x* is at the tolerance of every equality, and beyond it by rounding
        check_optimum(&problem, 1e-15, 1e-11);
        let x = problem.optimum().expect("known").solutions()[0].clone();
        let constraints = problem.constraints(&x);
        assert_active(&constraints, &[1], 1e-12);
        for h in constraints.equalities() {
            assert!((h.abs() - EQUALITY_TOLERANCE).abs() < 1e-11, "{h}");
        }
        // at x₄ = 100, x₅ = ln 800, x₆ = ln 400 and x₇ = ln 500, h₂…h₅ are 0 and
        // h₁ = 5000 (x₅ − x₆) = 5000 ln 2; g₁ = −x₁
        let point = reals(&[
            50.0,
            0.0,
            0.0,
            100.0,
            math::ln(800.0),
            math::ln(400.0),
            math::ln(500.0),
        ]);
        let constraints = problem.constraints(&point);
        assert_eq!(constraints.inequalities(), [-50.0]);
        let h = constraints.equalities();
        assert_close(h[0], 5000.0 * std::f64::consts::LN_2, 1e-12);
        assert!(h[1..].iter().all(|&h| h.abs() < 1e-12), "{h:?}");
        let (value, violation) = problem.evaluate(&point);
        assert_eq!(value, 50.0);
        assert_close(
            violation,
            5000.0 * std::f64::consts::LN_2 - EQUALITY_TOLERANCE,
            1e-12,
        );
        assert!(G21::with_tolerance(0.0).optimum().is_none());
        // at x₄ = 100, x₂ = 0 and x₃ = 25 ln 2 meet every equality exactly, with g₁ active (x₁
        // nudged up by 1e-9, above its rounding): f = 35 (25 ln 2)^0.6
        let x3 = 25.0 * std::f64::consts::LN_2;
        let x1 = 35.0 * math::powf(x3, 0.6) + 1e-9;
        let point = reals(&[
            x1,
            0.0,
            x3,
            100.0,
            math::ln(800.0),
            math::ln(400.0),
            math::ln(500.0),
        ]);
        let (value, violation) = problem.evaluate(&point);
        assert_eq!(violation, 0.0);
        assert_close(value - 1e-9, 193.788, 1e-5);
        assert!(
            problem
                .constraints(&point)
                .equalities()
                .iter()
                .all(|h| h.abs() < 1e-11)
        );
    }

    #[test]
    fn g22() {
        let problem = G22::default();
        check_optimum(&problem, 1e-15, 0.0);
        let x = problem.optimum().expect("known").solutions()[0].clone();
        let constraints = problem.constraints(&x);
        assert_eq!(constraints.len(), 20);
        // g₁ is nearly active
        let g1 = constraints.inequalities()[0];
        assert!((-3e-7..-1e-7).contains(&g1), "{g1}");
        for h in constraints.equalities() {
            assert!(h.abs() <= EQUALITY_TOLERANCE, "{h}");
        }
        // at the lower corner: h₁ = −10⁷ + 10⁷, h₂ = 10⁷ − 10⁷, h₃ = 10⁷ − 5·10⁷,
        // h₇ = h₈ = h₉ = 0, h₁₀ = h₁₁ = 0.01 and h₁₂ = 4.7 + ln(100.01 − 100); g₁ = 0
        let low: Reals = problem
            .representation()
            .bounds()
            .iter()
            .map(|range| *range.start())
            .collect();
        let constraints = problem.constraints(&low);
        assert_eq!(constraints.inequalities(), [0.0]);
        let h = constraints.equalities();
        assert_eq!(h[..3], [0.0, 0.0, -4e7]);
        assert_eq!(h[6..9], [0.0, 0.0, 0.0]);
        assert_close(h[9], 0.01, 1e-15);
        assert_close(h[10], 0.01, 1e-15);
        assert_close(h[11], 4.7 + math::ln(100.01 - 100.0), 1e-15);
        assert_eq!(problem.evaluate(&low).0, 0.0);
        assert_eq!(problem.representation().bounds()[9], 100.01..=300.0);
        assert!(G22::with_tolerance(0.0).optimum().is_none());
        // the equalities solved for all but x₁, x₈ and x₉, at x₈ = 130 and x₉ = 170: every one is
        // met to rounding, g₁ is active (x₁ nudged up by 1e-9, above its rounding), and f is
        // 0.0607 below the report's best known
        let (x8, x9) = (130.0, 170.0);
        let (x5, x6, x7) = (1e5 * x8 - 1e7, 1e5 * (x9 - x8), 5e7 - 1e5 * x9);
        let (x10, x11, x12) = (430.0 - x8, 440.0 - x9 + x8, 160.0 + x9);
        let (x16, x17) = (x11 - x8, x12 - x9);
        let (x18, x19) = (math::ln(x10 - 100.0), math::ln(300.0 - x8));
        let (x20, x21, x22) = (math::ln(x16), math::ln(400.0 - x9), math::ln(x17));
        let (x13, x14) = (30.0 / (x18 - x19), 40.0 / (x20 - x21));
        let x15 = 60.0 / (x22 - 4.605_17);
        let (x2, x3, x4) = (x5 / (120.0 * x13), x6 / (80.0 * x14), x7 / (40.0 * x15));
        let x1 = math::powf(x2, 0.6) + math::powf(x3, 0.6) + math::powf(x4, 0.6) + 1e-9;
        let point = reals(&[
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19,
            x20, x21, x22,
        ]);
        problem
            .representation()
            .validate(&point)
            .expect("in bounds");
        let (value, violation) = problem.evaluate(&point);
        assert_eq!(violation, 0.0);
        assert_close(value - 1e-9, 236.370_313_314_566, 1e-14);
        for h in problem.constraints(&point).equalities() {
            assert!(h.abs() < 1e-8, "{h}");
        }
        assert!(value < problem.optimum().expect("known").value() - 0.06);
    }

    #[test]
    fn g23() {
        let problem = G23::default();
        check_optimum(&problem, 1e-15, 1e-13);
        let x = problem.optimum().expect("known").solutions()[0].clone();
        let constraints = problem.constraints(&x);
        assert_active(&constraints, &[2], 0.0);
        for h in constraints.equalities() {
            assert!(h.abs() <= EQUALITY_TOLERANCE * (1.0 + 1e-9), "{h}");
        }
        // the report's x₈ and x₉, printed as one number
        assert_eq!((x[7], x[8]), (200.0, 0.010_000_010_000_010_000_8));
        // at the lower corner every constraint is 0: f = 0, feasible
        let low = reals(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01]);
        assert_eq!(problem.evaluate(&low), (0.0, 0.0));
        // x₂ = x₄ = x₇ = 100, x₈ = 200, x₉ = 0.01 meets every constraint exactly, g₂ active:
        // f = −15·200 + 16·100 + 10·100 = −400, 0.0551 above the best known, which uses the
        // tolerance
        let point = reals(&[0.0, 100.0, 0.0, 100.0, 0.0, 0.0, 100.0, 200.0, 0.01]);
        assert_eq!(problem.evaluate(&point), (-400.0, 0.0));
        assert_eq!(problem.constraints(&point).inequalities(), [0.0, 0.0]);
        assert!(G23::with_tolerance(0.0).optimum().is_none());
    }

    #[test]
    fn g24() {
        check_optimum(&G24, 1e-15, 1e-12);
        let x = G24.optimum().expect("known").solutions()[0].clone();
        assert_active(&G24.constraints(&x), &[1, 2], 1e-12);
        // x₁* is where the two bounds on x₂ cross, a root of x₁⁴ − 12x₁³ + 40x₁² − 48x₁ + 17
        let x1 = x[0];
        let quartic = (((x1 - 12.0) * x1 + 40.0) * x1 - 48.0) * x1 + 17.0;
        assert!(quartic.abs() < 1e-12, "{quartic}");
        // the bounds on x₂: 2(x₁(x₁ − 2))² + 2 and 4((x₁ − 1)(x₁ − 3))²
        let first = |x1: f64| 2.0 * (x1 * (x1 - 2.0)).powi(2) + 2.0;
        let second = |x1: f64| 4.0 * ((x1 - 1.0) * (x1 - 3.0)).powi(2);
        for x1 in [0.0, 0.5, 1.0, 2.0, 2.5, 3.0] {
            let g = G24.constraints(&reals(&[x1, 0.0]));
            assert_close(g.inequalities()[0], -first(x1), 1e-14);
            assert_close(g.inequalities()[1], -second(x1), 1e-14);
        }
        // the two parts of the feasible region meet at (1, 0)
        assert_eq!(G24.evaluate(&reals(&[1.0, 0.0])), (-1.0, 0.0));
        assert_eq!(G24.evaluate(&reals(&[1.0, 0.5])), (-1.5, 0.5));
        // the minimum over x₁ of −x₁ − min(4, both bounds), on a grid: none below f*
        let optimum = G24.optimum().expect("known").value();
        for i in 0..=30_000 {
            let x1 = 3.0 * f64::from(i) / 30_000.0;
            let x2 = first(x1).min(second(x1)).min(4.0);
            assert!(-x1 - x2 >= optimum - 1e-12, "{x1}");
        }
    }
}
