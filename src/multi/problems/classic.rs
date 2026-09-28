//! The classic problems with two or three objectives: Schaffer's, Fonseca and Fleming's,
//! Kursawe's, Poloni's and Viennet's, unconstrained, and BNH, SRN, TNK, OSY and CONSTR,
//! constrained.

use super::{
    MultiProblem, Piece, das_dennis, divisions_for, evenly, non_dominated, pieces_front,
    spread_over,
};
use crate::constraint::at_most;
use crate::genome::{Real, Reals};
use crate::multi::MultiFitnessFunction;
use crate::problems::Constraints;
use std::f64::consts::{FRAC_PI_2, PI};

// the NSGA-II paper, which restates several of the problems in its tables I and V
const NSGA2: &str = "Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002). A fast and elitist \
                     multiobjective genetic algorithm: NSGA-II. IEEE Transactions on \
                     Evolutionary Computation 6(2): 182-197.";
const NSGA2_URL: &str = "https://doi.org/10.1109/4235.996017";

// bounds that are valid by construction
fn bounds<const N: usize>(ranges: [(f64, f64); N]) -> Real {
    Real::new(ranges.map(|(low, high)| low..=high)).expect("valid bounds")
}

// the total violation of constraints g(x) <= 0, as `Constraints::violation` adds it up
fn violation(constraints: &[f64]) -> f64 {
    constraints.iter().map(|&g| at_most(g, 0.0)).sum()
}

// the metadata of an unconstrained problem
macro_rules! metadata {
    ($name:literal, $reference:expr, $url:expr) => {
        fn name(&self) -> &'static str {
            $name
        }

        fn reference(&self) -> &'static str {
            $reference
        }

        fn reference_url(&self) -> Option<&'static str> {
            $url
        }
    };
}

// a constrained problem's fitness and constraints, from its objectives and its constraint values
// g(x) <= 0
macro_rules! constrained {
    ($name:ident, $count:literal) => {
        impl MultiFitnessFunction<Reals, 2> for $name {
            type Output = ([f64; 2], f64);

            /// The objective values of `x` and its constraint violation, 0 when it's feasible.
            ///
            /// # Panics
            ///
            /// If `x` has fewer than the problem's variables.
            fn evaluate(&self, x: &Reals) -> ([f64; 2], f64) {
                (self.objectives(x), violation(&self.values(x)))
            }
        }

        impl $name {
            /// The number of constraints.
            pub const CONSTRAINTS: usize = $count;
        }
    };
}

// the constraint methods of a constrained problem's `MultiProblem` implementation
macro_rules! constraint_methods {
    () => {
        fn constraint_count(&self) -> usize {
            Self::CONSTRAINTS
        }

        fn constraints(&self, genome: &Reals) -> Constraints {
            Constraints::new(self.values(genome).to_vec(), Vec::new())
        }
    };
}

// ---- Schaffer ------------------------------------------------------------------------------------

/// Schaffer's first problem (SCH1): `f₁ = x²`, `f₂ = (x − 2)²`, on one variable.
///
/// Bounds [−1000, 1000]. The optimal solutions are x in [0, 2], and the front is
/// `f₂ = (√f₁ − 2)²` for f₁ in [0, 4].
///
/// Schaffer, J. D. (1985). Multiple objective optimization with vector evaluated genetic
/// algorithms. *Proceedings of the First International Conference on Genetic Algorithms*: 93-100.
/// Definition and bounds as restated in Deb, Pratap, Agarwal and Meyarivan (2002, NSGA-II,
/// table I); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)). Other papers use other bounds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Schaffer1;

impl MultiFitnessFunction<Reals, 2> for Schaffer1 {
    type Output = [f64; 2];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` is empty.
    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        [x[0] * x[0], (x[0] - 2.0) * (x[0] - 2.0)]
    }
}

impl MultiProblem<2> for Schaffer1 {
    type Representation = Real;

    metadata!(
        "SCH1",
        "Schaffer, J. D. (1985). Multiple objective optimization with vector evaluated genetic \
         algorithms. Proceedings of the First International Conference on Genetic Algorithms: \
         93-100.",
        None
    );

    fn representation(&self) -> Real {
        bounds([(-1000.0, 1000.0)])
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
        let front = (0..points).map(|i| {
            let x = 2.0 * evenly(i, points);
            [x * x, (x - 2.0) * (x - 2.0)]
        });
        Some(front.collect())
    }

    fn ideal_point(&self) -> Option<[f64; 2]> {
        Some([0.0, 0.0])
    }

    fn nadir_point(&self) -> Option<[f64; 2]> {
        Some([4.0, 4.0])
    }
}

/// Schaffer's second problem (SCH2), on one variable: `f₁ = −x` for x ≤ 1, `x − 2` for
/// 1 < x ≤ 3, `4 − x` for 3 < x ≤ 4 and `x − 4` for x > 4; `f₂ = (x − 5)²`.
///
/// Bounds [−5, 10]. The optimal solutions are x in [1, 2) and [4, 5], and the front is in two
/// pieces: `f₂ = (f₁ − 3)²` for f₁ in [−1, 0), and `f₂ = (f₁ − 1)²` for f₁ in [0, 1]. At x = 2,
/// (0, 9) is dominated by x = 4, (0, 1).
///
/// Schaffer, J. D. (1985). Multiple objective optimization with vector evaluated genetic
/// algorithms. *Proceedings of the First International Conference on Genetic Algorithms*: 93-100.
/// Definition and bounds as restated in Van Veldhuizen, D. A. (1999). *Multiobjective Evolutionary
/// Algorithms: Classifications, Analyses, and New Innovations.* PhD thesis AFIT/DS/ENG/99-01, Air
/// Force Institute of Technology, table B.1, after Srinivas and Deb (1994); not yet checked against
/// the original ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Schaffer2;

impl MultiFitnessFunction<Reals, 2> for Schaffer2 {
    type Output = [f64; 2];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` is empty.
    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let x = x[0];
        let f1 = if x <= 1.0 {
            -x
        } else if x <= 3.0 {
            x - 2.0
        } else if x <= 4.0 {
            4.0 - x
        } else {
            x - 4.0
        };
        [f1, (x - 5.0) * (x - 5.0)]
    }
}

impl MultiProblem<2> for Schaffer2 {
    type Representation = Real;

    metadata!(
        "SCH2",
        "Schaffer, J. D. (1985). Multiple objective optimization with vector evaluated genetic \
         algorithms. Proceedings of the First International Conference on Genetic Algorithms: \
         93-100.",
        None
    );

    fn representation(&self) -> Real {
        bounds([(-5.0, 10.0)])
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
        // x in [1, 2), then x in [4, 5]
        let left = |t: f64| [t - 1.0, (t - 4.0) * (t - 4.0)];
        let right = |t: f64| [t, (t - 1.0) * (t - 1.0)];
        let pieces = [
            Piece {
                curve: &left,
                with_end: false,
            },
            Piece {
                curve: &right,
                with_end: true,
            },
        ];
        Some(pieces_front(&pieces, points))
    }

    fn ideal_point(&self) -> Option<[f64; 2]> {
        Some([-1.0, 0.0])
    }

    fn nadir_point(&self) -> Option<[f64; 2]> {
        Some([1.0, 16.0])
    }
}

// ---- Fonseca and Fleming ---------------------------------------------------------------------

/// Fonseca and Fleming's problem (FON): `f₁ = 1 − exp(−Σ (xᵢ − 1/√n)²)`,
/// `f₂ = 1 − exp(−Σ (xᵢ + 1/√n)²)`, in n variables.
///
/// Bounds [−4, 4]ⁿ; 3 variables by default. The optimal solutions have all variables equal, to
/// t in [−1/√n, 1/√n], and the front, the same for every n, is `f₁ = 1 − exp(−(s − 1)²)`,
/// `f₂ = 1 − exp(−(s + 1)²)` for s = t√n in [−1, 1] (derived from the definition): concave, from
/// (0, 1 − e⁻⁴) to (1 − e⁻⁴, 0).
///
/// Fonseca, C. M. and Fleming, P. J. (1995). An overview of evolutionary algorithms in
/// multiobjective optimization. *Evolutionary Computation* 3(1): 1-16. Definition and bounds as
/// restated in Deb, Thiele, Laumanns and Zitzler (2001, TIK-Report 112, eq. 1) and, for 3
/// variables, in Deb, Pratap, Agarwal and Meyarivan (2002, NSGA-II, table I); not yet checked
/// against the original ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FonsecaFleming {
    variables: usize,
}

impl FonsecaFleming {
    /// The problem with `variables` variables, at least 1.
    ///
    /// # Panics
    ///
    /// If `variables` is 0.
    pub fn new(variables: usize) -> Self {
        assert!(variables >= 1, "FON needs at least 1 variable");
        Self { variables }
    }

    /// The number of variables.
    pub fn variables(&self) -> usize {
        self.variables
    }
}

impl Default for FonsecaFleming {
    /// The problem with 3 variables.
    fn default() -> Self {
        Self::new(3)
    }
}

impl MultiFitnessFunction<Reals, 2> for FonsecaFleming {
    type Output = [f64; 2];

    /// The objective values of `x`.
    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let shift = 1.0 / (x.len() as f64).sqrt();
        let distance = |sign: f64| {
            x.iter()
                .map(|xi| (xi - sign * shift) * (xi - sign * shift))
                .sum::<f64>()
        };
        [1.0 - (-distance(1.0)).exp(), 1.0 - (-distance(-1.0)).exp()]
    }
}

// the front of FON at s = t√n in [−1, 1]
fn fonseca_fleming_front(s: f64) -> [f64; 2] {
    [
        1.0 - (-(s - 1.0) * (s - 1.0)).exp(),
        1.0 - (-(s + 1.0) * (s + 1.0)).exp(),
    ]
}

impl MultiProblem<2> for FonsecaFleming {
    type Representation = Real;

    metadata!(
        "FON",
        "Fonseca, C. M. and Fleming, P. J. (1995). An overview of evolutionary algorithms in \
         multiobjective optimization. Evolutionary Computation 3(1): 1-16.",
        Some("https://doi.org/10.1162/evco.1995.3.1.1")
    );

    fn representation(&self) -> Real {
        Real::uniform(self.variables, -4.0..=4.0).expect("valid bounds")
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
        let front = (0..points).map(|i| fonseca_fleming_front(1.0 - 2.0 * evenly(i, points)));
        Some(front.collect())
    }

    fn ideal_point(&self) -> Option<[f64; 2]> {
        Some([0.0, 0.0])
    }

    fn nadir_point(&self) -> Option<[f64; 2]> {
        let worst = 1.0 - (-4.0f64).exp();
        Some([worst, worst])
    }
}

// ---- Kursawe -------------------------------------------------------------------------------------

/// Kursawe's problem (KUR): `f₁ = Σᵢ₌₁ⁿ⁻¹ −10 exp(−0.2 √(xᵢ² + xᵢ₊₁²))`,
/// `f₂ = Σᵢ₌₁ⁿ (|xᵢ|^0.8 + 5 sin(xᵢ³))`, in n variables.
///
/// Bounds [−5, 5]ⁿ; 3 variables by default. The front is disconnected: for 3 variables, the point
/// (−20, 0) at x = 0 and three curves. It isn't known in closed form:
/// [`optimal_front`](MultiProblem::optimal_front) is `None`. Deb et al. (2002) and Van Veldhuizen
/// (1999) describe three regions, and plot the point apart from them.
///
/// Its ends are known (derived from the definition): f₁ is smallest, −10(n − 1), only at x = 0,
/// where f₂ = 0. f₂ is a sum of one term per variable, `|x|^0.8 + 5 sin(x³)`, whose minimum
/// over [−5, 5] is h* = −3.8757622790462816, only at x* = −1.1527408475499261 (a root of its
/// derivative, polished to 50 digits: its next best local minimum is 0.09 higher). So f₂ is
/// smallest, n h*, only at x = (x*, …, x*), where f₁ = −10(n − 1) exp(−0.2 √2 |x*|). The
/// [`ideal_point`](MultiProblem::ideal_point) is (−10(n − 1), n h*) and the
/// [`nadir_point`](MultiProblem::nadir_point) (−10(n − 1) exp(−0.2 √2 |x*|), 0): for 3 variables,
/// (−20, −11.627286837138845) and (−14.435463549038639, 0). The non-dominated points of a
/// 401 × 401 × 401 grid over the box, with x*, reach both and stay between them.
///
/// Kursawe, F. (1991). A variant of evolution strategies for vector optimization. *Parallel
/// Problem Solving from Nature*, LNCS 496: 193-197. The original (p. 196) prints f₁ summed to n
/// and `f₂ = Σ (|xᵢ|^0.8 + 5 sin(xᵢ)³)`, with no bounds or number of variables; its figure 2
/// looks like sin(xᵢ)³ with 2 variables. The definition here, with sin(xᵢ³), 3 variables and
/// bounds [−5, 5], is Deb, Pratap, Agarwal and Meyarivan's (2002, NSGA-II, table I), the form
/// the literature uses; Van Veldhuizen (1999, PhD thesis, table B.1) keeps sin(xᵢ)³ and sums f₁
/// to n − 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Kursawe {
    variables: usize,
}

impl Kursawe {
    /// The problem with `variables` variables, at least 2.
    ///
    /// # Panics
    ///
    /// If `variables` is below 2.
    pub fn new(variables: usize) -> Self {
        assert!(variables >= 2, "KUR needs at least 2 variables");
        Self { variables }
    }

    /// The number of variables.
    pub fn variables(&self) -> usize {
        self.variables
    }
}

impl Default for Kursawe {
    /// The problem with 3 variables.
    fn default() -> Self {
        Self::new(3)
    }
}

impl MultiFitnessFunction<Reals, 2> for Kursawe {
    type Output = [f64; 2];

    /// The objective values of `x`.
    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let f1 = x
            .windows(2)
            .map(|pair| -10.0 * (-0.2 * (pair[0] * pair[0] + pair[1] * pair[1]).sqrt()).exp())
            .sum();
        let f2 = x
            .iter()
            .map(|xi| xi.abs().powf(0.8) + 5.0 * (xi * xi * xi).sin())
            .sum();
        [f1, f2]
    }
}

impl MultiProblem<2> for Kursawe {
    type Representation = Real;

    metadata!(
        "KUR",
        "Kursawe, F. (1991). A variant of evolution strategies for vector optimization. Parallel \
         Problem Solving from Nature, LNCS 496: 193-197.",
        Some("https://doi.org/10.1007/BFb0029752")
    );

    fn representation(&self) -> Real {
        Real::uniform(self.variables, -5.0..=5.0).expect("valid bounds")
    }

    fn optimal_front(&self, _points: usize) -> Option<Vec<[f64; 2]>> {
        None
    }

    // f₁'s minimum, at x = 0, and f₂'s, n times the least of its term
    fn ideal_point(&self) -> Option<[f64; 2]> {
        let n = self.variables as f64;
        Some([-10.0 * (n - 1.0), n * KURSAWE_TERM])
    }

    // f₁ at f₂'s minimum, and f₂ at f₁'s
    fn nadir_point(&self) -> Option<[f64; 2]> {
        let n = self.variables as f64;
        Some([-10.0 * (n - 1.0) * KURSAWE_DECAY, 0.0])
    }
}

// the least of KUR's term of f₂, |x|^0.8 + 5 sin(x³), over [−5, 5], at x* = −1.1527408475499261
const KURSAWE_TERM: f64 = -3.875_762_279_046_281_6;
// exp(−0.2 √2 |x*|): a term of f₁, over −10, where every variable is x*
const KURSAWE_DECAY: f64 = 0.721_773_177_451_931_9;

// ---- Poloni --------------------------------------------------------------------------------------

/// Poloni's problem (POL): `f₁ = 1 + (A₁ − B₁)² + (A₂ − B₂)²`, `f₂ = (x₁ + 3)² + (x₂ + 1)²`,
/// with `A₁ = 0.5 sin 1 − 2 cos 1 + sin 2 − 1.5 cos 2`, `A₂ = 1.5 sin 1 − cos 1 + 2 sin 2 −
/// 0.5 cos 2`, `B₁ = 0.5 sin x₁ − 2 cos x₁ + sin x₂ − 1.5 cos x₂` and `B₂ = 1.5 sin x₁ − cos x₁ +
/// 2 sin x₂ − 0.5 cos x₂`.
///
/// Bounds [−π, π]². f₁ is 1 at (1, 2), where B = A. The front is disconnected and not known in
/// closed form: [`optimal_front`](MultiProblem::optimal_front) is `None`.
///
/// Its ends are known (derived from the definition): f₂ is 0 only at (−3, −1), where
/// f₁ = 16.772337779156782, and f₁ is 1 where B = A, which in the box is at (1, 2), where f₂ = 25,
/// and at (2.0228, 0.7307), where f₂ = 28.2237: (1, 2) dominates it. The
/// [`ideal_point`](MultiProblem::ideal_point) is (1, 0) and the
/// [`nadir_point`](MultiProblem::nadir_point) (16.772337779156782, 25). Newton's method from each
/// of the 2,696 points of a 2,001 × 2,001 grid over the box with f₁ < 1.01 finds only these two
/// solutions of B = A, and the non-dominated points of a grid over the box, with the two ends,
/// stay between the two points.
///
/// Poloni, C., Giurgevich, A., Onesti, L. and Pediroda, V. (2000). Hybridization of a
/// multi-objective genetic algorithm, a neural network and a classical optimizer for a complex
/// design problem in fluid dynamics. *Computer Methods in Applied Mechanics and Engineering*
/// 186(2-4): 403-420. It first appeared in Poloni et al. (1996, ECCOMAS '96, Wiley: 258-264)
/// and Poloni (1997, in *Genetic Algorithms in Engineering and Computer Science*, Wiley:
/// 397-414), which, as Van Veldhuizen (1999, PhD thesis, table B.1) notes, print it mistyped.
/// Definition and bounds as restated in Deb, Pratap, Agarwal and Meyarivan (2002, NSGA-II, table
/// I), minimized; Van Veldhuizen restates it as the maximization of the negated objectives, and
/// Rigoni and Poles (2005, Dagstuhl Seminar Proceedings 04461) minimize it with the same
/// constants. Not yet checked against the originals
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Poloni;

// A₁ and A₂, or B₁ and B₂ at (x₁, x₂)
fn poloni_terms(x1: f64, x2: f64) -> (f64, f64) {
    let (s1, c1, s2, c2) = (x1.sin(), x1.cos(), x2.sin(), x2.cos());
    (
        0.5 * s1 - 2.0 * c1 + s2 - 1.5 * c2,
        1.5 * s1 - c1 + 2.0 * s2 - 0.5 * c2,
    )
}

impl MultiFitnessFunction<Reals, 2> for Poloni {
    type Output = [f64; 2];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 2 genes.
    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let (x1, x2) = (x[0], x[1]);
        let (a1, a2) = poloni_terms(1.0, 2.0);
        let (b1, b2) = poloni_terms(x1, x2);
        [
            1.0 + (a1 - b1) * (a1 - b1) + (a2 - b2) * (a2 - b2),
            (x1 + 3.0) * (x1 + 3.0) + (x2 + 1.0) * (x2 + 1.0),
        ]
    }
}

impl MultiProblem<2> for Poloni {
    type Representation = Real;

    metadata!(
        "POL",
        "Poloni, C., Giurgevich, A., Onesti, L. and Pediroda, V. (2000). Hybridization of a \
         multi-objective genetic algorithm, a neural network and a classical optimizer for a \
         complex design problem in fluid dynamics. Computer Methods in Applied Mechanics and \
         Engineering 186(2-4): 403-420.",
        Some("https://doi.org/10.1016/S0045-7825(99)00394-1")
    );

    fn representation(&self) -> Real {
        bounds([(-PI, PI), (-PI, PI)])
    }

    fn optimal_front(&self, _points: usize) -> Option<Vec<[f64; 2]>> {
        None
    }

    // f₁'s minimum at (1, 2), and f₂'s at (−3, −1)
    fn ideal_point(&self) -> Option<[f64; 2]> {
        Some([1.0, 0.0])
    }

    // f₁ at (−3, −1), to 50 digits, and f₂ at (1, 2)
    fn nadir_point(&self) -> Option<[f64; 2]> {
        Some([16.772_337_779_156_782, 25.0])
    }
}

// ---- Viennet -------------------------------------------------------------------------------------

const VIENNET: &str = "Viennet, R., Fonteix, C. and Marc, I. (1996). Multicriteria optimization \
                       using a genetic algorithm for determining a Pareto set. International \
                       Journal of Systems Science 27(2): 255-260.";
const VIENNET_URL: &str = "https://doi.org/10.1080/00207729608929211";

/// Viennet's first problem (VNT1), with three objectives: `f₁ = x₁² + (x₂ − 1)²`,
/// `f₂ = x₁² + (x₂ + 1)² + 1`, `f₃ = (x₁ − 1)² + x₂² + 2`.
///
/// Bounds [−2, 2]². Each objective is a squared distance to a point, (0, 1), (0, −1) and (1, 0),
/// plus a constant, so the optimal solutions are the triangle with these corners, and the front
/// its image (derived from the definition).
///
/// Viennet, R., Fonteix, C. and Marc, I. (1996). Multicriteria optimization using a genetic
/// algorithm for determining a Pareto set. *International Journal of Systems Science* 27(2):
/// 255-260. Definition and bounds as restated in Van Veldhuizen, D. A. (1999). *Multiobjective
/// Evolutionary Algorithms: Classifications, Analyses, and New Innovations.* PhD thesis
/// AFIT/DS/ENG/99-01, Air Force Institute of Technology, table B.1; not yet checked against the
/// original ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Viennet1;

// the minima of VNT1's objectives, the corners of its optimal solutions
const VIENNET1_CORNERS: [[f64; 2]; 3] = [[0.0, 1.0], [0.0, -1.0], [1.0, 0.0]];

impl MultiFitnessFunction<Reals, 3> for Viennet1 {
    type Output = [f64; 3];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 2 genes.
    fn evaluate(&self, x: &Reals) -> [f64; 3] {
        let (x1, x2) = (x[0], x[1]);
        [
            x1 * x1 + (x2 - 1.0) * (x2 - 1.0),
            x1 * x1 + (x2 + 1.0) * (x2 + 1.0) + 1.0,
            (x1 - 1.0) * (x1 - 1.0) + x2 * x2 + 2.0,
        ]
    }
}

impl MultiProblem<3> for Viennet1 {
    type Representation = Real;

    metadata!("VNT1", VIENNET, Some(VIENNET_URL));

    fn representation(&self) -> Real {
        bounds([(-2.0, 2.0), (-2.0, 2.0)])
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 3]>> {
        // the images of Das and Dennis's points, as weights of the corners
        let front = das_dennis::<3>(divisions_for::<3>(points))
            .into_iter()
            .map(|weights| {
                let x: Reals = (0..2)
                    .map(|i| (0..3).map(|j| weights[j] * VIENNET1_CORNERS[j][i]).sum())
                    .collect();
                self.evaluate(&x)
            });
        Some(front.collect())
    }

    fn ideal_point(&self) -> Option<[f64; 3]> {
        Some([0.0, 1.0, 2.0])
    }

    // the worst values are at the corners: 4, 5 and 4 for the three objectives
    fn nadir_point(&self) -> Option<[f64; 3]> {
        Some([4.0, 5.0, 4.0])
    }
}

/// Viennet's second problem (VNT2), with three objectives:
/// `f₁ = (x₁ − 2)²/2 + (x₂ + 1)²/13 + 3`, `f₂ = (x₁ + x₂ − 3)²/36 + (−x₁ + x₂ + 2)²/8 − 17`,
/// `f₃ = (x₁ + 2x₂ − 1)²/175 + (2x₂ − x₁)²/17 − 13`.
///
/// Bounds [−4, 4]². The front is not known in closed form:
/// [`optimal_front`](MultiProblem::optimal_front) is `None`.
///
/// Its ideal and nadir points are known (derived from the definition): the objectives are convex
/// quadratics, so the optimal solutions are the minima of the weighted sums w₁f₁ + w₂f₂ + w₃f₃ with
/// w ≥ 0, each the solution of a 2 × 2 linear system: a curved triangle whose corners are the
/// objectives' minima, (2, −1), (2.5, 0.5) and (0.5, 0.25). The
/// [`ideal_point`](MultiProblem::ideal_point) is (3, −17, −13), and the
/// [`nadir_point`](MultiProblem::nadir_point) (883/208, −2109/128, −35858/2975) ≈ (4.2452,
/// −16.4766, −12.0531): f₁ and f₂ are worst at f₃'s minimum, and f₃ at f₁'s. The minima of 80,601
/// weighted sums, Das and Dennis's weights with 400 divisions, and the non-dominated points of a
/// grid over the box stay between the two points.
///
/// Viennet, R., Fonteix, C. and Marc, I. (1996). Multicriteria optimization using a genetic
/// algorithm for determining a Pareto set. *International Journal of Systems Science* 27(2):
/// 255-260. Definition and bounds as restated in Van Veldhuizen (1999, PhD thesis, table B.1; its
/// table 5.3 has wider bounds); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Viennet2;

impl MultiFitnessFunction<Reals, 3> for Viennet2 {
    type Output = [f64; 3];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 2 genes.
    fn evaluate(&self, x: &Reals) -> [f64; 3] {
        let (x1, x2) = (x[0], x[1]);
        [
            (x1 - 2.0).powi(2) / 2.0 + (x2 + 1.0).powi(2) / 13.0 + 3.0,
            (x1 + x2 - 3.0).powi(2) / 36.0 + (-x1 + x2 + 2.0).powi(2) / 8.0 - 17.0,
            (x1 + 2.0 * x2 - 1.0).powi(2) / 175.0 + (2.0 * x2 - x1).powi(2) / 17.0 - 13.0,
        ]
    }
}

impl MultiProblem<3> for Viennet2 {
    type Representation = Real;

    metadata!("VNT2", VIENNET, Some(VIENNET_URL));

    fn representation(&self) -> Real {
        bounds([(-4.0, 4.0), (-4.0, 4.0)])
    }

    fn optimal_front(&self, _points: usize) -> Option<Vec<[f64; 3]>> {
        None
    }

    // the minima, at (2, −1), (2.5, 0.5) and (0.5, 0.25)
    fn ideal_point(&self) -> Option<[f64; 3]> {
        Some([3.0, -17.0, -13.0])
    }

    // f₁ and f₂ at f₃'s minimum (0.5, 0.25), and f₃ at f₁'s (2, −1)
    fn nadir_point(&self) -> Option<[f64; 3]> {
        Some([883.0 / 208.0, -2109.0 / 128.0, -35858.0 / 2975.0])
    }
}

/// Viennet's third problem (VNT3), with three objectives:
/// `f₁ = 0.5 (x₁² + x₂²) + sin(x₁² + x₂²)`,
/// `f₂ = (3x₁ − 2x₂ + 4)²/8 + (x₁ − x₂ + 1)²/27 + 15`,
/// `f₃ = 1 / (x₁² + x₂² + 1) − 1.1 exp(−(x₁² + x₂²))`.
///
/// Bounds [−3, 3]². The front is not known in closed form:
/// [`optimal_front`](MultiProblem::optimal_front) is `None`.
///
/// Its ideal and nadir points are known (derived from the definition, and checked numerically): f₁
/// and f₃ depend only on t = x₁² + x₂², so an optimal solution has the least f₂ on its circle, and
/// the front is the image of one curve in t, two pieces of which are optimal: t from 0 to about
/// 1.5, and from 4π/3, where f₁ has a local minimum, to about 17.16. The
/// [`ideal_point`](MultiProblem::ideal_point) is (0, 15, −0.1): f₁ and f₃ are smallest at the
/// origin, and f₂ at (−2, −1). The [`nadir_point`](MultiProblem::nadir_point) is (7π/3 + √3/2,
/// 460/27, 1/(1 + 4π/3) − 1.1 exp(−4π/3)) ≈ (8.1964, 17.0370, 0.1760): f₁ is worst at its local
/// maximum t = 14π/3, on the second piece; f₂ at the origin, where f₁ is 0; and f₃ at the second
/// piece's start, since it falls from there on and stays below 0.155 on the first. Checked against
/// 18,001 values of t, each with the least f₂ on its circle, and against the non-dominated points
/// of a grid over the box.
///
/// Viennet, R., Fonteix, C. and Marc, I. (1996). Multicriteria optimization using a genetic
/// algorithm for determining a Pareto set. *International Journal of Systems Science* 27(2):
/// 255-260. Definition and bounds as restated in Van Veldhuizen (1999, PhD thesis, table B.1; its
/// table 5.3 has wider bounds); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Viennet3;

impl MultiFitnessFunction<Reals, 3> for Viennet3 {
    type Output = [f64; 3];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 2 genes.
    fn evaluate(&self, x: &Reals) -> [f64; 3] {
        let (x1, x2) = (x[0], x[1]);
        let squares = x1 * x1 + x2 * x2;
        [
            0.5 * squares + squares.sin(),
            (3.0 * x1 - 2.0 * x2 + 4.0).powi(2) / 8.0 + (x1 - x2 + 1.0).powi(2) / 27.0 + 15.0,
            1.0 / (squares + 1.0) - 1.1 * (-squares).exp(),
        ]
    }
}

impl MultiProblem<3> for Viennet3 {
    type Representation = Real;

    metadata!("VNT3", VIENNET, Some(VIENNET_URL));

    fn representation(&self) -> Real {
        bounds([(-3.0, 3.0), (-3.0, 3.0)])
    }

    fn optimal_front(&self, _points: usize) -> Option<Vec<[f64; 3]>> {
        None
    }

    // f₁ and f₃ at the origin, and f₂ at (−2, −1)
    fn ideal_point(&self) -> Option<[f64; 3]> {
        Some([0.0, 15.0, -0.1])
    }

    // f₁ at t = 14π/3, f₂ at the origin and f₃ at t = 4π/3, to 50 digits
    fn nadir_point(&self) -> Option<[f64; 3]> {
        Some([8.196_408_262_160_622, 460.0 / 27.0, 0.176_042_069_506_621_9])
    }
}

// ---- BNH -----------------------------------------------------------------------------------------

/// Binh and Korn's problem (BNH): `f₁ = 4x₁² + 4x₂²`, `f₂ = (x₁ − 5)² + (x₂ − 5)²`, subject to
/// `(x₁ − 5)² + x₂² ≤ 25` and `(x₁ − 8)² + (x₂ + 3)² ≥ 7.7`.
///
/// Bounds [−15, 30]². The optimal solutions, derived from the definition, are x₁ = x₂ = t for t
/// in [0, 5], where each objective is a squared distance to (0, 0) or (5, 5); both ends lie on the
/// first constraint's boundary. The front is `f = (8t², 2(t − 5)²)`, convex, from (0, 50) to
/// (200, 0). Later papers, e.g. Van Veldhuizen (1999, PhD thesis AFIT/DS/ENG/99-01, table B.2),
/// use the bounds x₁ in [0, 5] and x₂ in [0, 3], which cut the front at x₂ = 3.
///
/// [`constraints`](MultiProblem::constraints) gives `(x₁ − 5)² + x₂² − 25` and
/// `7.7 − (x₁ − 8)² − (x₂ + 3)²`.
///
/// Binh, T. T. and Korn, U. (1997). MOBES: a multiobjective evolution strategy for constrained
/// optimization problems. *Proceedings of the Third International Conference on Genetic Algorithms
/// (Mendel 97)*, Brno: 176-182. Definition and bounds from its section 5.2, in the authors' version
/// of the paper.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Bnh;

impl Bnh {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        let (x1, x2) = (x[0], x[1]);
        [
            4.0 * x1 * x1 + 4.0 * x2 * x2,
            (x1 - 5.0) * (x1 - 5.0) + (x2 - 5.0) * (x2 - 5.0),
        ]
    }

    fn values(&self, x: &Reals) -> [f64; 2] {
        let (x1, x2) = (x[0], x[1]);
        [
            (x1 - 5.0) * (x1 - 5.0) + x2 * x2 - 25.0,
            7.7 - (x1 - 8.0) * (x1 - 8.0) - (x2 + 3.0) * (x2 + 3.0),
        ]
    }
}

constrained!(Bnh, 2);

impl MultiProblem<2> for Bnh {
    type Representation = Real;

    metadata!(
        "BNH",
        "Binh, T. T. and Korn, U. (1997). MOBES: a multiobjective evolution strategy for \
         constrained optimization problems. Proceedings of the Third International Conference on \
         Genetic Algorithms (Mendel 97), Brno: 176-182.",
        None
    );

    constraint_methods!();

    fn representation(&self) -> Real {
        bounds([(-15.0, 30.0), (-15.0, 30.0)])
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
        let diagonal = |t: f64| {
            let t = 5.0 * t;
            [8.0 * t * t, 2.0 * (t - 5.0) * (t - 5.0)]
        };
        let pieces = [Piece {
            curve: &diagonal,
            with_end: true,
        }];
        Some(pieces_front(&pieces, points))
    }

    fn ideal_point(&self) -> Option<[f64; 2]> {
        Some([0.0, 0.0])
    }

    fn nadir_point(&self) -> Option<[f64; 2]> {
        Some([200.0, 50.0])
    }
}

// ---- SRN -----------------------------------------------------------------------------------------

// where the front of SRN leaves the circle x₁² + x₂² = 225: x₁ = −a with
// 9 √(225 − a²) = 2a (√(225 − a²) − 1), where ∂f₂/∂x₁ along the circle is 0, solved to 40 digits
// by Newton's method and rounded
const SRN_END_X1: f64 = -4.840_977_370_874_673;
const SRN_END_X2: f64 = 14.197_356_729_147_836;

/// Srinivas and Deb's problem (SRN): `f₁ = (x₁ − 2)² + (x₂ − 1)² + 2`,
/// `f₂ = 9x₁ − (x₂ − 1)²`, subject to `x₁² + x₂² ≤ 225` and `x₁ − 3x₂ ≤ −10`.
///
/// Bounds [−20, 20]². The front, derived from the definition, has three pieces: the second
/// constraint's boundary x₁ = 3x₂ − 10 for x₂ from 3.7 down to 2.5; the line x₁ = −2.5 for x₂
/// from 2.5 to √218.75 ≈ 14.79, on the first constraint's circle; and that circle up to
/// x₁ ≈ −4.841, x₂ ≈ 14.197, where f₂ stops decreasing along it. The optimal solutions on
/// x₁ = −2.5 are those usually quoted, but they are only part of the front. It runs from
/// (10.1, 2.61) to (222.969, −217.739).
///
/// [`constraints`](MultiProblem::constraints) gives `x₁² + x₂² − 225` and `x₁ − 3x₂ + 10`.
///
/// Srinivas, N. and Deb, K. (1994). Multiobjective optimization using nondominated sorting in
/// genetic algorithms. *Evolutionary Computation* 2(3): 221-248. Definition and bounds as
/// restated in Deb, Pratap, Agarwal and Meyarivan (2002, NSGA-II, table V) and Binh and Korn
/// (1997, section 5.1); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Srn;

impl Srn {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        let (x1, x2) = (x[0], x[1]);
        [
            (x1 - 2.0) * (x1 - 2.0) + (x2 - 1.0) * (x2 - 1.0) + 2.0,
            9.0 * x1 - (x2 - 1.0) * (x2 - 1.0),
        ]
    }

    fn values(&self, x: &Reals) -> [f64; 2] {
        let (x1, x2) = (x[0], x[1]);
        [x1 * x1 + x2 * x2 - 225.0, x1 - 3.0 * x2 + 10.0]
    }

    // the objectives at (x₁, x₂)
    fn at(x1: f64, x2: f64) -> [f64; 2] {
        Self.objectives(&Reals::from(vec![x1, x2]))
    }
}

constrained!(Srn, 2);

impl MultiProblem<2> for Srn {
    type Representation = Real;

    metadata!(
        "SRN",
        "Srinivas, N. and Deb, K. (1994). Multiobjective optimization using nondominated sorting \
         in genetic algorithms. Evolutionary Computation 2(3): 221-248.",
        Some("https://doi.org/10.1162/evco.1994.2.3.221")
    );

    constraint_methods!();

    fn representation(&self) -> Real {
        bounds([(-20.0, 20.0), (-20.0, 20.0)])
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
        let top = 218.75f64.sqrt();
        // x₂ from 3.7 to 2.5 on x₁ = 3x₂ − 10
        let boundary = |t: f64| {
            let x2 = 3.7 - 1.2 * t;
            Self::at(3.0 * x2 - 10.0, x2)
        };
        // x₂ from 2.5 to √218.75 on x₁ = −2.5
        let line = |t: f64| Self::at(-2.5, 2.5 + (top - 2.5) * t);
        // the circle of radius 15, by angle
        let (start, end) = (top.atan2(-2.5), SRN_END_X2.atan2(SRN_END_X1));
        let circle = |t: f64| {
            let angle = start + (end - start) * t;
            Self::at(15.0 * angle.cos(), 15.0 * angle.sin())
        };
        let pieces = [
            Piece {
                curve: &boundary,
                with_end: false,
            },
            Piece {
                curve: &line,
                with_end: false,
            },
            Piece {
                curve: &circle,
                with_end: true,
            },
        ];
        Some(pieces_front(&pieces, points))
    }

    fn ideal_point(&self) -> Option<[f64; 2]> {
        Some([Self::at(1.1, 3.7)[0], Self::at(SRN_END_X1, SRN_END_X2)[1]])
    }

    fn nadir_point(&self) -> Option<[f64; 2]> {
        Some([Self::at(SRN_END_X1, SRN_END_X2)[0], Self::at(1.1, 3.7)[1]])
    }
}

// ---- TNK -----------------------------------------------------------------------------------------

/// Tanaka's problem (TNK): `f₁ = x₁`, `f₂ = x₂`, subject to
/// `x₁² + x₂² − 1 − 0.1 cos(16 arctan(x₁/x₂)) ≥ 0` and `(x₁ − 0.5)² + (x₂ − 0.5)² ≤ 0.5`.
///
/// Bounds [0, π]². The angle arctan(x₁/x₂) is taken as `atan2(x₁, x₂)`, π/2 at x₂ = 0 (the
/// limit) and 0 at the origin, which is infeasible either way. The front lies on the first
/// constraint's boundary, the curve of radius √(1 + 0.1 cos 16φ) at the angle φ from the x₂ axis,
/// in disconnected pieces; [`optimal_front`](MultiProblem::optimal_front) samples the curve
/// densely and keeps the feasible non-dominated points. It is symmetric in x₁ and x₂, and runs
/// from about (0.0417, 1.0384), where the curve leaves the second constraint's circle, to
/// (1.0384, 0.0417).
///
/// [`constraints`](MultiProblem::constraints) gives
/// `1 + 0.1 cos(16 arctan(x₁/x₂)) − x₁² − x₂²` and `(x₁ − 0.5)² + (x₂ − 0.5)² − 0.5`.
///
/// Tanaka, M., Watanabe, H., Furukawa, Y. and Tanino, T. (1995). GA-based decision support
/// system for multicriteria optimization. *Proceedings of the IEEE International Conference on
/// Systems, Man and Cybernetics* 2: 1556-1561. Definition and bounds as restated in Deb, Pratap,
/// Agarwal and Meyarivan (2002, NSGA-II, table V) and Deb, Pratap and Meyarivan (2001, EMO 2001,
/// eq. 2); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Tnk;

impl Tnk {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        [x[0], x[1]]
    }

    fn values(&self, x: &Reals) -> [f64; 2] {
        let (x1, x2) = (x[0], x[1]);
        [
            1.0 + 0.1 * (16.0 * x1.atan2(x2)).cos() - x1 * x1 - x2 * x2,
            (x1 - 0.5) * (x1 - 0.5) + (x2 - 0.5) * (x2 - 0.5) - 0.5,
        ]
    }

    // the point of the first constraint's boundary at the angle φ from the x₂ axis
    fn boundary(angle: f64) -> [f64; 2] {
        let radius = (1.0 + 0.1 * (16.0 * angle).cos()).sqrt();
        [radius * angle.sin(), radius * angle.cos()]
    }

    // the second constraint at the boundary point at `angle`
    fn outside_circle(angle: f64) -> f64 {
        let [x1, x2] = Self::boundary(angle);
        (x1 - 0.5) * (x1 - 0.5) + (x2 - 0.5) * (x2 - 0.5) - 0.5
    }

    // the angle where the front starts: where the boundary enters the second constraint's
    // circle, between 0 (outside) and 0.2 (inside), by bisection to the precision of f64
    fn start() -> f64 {
        let (mut outside, mut inside) = (0.0, 0.2);
        while inside - outside > f64::EPSILON {
            let middle = 0.5 * (outside + inside);
            if middle <= outside || middle >= inside {
                break;
            }
            if Self::outside_circle(middle) > 0.0 {
                outside = middle;
            } else {
                inside = middle;
            }
        }
        inside
    }
}

constrained!(Tnk, 2);

impl MultiProblem<2> for Tnk {
    type Representation = Real;

    metadata!(
        "TNK",
        "Tanaka, M., Watanabe, H., Furukawa, Y. and Tanino, T. (1995). GA-based decision support \
         system for multicriteria optimization. Proceedings of the IEEE International Conference \
         on Systems, Man and Cybernetics 2: 1556-1561.",
        Some("https://doi.org/10.1109/ICSMC.1995.537993")
    );

    constraint_methods!();

    fn representation(&self) -> Real {
        bounds([(0.0, PI), (0.0, PI)])
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
        let start = Self::start();
        let end = FRAC_PI_2 - start;
        let mut samples = 100_000.max(20 * points);
        loop {
            let curve = (0..samples)
                .map(|i| start + (end - start) * evenly(i, samples))
                .filter(|&angle| Self::outside_circle(angle) <= 1e-12)
                .map(Self::boundary)
                .collect();
            let front = non_dominated(curve);
            if front.len() >= points {
                return Some(spread_over(&front, points));
            }
            samples *= 2;
        }
    }

    fn ideal_point(&self) -> Option<[f64; 2]> {
        let [low, _] = Self::boundary(Self::start());
        Some([low, low])
    }

    fn nadir_point(&self) -> Option<[f64; 2]> {
        let [_, high] = Self::boundary(Self::start());
        Some([high, high])
    }
}

// ---- OSY -----------------------------------------------------------------------------------------

// where the front of OSY passes from its third piece to its fourth: x₁ on the third, x₃ on the
// fourth, where both give the same objectives, solved to 40 digits by Newton's method and rounded
const OSY_THIRD_START: f64 = 4.056_543_012_431_284;
const OSY_FOURTH_END: f64 = 3.731_684_756_057_475;

/// Osyczka and Kundu's problem (OSY), in six variables:
/// `f₁ = −[25 (x₁ − 2)² + (x₂ − 2)² + (x₃ − 1)² + (x₄ − 4)² + (x₅ − 1)²]`, `f₂ = Σ xᵢ²`,
/// subject to `x₁ + x₂ ≥ 2`, `x₁ + x₂ ≤ 6`, `x₂ − x₁ ≤ 2`, `x₁ − 3x₂ ≤ 2`,
/// `(x₃ − 3)² + x₄ ≤ 4` and `(x₅ − 3)² + x₆ ≥ 4`.
///
/// Bounds x₁, x₂, x₆ in [0, 10], x₃, x₅ in [1, 5], x₄ in [0, 6]. The front, derived from the
/// definition, has x₄ = x₆ = 0 and five pieces, from (−274, 76) to (−42, 4):
///
/// | Piece | x₁ | x₂ | x₃ | x₅ |
/// |---|---|---|---|---|
/// | 1 | 5 | 1 | 5 to 1 | 5 |
/// | 2 | 5 | 1 | 5 to 1 | 1 |
/// | 3 | 5 to 4.0565 | (x₁ − 2)/3 | 1 | 1 |
/// | 4 | 0 | 2 | 3.7317 to 1 | 1 |
/// | 5 | 0 to 1 | 2 − x₁ | 1 | 1 |
///
/// The third and fourth pieces meet where they reach the same objectives.
///
/// [`constraints`](MultiProblem::constraints) gives the six constraints in this order, as
/// `2 − x₁ − x₂`, `x₁ + x₂ − 6`, `x₂ − x₁ − 2`, `x₁ − 3x₂ − 2`, `(x₃ − 3)² + x₄ − 4` and
/// `4 − (x₅ − 3)² − x₆`.
///
/// Osyczka, A. and Kundu, S. (1995). A new method to solve generalized multicriteria optimization
/// problems using the simple genetic algorithm. *Structural Optimization* 10(2): 94-99. Definition
/// and bounds as restated in Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test
/// problems for multi-objective evolutionary optimization. *Evolutionary Multi-Criterion
/// Optimization (EMO 2001)*, LNCS 1993: 284-298 (KanGAL report 200002, eq. 3), whose table 1 lists
/// the same five pieces, with the ends 4.056 and 3.732; not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Osy;

impl Osy {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        let f1 = -(25.0 * (x[0] - 2.0).powi(2)
            + (x[1] - 2.0).powi(2)
            + (x[2] - 1.0).powi(2)
            + (x[3] - 4.0).powi(2)
            + (x[4] - 1.0).powi(2));
        let f2 = x[..6].iter().map(|xi| xi * xi).sum();
        [f1, f2]
    }

    fn values(&self, x: &Reals) -> [f64; 6] {
        [
            2.0 - x[0] - x[1],
            x[0] + x[1] - 6.0,
            x[1] - x[0] - 2.0,
            x[0] - 3.0 * x[1] - 2.0,
            (x[2] - 3.0).powi(2) + x[3] - 4.0,
            4.0 - (x[4] - 3.0).powi(2) - x[5],
        ]
    }

    // the objectives at (x₁, x₂, x₃, 0, x₅, 0)
    fn at(x1: f64, x2: f64, x3: f64, x5: f64) -> [f64; 2] {
        Self.objectives(&Reals::from(vec![x1, x2, x3, 0.0, x5, 0.0]))
    }
}

constrained!(Osy, 6);

impl MultiProblem<2> for Osy {
    type Representation = Real;

    metadata!(
        "OSY",
        "Osyczka, A. and Kundu, S. (1995). A new method to solve generalized multicriteria \
         optimization problems using the simple genetic algorithm. Structural Optimization \
         10(2): 94-99.",
        Some("https://doi.org/10.1007/BF01743536")
    );

    constraint_methods!();

    fn representation(&self) -> Real {
        bounds([
            (0.0, 10.0),
            (0.0, 10.0),
            (1.0, 5.0),
            (0.0, 6.0),
            (1.0, 5.0),
            (0.0, 10.0),
        ])
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
        let first = |t: f64| Self::at(5.0, 1.0, 5.0 - 4.0 * t, 5.0);
        let second = |t: f64| Self::at(5.0, 1.0, 5.0 - 4.0 * t, 1.0);
        let third = |t: f64| {
            let x1 = 5.0 - (5.0 - OSY_THIRD_START) * t;
            Self::at(x1, (x1 - 2.0) / 3.0, 1.0, 1.0)
        };
        let fourth = |t: f64| Self::at(0.0, 2.0, OSY_FOURTH_END - (OSY_FOURTH_END - 1.0) * t, 1.0);
        let fifth = |t: f64| Self::at(t, 2.0 - t, 1.0, 1.0);
        let pieces = [
            Piece {
                curve: &first,
                with_end: false,
            },
            Piece {
                curve: &second,
                with_end: false,
            },
            Piece {
                curve: &third,
                with_end: false,
            },
            Piece {
                curve: &fourth,
                with_end: false,
            },
            Piece {
                curve: &fifth,
                with_end: true,
            },
        ];
        Some(pieces_front(&pieces, points))
    }

    fn ideal_point(&self) -> Option<[f64; 2]> {
        Some([-274.0, 4.0])
    }

    fn nadir_point(&self) -> Option<[f64; 2]> {
        Some([-42.0, 76.0])
    }
}

// ---- CONSTR --------------------------------------------------------------------------------------

/// Deb's CONSTR: `f₁ = x₁`, `f₂ = (1 + x₂)/x₁`, subject to `x₂ + 9x₁ ≥ 6` and `−x₂ + 9x₁ ≥ 1`.
///
/// Bounds x₁ in [0.1, 1], x₂ in [0, 5]. The front, derived from the definition, is the first
/// constraint's boundary x₂ = 6 − 9x₁ for x₁ in [7/18, 2/3], `f₂ = (7 − 9f₁)/f₁`, then x₂ = 0 for
/// x₁ in [2/3, 1], `f₂ = 1/f₁`: from (7/18, 9) to (1, 1).
///
/// [`constraints`](MultiProblem::constraints) gives `6 − x₂ − 9x₁` and `1 + x₂ − 9x₁`.
///
/// Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002). A fast and elitist multiobjective
/// genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation* 6(2): 182-197,
/// table V.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Constr;

impl Constr {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        [x[0], (1.0 + x[1]) / x[0]]
    }

    fn values(&self, x: &Reals) -> [f64; 2] {
        let (x1, x2) = (x[0], x[1]);
        [6.0 - x2 - 9.0 * x1, 1.0 + x2 - 9.0 * x1]
    }
}

constrained!(Constr, 2);

impl MultiProblem<2> for Constr {
    type Representation = Real;

    metadata!("CONSTR", NSGA2, Some(NSGA2_URL));

    constraint_methods!();

    fn representation(&self) -> Real {
        bounds([(0.1, 1.0), (0.0, 5.0)])
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
        let boundary = |t: f64| {
            let x1 = 7.0 / 18.0 + (2.0 / 3.0 - 7.0 / 18.0) * t;
            [x1, (7.0 - 9.0 * x1) / x1]
        };
        let bound = |t: f64| {
            let x1 = 2.0 / 3.0 + t / 3.0;
            [x1, 1.0 / x1]
        };
        let pieces = [
            Piece {
                curve: &boundary,
                with_end: false,
            },
            Piece {
                curve: &bound,
                with_end: true,
            },
        ];
        Some(pieces_front(&pieces, points))
    }

    fn ideal_point(&self) -> Option<[f64; 2]> {
        Some([7.0 / 18.0, 1.0])
    }

    fn nadir_point(&self) -> Option<[f64; 2]> {
        Some([1.0, 9.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamRng;
    use crate::genome::Representation;
    use crate::multi::IntoScores;

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

    // `steps + 1` evenly spaced values from `low` to `high`, and `extra`
    fn axis(low: f64, high: f64, steps: usize, extra: &[f64]) -> Vec<f64> {
        let mut values: Vec<f64> = (0..=steps)
            .map(|i| low + (high - low) * i as f64 / steps as f64)
            .collect();
        values.extend_from_slice(extra);
        values
    }

    // the non-dominated points among `points`, with any number of objectives: sorted by f₁,
    // then each point against a staircase of the ones before it, in f₂ and the rest (for 3
    // objectives, exact; a copy of an earlier point is dropped)
    fn non_dominated_3(mut points: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
        use std::collections::BTreeMap;
        // f64s in their total order, as integers
        fn key(x: f64) -> i64 {
            let bits = x.to_bits() as i64;
            bits ^ (((bits >> 63) as u64) >> 1) as i64
        }
        points.sort_by(|a, b| {
            (a[0].total_cmp(&b[0]))
                .then(a[1].total_cmp(&b[1]))
                .then(a[2].total_cmp(&b[2]))
        });
        // f₂ → f₃ of the non-dominated points so far, f₃ falling as f₂ rises
        let mut staircase: BTreeMap<i64, f64> = BTreeMap::new();
        let mut front = Vec::new();
        for point in points {
            let below = staircase.range(..=key(point[1])).next_back();
            if below.is_some_and(|(_, &f3)| f3 <= point[2]) {
                continue;
            }
            let covered: Vec<i64> = staircase
                .range(key(point[1])..)
                .take_while(|&(_, &f3)| f3 >= point[2])
                .map(|(&k, _)| k)
                .collect();
            for k in covered {
                staircase.remove(&k);
            }
            staircase.insert(key(point[1]), point[2]);
            front.push(point);
        }
        front
    }

    // the non-dominated points of `sample` lie between the ideal and nadir points, to
    // `tolerance` of each objective's range, and reach both to `reach`: the points hold for the
    // front, not only the box
    fn check_extremes<const M: usize>(
        name: &str,
        front: &[[f64; M]],
        ideal: [f64; M],
        nadir: [f64; M],
        tolerance: f64,
        reach: f64,
    ) {
        for j in 0..M {
            let range = nadir[j] - ideal[j];
            let values = front.iter().map(|point| point[j]);
            let low = values.clone().fold(f64::INFINITY, f64::min);
            let high = values.fold(f64::NEG_INFINITY, f64::max);
            assert!(
                low >= ideal[j] - tolerance * range,
                "{name}: f{} {low}",
                j + 1
            );
            assert!(
                high <= nadir[j] + tolerance * range,
                "{name}: f{} {high}",
                j + 1
            );
            assert!(low <= ideal[j] + reach * range, "{name}: f{} {low}", j + 1);
            assert!(
                high >= nadir[j] - reach * range,
                "{name}: f{} {high}",
                j + 1
            );
        }
    }

    // no feasible genome in the bounds dominates a point of the optimal front by more than
    // `tolerance` in both objectives: a check of the derived fronts against 200,000 random
    // genomes
    fn no_genome_dominates_the_front<P>(problem: &P, tolerance: f64)
    where
        P: MultiProblem<2, Representation = Real> + MultiFitnessFunction<Reals, 2>,
    {
        let front = problem.optimal_front(400).expect("known");
        let real = problem.representation();
        let mut rng = StreamRng::seed_from_u64(3);
        for _ in 0..200_000 {
            let genome = real.random_genome(&mut rng);
            let scores = problem.evaluate(&genome).into_scores().expect("valid");
            if !scores.is_feasible() {
                continue;
            }
            let [f1, f2] = scores.values().expect("valid");
            for point in &front {
                assert!(
                    !(f1 < point[0] - tolerance && f2 < point[1] - tolerance),
                    "{}: {genome:?} at ({f1}, {f2}) dominates {point:?}",
                    problem.name()
                );
            }
        }
    }

    #[test]
    fn schaffer_1() {
        // x² and (x − 2)² at 0, 1 and 2
        assert_eq!(Schaffer1.evaluate(&at(&[0.0])), [0.0, 4.0]);
        assert_eq!(Schaffer1.evaluate(&at(&[1.0])), [1.0, 1.0]);
        assert_eq!(Schaffer1.evaluate(&at(&[2.0])), [4.0, 0.0]);
        let front = Schaffer1.optimal_front(11).expect("known");
        assert_eq!((front[0], front[10]), ([0.0, 4.0], [4.0, 0.0]));
        for [f1, f2] in front {
            assert!((f2 - (f1.sqrt() - 2.0).powi(2)).abs() < 1e-12);
        }
        assert_eq!(Schaffer1.representation().bounds(), [-1000.0..=1000.0]);
        no_genome_dominates_the_front(&Schaffer1, 1e-9);
    }

    #[test]
    fn schaffer_2() {
        // each branch of f₁, and f₂ = (x − 5)²
        for (x, expected) in [
            (-1.0, [1.0, 36.0]),
            (1.0, [-1.0, 16.0]),
            (1.5, [-0.5, 12.25]),
            (3.0, [1.0, 4.0]),
            (3.5, [0.5, 2.25]),
            (4.0, [0.0, 1.0]),
            (5.0, [1.0, 0.0]),
            (7.0, [3.0, 4.0]),
        ] {
            assert_eq!(Schaffer2.evaluate(&at(&[x])), expected, "x = {x}");
        }
        // x = 2 is dominated by x = 4
        assert_eq!(Schaffer2.evaluate(&at(&[2.0])), [0.0, 9.0]);
        let front = Schaffer2.optimal_front(100).expect("known");
        assert_eq!(front[0], [-1.0, 16.0]);
        assert_eq!(front[99], [1.0, 0.0]);
        for [f1, f2] in front {
            let expected = if f1 < 0.0 { f1 - 3.0 } else { f1 - 1.0 };
            assert!((f2 - expected * expected).abs() < 1e-12);
        }
        no_genome_dominates_the_front(&Schaffer2, 1e-9);
    }

    #[test]
    fn fonseca_fleming() {
        let problem = FonsecaFleming::default();
        // the minimum of f₁, at xᵢ = 1/√3: Σ (2/√3)² = 4 in f₂
        let x = 1.0 / 3f64.sqrt();
        assert_close(
            &problem.evaluate(&at(&[x; 3])),
            &[0.0, 1.0 - (-4.0f64).exp()],
        );
        // at the origin, Σ (1/√3)² = 1 in both
        let value = 1.0 - (-1.0f64).exp();
        assert_close(&problem.evaluate(&at(&[0.0; 3])), &[value, value]);
        // equal variables in [−1/√n, 1/√n] lie on the front, for any n
        for n in [1, 3, 5] {
            let problem = FonsecaFleming::new(n);
            for s in [-1.0, -0.3, 0.0, 0.8, 1.0] {
                let t = s / (n as f64).sqrt();
                let f = problem.evaluate(&Reals::from(vec![t; n]));
                assert_close(&f, &fonseca_fleming_front(s));
            }
        }
        assert_eq!(problem.representation().bounds()[2], -4.0..=4.0);
        no_genome_dominates_the_front(&problem, 1e-9);
    }

    #[test]
    fn kursawe() {
        let problem = Kursawe::default();
        // at the origin, two terms −10 e⁰, and 0
        assert_eq!(problem.evaluate(&at(&[0.0; 3])), [-20.0, 0.0]);
        // at (1, 1, 1): two terms −10 exp(−0.2 √2), and three 1 + 5 sin 1
        let f1 = -20.0 * (-0.2 * 2f64.sqrt()).exp();
        let f2 = 3.0 * (1.0 + 5.0 * 1f64.sin());
        assert_close(&problem.evaluate(&at(&[1.0; 3])), &[f1, f2]);
        // sin(xᵢ³), not sin³(xᵢ): at (2, 0, 0), 2^0.8 + 5 sin 8
        let f2 = 2f64.powf(0.8) + 5.0 * 8f64.sin();
        assert_close(&problem.evaluate(&at(&[2.0, 0.0, 0.0]))[1..], &[f2]);
        assert!(problem.optimal_front(10).is_none());
    }

    // where KUR's term of f₂ is least, x*
    const KURSAWE_X: f64 = -1.152_740_847_549_926_1;

    #[test]
    fn kursawe_extremes() {
        // the term of f₂ is least at KURSAWE_X: its derivative is 0 there, and a grid of
        // 1,000,001 values over [−5, 5] has nothing lower
        let term = |x: f64| x.abs().powf(0.8) + 5.0 * (x * x * x).sin();
        let slope = |x: f64| -0.8 * (-x).powf(-0.2) + 15.0 * x * x * (x * x * x).cos();
        assert!(slope(KURSAWE_X).abs() < 1e-12);
        assert!((term(KURSAWE_X) - KURSAWE_TERM).abs() < 1e-14);
        let x = KURSAWE_X.abs();
        assert!(((-0.2 * (2.0 * x * x).sqrt()).exp() - KURSAWE_DECAY).abs() < 1e-15);
        let least = (0..=1_000_000)
            .map(|i| term(-5.0 + i as f64 * 1e-5))
            .fold(f64::INFINITY, f64::min);
        assert!((KURSAWE_TERM..KURSAWE_TERM + 1e-9).contains(&least));
        // the ends of the front, for 2 to 5 variables
        for n in 2..=5 {
            let problem = Kursawe::new(n);
            let ideal = problem.ideal_point().expect("known");
            let nadir = problem.nadir_point().expect("known");
            assert_eq!(
                problem.evaluate(&Reals::from(vec![0.0; n])),
                [ideal[0], 0.0]
            );
            assert_eq!(nadir[1], 0.0);
            let f = problem.evaluate(&Reals::from(vec![KURSAWE_X; n]));
            assert_close(&f, &[nadir[0], ideal[1]]);
        }
        let problem = Kursawe::default();
        let (ideal, nadir) = (
            problem.ideal_point().unwrap(),
            problem.nadir_point().unwrap(),
        );
        assert_eq!(ideal, [-20.0, -11.627_286_837_138_845]);
        assert_eq!(nadir, [-14.435_463_549_038_639, 0.0]);
        // the non-dominated points of a 101 × 101 × 101 grid over the box, with KURSAWE_X
        let values = axis(-5.0, 5.0, 100, &[KURSAWE_X]);
        let mut sample = Vec::new();
        for &x1 in &values {
            for &x2 in &values {
                for &x3 in &values {
                    sample.push(problem.evaluate(&at(&[x1, x2, x3])));
                }
            }
        }
        let front = non_dominated(sample);
        check_extremes("KUR", &front, ideal, nadir, 1e-12, 1e-12);
    }

    #[test]
    fn poloni() {
        // B = A at (1, 2): f₁ is 1, its minimum; f₂ = 4² + 3²
        assert_close(&Poloni.evaluate(&at(&[1.0, 2.0])), &[1.0, 25.0]);
        // at the origin: B₁ = −2 − 1.5 and B₂ = −1 − 0.5; f₂ = 9 + 1
        let (s1, c1, s2, c2) = (1f64.sin(), 1f64.cos(), 2f64.sin(), 2f64.cos());
        let a1 = 0.5 * s1 - 2.0 * c1 + s2 - 1.5 * c2;
        let a2 = 1.5 * s1 - c1 + 2.0 * s2 - 0.5 * c2;
        let f1 = 1.0 + (a1 + 3.5).powi(2) + (a2 + 1.5).powi(2);
        assert_close(&Poloni.evaluate(&at(&[0.0, 0.0])), &[f1, 10.0]);
        assert_eq!(Poloni.representation().bounds()[1], -PI..=PI);
        // the ends of the front: f₁ at (−3, −1), and f₂ at (1, 2)
        let (ideal, nadir) = (Poloni.ideal_point().unwrap(), Poloni.nadir_point().unwrap());
        assert_close(&Poloni.evaluate(&at(&[-3.0, -1.0])), &[nadir[0], ideal[1]]);
        assert_close(&Poloni.evaluate(&at(&[1.0, 2.0])), &[ideal[0], nadir[1]]);
        // the other solution of B = A has f₁ = 1 too, and a larger f₂: dominated
        let other = Poloni.evaluate(&at(&[2.022_785_254_123_813, 0.730_709_903_108_889_5]));
        assert!((other[0] - 1.0).abs() < 1e-14 && (other[1] - 28.223_728).abs() < 1e-6);
        // the non-dominated points of a 1001 × 1001 grid over the box, with the two ends
        let values = axis(-PI, PI, 1000, &[]);
        let mut sample = vec![
            Poloni.evaluate(&at(&[-3.0, -1.0])),
            Poloni.evaluate(&at(&[1.0, 2.0])),
        ];
        for &x1 in &values {
            for &x2 in &values {
                sample.push(Poloni.evaluate(&at(&[x1, x2])));
            }
        }
        check_extremes("POL", &non_dominated(sample), ideal, nadir, 1e-12, 1e-12);
    }

    #[test]
    fn viennet_1() {
        // at the minima of the three objectives, and at the origin
        assert_eq!(Viennet1.evaluate(&at(&[0.0, 1.0])), [0.0, 5.0, 4.0]);
        assert_eq!(Viennet1.evaluate(&at(&[0.0, -1.0])), [4.0, 1.0, 4.0]);
        assert_eq!(Viennet1.evaluate(&at(&[1.0, 0.0])), [2.0, 3.0, 2.0]);
        assert_eq!(Viennet1.evaluate(&at(&[0.0, 0.0])), [1.0, 2.0, 3.0]);
        let front = Viennet1.optimal_front(15).expect("known");
        assert_eq!(front.len(), 15);
        // no genome in the bounds dominates a point of the front
        let mut rng = StreamRng::seed_from_u64(5);
        for _ in 0..50_000 {
            let f = Viennet1.evaluate(&Viennet1.representation().random_genome(&mut rng));
            for point in &front {
                assert!(
                    !(0..3).all(|j| f[j] < point[j] - 1e-9),
                    "{f:?} dominates {point:?}"
                );
            }
        }
    }

    #[test]
    fn viennet_2_and_3() {
        // VNT2 at (2, −1), where f₁ is smallest: 3; (−2)²/36 + (−1)²/8 − 17;
        // (−1)²/175 + 4²/17 − 13
        let expected = [
            3.0,
            4.0 / 36.0 + 1.0 / 8.0 - 17.0,
            1.0 / 175.0 + 16.0 / 17.0 - 13.0,
        ];
        assert_close(&Viennet2.evaluate(&at(&[2.0, -1.0])), &expected);
        // VNT3 at the origin: 0; 4²/8 + 1/27 + 15; 1 − 1.1
        let expected = [0.0, 17.0 + 1.0 / 27.0, -0.1];
        assert_close(&Viennet3.evaluate(&at(&[0.0, 0.0])), &expected);
        assert_eq!(Viennet2.representation().bounds()[0], -4.0..=4.0);
        assert_eq!(Viennet3.representation().bounds()[0], -3.0..=3.0);
    }

    #[test]
    fn viennet_2_extremes() {
        let (ideal, nadir) = (
            Viennet2.ideal_point().unwrap(),
            Viennet2.nadir_point().unwrap(),
        );
        // the minima of the objectives, the corners of the optimal solutions
        let [f1, f2, f3] =
            [[2.0, -1.0], [2.5, 0.5], [0.5, 0.25]].map(|x| Viennet2.evaluate(&at(&x)));
        assert_close(&[f1[0], f2[1], f3[2]], &ideal);
        assert_close(&[f3[0], f3[1], f1[2]], &nadir);
        // the minima of 80,601 weighted sums, Das and Dennis's weights with 400 divisions: each
        // objective is (x − c)ᵀ A (x − c) / 2 plus a constant, with the Hessian A and minimum c
        let hessians = [
            [[1.0, 0.0], [0.0, 2.0 / 13.0]],
            [
                [2.0 / 36.0 + 2.0 / 8.0, 2.0 / 36.0 - 2.0 / 8.0],
                [2.0 / 36.0 - 2.0 / 8.0, 2.0 / 36.0 + 2.0 / 8.0],
            ],
            [
                [2.0 / 175.0 + 2.0 / 17.0, 4.0 / 175.0 - 4.0 / 17.0],
                [4.0 / 175.0 - 4.0 / 17.0, 8.0 / 175.0 + 8.0 / 17.0],
            ],
        ];
        let minima = [[2.0, -1.0], [2.5, 0.5], [0.5, 0.25]];
        let front: Vec<[f64; 3]> = das_dennis::<3>(400)
            .into_iter()
            .map(|w| {
                // Σ wᵢ Aᵢ x = Σ wᵢ Aᵢ cᵢ
                let mut a = [[0.0; 2]; 2];
                let mut b = [0.0; 2];
                for i in 0..3 {
                    for r in 0..2 {
                        for c in 0..2 {
                            a[r][c] += w[i] * hessians[i][r][c];
                            b[r] += w[i] * hessians[i][r][c] * minima[i][c];
                        }
                    }
                }
                let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
                let x1 = (b[0] * a[1][1] - a[0][1] * b[1]) / det;
                let x2 = (a[0][0] * b[1] - a[1][0] * b[0]) / det;
                Viennet2.evaluate(&at(&[x1, x2]))
            })
            .collect();
        assert_eq!(front.len(), 80_601);
        check_extremes("VNT2", &front, ideal, nadir, 1e-12, 1e-12);
        // and the non-dominated points of a 801 × 801 grid over the box
        let values = axis(-4.0, 4.0, 800, &[]);
        let mut sample = Vec::new();
        for &x1 in &values {
            for &x2 in &values {
                sample.push(Viennet2.evaluate(&at(&[x1, x2])));
            }
        }
        check_extremes("VNT2", &non_dominated_3(sample), ideal, nadir, 1e-12, 1e-12);
    }

    #[test]
    fn viennet_3_extremes() {
        let (ideal, nadir) = (
            Viennet3.ideal_point().unwrap(),
            Viennet3.nadir_point().unwrap(),
        );
        // f₁ and f₃ at the origin, f₂ at (−2, −1)
        let origin = Viennet3.evaluate(&at(&[0.0, 0.0]));
        assert_close(&[origin[0], origin[2]], &[ideal[0], ideal[2]]);
        assert_eq!(Viennet3.evaluate(&at(&[-2.0, -1.0]))[1], ideal[1]);
        // f₁ at t = 14π/3, f₂ at the origin and f₃ at t = 4π/3, on circles of radius √t
        let on_circle = |t: f64| Viennet3.evaluate(&at(&[t.sqrt(), 0.0]));
        assert_close(
            &[
                on_circle(14.0 * PI / 3.0)[0],
                origin[1],
                on_circle(4.0 * PI / 3.0)[2],
            ],
            &nadir,
        );
        // the non-dominated points of a 1201 × 1201 grid over the box: next to t = 4π/3, a grid
        // point that the grid doesn't dominate has f₃ up to 7e-4 of its range above the nadir
        let values = axis(-3.0, 3.0, 1200, &[]);
        let mut sample = Vec::new();
        for &x1 in &values {
            for &x2 in &values {
                sample.push(Viennet3.evaluate(&at(&[x1, x2])));
            }
        }
        check_extremes("VNT3", &non_dominated_3(sample), ideal, nadir, 1e-3, 1e-4);
    }

    #[test]
    fn bnh() {
        // at the origin: f = (0, 50); g₁ = 25 − 25, g₂ = 7.7 − 64 − 9
        assert_eq!(Bnh.evaluate(&at(&[0.0, 0.0])), ([0.0, 50.0], 0.0));
        assert_close(
            Bnh.constraints(&at(&[0.0, 0.0])).inequalities(),
            &[0.0, -65.3],
        );
        // (5, 5), the other end of the front, is on the first boundary too: 0 + 25 − 25
        assert_eq!(Bnh.evaluate(&at(&[5.0, 5.0])), ([200.0, 0.0], 0.0));
        // (0, 3) is outside the first circle: 25 + 9 − 25 = 9
        assert_eq!(Bnh.evaluate(&at(&[0.0, 3.0])), ([36.0, 29.0], 9.0));
        // (8, −3), the center of the second circle, violates it by 7.7, and the first by
        // 9 + 9 − 25 < 0 not at all
        assert_close(&[Bnh.evaluate(&at(&[8.0, -3.0])).1], &[7.7]);
        // the optimal solutions x₁ = x₂ = t reach the front
        for t in [0.0, 1.0, 2.5, 5.0] {
            let (f, violation) = Bnh.evaluate(&at(&[t, t]));
            assert_close(&f, &[8.0 * t * t, 2.0 * (t - 5.0).powi(2)]);
            assert_eq!(violation, 0.0);
        }
        assert_eq!(Bnh.representation().bounds()[1], -15.0..=30.0);
        no_genome_dominates_the_front(&Bnh, 1e-9);
    }

    #[test]
    fn srn() {
        // the corner of the front's first two pieces, on the second constraint's boundary
        let (f, violation) = Srn.evaluate(&at(&[-2.5, 2.5]));
        assert_close(&f, &[24.5, -24.75]);
        assert_eq!(violation, 0.0);
        assert_close(
            Srn.constraints(&at(&[-2.5, 2.5])).inequalities(),
            &[-212.5, 0.0],
        );
        // the front's ends: (1.1, 3.7), where f₁ is smallest on x₁ = 3x₂ − 10, and the end on
        // the circle
        assert_close(&Srn.evaluate(&at(&[1.1, 3.7])).0, &[10.1, 2.61]);
        let [x1, x2] = [SRN_END_X1, SRN_END_X2];
        assert!((x1 * x1 + x2 * x2 - 225.0).abs() < 1e-12);
        // there, f₂ is smallest along the circle
        let f2 = |angle: f64| Srn::at(15.0 * angle.cos(), 15.0 * angle.sin())[1];
        let angle = x2.atan2(x1);
        assert!(f2(angle) < f2(angle - 1e-4) && f2(angle) < f2(angle + 1e-4));
        // the origin violates the second constraint by 10, (15, 15) the first by 225
        assert_eq!(Srn.evaluate(&at(&[0.0, 0.0])).1, 10.0);
        assert_eq!(Srn.evaluate(&at(&[15.0, 15.0])).1, 225.0);
        let ideal = Srn.ideal_point().expect("known");
        let nadir = Srn.nadir_point().expect("known");
        assert!((ideal[0] - 10.1).abs() < 1e-12 && (nadir[1] - 2.61).abs() < 1e-12);
        assert!((nadir[0] - 222.969_196).abs() < 1e-6);
        assert!((ideal[1] + 217.739_021).abs() < 1e-6);
        no_genome_dominates_the_front(&Srn, 1e-9);
    }

    #[test]
    fn tnk() {
        // (1, 1): 1 + 0.1 cos 4π − 2 and 0.5 − 0.5, feasible on the second boundary
        assert_eq!(Tnk.evaluate(&at(&[1.0, 1.0])), ([1.0, 1.0], 0.0));
        assert_close(
            Tnk.constraints(&at(&[1.0, 1.0])).inequalities(),
            &[-0.9, 0.0],
        );
        // (0.5, 0.5) is inside the first constraint's curve: 1 + 0.1 − 0.5
        assert_close(&[Tnk.evaluate(&at(&[0.5, 0.5])).1], &[0.6]);
        // x₂ = 0: the angle is π/2, and cos 8π = 1
        let g = Tnk.constraints(&at(&[PI, 0.0]));
        assert_close(&g.inequalities()[..1], &[1.1 - PI * PI]);
        // the front lies on the first boundary, inside the second circle
        let front = Tnk.optimal_front(200).expect("known");
        for point in &front {
            let g = Tnk.constraints(&at(point));
            assert!(g.inequalities()[0].abs() < 1e-12, "{point:?}");
            assert!(g.inequalities()[1] <= 1e-12, "{point:?}");
        }
        // symmetric, from about (0.0417, 1.0384)
        let low = Tnk.ideal_point().expect("known")[0];
        let high = Tnk.nadir_point().expect("known")[0];
        assert!((low - 0.041_664).abs() < 1e-6 && (high - 1.038_450).abs() < 1e-6);
        assert!((front[0][0] - low).abs() < 1e-12 && (front[0][1] - high).abs() < 1e-12);
        no_genome_dominates_the_front(&Tnk, 1e-4);
    }

    #[test]
    fn osy() {
        // on the fifth piece, x₁ = 0.5: −(25 × 2.25 + 0.25 + 16) and 0.25 + 2.25 + 1 + 1
        let x = at(&[0.5, 1.5, 1.0, 0.0, 1.0, 0.0]);
        assert_eq!(Osy.evaluate(&x), ([-72.5, 4.5], 0.0));
        // (0, 0, 3, 0, 3, 0) violates the first constraint by 2 and the sixth by 4
        let x = at(&[0.0, 0.0, 3.0, 0.0, 3.0, 0.0]);
        assert_close(
            Osy.constraints(&x).inequalities(),
            &[2.0, -6.0, -2.0, -2.0, -4.0, 4.0],
        );
        assert_eq!(Osy.evaluate(&x).1, 6.0);
        // the genomes of the five pieces are feasible, and the pieces meet
        let pieces: [&dyn Fn(f64) -> [f64; 6]; 5] = [
            &|t| [5.0, 1.0, 5.0 - 4.0 * t, 0.0, 5.0, 0.0],
            &|t| [5.0, 1.0, 5.0 - 4.0 * t, 0.0, 1.0, 0.0],
            &|t| {
                let x1 = 5.0 - (5.0 - OSY_THIRD_START) * t;
                [x1, (x1 - 2.0) / 3.0, 1.0, 0.0, 1.0, 0.0]
            },
            &|t| {
                let x3 = OSY_FOURTH_END - (OSY_FOURTH_END - 1.0) * t;
                [0.0, 2.0, x3, 0.0, 1.0, 0.0]
            },
            &|t| [t, 2.0 - t, 1.0, 0.0, 1.0, 0.0],
        ];
        let objectives = |x: [f64; 6]| {
            let (f, violation) = Osy.evaluate(&at(&x));
            assert!(violation < 1e-12, "{x:?}");
            f
        };
        for t in [0.0, 0.3, 1.0] {
            for piece in pieces {
                objectives(piece(t));
            }
        }
        for pair in pieces.windows(2) {
            assert_close(&objectives(pair[0](1.0)), &objectives(pair[1](0.0)));
        }
        assert_eq!(objectives(pieces[0](0.0)), [-274.0, 76.0]);
        assert_eq!(objectives(pieces[4](1.0)), [-42.0, 4.0]);
        no_genome_dominates_the_front(&Osy, 1e-9);
    }

    #[test]
    fn constr() {
        // (0.5, 1.5) is on the first constraint's boundary: 6 − 1.5 − 4.5
        assert_eq!(Constr.evaluate(&at(&[0.5, 1.5])), ([0.5, 5.0], 0.0));
        assert_close(
            Constr.constraints(&at(&[0.5, 1.5])).inequalities(),
            &[0.0, -2.0],
        );
        // (0.1, 0) violates both: 6 − 0.9 and 1 − 0.9
        assert_close(&[Constr.evaluate(&at(&[0.1, 0.0])).1], &[5.2]);
        // the front's ends: x = (7/18, 2.5) and (1, 0)
        assert_close(
            &Constr.evaluate(&at(&[7.0 / 18.0, 2.5])).0,
            &[7.0 / 18.0, 9.0],
        );
        assert_eq!(Constr.evaluate(&at(&[1.0, 0.0])), ([1.0, 1.0], 0.0));
        no_genome_dominates_the_front(&Constr, 1e-9);
    }
}
