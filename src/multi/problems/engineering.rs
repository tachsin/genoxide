//! Engineering design problems with several objectives: the two-bar and four-bar trusses, the
//! welded beam, the disc brake, the speed reducer, the car side impact, water resource planning,
//! the rocket injector, vehicle crashworthiness and conceptual marine design.
//!
//! Every objective is minimized, in the units of the paper that states it. A constrained
//! problem's fitness is `([f64; M], violation)`, the violation being `Σ max(0, gᵢ(x))` over its
//! constraints written as `gᵢ(x) ≤ 0`, 0 when it's feasible; the constraints keep their paper's
//! scale, as in the single-objective [`problems::engineering`](crate::problems::engineering).
//!
//! | Problem | Variables | Objectives | Constraints | Optimal front |
//! |---|---|---|---|---|
//! | [`TwoBarTruss`] | 3 | 2 | 1 | two pieces, derived |
//! | [`WeldedBeam`] | 4 | 2 | 4 | not known |
//! | [`DiscBrake`] | 4 (one integer) | 2 | 5 | not known |
//! | [`SpeedReducer`] | 7 (one integer) | 2 | 11 | not known |
//! | [`FourBarTruss`] | 4 | 2 | | three pieces, derived |
//! | [`CarSideImpact`] | 7 | 3 | 10 | not known |
//! | [`RocketInjector`] | 4 | 3 | | not known |
//! | [`VehicleCrashworthiness`] | 5 | 3 | | not known |
//! | [`MarineDesign`] | 6 | 3 | 9 | not known |
//! | [`WaterResourcePlanning`] | 3 | 5 | 7 | a surface, derived |
//!
//! The fronts of the two trusses and of water resource planning are derived from their
//! definitions, and their docs say how.
//! The others aren't known in closed form: their [`optimal_front`](MultiProblem::optimal_front)
//! is `None`, and their docs give their ideal point where genoxide computed it, each objective
//! minimized alone over the feasible region, and for two objectives their nadir point, the
//! other objective at those minima. These are best known values, not proven optima.
//!
//! **Discrete variables.** [`DiscBrake`] and [`SpeedReducer`] each have an integer variable (the
//! number of friction surfaces, the number of teeth): their genome is real, its integer gene is
//! rounded to the nearest integer when it's evaluated, and their `design` method gives the rounded
//! design, as for the single-objective [`SpeedReducer`](crate::problems::engineering::SpeedReducer).
//!
//! **Sources.** The two-bar truss and the welded beam were checked in Deb, Pratap and Moitra's
//! KanGAL report 200002 (2000), the preprint of their PPSN VI paper; the car side impact's three
//! objectives in Jain and Deb's accepted manuscript (2014, appendix); the rocket injector's
//! response surfaces in Vaidyanathan, Tucker, Papila and Shyy's AIAA paper 2003-296 (appendix,
//! eqs. A1-A4); water resource planning in the NSGA-II paper (table V, and the same table in the
//! preprint, KanGAL report 200001) and in Jain and Deb's appendix; and the marine design in
//! Parsons and Scott (2004, appendix and the Panamax case). The other originals (Osyczka and
//! Kundu 1995, Kurpati, Azarm and Wu 2002, Stadler and Dauer 1993, Liao et al. 2008) couldn't be
//! read: those definitions are taken from later papers that restate them, named in each
//! problem's docs, and are still to be checked against the originals
//! ([#168](https://github.com/tachsin/genoxide/issues/168)).

use super::{MultiProblem, Piece, pieces_front};
use crate::constraint::at_most;
use crate::genome::{Real, Reals};
use crate::math;
use crate::multi::MultiFitnessFunction;
use crate::problems::Constraints;
use crate::problems::engineering as single;
use std::f64::consts::SQRT_2;

// bounds that are valid by construction
fn bounds<const N: usize>(ranges: [(f64, f64); N]) -> Real {
    Real::new(ranges.map(|(low, high)| low..=high)).expect("valid bounds")
}

// the total violation of constraints g(x) <= 0, as `Constraints::violation` adds it up
fn violation(constraints: &[f64]) -> f64 {
    constraints.iter().map(|&g| at_most(g, 0.0)).sum()
}

// the metadata of a problem
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

// a constrained problem's fitness, from its objectives and its constraint values g(x) <= 0, and
// its number of constraints
macro_rules! constrained {
    ($name:ident, $objectives:literal, $count:literal) => {
        impl MultiFitnessFunction<Reals, $objectives> for $name {
            type Output = ([f64; $objectives], f64);

            /// The objective values of `x` and its constraint violation, 0 when it's feasible.
            ///
            /// # Panics
            ///
            /// If `x` has fewer than the problem's variables.
            fn evaluate(&self, x: &Reals) -> ([f64; $objectives], f64) {
                (self.objectives(x), violation(&self.values(x)))
            }
        }

        impl $name {
            /// The number of constraints.
            pub const CONSTRAINTS: usize = $count;
        }
    };
}

// an unconstrained problem's fitness, from its objectives
macro_rules! unconstrained {
    ($name:ident, $objectives:literal) => {
        impl MultiFitnessFunction<Reals, $objectives> for $name {
            type Output = [f64; $objectives];

            /// The objective values of `x`.
            ///
            /// # Panics
            ///
            /// If `x` has fewer than the problem's variables.
            fn evaluate(&self, x: &Reals) -> [f64; $objectives] {
                self.objectives(x)
            }
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

const DEB_PRATAP_MOITRA: &str = "Deb, K., Pratap, A. and Moitra, S. (2000). Mechanical component \
                                 design for multiple objectives using elitist non-dominated \
                                 sorting GA. Parallel Problem Solving from Nature (PPSN VI), \
                                 LNCS 1917: 859-868.";
const DEB_PRATAP_MOITRA_URL: &str = "https://doi.org/10.1007/3-540-45356-3_84";

// `points` points of a front made of `pieces` in objectives scaled to [0, 1] by `ideal` and
// `nadir`, spread over the pieces by their lengths there: the pieces' curves give scaled points
fn scaled_front(
    pieces: &[Piece<'_>],
    points: usize,
    ideal: [f64; 2],
    nadir: [f64; 2],
) -> Vec<[f64; 2]> {
    pieces_front(pieces, points)
        .into_iter()
        .map(|p| [0, 1].map(|j| ideal[j] + p[j] * (nadir[j] - ideal[j])))
        .collect()
}

// the point where the arc length of a scaled curve reaches the share `t` of its whole length:
// `curve` over [0, 1] reparametrized, so that points evenly spread in t are evenly spread along
// it
fn by_length(curve: &dyn Fn(f64) -> [f64; 2], t: f64) -> [f64; 2] {
    const STEPS: usize = 4_000;
    let mut lengths = Vec::with_capacity(STEPS + 1);
    lengths.push(0.0);
    let mut last = curve(0.0);
    for step in 1..=STEPS {
        let point = curve(step as f64 / STEPS as f64);
        let length = ((point[0] - last[0]).powi(2) + (point[1] - last[1]).powi(2)).sqrt();
        lengths.push(lengths[step - 1] + length);
        last = point;
    }
    let target = t * lengths[STEPS];
    let step = lengths
        .partition_point(|&length| length < target)
        .clamp(1, STEPS);
    let (before, after) = (lengths[step - 1], lengths[step]);
    let within = if after > before {
        (target - before) / (after - before)
    } else {
        0.0
    };
    curve((step as f64 - 1.0 + within) / STEPS as f64)
}

// the ideal and nadir points of a two-objective front whose ends are the designs `first` (the
// least f₁, and the least f₂ there) and `second` (the least f₂, and the least f₁ there)
fn extremes(
    objectives: impl Fn(&Reals) -> [f64; 2],
    first: &[f64],
    second: &[f64],
) -> [[f64; 2]; 2] {
    let [a1, a2] = objectives(&Reals::from(first.to_vec()));
    let [b1, b2] = objectives(&Reals::from(second.to_vec()));
    [[a1, b2], [b1, a2]]
}

// the ideal point of a front, from a design that minimizes each objective
fn ideal<const M: usize, const N: usize>(
    objectives: impl Fn(&Reals) -> [f64; M],
    designs: &[[f64; N]; M],
) -> [f64; M] {
    std::array::from_fn(|j| objectives(&Reals::from(designs[j].to_vec()))[j])
}

// ---- two-bar truss -------------------------------------------------------------------------------

/// The two-bar truss: the lightest truss of two bars that carries a load of 100 kN, and the one
/// whose bars are least stressed (Deb, Pratap and Moitra, 2000).
///
/// The bars AC and BC join at C, which carries the load, y metres below the supports A and B,
/// 4 m and 1 m to the side. The genes are the bars' cross-sections x₁ (AC) and x₂ (BC), in m², and
/// y: x = (x₁, x₂, y). The objectives are the volume `f₁ = x₁ √(16 + y²) + x₂ √(1 + y²)`, in m³,
/// and the larger stress `f₂ = max(σ_AC, σ_BC)`, in kPa, with `σ_AC = 20 √(16 + y²) / (y x₁)` and
/// `σ_BC = 80 √(1 + y²) / (y x₂)`, subject to `max(σ_AC, σ_BC) ≤ 10⁵`.
///
/// Bounds x₁, x₂ ∈ [0, 0.01] and y ∈ [1, 3], as in the paper: at x₁ = 0 or x₂ = 0 a bar has no
/// section, and the stress, and the violation, are infinite.
///
/// **The front**, derived from the definition (and by Deb and Srinivasan, below): on the optimal
/// solutions both bars carry the same stress S. For y and S given, the least volume is then
/// `(400 + 100y²) / (yS)`, smallest at y = 2, so that `f₁ f₂ = 400`, with `x₁ = 20√5 / S` and
/// `x₂ = 40√5 / S`, for S from 10⁵ down to 4000√5 ≈ 8944.27, where x₂ reaches 0.01 (volume
/// 0.0044721). Less stress needs y > 2, where x₂ stays at 0.01 and y is the smallest that keeps
/// σ_BC at S: `S = 8000 √(1 + y²) / y` and `f₁ = (4 + y²) / (80 √(1 + y²))` for y from 2 to 3,
/// down to 8000√10/3 ≈ 8432.74 at a volume of 0.0513870. The front runs from (0.004, 10⁵) to
/// there; [`optimal_front`](MultiProblem::optimal_front) spreads its points evenly along it in
/// objectives scaled by the ideal and nadir points.
///
/// Deb, Pratap and Moitra report NSGA-II solutions from (0.00407, 99755) to (0.05304, 8439). Deb
/// and Srinivasan (2006, KanGAL report 2005007, table 1) tabulate three: (4.60·10⁻⁴, 9.05·10⁻⁴,
/// 1.935) at (0.004013, 99,937), (49.30·10⁻⁴, 99.89·10⁻⁴, 2.035) at (0.044779, 8945.6) and
/// (39.54·10⁻⁴, 100·10⁻⁴, 3.0) at (0.051391, 8432.74), and derive the same two pieces.
///
/// [`constraints`](MultiProblem::constraints) gives `max(σ_AC, σ_BC) − 10⁵`.
///
/// Deb, K., Pratap, A. and Moitra, S. (2000). Mechanical component design for multiple objectives
/// using elitist non-dominated sorting GA. *Parallel Problem Solving from Nature (PPSN VI)*, LNCS
/// 1917: 859-868, eq. 1 and section 4.1, checked in the authors' preprint (KanGAL report 200002),
/// which calls x₁ and x₂ the bars' lengths but uses them as areas; after Palli, Azarm, McCluskey
/// and Sundararajan's ε-constraint study (1998, *Journal of Mechanical Design* 120(4): 678-686,
/// which Deb et al. date 1999; not read). Bounds and the front's derivation as in Deb,
/// K. and Srinivasan, A. (2006). Innovization: innovating design principles through
/// optimization. *GECCO 2006*: 1629-1636 (KanGAL report 2005007, eqs. 1-7).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TwoBarTruss;

impl TwoBarTruss {
    // the stresses in the bars AC and BC, in kPa
    fn stresses(&self, x: &Reals) -> (f64, f64) {
        let (x1, x2, y) = (x[0], x[1], x[2]);
        (
            20.0 * (16.0 + y * y).sqrt() / (y * x1),
            80.0 * (1.0 + y * y).sqrt() / (y * x2),
        )
    }

    fn objectives(&self, x: &Reals) -> [f64; 2] {
        let (x1, x2, y) = (x[0], x[1], x[2]);
        let (ac, bc) = self.stresses(x);
        [
            x1 * (16.0 + y * y).sqrt() + x2 * (1.0 + y * y).sqrt(),
            ac.max(bc),
        ]
    }

    fn values(&self, x: &Reals) -> [f64; 1] {
        let (ac, bc) = self.stresses(x);
        [ac.max(bc) - 1e5]
    }
}

constrained!(TwoBarTruss, 2, 1);

impl MultiProblem<2> for TwoBarTruss {
    type Representation = Real;

    metadata!(
        "TwoBarTruss",
        DEB_PRATAP_MOITRA,
        Some(DEB_PRATAP_MOITRA_URL)
    );

    constraint_methods!();

    fn representation(&self) -> Real {
        bounds([(0.0, 0.01), (0.0, 0.01), (1.0, 3.0)])
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
        let (ideal, nadir) = (self.ideal_point()?, self.nadir_point()?);
        let scale = |f: [f64; 2]| [0, 1].map(|j| (f[j] - ideal[j]) / (nadir[j] - ideal[j]));
        // y = 2: f₁ f₂ = 400, the stress from 10⁵ down to 4000√5
        let hyperbola = |t: f64| {
            let stress = 1e5 + t * (4000.0 * 5f64.sqrt() - 1e5);
            scale([400.0 / stress, stress])
        };
        // x₂ = 0.01, y from 2 to 3
        let bound = |t: f64| {
            let y = 2.0 + t;
            let root = (1.0 + y * y).sqrt();
            scale([(4.0 + y * y) / (80.0 * root), 8000.0 * root / y])
        };
        let even_hyperbola = |t: f64| by_length(&hyperbola, t);
        let even_bound = |t: f64| by_length(&bound, t);
        let pieces = [
            Piece {
                curve: &even_hyperbola,
                with_end: false,
            },
            Piece {
                curve: &even_bound,
                with_end: true,
            },
        ];
        Some(scaled_front(&pieces, points, ideal, nadir))
    }

    // the least volume, 400 / 10⁵, and the least stress, 8000 √10 / 3, at y = 3
    fn ideal_point(&self) -> Option<[f64; 2]> {
        Some([0.004, 8000.0 * 10f64.sqrt() / 3.0])
    }

    // the volume at the least stress, 13 / (80 √10), and the stress limit
    fn nadir_point(&self) -> Option<[f64; 2]> {
        Some([13.0 / (80.0 * 10f64.sqrt()), 1e5])
    }
}

// ---- welded beam ---------------------------------------------------------------------------------

/// The welded beam with two objectives: the cheapest beam welded to a support that carries 6000
/// lb at 14 in, and the one whose end deflects least (Deb, Pratap and Moitra, 2000).
///
/// The genes are the weld's thickness h and length l and the bar's height t and thickness b, in
/// inches: x = (h, l, t, b). The objectives are the cost `1.10471 h² l + 0.04811 t b (14 + l)` and
/// the end deflection `δ = 2.1952 / (t³ b)`, in inches, subject to the first four constraints of
/// [`WeldedBeamRagsdell`](crate::problems::engineering::WeldedBeamRagsdell), whose code this
/// problem shares: the weld's shear stress τ ≤ 13,600 psi, the bar's bending stress
/// σ = 504,000 / (t² b) ≤ 30,000 psi, h ≤ b, and the buckling load
/// `P_c = 64,746.022 (1 − 0.0282346 t) t b³ ≥ 6000`; the deflection is an objective, not a
/// constraint.
///
/// Bounds h, b ∈ [0.125, 5] and l, t ∈ [0.1, 10], the paper's. The front isn't known. Its ends,
/// found with genoxide's SHADE and best known, give the ideal and nadir points: the cheapest
/// design, the single-objective problem's (whose deflection limit isn't active there), costs
/// 2.3811341169 at (0.24436895, 6.2186069, 8.2914718, 0.24436895), with a deflection of
/// 0.0157592; the least deflection is 2.1952 / (10³ · 5) = 0.00043904, at t = 10 and b = 5, where
/// the cheapest design costs 36.421245, at h = 1.7345106 and l = 0.4790053. Deb, Pratap and
/// Moitra's NSGA-II front runs from a cost of 2.79 to about 37, with deflections from 0.0088 down
/// to 0.0004.
///
/// This is Ragsdell and Phillips's form of the beam, which the paper prints with its constants
/// (eq. 2). Tanabe and Ishibuchi's restatement (2020, *Applied Soft Computing* 89: 106078,
/// problem CRE2-4-2) writes the deflection as `4PL³ / (E t³ b)`, the same, but takes the
/// shear stress and the buckling load of the other form,
/// [`WeldedBeam`](crate::problems::engineering::WeldedBeam)'s: not the same front.
///
/// [`constraints`](MultiProblem::constraints) gives the four constraints in this order, as
/// `g(x) ≤ 0`.
///
/// Deb, K., Pratap, A. and Moitra, S. (2000). Mechanical component design for multiple objectives
/// using elitist non-dominated sorting GA. *Parallel Problem Solving from Nature (PPSN VI)*, LNCS
/// 1917: 859-868, eq. 2 and section 4.4, checked in the authors' preprint (KanGAL report 200002).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WeldedBeam;

impl WeldedBeam {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        let (t, b) = (x[2], x[3]);
        [single::WeldedBeam.value(x), 2.1952 / (math::powi(t, 3) * b)]
    }

    fn values(&self, x: &Reals) -> [f64; 4] {
        let [shear, bending, thickness, buckling, _] = single::WeldedBeamRagsdell.values(x);
        [shear, bending, thickness, buckling]
    }
}

constrained!(WeldedBeam, 2, 4);

impl MultiProblem<2> for WeldedBeam {
    type Representation = Real;

    metadata!("WeldedBeam", DEB_PRATAP_MOITRA, Some(DEB_PRATAP_MOITRA_URL));

    constraint_methods!();

    fn representation(&self) -> Real {
        bounds([(0.125, 5.0), (0.1, 10.0), (0.1, 10.0), (0.125, 5.0)])
    }

    fn optimal_front(&self, _points: usize) -> Option<Vec<[f64; 2]>> {
        None
    }

    fn ideal_point(&self) -> Option<[f64; 2]> {
        Some(self.extremes()[0])
    }

    fn nadir_point(&self) -> Option<[f64; 2]> {
        Some(self.extremes()[1])
    }
}

impl WeldedBeam {
    // the ends of the front, best known: the cheapest design, and the cheapest with the least
    // deflection, t = 10 and b = 5, both found with genoxide's SHADE
    const CHEAPEST: [f64; 4] = [
        0.244_368_953_448_378_53,
        6.218_606_918_428_453,
        8.291_471_769_713_24,
        0.244_368_953_448_379_22,
    ];
    const STIFFEST: [f64; 4] = [1.734_510_586_893_948_5, 0.479_005_327_438_575_4, 10.0, 5.0];

    fn extremes(&self) -> [[f64; 2]; 2] {
        extremes(|x| self.objectives(x), &Self::CHEAPEST, &Self::STIFFEST)
    }
}

// ---- disc brake ----------------------------------------------------------------------------------

/// The disc brake: the lightest multiple-disc brake, and the one that stops fastest (Osyczka and
/// Kundu, 1995).
///
/// The genes are the discs' inner radius r and outer radius R, in mm, the engaging force F, in N,
/// and the number of friction surfaces s, an integer: x = (r, R, F, s).
/// [`evaluate`](MultiFitnessFunction::evaluate) rounds gene 3 to the nearest integer, and
/// [`design`](DiscBrake::design) gives the rounded design. The objectives are the mass
/// `f₁ = 4.9·10⁻⁵ (R² − r²)(s − 1)`, in kg, and the stopping time
/// `f₂ = 9.82·10⁶ (R² − r²) / (F s (R³ − r³))`, in s, subject to five constraints: the discs'
/// width `R − r ≥ 20`, the brake's length `2.5 (s + 1) ≤ 30`, the pressure
/// `F / (3.14 (R² − r²)) ≤ 0.4`, the temperature `2.22·10⁻³ F (R³ − r³) / (R² − r²)² ≤ 1`, and the
/// torque `0.0266 F s (R³ − r³) / (R² − r²) ≥ 900`.
///
/// Bounds r ∈ [55, 80], R ∈ [75, 110], F ∈ [1000, 3000] and s ∈ [2, 20]. With R ≤ r, which the
/// bounds allow and the first constraint forbids, the mass is 0 or negative, and at R = r the
/// stopping time is 0/0: NaN, which genoxide's algorithms treat as invalid. The front isn't
/// known; the brake's length allows at most 11 surfaces. Its ends follow from the definition:
/// the lightest brake, 0.1274 kg, has the narrowest discs at the smallest radii, r = 55 and
/// R = 75, and two surfaces, and stops in 16.654925 s at the largest force, 3000; the fastest,
/// 2.0710401 s, has the widest discs, r = 80 and R = 110, eleven surfaces and the largest force,
/// and weighs 2.793 kg. Both meet the constraints.
///
/// [`constraints`](MultiProblem::constraints) gives the five constraints of the design in this
/// order, as `g(x) ≤ 0`.
///
/// Osyczka, A. and Kundu, S. (1995). A new method to solve generalized multicriteria optimization
/// problems using the simple genetic algorithm. *Structural Optimization* 10(2): 94-99; also in
/// Ray, T. and Liew, K. M. (2002). A swarm metaphor for multiobjective design optimization.
/// *Engineering Optimization* 34(2): 141-153. Neither was read: definition and bounds as restated
/// in Yang, X.-S., Karamanoglu, M. and He, X. (2013). Multi-objective flower algorithm for
/// optimization. *Procedia Computer Science* 18: 861-868 (eqs. 10-12, arXiv:1404.0695), and the
/// same in Tanabe and Ishibuchi (2020, *Applied Soft Computing* 89: 106078, supplement, problem
/// RE3-4-3), except that they drop the length constraint and bound s by [11, 20], where their note
/// says the constraint gives s ≤ 11; not yet checked against the originals
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DiscBrake;

impl DiscBrake {
    // the ends of the front: the lightest brake that stops fastest, and the fastest
    const LIGHTEST: [f64; 4] = [55.0, 75.0, 3000.0, 2.0];
    const FASTEST: [f64; 4] = [80.0, 110.0, 3000.0, 11.0];

    /// The design of `genome`: its genes, with the number of friction surfaces s rounded to the
    /// nearest integer.
    ///
    /// # Panics
    ///
    /// If `genome` has fewer than 4 genes.
    pub fn design(&self, genome: &Reals) -> [f64; 4] {
        [genome[0], genome[1], genome[2], genome[3].round()]
    }

    fn objectives(&self, x: &Reals) -> [f64; 2] {
        let [r, outer, force, s] = self.design(x);
        let squares = outer * outer - r * r;
        let cubes = math::powi(outer, 3) - math::powi(r, 3);
        [
            4.9e-5 * squares * (s - 1.0),
            9.82e6 * squares / (force * s * cubes),
        ]
    }

    // 3.14 is the paper's constant, which is close to π
    #[expect(clippy::approx_constant)]
    fn values(&self, x: &Reals) -> [f64; 5] {
        let [r, outer, force, s] = self.design(x);
        let squares = outer * outer - r * r;
        let cubes = math::powi(outer, 3) - math::powi(r, 3);
        [
            20.0 - (outer - r),
            2.5 * (s + 1.0) - 30.0,
            force / (3.14 * squares) - 0.4,
            2.22e-3 * force * cubes / (squares * squares) - 1.0,
            900.0 - 0.0266 * force * s * cubes / squares,
        ]
    }
}

constrained!(DiscBrake, 2, 5);

impl MultiProblem<2> for DiscBrake {
    type Representation = Real;

    metadata!(
        "DiscBrake",
        "Osyczka, A. and Kundu, S. (1995). A new method to solve generalized multicriteria \
         optimization problems using the simple genetic algorithm. Structural Optimization \
         10(2): 94-99.",
        Some("https://doi.org/10.1007/BF01743536")
    );

    constraint_methods!();

    fn representation(&self) -> Real {
        bounds([(55.0, 80.0), (75.0, 110.0), (1000.0, 3000.0), (2.0, 20.0)])
    }

    fn optimal_front(&self, _points: usize) -> Option<Vec<[f64; 2]>> {
        None
    }

    fn ideal_point(&self) -> Option<[f64; 2]> {
        Some(extremes(|x| self.objectives(x), &Self::LIGHTEST, &Self::FASTEST)[0])
    }

    fn nadir_point(&self) -> Option<[f64; 2]> {
        Some(extremes(|x| self.objectives(x), &Self::LIGHTEST, &Self::FASTEST)[1])
    }
}

// ---- speed reducer -------------------------------------------------------------------------------

/// The speed reducer with two objectives: the lightest gearbox, and the one whose first shaft is
/// least stressed (Kurpati, Azarm and Wu, 2002).
///
/// The genes are those of Golinski's single-objective
/// [`SpeedReducer`](crate::problems::engineering::SpeedReducer): the face width x₁, the module of
/// the teeth x₂, the number of teeth on the pinion x₃ (an integer:
/// [`evaluate`](MultiFitnessFunction::evaluate) rounds gene 2 to the nearest integer, and
/// [`design`](SpeedReducer::design) gives the rounded design), the lengths of the shafts between
/// bearings x₄ and x₅, and the diameters of the shafts x₆ and x₇. The objectives are the volume
/// `f₁ = 0.7854 x₁x₂² (10x₃²/3 + 14.933 x₃ − 43.0934) − 1.508 x₁ (x₆² + x₇²) + 7.477 (x₆³ + x₇³)
/// + 0.7854 (x₄x₆² + x₅x₇²)` and the first shaft's stress
/// `f₂ = √((745 x₄ / (x₂x₃))² + 1.69·10⁷) / (0.1 x₆³)`, subject to eleven constraints `gᵢ ≤ 0`:
/// `1/(x₁x₂²x₃) − 1/27`, `1/(x₁x₂²x₃²) − 1/397.5`, `x₄³/(x₂x₃x₆⁴) − 1/1.93`,
/// `x₅³/(x₂x₃x₇⁴) − 1/1.93`, `x₂x₃ − 40`, `x₁/x₂ − 12`, `5 − x₁/x₂`, `1.9 − x₄ + 1.5x₆`,
/// `1.9 − x₅ + 1.1x₇`, `f₂ − 1300`, and `√((745 x₅ / (x₂x₃))² + 1.575·10⁸) / (0.1 x₇³) − 1100`.
///
/// Bounds x₁ ∈ [2.6, 3.6], x₂ ∈ [0.7, 0.8], x₃ ∈ [17, 28], x₄, x₅ ∈ [7.3, 8.3], x₆ ∈ [2.9, 3.9],
/// x₇ ∈ [5, 5.5]. The front isn't known. Its ends, found with genoxide's SHADE and best known,
/// give the ideal and nadir points: the lightest design, 2771.9151 at (3.5, 0.7, 17, 7.3, 7.4,
/// 3.1687580, 5), where the first shaft's stress is at its limit, 1300; and the least stress,
/// 694.70574, at (3.6, 0.72, 28, 7.75, 7.4, 3.9, 5), where the lightest design weighs 5777.9203.
/// The second shaft's length is at 1.9 + 1.1x₇ = 7.4 in both, and at the second end the first's
/// is at 1.9 + 1.5x₆, and x₁/x₂ ≥ 5 keeps x₂ at 0.72.
///
/// The constants differ from Golinski's form: 7.477 and 14.933 for 7.4777 and 14.9334, stress
/// limits of 1300 and 1100 for 1100 and 850, and x₅ down to 7.3. Both restatements below agree
/// on them, except the second shaft's constant: 1.575·10⁸ in Tanabe and Ishibuchi, 1.275·10⁸ in
/// the other. genoxide follows 1.575·10⁸, which is Golinski's 157.5·10⁶ for the same shaft.
///
/// [`constraints`](MultiProblem::constraints) gives g₁…g₁₁ of the design in this order.
///
/// Kurpati, A., Azarm, S. and Wu, J. (2002). Constraint handling improvements for multiobjective
/// genetic algorithms. *Structural and Multidisciplinary Optimization* 23(3): 204-213, not read.
/// Definition and bounds as restated in Tanabe, R. and Ishibuchi, H. (2020). An easy-to-use
/// real-world multi-objective optimization problem suite. *Applied Soft Computing* 89: 106078
/// (supplement, problem RE3-7-5 and its constrained form CRE2-7-4), and in Saad, Emam and Houssein
/// (2025, *Scientific Reports* 15: 5126, eqs. 22-23); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SpeedReducer;

impl SpeedReducer {
    // the ends of the front, best known: the lightest design, and the lightest with the least
    // stress, both found with genoxide's SHADE
    const LIGHTEST: [f64; 7] = [
        3.5,
        0.7,
        17.0,
        7.3,
        7.400_000_000_000_013,
        3.168_758_048_542_292,
        5.0,
    ];
    const LEAST_STRESSED: [f64; 7] = [
        3.599_999_999_999_967_2,
        0.719_999_999_999_993_4,
        28.0,
        7.749_999_999_999_999,
        7.400_000_000_000_013,
        3.9,
        5.0,
    ];

    /// The design of `genome`: its genes, with the number of teeth x₃ rounded to the nearest
    /// integer.
    ///
    /// # Panics
    ///
    /// If `genome` has fewer than 7 genes.
    pub fn design(&self, genome: &Reals) -> [f64; 7] {
        single::SpeedReducer.design(genome)
    }

    // the stress in the second shaft
    fn second_stress(x: [f64; 7]) -> f64 {
        let [_, x2, x3, _, x5, _, x7] = x;
        (math::powi(745.0 * x5 / (x2 * x3), 2) + 1.575e8).sqrt() / (0.1 * math::powi(x7, 3))
    }

    // 0.7854 is the paper's constant, which is close to π/4
    #[expect(clippy::approx_constant)]
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        let design = self.design(x);
        let [x1, x2, x3, x4, x5, x6, x7] = design;
        [
            0.7854 * x1 * x2 * x2 * (10.0 * x3 * x3 / 3.0 + 14.933 * x3 - 43.0934)
                - 1.508 * x1 * (x6 * x6 + x7 * x7)
                + 7.477 * (math::powi(x6, 3) + math::powi(x7, 3))
                + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7),
            (math::powi(745.0 * x4 / (x2 * x3), 2) + 1.69e7).sqrt() / (0.1 * math::powi(x6, 3)),
        ]
    }

    fn values(&self, x: &Reals) -> [f64; 11] {
        let design = self.design(x);
        let [x1, x2, x3, x4, x5, x6, x7] = design;
        let [_, stress] = self.objectives(x);
        [
            1.0 / (x1 * x2 * x2 * x3) - 1.0 / 27.0,
            1.0 / (x1 * x2 * x2 * x3 * x3) - 1.0 / 397.5,
            math::powi(x4, 3) / (x2 * x3 * math::powi(x6, 4)) - 1.0 / 1.93,
            math::powi(x5, 3) / (x2 * x3 * math::powi(x7, 4)) - 1.0 / 1.93,
            x2 * x3 - 40.0,
            x1 / x2 - 12.0,
            5.0 - x1 / x2,
            1.9 - x4 + 1.5 * x6,
            1.9 - x5 + 1.1 * x7,
            stress - 1300.0,
            Self::second_stress(design) - 1100.0,
        ]
    }
}

constrained!(SpeedReducer, 2, 11);

impl MultiProblem<2> for SpeedReducer {
    type Representation = Real;

    metadata!(
        "SpeedReducer",
        "Kurpati, A., Azarm, S. and Wu, J. (2002). Constraint handling improvements for \
         multiobjective genetic algorithms. Structural and Multidisciplinary Optimization 23(3): \
         204-213.",
        Some("https://doi.org/10.1007/s00158-002-0178-2")
    );

    constraint_methods!();

    fn representation(&self) -> Real {
        bounds([
            (2.6, 3.6),
            (0.7, 0.8),
            (17.0, 28.0),
            (7.3, 8.3),
            (7.3, 8.3),
            (2.9, 3.9),
            (5.0, 5.5),
        ])
    }

    fn optimal_front(&self, _points: usize) -> Option<Vec<[f64; 2]>> {
        None
    }

    fn ideal_point(&self) -> Option<[f64; 2]> {
        let objectives = |x: &Reals| self.objectives(x);
        Some(extremes(objectives, &Self::LIGHTEST, &Self::LEAST_STRESSED)[0])
    }

    fn nadir_point(&self) -> Option<[f64; 2]> {
        let objectives = |x: &Reals| self.objectives(x);
        Some(extremes(objectives, &Self::LIGHTEST, &Self::LEAST_STRESSED)[1])
    }
}

// ---- four-bar truss ------------------------------------------------------------------------------

// the four-bar truss's load F (kN), bar length L (cm), Young's modulus E (kN/cm²) and allowed
// stress σ (kN/cm²), which gives the unit of the areas, F/σ = 1 cm²
const FOUR_BAR_LOAD: f64 = 10.0;
const FOUR_BAR_LENGTH: f64 = 200.0;
const FOUR_BAR_YOUNG: f64 = 2e5;

/// The four-bar truss: the truss of four bars with the least volume, and the one whose loaded
/// joint moves least (Stadler and Dauer, 1993).
///
/// The genes are the bars' cross-sections x₁ … x₄, in cm². The objectives are the volume
/// `f₁ = L (2x₁ + √2 x₂ + √2 x₃ + x₄)`, in cm³, and the joint's displacement
/// `f₂ = (FL/E) (2/x₁ + 2√2/x₂ − 2√2/x₃ + 2/x₄)`, in cm, with F = 10 kN, L = 200 cm and
/// E = 2·10⁵ kN/cm². There are no constraints besides the bounds, which keep each bar's stress
/// under σ = 10 kN/cm²: x₁, x₄ ∈ [F/σ, 3F/σ] = [1, 3] and x₂, x₃ ∈ [√2 F/σ, 3F/σ] = [√2, 3].
///
/// **The front**, derived from the definition: x₃ adds to the volume and takes from the
/// displacement's third term, so both are least at x₃ = √2. The rest is a convex problem, whose
/// optimal solutions minimize `f₁ + λ f₂` for some λ > 0: x₁ = √(λ/20000), x₂ = x₄ = √(λ/10000),
/// each clamped to its bounds. That gives three pieces: x₁ = 1, x₂ = √2 and x₄ from 1 to √2; then
/// x₂ = x₄ = √2 x₁ from √2 to 3; then x₂ = x₄ = 3 and x₁ from 3/√2 to 3. The front runs from
/// (1400, 0.04) at x = (1, √2, √2, 1) to (2200 + 600√2, (2√2 − 2)/300) ≈ (3048.53, 0.0027614) at
/// x = (3, 3, √2, 3); [`optimal_front`](MultiProblem::optimal_front) spreads its points evenly
/// along it in objectives scaled by the ideal and nadir points.
///
/// Tanabe and Ishibuchi's restatement (2020, *Applied Soft Computing* 89: 106078, problem RE2-4-1)
/// prints and computes the volume with √x₃ for √2 x₃: not the same front.
///
/// Stadler, W. and Dauer, J. (1993). Multicriteria optimization in engineering: a tutorial and
/// survey. In *Structural Optimization: Status and Promise*, Progress in Astronautics and
/// Aeronautics 150, AIAA: 209-249; and Cheng, F. Y. and Li, X. S. (1999). Generalized center
/// method for multiobjective engineering optimization. *Engineering Optimization* 31(5): 641-661.
/// Neither was read: definition, constants and bounds as restated in Costa, M. F. P. and
/// Fernandes, E. M. G. P. (2009). Practical evaluation of an interior point three-D filter line
/// search method using engineering design problems. *8th World Congress on Structural and
/// Multidisciplinary Optimization*, Lisbon, problem (4-truss), after Stadler and Dauer, whose
/// displacement becomes a constraint there; not yet checked against the originals
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FourBarTruss;

impl FourBarTruss {
    fn objectives(&self, x: &Reals) -> [f64; 2] {
        let (x1, x2, x3, x4) = (x[0], x[1], x[2], x[3]);
        [
            FOUR_BAR_LENGTH * (2.0 * x1 + SQRT_2 * x2 + SQRT_2 * x3 + x4),
            FOUR_BAR_LOAD * FOUR_BAR_LENGTH / FOUR_BAR_YOUNG
                * (2.0 / x1 + 2.0 * SQRT_2 / x2 - 2.0 * SQRT_2 / x3 + 2.0 / x4),
        ]
    }

    // the optimal solution with x₄ = p for p in [1, √2] (first piece), x₂ = x₄ = p for p in
    // [√2, 3] (second), or x₁ = p / √2 for p in [3, 3√2] (third)
    fn optimal(p: f64) -> Reals {
        let x = if p <= SQRT_2 {
            [1.0, SQRT_2, SQRT_2, p]
        } else if p <= 3.0 {
            [p / SQRT_2, p, SQRT_2, p]
        } else {
            [p / SQRT_2, 3.0, SQRT_2, 3.0]
        };
        Reals::from(x.to_vec())
    }
}

unconstrained!(FourBarTruss, 2);

impl MultiProblem<2> for FourBarTruss {
    type Representation = Real;

    metadata!(
        "FourBarTruss",
        "Stadler, W. and Dauer, J. (1993). Multicriteria optimization in engineering: a tutorial \
         and survey. In Structural Optimization: Status and Promise, Progress in Astronautics and \
         Aeronautics 150, AIAA: 209-249.",
        Some("https://doi.org/10.2514/5.9781600866234.0209.0249")
    );

    fn representation(&self) -> Real {
        bounds([(1.0, 3.0), (SQRT_2, 3.0), (SQRT_2, 3.0), (1.0, 3.0)])
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
        let (ideal, nadir) = (self.ideal_point()?, self.nadir_point()?);
        let scale = |f: [f64; 2]| [0, 1].map(|j| (f[j] - ideal[j]) / (nadir[j] - ideal[j]));
        let curve = |from: f64, to: f64| {
            move |t: f64| scale(self.objectives(&Self::optimal(from + t * (to - from))))
        };
        let (first, second, third) = (
            curve(1.0, SQRT_2),
            curve(SQRT_2, 3.0),
            curve(3.0, 3.0 * SQRT_2),
        );
        let first = |t: f64| by_length(&first, t);
        let second = |t: f64| by_length(&second, t);
        let third = |t: f64| by_length(&third, t);
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
                with_end: true,
            },
        ];
        Some(scaled_front(&pieces, points, ideal, nadir))
    }

    fn ideal_point(&self) -> Option<[f64; 2]> {
        Some([1400.0, (2.0 * SQRT_2 - 2.0) / 300.0])
    }

    fn nadir_point(&self) -> Option<[f64; 2]> {
        Some([2200.0 + 600.0 * SQRT_2, 0.04])
    }
}

// ---- car side impact -----------------------------------------------------------------------------

/// The car side impact with three objectives: the car's weight, the pubic force a passenger
/// feels, and the mean velocity of the B-pillar and the front door in the side impact (Jain and
/// Deb, 2014).
///
/// The genes, bounds and ten constraints are those of the single-objective
/// [`CarSideImpact`](crate::problems::engineering::CarSideImpact), whose code this problem shares:
/// the thicknesses of seven parts of the car body, and response surfaces fitted to crash
/// simulations. The objectives are the weight `f₁ = 1.98 + 4.9x₁ + 6.67x₂ + 6.98x₃ + 4.01x₄ +
/// 1.78x₅ + 0.00001x₆ + 2.73x₇`, the pubic force `f₂ = F = 4.72 − 0.5x₄ − 0.19x₂x₃`, in kN, and
/// `f₃ = (V_MBP + V_FD) / 2`, the mean of the B-pillar middle point's velocity
/// `V_MBP = 10.58 − 0.674x₁x₂ − 0.67275x₂` and the front door's `V_FD = 16.45 − 0.489x₃x₇ −
/// 0.843x₅x₆`, in mm/ms. The pubic force and both velocities stay constrained too, by F ≤ 4,
/// V_MBP ≤ 9.9 and V_FD ≤ 15.7.
///
/// The front isn't known. Jain and Deb draw theirs from about (21, 3.6, 10.5) to (45, 4, 12.5)
/// (their figure 19), where only 95 of NSGA-III's 153 reference directions find a solution. The
/// least weight is the single-objective problem's best known, 23.585658.
///
/// [`constraints`](MultiProblem::constraints) gives g₁…g₁₀ as
/// [`CarSideImpact`](crate::problems::engineering::CarSideImpact) does, each response minus its
/// limit.
///
/// Gu, L., Yang, R. J., Tho, C. H., Makowski, M., Faruque, O. and Li, Y. (2001). Optimisation and
/// robustness for crashworthiness of side impact. *International Journal of Vehicle Design*
/// 26(4): 348-360 (not read), with the three objectives of Jain, H. and Deb, K. (2014). An
/// evolutionary many-objective optimization algorithm using reference-point based nondominated
/// sorting approach, part II. *IEEE Transactions on Evolutionary Computation* 18(4): 602-622
/// (section V-F and appendix, checked in the authors' accepted manuscript).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CarSideImpact;

impl CarSideImpact {
    // a design that minimizes each objective, found with genoxide's SHADE: the single-objective
    // problem's best known, the least pubic force (x₂ = 1.35, x₃ = x₄ = 1.5) and the least mean
    // velocity (x₁ = x₂ = x₃ = 1.5 at their upper bounds, x₅ = 2.625, x₆ = x₇ = 1.2)
    const EXTREMES: [[f64; 7]; 3] = [
        [
            0.5,
            1.225_732_323_232_322_5,
            0.5,
            1.207_110_858_585_857_6,
            0.875,
            0.884_189_120_488_052_4,
            0.4,
        ],
        [
            1.445_912_100_164_925_8,
            1.35,
            1.5,
            1.5,
            1.117_796_324_519_709_1,
            0.462_415_417_518_881_2,
            1.114_772_370_363_288_6,
        ],
        [1.5, 1.35, 1.5, 0.853_432_002_066_908_5, 2.625, 1.2, 1.2],
    ];

    fn objectives(&self, x: &Reals) -> [f64; 3] {
        let responses = single::CarSideImpact.responses(x);
        let (pubic, pillar, door) = (responses[7], responses[8], responses[9]);
        [single::CarSideImpact.value(x), pubic, 0.5 * (pillar + door)]
    }

    fn values(&self, x: &Reals) -> [f64; 10] {
        single::CarSideImpact.values(x)
    }
}

constrained!(CarSideImpact, 3, 10);

impl MultiProblem<3> for CarSideImpact {
    type Representation = Real;

    metadata!(
        "CarSideImpact",
        "Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using \
         reference-point based nondominated sorting approach, part II: handling constraints and \
         extending to an adaptive approach. IEEE Transactions on Evolutionary Computation 18(4): \
         602-622.",
        Some("https://doi.org/10.1109/TEVC.2013.2281534")
    );

    constraint_methods!();

    fn representation(&self) -> Real {
        use crate::problems::Problem;
        single::CarSideImpact.representation()
    }

    fn optimal_front(&self, _points: usize) -> Option<Vec<[f64; 3]>> {
        None
    }

    fn ideal_point(&self) -> Option<[f64; 3]> {
        Some(ideal(|x| self.objectives(x), &Self::EXTREMES))
    }
}

// ---- rocket injector -----------------------------------------------------------------------------

/// The rocket injector: the design of a single-element injector of hydrogen and oxygen for the
/// longest life and the best performance, through response surfaces fitted to CFD simulations
/// (Vaidyanathan, Tucker, Papila and Shyy, 2003).
///
/// The genes are the hydrogen's flow angle α, the changes in the hydrogen's and the oxygen's flow
/// areas ΔHA and ΔOA, and the oxidizer post tip's thickness OPTT, each scaled to [0, 1] over its
/// range: x = (α, ΔHA, ΔOA, OPTT). The objectives are the injector face's highest temperature
/// TF_max, the oxidizer post tip's highest temperature TT_max, and the combustion length X_cc, the
/// distance from the inlet where combustion is 99% complete, each a quadratic or cubic response
/// surface, scaled as in the paper:
///
/// ```text
/// TF_max = 0.692 + 0.477α − 0.687ΔHA − 0.080ΔOA − 0.0650OPTT − 0.167α² − 0.0129ΔHAα
///          + 0.0796ΔHA² − 0.0634ΔOAα − 0.0257ΔOAΔHA + 0.0877ΔOA² − 0.0521OPTTα
///          + 0.00156OPTTΔHA + 0.00198OPTTΔOA + 0.0184OPTT²
/// TT_max = 0.370 − 0.205α + 0.0307ΔHA + 0.108ΔOA + 1.019OPTT − 0.135α² + 0.0141ΔHAα
///          + 0.0998ΔHA² + 0.208ΔOAα − 0.0301ΔOAΔHA − 0.226ΔOA² + 0.353OPTTα
///          − 0.0497OPTTΔOA − 0.423OPTT² + 0.202ΔHAα² − 0.281ΔOAα² − 0.342ΔHA²α
///          − 0.245ΔHA²ΔOA + 0.281ΔOA²ΔHA − 0.184OPTT²α − 0.281ΔHAαΔOA
/// X_cc   = 0.153 − 0.322α + 0.396ΔHA + 0.424ΔOA + 0.0226OPTT + 0.175α² + 0.0185ΔHAα
///          − 0.0701ΔHA² − 0.251ΔOAα + 0.179ΔOAΔHA + 0.0150ΔOA² + 0.0134OPTTα
///          + 0.0296OPTTΔHA + 0.0752OPTTΔOA + 0.0192OPTT²
/// ```
///
/// The paper has a fourth objective, the wall temperature three inches from the face, TW₄ (its
/// eq. A2), left out here as in the three-objective form of Goel et al. (2007, *Computer Methods
/// in Applied Mechanics and Engineering* 196: 879-893, not read), which Tanabe and Ishibuchi
/// (2020, problem RE3-4-7) restate. The order here is the paper's. There are no constraints
/// besides the bounds [0, 1]⁴. The front isn't known; its ideal point, found with genoxide's
/// SHADE, is (0.0088934, −0.4315, 0.00488).
///
/// Vaidyanathan, R., Tucker, P. K., Papila, N. and Shyy, W. (2003). CFD-based design optimization
/// for single element rocket injector. *41st AIAA Aerospace Sciences Meeting*, AIAA paper
/// 2003-296, eqs. A1, A3 and A4, checked in the authors' copy (NASA NTRS 20030060421).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RocketInjector;

impl RocketInjector {
    // a design that minimizes each objective: TF_max's minimum, 0.0088934, found with genoxide's
    // SHADE at ΔOA = 0.5913341 and the others at their bounds; TT_max's, −0.4315, at (1, 1, 1, 0);
    // and X_cc's, 0.00488, at α = 0.92 and the others 0, where 0.153 − 0.322α + 0.175α² is least
    const EXTREMES: [[f64; 4]; 3] = [
        [0.0, 1.0, 0.591_334_091_183_096_3, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.92, 0.0, 0.0, 0.0],
    ];

    fn objectives(&self, x: &Reals) -> [f64; 3] {
        let (a, h, o, t) = (x[0], x[1], x[2], x[3]);
        let face =
            0.692 + 0.477 * a - 0.687 * h - 0.080 * o - 0.0650 * t - 0.167 * a * a - 0.0129 * h * a
                + 0.0796 * h * h
                - 0.0634 * o * a
                - 0.0257 * o * h
                + 0.0877 * o * o
                - 0.0521 * t * a
                + 0.001_56 * t * h
                + 0.001_98 * t * o
                + 0.0184 * t * t;
        let tip = 0.370 - 0.205 * a + 0.0307 * h + 0.108 * o + 1.019 * t - 0.135 * a * a
            + 0.0141 * h * a
            + 0.0998 * h * h
            + 0.208 * o * a
            - 0.0301 * o * h
            - 0.226 * o * o
            + 0.353 * t * a
            - 0.0497 * t * o
            - 0.423 * t * t
            + 0.202 * h * a * a
            - 0.281 * o * a * a
            - 0.342 * h * h * a
            - 0.245 * h * h * o
            + 0.281 * o * o * h
            - 0.184 * t * t * a
            - 0.281 * h * a * o;
        let length =
            0.153 - 0.322 * a + 0.396 * h + 0.424 * o + 0.0226 * t + 0.175 * a * a + 0.0185 * h * a
                - 0.0701 * h * h
                - 0.251 * o * a
                + 0.179 * o * h
                + 0.0150 * o * o
                + 0.0134 * t * a
                + 0.0296 * t * h
                + 0.0752 * t * o
                + 0.0192 * t * t;
        [face, tip, length]
    }
}

unconstrained!(RocketInjector, 3);

impl MultiProblem<3> for RocketInjector {
    type Representation = Real;

    metadata!(
        "RocketInjector",
        "Vaidyanathan, R., Tucker, P. K., Papila, N. and Shyy, W. (2003). CFD-based design \
         optimization for single element rocket injector. 41st AIAA Aerospace Sciences Meeting, \
         AIAA paper 2003-296.",
        Some("https://doi.org/10.2514/6.2003-296")
    );

    fn representation(&self) -> Real {
        bounds([(0.0, 1.0); 4])
    }

    fn optimal_front(&self, _points: usize) -> Option<Vec<[f64; 3]>> {
        None
    }

    fn ideal_point(&self) -> Option<[f64; 3]> {
        Some(ideal(|x| self.objectives(x), &Self::EXTREMES))
    }
}

// ---- vehicle crashworthiness ---------------------------------------------------------------------

/// Vehicle crashworthiness: the lightest car front that best protects its occupants in frontal
/// crashes, through response surfaces fitted to crash simulations (Liao, Li, Yang, Zhang and Li,
/// 2008).
///
/// The genes are the thicknesses of five reinforcing members around the car's front, in mm, each
/// in [1, 3]. The objectives are the car's mass
/// `f₁ = 1640.2823 + 2.3573285x₁ + 2.3220035x₂ + 4.5688768x₃ + 7.7213633x₄ + 4.4559504x₅`, in kg,
/// the integral of the deceleration in the full frontal crash
/// `f₂ = 6.5856 + 1.15x₁ − 1.0427x₂ + 0.9738x₃ + 0.8364x₄ − 0.3695x₁x₄ + 0.0861x₁x₅ + 0.3628x₂x₄
/// − 0.1106x₁² − 0.3437x₃² + 0.1764x₄²`, and the toe board's intrusion in the 40% offset frontal
/// crash `f₃ = −0.0551 + 0.0181x₁ + 0.1024x₂ + 0.0421x₃ − 0.0073x₁x₂ + 0.024x₂x₃ − 0.0118x₂x₄
/// − 0.0204x₃x₄ − 0.008x₃x₅ − 0.0241x₂² + 0.0109x₄²`. There are no constraints besides the
/// bounds. The front isn't known; Deb and Jain (2014, part I, figure 30) draw theirs over masses
/// from about 1660 to 1700.
///
/// Liao, X., Li, Q., Yang, X., Zhang, W. and Li, W. (2008). Multiobjective optimization for crash
/// safety design of vehicles using stepwise regression model. *Structural and Multidisciplinary
/// Optimization* 35(6): 561-569, not read. Definition and bounds as restated in Tanabe and
/// Ishibuchi (2020, *Applied Soft Computing* 89: 106078, supplement, problem RE3-5-4), and the
/// same in de Carvalho, V. R. and Sichman, J. S. (2018). Solving real-world multi-objective
/// engineering optimization problems with an election-based hyper-heuristic. *OptMAS 2018*
/// (eqs. 1-3); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VehicleCrashworthiness;

impl VehicleCrashworthiness {
    // a design that minimizes each objective, at bounds, found with genoxide's SHADE: the
    // thinnest members; x₂ = x₃ = 3; and x₃ = x₄ = x₅ = 3
    const EXTREMES: [[f64; 5]; 3] = [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 3.0, 3.0, 1.0, 1.0],
        [1.0, 1.0, 3.0, 3.0, 3.0],
    ];

    fn objectives(&self, x: &Reals) -> [f64; 3] {
        let (x1, x2, x3, x4, x5) = (x[0], x[1], x[2], x[3], x[4]);
        [
            1640.2823
                + 2.357_328_5 * x1
                + 2.322_003_5 * x2
                + 4.568_876_8 * x3
                + 7.721_363_3 * x4
                + 4.455_950_4 * x5,
            6.5856 + 1.15 * x1 - 1.0427 * x2 + 0.9738 * x3 + 0.8364 * x4 - 0.3695 * x1 * x4
                + 0.0861 * x1 * x5
                + 0.3628 * x2 * x4
                - 0.1106 * x1 * x1
                - 0.3437 * x3 * x3
                + 0.1764 * x4 * x4,
            -0.0551 + 0.0181 * x1 + 0.1024 * x2 + 0.0421 * x3 - 0.0073 * x1 * x2 + 0.024 * x2 * x3
                - 0.0118 * x2 * x4
                - 0.0204 * x3 * x4
                - 0.008 * x3 * x5
                - 0.0241 * x2 * x2
                + 0.0109 * x4 * x4,
        ]
    }
}

unconstrained!(VehicleCrashworthiness, 3);

impl MultiProblem<3> for VehicleCrashworthiness {
    type Representation = Real;

    metadata!(
        "VehicleCrashworthiness",
        "Liao, X., Li, Q., Yang, X., Zhang, W. and Li, W. (2008). Multiobjective optimization for \
         crash safety design of vehicles using stepwise regression model. Structural and \
         Multidisciplinary Optimization 35(6): 561-569.",
        Some("https://doi.org/10.1007/s00158-007-0163-x")
    );

    fn representation(&self) -> Real {
        bounds([(1.0, 3.0); 5])
    }

    fn optimal_front(&self, _points: usize) -> Option<Vec<[f64; 3]>> {
        None
    }

    fn ideal_point(&self) -> Option<[f64; 3]> {
        Some(ideal(|x| self.objectives(x), &Self::EXTREMES))
    }
}

// ---- water resource planning ---------------------------------------------------------------------

/// Water resource planning (WATER): the design of a storm drainage system, with five costs and
/// losses to minimize under seven constraints (Musselman and Talavage, 1980).
///
/// The genes are the local detention storage capacity x₁, the maximum treatment rate x₂ and the
/// maximum allowable overflow rate x₃. The objectives are the drainage network's cost
/// `f₁ = 106780.37 (x₂ + x₃) + 61704.67`, the storage facility's `f₂ = 3000 x₁`, the treatment
/// facility's `f₃ = 305700 · 2289 x₂ / (0.06 · 2289)^0.65`, the expected flood damage
/// `f₄ = 250 · 2289 exp(−39.75 x₂ + 9.9 x₃ + 2.74)` and the expected economic loss from floods
/// `f₅ = 25 (1.39 / (x₁x₂) + 4940 x₃ − 80)`, subject to
///
/// ```text
/// g₁ = 0.00139/(x₁x₂) + 4.94x₃ − 0.08       ≤ 1
/// g₂ = 0.000306/(x₁x₂) + 1.082x₃ − 0.0986   ≤ 1
/// g₃ = 12.307/(x₁x₂) + 49408.24x₃ + 4051.02 ≤ 50000
/// g₄ = 2.098/(x₁x₂) + 8046.33x₃ − 696.71    ≤ 16000
/// g₅ = 2.138/(x₁x₂) + 7883.39x₃ − 705.04    ≤ 10000
/// g₆ = 0.417x₁x₂ + 1721.26x₃ − 136.54       ≤ 2000
/// g₇ = 0.164/(x₁x₂) + 631.13x₃ − 54.48      ≤ 550
/// ```
///
/// Bounds x₁ ∈ [0.01, 0.45] and x₂, x₃ ∈ [0.01, 0.1]. g₆ is printed with a product x₁x₂ where
/// the others divide by it, in every restatement; as printed it holds everywhere in the bounds,
/// and read as a quotient, `0.417/(x₁x₂)`, it would still hold wherever g₇ does, so the reading
/// doesn't change the problem.
///
/// **The front**, derived from the definition: f₁, f₄ and f₅ grow with x₃, which the others
/// don't depend on, and every constraint is easier to meet with a smaller x₃, so the optimal
/// solutions have x₃ = 0.01. There, the constraints come down to a least x₁x₂, g₁'s
/// `0.00139 / (1.08 − 0.0494) ≈ 0.0013487`. And no two such solutions dominate each other: f₂
/// grows with x₁ and f₁ with x₂, so a solution that dominates another has neither a larger x₁
/// nor a larger x₂; f₄ falls as x₂ grows, so it has the same x₂, and then f₅ falls as x₁ grows,
/// so it has the same x₁. The optimal solutions are thus every (x₁, x₂, 0.01) with
/// x₁x₂ ≥ 0.0013487, and the front is their image, a surface in five dimensions.
/// [`optimal_front`](MultiProblem::optimal_front) samples it on a grid over x₁ and x₂ and along
/// the curve x₁x₂ = 0.0013487, its edge. Its ideal point is (63840.2774, 40.4619, 285346.896,
/// 183749.967, 7.2222) and its nadir point (73450.5107, 1350, 2853468.96, 6575303.1, 25000):
/// f₅ reaches 25000 on the edge, where `1.39 / (x₁x₂) = 1030.6`.
///
/// [`constraints`](MultiProblem::constraints) gives g₁…g₇ in this order, each minus its limit.
///
/// Musselman, K. and Talavage, J. (1980). A tradeoff cut approach to multiple objective
/// optimization. *Operations Research* 28(6): 1424-1435, and Ray, T., Tai, K. and Seow, K. C.
/// (2001). Multiobjective design optimization by an evolutionary algorithm. *Engineering
/// Optimization* 33(4): 399-424; neither was read. Definition and bounds as restated in Deb, K.,
/// Pratap, A., Agarwal, S. and Meyarivan, T. (2002). A fast and elitist multiobjective genetic
/// algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation* 6(2): 182-197 (table V),
/// the same in its preprint (KanGAL report 200001, table VI) and in Jain and Deb (2014, part II,
/// appendix).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WaterResourcePlanning;

impl WaterResourcePlanning {
    // the least x₁x₂ of a feasible solution with x₃ = 0.01, from g₁
    fn least_area() -> f64 {
        0.001_39 / (1.08 - 4.94 * 0.01)
    }

    // the least x₁ of a feasible solution with x₂ and x₃ = 0.01: the least area divided by x₂,
    // moved up to the first value that meets g₁ when evaluated
    fn least_x1(x2: f64) -> f64 {
        let mut x1 = Self::least_area() / x2;
        while at_most(Self.values(&Reals::from(vec![x1, x2, 0.01]))[0], 0.0) > 0.0 {
            x1 = x1.next_up();
        }
        x1
    }

    // an optimal design that minimizes (`best`) or maximizes each objective on the front
    fn extremes(&self, best: bool) -> [[f64; 3]; 5] {
        let (least, top) = (Self::least_x1(0.1), Self::least_x1(0.01));
        if best {
            [
                [0.45, 0.01, 0.01],
                [least, 0.1, 0.01],
                [0.45, 0.01, 0.01],
                [0.45, 0.1, 0.01],
                [0.45, 0.1, 0.01],
            ]
        } else {
            [
                [0.45, 0.1, 0.01],
                [0.45, 0.1, 0.01],
                [0.45, 0.1, 0.01],
                [top, 0.01, 0.01],
                [least, 0.1, 0.01],
            ]
        }
    }

    fn objectives(&self, x: &Reals) -> [f64; 5] {
        let (x1, x2, x3) = (x[0], x[1], x[2]);
        [
            106_780.37 * (x2 + x3) + 61_704.67,
            3000.0 * x1,
            305_700.0 * 2289.0 * x2 / math::powf(0.06 * 2289.0, 0.65),
            250.0 * 2289.0 * math::exp(-39.75 * x2 + 9.9 * x3 + 2.74),
            25.0 * (1.39 / (x1 * x2) + 4940.0 * x3 - 80.0),
        ]
    }

    fn values(&self, x: &Reals) -> [f64; 7] {
        let (x1, x2, x3) = (x[0], x[1], x[2]);
        let area = x1 * x2;
        [
            0.001_39 / area + 4.94 * x3 - 0.08 - 1.0,
            0.000_306 / area + 1.082 * x3 - 0.0986 - 1.0,
            12.307 / area + 49_408.24 * x3 + 4051.02 - 50_000.0,
            2.098 / area + 8046.33 * x3 - 696.71 - 16_000.0,
            2.138 / area + 7883.39 * x3 - 705.04 - 10_000.0,
            0.417 * area + 1721.26 * x3 - 136.54 - 2000.0,
            0.164 / area + 631.13 * x3 - 54.48 - 550.0,
        ]
    }
}

constrained!(WaterResourcePlanning, 5, 7);

impl MultiProblem<5> for WaterResourcePlanning {
    type Representation = Real;

    metadata!(
        "WaterResourcePlanning",
        "Musselman, K. and Talavage, J. (1980). A tradeoff cut approach to multiple objective \
         optimization. Operations Research 28(6): 1424-1435.",
        Some("https://doi.org/10.1287/opre.28.6.1424")
    );

    constraint_methods!();

    fn representation(&self) -> Real {
        bounds([(0.01, 0.45), (0.01, 0.1), (0.01, 0.1)])
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 5]>> {
        // a k × k grid over x₁ and x₂, its feasible points, and k points on the edge
        // x₁x₂ = the least area, from x₂ = 0.1 to 0.01, until there are enough
        let area = Self::least_area();
        let mut k = 2;
        loop {
            let mut front = Vec::new();
            for i in 0..k {
                for j in 0..k {
                    let x1 = 0.01 + 0.44 * i as f64 / (k - 1) as f64;
                    let x2 = 0.01 + 0.09 * j as f64 / (k - 1) as f64;
                    if x1 * x2 >= area {
                        front.push(self.objectives(&Reals::from(vec![x1, x2, 0.01])));
                    }
                }
            }
            for i in 0..k {
                let x2 = 0.1 - 0.09 * i as f64 / (k - 1) as f64;
                front.push(self.objectives(&Reals::from(vec![Self::least_x1(x2), x2, 0.01])));
            }
            if front.len() >= points {
                return Some(front);
            }
            k += 1;
        }
    }

    fn ideal_point(&self) -> Option<[f64; 5]> {
        Some(ideal(|x| self.objectives(x), &self.extremes(true)))
    }

    fn nadir_point(&self) -> Option<[f64; 5]> {
        let designs = self.extremes(false);
        let values = designs.map(|design| self.objectives(&Reals::from(design.to_vec())));
        Some(std::array::from_fn(|j| values[j][j]))
    }
}

// ---- conceptual marine design --------------------------------------------------------------------

/// Conceptual marine design: the bulk carrier that carries cargo the cheapest, with the lightest
/// ship and the most cargo a year, in the Panamax case of Parsons and Scott (2004).
///
/// The genes are the ship's length L, beam B, depth D and draft T, in m, its block coefficient
/// C_B and its speed V_k, in knots: x = (L, B, D, T, C_B, V_k), the paper's order. The objectives
/// are the transportation cost `f₁ = annual cost / annual cargo`, in £/t, the light ship weight
/// `f₂ = W_s + W_o + W_m`, in t, and the annual cargo, maximized, so `f₃ = −annual cargo`, in t a
/// year. The model, Parsons and Scott's appendix (after Sen and Yang, with the Froude number in
/// consistent units):
///
/// ```text
/// displacement Δ = 1.025 L B T C_B                       V = 0.5144 V_k m/s, g = 9.8065 m/s²
/// Froude number F_n = V / √(g L)
/// power P = Δ^(2/3) V_k³ / (a + b F_n),   a = 4977.06 C_B² − 8105.61 C_B + 4456.51
///                                         b = −10847.2 C_B² + 12817 C_B − 6960.32
/// steel W_s = 0.034 L^1.7 B^0.7 D^0.4 C_B^0.5,  outfit W_o = L^0.8 B^0.6 D^0.3 C_B^0.1,
/// machinery W_m = 0.17 P^0.9,   deadweight DWT = Δ − (W_s + W_o + W_m)
/// daily consumption = 0.19 P · 24 / 1000 + 0.2,   sea days = 5000 / (24 V_k)
/// cargo deadweight = DWT − daily consumption (sea days + 5) − 2 DWT^0.5
/// port days = 2 (cargo deadweight / 8000 + 0.5)
/// round trips a year RTPA = 350 / (sea days + port days)
/// annual cost = 0.2 · 1.3 (2000 W_s^0.85 + 3500 W_o + 2400 P^0.8) + 40000 DWT^0.3
///             + (1.05 · daily consumption · sea days · 100 + 6.3 DWT^0.8) RTPA
/// annual cargo = cargo deadweight · RTPA
/// ```
///
/// subject to nine constraints: `L/B ≥ 6`, `L/D ≤ 15`, `L/T ≤ 19`, `T ≤ 0.45 DWT^0.31`,
/// `T ≤ 0.7 D + 0.7`, `25,000 ≤ DWT ≤ 500,000`, `F_n ≤ 0.32` and the metacentric height
/// `GM_T = 0.53 T + (0.085 C_B − 0.002) B² / (T C_B) − (1 + 0.52 D) ≥ 0.07 B`.
///
/// Bounds L ∈ [150, 274.32], B ∈ [20, 32.31], D ∈ [10, 25], T ∈ [8, 11.71], C_B ∈ [0.63, 0.75]
/// and V_k ∈ [14, 18]. The paper's base model has thirteen constraints and no bounds on L, B, D
/// and T (its variables are only nonnegative); its Panamax case 2 adds L ≤ 274.32 m, B ≤ 32.31 m
/// and T ≤ 11.71 m and raises the least deadweight from 3000 to 25,000 t. Here those limits and
/// the base model's 0.63 ≤ C_B ≤ 0.75 and 14 ≤ V_k ≤ 18 are the bounds, and the lower bounds of L,
/// B, D and T and the upper bound of D, which the paper doesn't give, are genoxide's: they hold
/// every feasible design, whose L is at least 150.73, B at least 22.23, D from 11.50 to 20.33 and
/// T at least 8.75 (each found by SLSQP from 300 random starts).
///
/// The front isn't known. Its ideal point is the paper's three single-criterion designs (table
/// 4), computed again: the least transportation cost, 8.376894 £/t at L = 6B = 193.86, B = 32.31,
/// D = 15.728571, T = 11.71, C_B = 0.680879 and V_k = 14 (the paper: 8.377 at C_B = 0.681); the
/// lightest ship, 5240.3356 t at (150.73, 25.12, 13.84, 10.39, 0.75, 14) with a deadweight of
/// exactly 25,000 (the paper: 5240.3); and the most cargo, 700,552.76 t a year at
/// (222.49, 32.31, 15.73, 11.71, 0.75, 18) (the paper: 700,553).
///
/// **Restatements.** Tanabe and Ishibuchi (2020, *Applied Soft Computing* 89: 106078, problem
/// RE4-6-2) add a fourth objective, the constraints' total violation, keep the least deadweight
/// at 3000 with bounds L ∈ [150, 274.32], B ∈ [20, 32.31], D ∈ [13, 25] and T ∈ [10, 11.71],
/// which cut off feasible designs, and their code computes the sea days as (5000/24) V_k. Kudela
/// (2023, *Computers* 12(11): 225) has V = 0.5114 V_k and the annual cargo from the deadweight.
/// The paper's tables decide: its designs (tables 3 to 6) give its printed criteria, deadweights
/// and powers to their printed digits with this model, and not with 0.5114.
///
/// [`constraints`](MultiProblem::constraints) gives the nine constraints in this order, as
/// `g(x) ≤ 0`.
///
/// Parsons, M. G. and Scott, R. L. (2004). Formulation of multicriterion design optimization
/// problems for solution with scalar numerical optimization methods. *Journal of Ship Research*
/// 48(1): 61-76, the numerical example (p. 68), its Panamax case 2 and table 4 (p. 69) and the
/// appendix (p. 76), after Sen, P. and Yang, J.-B. (1998). *Multiple Criteria Decision Support in
/// Engineering Design*. Springer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MarineDesign;

// the quantities of a ship that the objectives and constraints need
struct Ship {
    froude: f64,
    light: f64,
    deadweight: f64,
    cost: f64,
    cargo: f64,
    metacentric: f64,
}

impl MarineDesign {
    // a design that minimizes each objective (see the docs): the least transportation cost at
    // L = 6B, B and T at their bounds, T = 0.7D + 0.7 and V_k = 14, C_B found by Brent's method;
    // the lightest ship at L = 6B, T = 0.7D + 0.7 = 0.45 DWT^0.31 and DWT = 25,000 (plus 1e-6, so
    // that it's feasible when evaluated); and the most cargo at L = 19T, with B, T, C_B and V_k at
    // their upper bounds and T = 0.7D + 0.7. SLSQP from 300 random starts finds none better
    const EXTREMES: [[f64; 6]; 3] = [
        [
            193.86,
            32.31,
            15.728_571_428_571_431,
            11.71,
            0.680_878_824_022_718_3,
            14.0,
        ],
        [
            150.726_427_459_813_46,
            25.121_071_243_302_243,
            13.841_434_387_783_172,
            10.389_004_071_448_22,
            0.75,
            14.0,
        ],
        [222.49, 32.31, 15.728_571_428_571_431, 11.71, 0.75, 18.0],
    ];

    fn ship(x: &Reals) -> Ship {
        let (length, beam, depth, draft, block, knots) = (x[0], x[1], x[2], x[3], x[4], x[5]);
        let displacement = 1.025 * length * beam * draft * block;
        let froude = 0.5144 * knots / (9.8065 * length).sqrt();
        let a = 4977.06 * block * block - 8105.61 * block + 4456.51;
        let b = -10847.2 * block * block + 12817.0 * block - 6960.32;
        let power = math::powf(displacement, 2.0 / 3.0) * math::powi(knots, 3) / (a + b * froude);
        let steel = 0.034
            * math::powf(length, 1.7)
            * math::powf(beam, 0.7)
            * math::powf(depth, 0.4)
            * block.sqrt();
        let outfit = math::powf(length, 0.8)
            * math::powf(beam, 0.6)
            * math::powf(depth, 0.3)
            * math::powf(block, 0.1);
        let machinery = 0.17 * math::powf(power, 0.9);
        let light = steel + outfit + machinery;
        let deadweight = displacement - light;
        let daily = 0.19 * power * 24.0 / 1000.0 + 0.2;
        let sea_days = 5000.0 / (24.0 * knots);
        let cargo_deadweight = deadweight - daily * (sea_days + 5.0) - 2.0 * deadweight.sqrt();
        let port_days = 2.0 * (cargo_deadweight / 8000.0 + 0.5);
        let trips = 350.0 / (sea_days + port_days);
        let ship_cost = 1.3
            * (2000.0 * math::powf(steel, 0.85)
                + 3500.0 * outfit
                + 2400.0 * math::powf(power, 0.8));
        let voyage = (1.05 * daily * sea_days * 100.0 + 6.3 * math::powf(deadweight, 0.8)) * trips;
        let cost = 0.2 * ship_cost + 40_000.0 * math::powf(deadweight, 0.3) + voyage;
        let metacentric = 0.53 * draft + (0.085 * block - 0.002) * beam * beam / (draft * block)
            - (1.0 + 0.52 * depth);
        Ship {
            froude,
            light,
            deadweight,
            cost,
            cargo: cargo_deadweight * trips,
            metacentric,
        }
    }

    fn objectives(&self, x: &Reals) -> [f64; 3] {
        let ship = Self::ship(x);
        [ship.cost / ship.cargo, ship.light, -ship.cargo]
    }

    fn values(&self, x: &Reals) -> [f64; 9] {
        let (length, beam, depth, draft) = (x[0], x[1], x[2], x[3]);
        let ship = Self::ship(x);
        [
            6.0 - length / beam,
            length / depth - 15.0,
            length / draft - 19.0,
            draft - 0.45 * math::powf(ship.deadweight, 0.31),
            draft - (0.7 * depth + 0.7),
            25_000.0 - ship.deadweight,
            ship.deadweight - 500_000.0,
            ship.froude - 0.32,
            0.07 * beam - ship.metacentric,
        ]
    }
}

constrained!(MarineDesign, 3, 9);

impl MultiProblem<3> for MarineDesign {
    type Representation = Real;

    metadata!(
        "MarineDesign",
        "Parsons, M. G. and Scott, R. L. (2004). Formulation of multicriterion design \
         optimization problems for solution with scalar numerical optimization methods. Journal \
         of Ship Research 48(1): 61-76.",
        Some("https://doi.org/10.5957/jsr.2004.48.1.61")
    );

    constraint_methods!();

    fn representation(&self) -> Real {
        bounds([
            (150.0, 274.32),
            (20.0, 32.31),
            (10.0, 25.0),
            (8.0, 11.71),
            (0.63, 0.75),
            (14.0, 18.0),
        ])
    }

    fn optimal_front(&self, _points: usize) -> Option<Vec<[f64; 3]>> {
        None
    }

    fn ideal_point(&self) -> Option<[f64; 3]> {
        Some(ideal(|x| self.objectives(x), &Self::EXTREMES))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamRng;
    use crate::genome::Representation;
    use crate::multi::IntoScores;
    use crate::problems::Problem;

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

    // the scores of random genomes in the bounds: finite objectives, and never below the ideal
    // point where feasible (a check of the ideal point itself)
    fn check_random<P, const M: usize>(problem: &P, samples: usize)
    where
        P: MultiProblem<M, Representation = Real> + MultiFitnessFunction<Reals, M>,
    {
        let real = problem.representation();
        let ideal = problem.ideal_point().expect("known");
        let mut rng = StreamRng::seed_from_u64(5);
        for _ in 0..samples {
            let genome = real.random_genome(&mut rng);
            let scores = problem.evaluate(&genome).into_scores().expect("valid");
            let values = scores.values().expect("valid");
            assert!(values.iter().all(|v| v.is_finite()), "{genome:?}");
            if scores.is_feasible() {
                for j in 0..M {
                    assert!(
                        values[j] >= ideal[j] - 1e-9 * ideal[j].abs().max(1.0),
                        "{}: {genome:?} is below the ideal point in f{}",
                        problem.name(),
                        j + 1
                    );
                }
            }
        }
    }

    // no random feasible genome dominates a point of the optimal front
    fn no_genome_dominates_the_front<P>(problem: &P, samples: usize, tolerance: f64)
    where
        P: MultiProblem<2, Representation = Real> + MultiFitnessFunction<Reals, 2>,
    {
        let front = problem.optimal_front(400).expect("known");
        let real = problem.representation();
        let mut rng = StreamRng::seed_from_u64(3);
        for _ in 0..samples {
            let genome = real.random_genome(&mut rng);
            let scores = problem.evaluate(&genome).into_scores().expect("valid");
            if !scores.is_feasible() {
                continue;
            }
            let [f1, f2] = scores.values().expect("valid");
            for point in &front {
                assert!(
                    !(f1 < point[0] * (1.0 - tolerance) && f2 < point[1] * (1.0 - tolerance)),
                    "{}: {genome:?} at ({f1}, {f2}) dominates {point:?}",
                    problem.name()
                );
            }
        }
    }

    // the front's points are mutually non-dominated, run from the ideal to the nadir point's
    // corners, and are spread evenly: no gap between neighbors, in scaled objectives, is more than
    // `gap` times the mean
    fn check_front<P>(problem: &P, gap: f64)
    where
        P: MultiProblem<2>,
    {
        let front = problem.optimal_front(200).expect("known");
        assert_eq!(front.len(), 200);
        let (ideal, nadir) = (
            problem.ideal_point().unwrap(),
            problem.nadir_point().unwrap(),
        );
        assert_close(&front[0], &[ideal[0], nadir[1]], 1e-12);
        assert_close(&front[199], &[nadir[0], ideal[1]], 1e-12);
        let scaled: Vec<[f64; 2]> = front
            .iter()
            .map(|p| [0, 1].map(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j])))
            .collect();
        let steps: Vec<f64> = scaled
            .windows(2)
            .map(|w| {
                assert!(w[1][0] > w[0][0] && w[1][1] < w[0][1], "{w:?}");
                ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt()
            })
            .collect();
        let mean = steps.iter().sum::<f64>() / steps.len() as f64;
        assert!(steps.iter().all(|&step| step <= gap * mean), "{steps:?}");
    }

    #[test]
    fn two_bar_truss() {
        // both bars at 0.01 m² and y = 1: volume 0.01 (√17 + √2), stresses 20√17 / 0.01 and
        // 80√2 / 0.01, the larger BC's
        let (f, violation) = TwoBarTruss.evaluate(&at(&[0.01, 0.01, 1.0]));
        assert_close(
            &f,
            &[0.01 * (17f64.sqrt() + SQRT_2), 8000.0 * SQRT_2],
            1e-12,
        );
        assert_eq!(violation, 0.0);
        // a bar with no section: infinite stress, infinitely infeasible, finite volume
        let (f, violation) = TwoBarTruss.evaluate(&at(&[0.0, 0.01, 2.0]));
        assert_eq!((f[1], violation), (f64::INFINITY, f64::INFINITY));
        assert!(f[0].is_finite());
        // too thin: 10⁻⁴ m² for AC at y = 2 carries 20√20 / (2 · 10⁻⁴) ≈ 447,214 kPa
        let g = TwoBarTruss.constraints(&at(&[1e-4, 0.01, 2.0]));
        assert_close(g.inequalities(), &[20.0 * 20f64.sqrt() / 2e-4 - 1e5], 1e-12);
        // Deb and Srinivasan's table 1, whose variables are printed to three or four digits
        for (x, expected) in [
            ([4.60e-4, 9.05e-4, 1.935], [0.004013, 99_937.031]),
            ([49.30e-4, 99.89e-4, 2.035], [0.044779, 8_945.610]),
            ([39.54e-4, 100.00e-4, 3.000], [0.051391, 8_432.740]),
        ] {
            let (f, _) = TwoBarTruss.evaluate(&at(&x));
            assert_close(&f, &expected, 3e-3);
        }
        // the optimal solutions: y = 2 with both bars at the stress S, then x₂ = 0.01 and σ_BC
        // = S for y from 2 to 3
        for stress in [1e5, 5e4, 1e4, 4000.0 * 5f64.sqrt()] {
            let x = [
                20.0 * 5f64.sqrt() / stress,
                40.0 * 5f64.sqrt() / stress,
                2.0,
            ];
            let (f, violation) = TwoBarTruss.evaluate(&at(&x));
            assert_close(&f, &[400.0 / stress, stress], 1e-12);
            assert_eq!(violation, 0.0);
        }
        for y in [2.0f64, 2.5, 3.0] {
            let stress = 8000.0 * (1.0 + y * y).sqrt() / y;
            let x = [20.0 * (16.0 + y * y).sqrt() / (y * stress), 0.01, y];
            let (f, _) = TwoBarTruss.evaluate(&at(&x));
            let volume = (4.0 + y * y) / (80.0 * (1.0 + y * y).sqrt());
            assert_close(&f, &[volume, stress], 1e-12);
        }
        let (ideal, nadir) = (
            TwoBarTruss.ideal_point().unwrap(),
            TwoBarTruss.nadir_point().unwrap(),
        );
        assert_close(&ideal, &[0.004, 8_432.740_427_115_68], 1e-12);
        assert_close(&nadir, &[0.051_387_011_977_736_16, 1e5], 1e-12);
        check_front(&TwoBarTruss, 2.0);
        // the transition at 4000√5, the fullest-stressed corner of both pieces, is on the front
        let front = TwoBarTruss.optimal_front(1000).unwrap();
        for point in &front {
            if point[1] >= 4000.0 * 5f64.sqrt() {
                assert!((point[0] * point[1] - 400.0).abs() < 1e-9);
            }
        }
        no_genome_dominates_the_front(&TwoBarTruss, 200_000, 1e-9);
        check_random(&TwoBarTruss, 10_000);
    }

    #[test]
    fn welded_beam() {
        // the cost is the single-objective welded beam's, the deflection 2.1952 / (t³ b), and the
        // constraints the first four of Ragsdell and Phillips's form
        let x = at(&[0.3, 5.0, 8.0, 0.4]);
        let (f, violation) = WeldedBeam.evaluate(&x);
        assert_eq!(f[0], single::WeldedBeam.value(&x));
        assert_close(&[f[1]], &[2.1952 / (512.0 * 0.4)], 1e-15);
        let all = single::WeldedBeamRagsdell.constraints(&x);
        assert_eq!(
            WeldedBeam.constraints(&x).inequalities(),
            &all.inequalities()[..4]
        );
        assert_eq!(
            violation,
            all.inequalities()[..4].iter().map(|&g| g.max(0.0)).sum()
        );
        // the cheapest end costs 1.3e-9 less than the single-objective problem's best known
        // design, and has a deflection far under that problem's limit of 0.25
        let [ideal, nadir] = WeldedBeam.extremes();
        let best = single::WeldedBeamRagsdell.optimum().unwrap().value();
        assert!(ideal[0] <= best && best - ideal[0] < 2e-9);
        assert_close(
            &[ideal[0], nadir[1]],
            &[2.381_134_116_891_8, 0.015_759_164],
            1e-9,
        );
        // the least deflection, at t = 10 and b = 5
        assert_close(
            &[ideal[1], nadir[0]],
            &[2.1952 / 5000.0, 36.421_245_392],
            1e-9,
        );
        for design in [WeldedBeam::CHEAPEST, WeldedBeam::STIFFEST] {
            assert_eq!(WeldedBeam.evaluate(&at(&design)).1, 0.0);
        }
        assert_eq!(WeldedBeam.representation().bounds()[3], 0.125..=5.0);
        check_random(&WeldedBeam, 100_000);
    }

    #[test]
    #[expect(clippy::approx_constant)]
    fn disc_brake() {
        // the lightest end: r = 55, R = 75, F = 3000, s = 2; R² − r² = 2600 and R³ − r³ = 255,500
        let (f, violation) = DiscBrake.evaluate(&at(&[55.0, 75.0, 3000.0, 2.0]));
        assert_close(
            &f,
            &[
                4.9e-5 * 2600.0,
                9.82e6 * 2600.0 / (3000.0 * 2.0 * 255_500.0),
            ],
            1e-14,
        );
        assert_eq!(violation, 0.0);
        // the fastest end: r = 80, R = 110, F = 3000, s = 11; 5700 and 819,000
        let (f, violation) = DiscBrake.evaluate(&at(&[80.0, 110.0, 3000.0, 11.0]));
        assert_close(
            &f,
            &[
                4.9e-5 * 5700.0 * 10.0,
                9.82e6 * 5700.0 / (33_000.0 * 819_000.0),
            ],
            1e-14,
        );
        assert_eq!(violation, 0.0);
        assert_close(
            &DiscBrake.ideal_point().unwrap(),
            &[0.1274, 2.071_040_071_04],
            1e-11,
        );
        assert_close(
            &DiscBrake.nadir_point().unwrap(),
            &[2.793, 16.654_924_983_692],
            1e-11,
        );
        // s is rounded: 10.6 is 11 surfaces, and 11.6 is 12, too long by 2.5
        let x = at(&[80.0, 110.0, 3000.0, 10.6]);
        assert_eq!(DiscBrake.design(&x), [80.0, 110.0, 3000.0, 11.0]);
        let g = DiscBrake.constraints(&at(&[80.0, 110.0, 3000.0, 11.6]));
        assert_close(
            g.inequalities(),
            &[
                -10.0,
                2.5,
                3000.0 / (3.14 * 5700.0) - 0.4,
                2.22e-3 * 3000.0 * 819_000.0 / (5700.0 * 5700.0) - 1.0,
                900.0 - 0.0266 * 3000.0 * 12.0 * 819_000.0 / 5700.0,
            ],
            1e-12,
        );
        // discs 10 mm wide are 10 too narrow; a force of 1000 on the lightest discs gives too
        // little torque: 900 − 0.0266 · 1000 · 2 · 255,500 / 2600 ≈ −4.7, just enough
        assert_close(
            &[DiscBrake
                .constraints(&at(&[70.0, 80.0, 2000.0, 2.0]))
                .inequalities()[0]],
            &[10.0],
            1e-12,
        );
        let g = DiscBrake.constraints(&at(&[55.0, 75.0, 1000.0, 2.0]));
        assert_close(
            &[g.inequalities()[4]],
            &[900.0 - 0.0266 * 1000.0 * 2.0 * 255_500.0 / 2600.0],
            1e-12,
        );
        check_random(&DiscBrake, 100_000);
    }

    #[test]
    #[expect(clippy::approx_constant)]
    fn speed_reducer() {
        // the design rounds x₃ as the single-objective problem does
        let x = at(&[3.5, 0.7, 17.4, 7.3, 7.8, 3.35, 5.29]);
        assert_eq!(SpeedReducer.design(&x)[2], 17.0);
        // the volume differs from Golinski's by its constants only
        let [x1, x2, x3, x4, x5, x6, x7] = SpeedReducer.design(&x);
        let volume = 0.7854 * x1 * x2 * x2 * (10.0 * x3 * x3 / 3.0 + 14.933 * x3 - 43.0934)
            - 1.508 * x1 * (x6 * x6 + x7 * x7)
            + 7.477 * (x6.powi(3) + x7.powi(3))
            + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7);
        let stress = ((745.0 * x4 / (x2 * x3)).powi(2) + 1.69e7).sqrt() / (0.1 * x6.powi(3));
        let (f, _) = SpeedReducer.evaluate(&x);
        assert_close(&f, &[volume, stress], 1e-12);
        let g = SpeedReducer.constraints(&x);
        assert_eq!(g.len(), 11);
        assert_close(&[g.inequalities()[9]], &[stress - 1300.0], 1e-12);
        let second = ((745.0 * x5 / (x2 * x3)).powi(2) + 1.575e8).sqrt() / (0.1 * x7.powi(3));
        assert_close(&[g.inequalities()[10]], &[second - 1100.0], 1e-12);
        // the ends: the lightest at the first shaft's stress limit, and the least stress with
        // x₆ = 3.9, x₄ = 1.9 + 1.5 · 3.9 and x₂x₃ = 0.72 · 28
        for design in [SpeedReducer::LIGHTEST, SpeedReducer::LEAST_STRESSED] {
            assert_eq!(SpeedReducer.evaluate(&at(&design)).1, 0.0);
        }
        let (ideal, nadir) = (
            SpeedReducer.ideal_point().unwrap(),
            SpeedReducer.nadir_point().unwrap(),
        );
        let least =
            ((745.0f64 * 7.75 / (0.72 * 28.0)).powi(2) + 1.69e7).sqrt() / (0.1 * 3.9f64.powi(3));
        assert_close(&ideal, &[2_771.915_149_505_594, least], 1e-9);
        assert_close(&nadir, &[5_777.920_286_173_699, 1300.0], 1e-9);
        check_random(&SpeedReducer, 100_000);
    }

    #[test]
    fn four_bar_truss() {
        // all bars at their lower bounds: 200 (2 + 2 + 2 + 1) and 0.01 (2 + 2 − 2 + 2)
        let x = at(&[1.0, SQRT_2, SQRT_2, 1.0]);
        assert_close(&FourBarTruss.evaluate(&x), &[1400.0, 0.04], 1e-12);
        // at their upper bounds but x₃ = √2: 200 (6 + 3√2 + 2 + 3), and
        // 0.01 (2/3 + 2√2/3 − 2 + 2/3)
        let x = at(&[3.0, 3.0, SQRT_2, 3.0]);
        let expected = [
            200.0 * (11.0 + 3.0 * SQRT_2),
            0.01 * (4.0 / 3.0 + 2.0 * SQRT_2 / 3.0 - 2.0),
        ];
        assert_close(&FourBarTruss.evaluate(&x), &expected, 1e-12);
        assert_close(
            &FourBarTruss.ideal_point().unwrap(),
            &[1400.0, expected[1]],
            1e-12,
        );
        assert_close(
            &FourBarTruss.nadir_point().unwrap(),
            &[expected[0], 0.04],
            1e-12,
        );
        // a larger x₃ is worse in both
        let wide = FourBarTruss.evaluate(&at(&[2.0, 2.0, 2.0, 2.0]));
        let narrow = FourBarTruss.evaluate(&at(&[2.0, 2.0, SQRT_2, 2.0]));
        assert!(narrow[0] < wide[0] && narrow[1] < wide[1]);
        // the pieces meet at x₄ = √2 and at x₂ = x₄ = 3
        let meet = FourBarTruss.evaluate(&FourBarTruss::optimal(SQRT_2));
        assert_close(
            &meet,
            &FourBarTruss.evaluate(&at(&[1.0, SQRT_2, SQRT_2, SQRT_2])),
            1e-15,
        );
        check_front(&FourBarTruss, 2.0);
        no_genome_dominates_the_front(&FourBarTruss, 200_000, 1e-9);
        check_random(&FourBarTruss, 10_000);
    }

    #[test]
    fn car_side_impact() {
        // the constraints and the weight are the single-objective problem's; the pubic force and
        // the mean velocity are its responses 8, 9 and 10
        let single = single::CarSideImpact;
        let best = single.optimum().unwrap().solutions()[0].clone();
        let (f, violation) = CarSideImpact.evaluate(&best);
        assert_eq!(violation, 0.0);
        assert_eq!(f[0], single.optimum().unwrap().value());
        let r = single.responses(&best);
        assert_eq!(f[1], r[7]);
        assert_eq!(f[2], 0.5 * (r[8] + r[9]));
        assert_eq!(
            CarSideImpact.constraints(&best).inequalities(),
            single.constraints(&best).inequalities()
        );
        // at x₁ … x₇ = 1: F = 4.72 − 0.5 − 0.19, V_MBP = 10.58 − 0.674 − 0.67275 and
        // V_FD = 16.45 − 0.489 − 0.843
        let (f, _) = CarSideImpact.evaluate(&at(&[1.0; 7]));
        assert_close(
            &f,
            &[
                1.98 + 4.9 + 6.67 + 6.98 + 4.01 + 1.78 + 0.00001 + 2.73,
                4.03,
                0.5 * (9.23325 + 15.118),
            ],
            1e-14,
        );
        // the least pubic force and mean velocity, at bounds
        let ideal = CarSideImpact.ideal_point().unwrap();
        assert_close(
            &ideal,
            &[
                23.585_657_980_780_084,
                4.72 - 0.75 - 0.19 * 1.35 * 1.5,
                0.5 * (10.58 - 0.674 * 1.5 * 1.35 - 0.672_75 * 1.35)
                    + 0.5 * (16.45 - 0.489 * 1.5 * 1.2 - 0.843 * 2.625 * 1.2),
            ],
            1e-14,
        );
        for design in CarSideImpact::EXTREMES {
            assert_eq!(CarSideImpact.evaluate(&at(&design)).1, 0.0, "{design:?}");
        }
        check_random(&CarSideImpact, 100_000);
    }

    #[test]
    fn rocket_injector() {
        // at the origin only the constants are left, and at all ones the sums of the
        // coefficients
        assert_eq!(
            RocketInjector.evaluate(&at(&[0.0; 4])),
            [0.692, 0.370, 0.153]
        );
        let face: f64 = [
            0.692, 0.477, -0.687, -0.080, -0.0650, -0.167, -0.0129, 0.0796, -0.0634, -0.0257,
            0.0877, -0.0521, 0.00156, 0.00198, 0.0184,
        ]
        .iter()
        .sum();
        let tip: f64 = [
            0.370, -0.205, 0.0307, 0.108, 1.019, -0.135, 0.0141, 0.0998, 0.208, -0.0301, -0.226,
            0.353, -0.0497, -0.423, 0.202, -0.281, -0.342, -0.245, 0.281, -0.184, -0.281,
        ]
        .iter()
        .sum();
        let length: f64 = [
            0.153, -0.322, 0.396, 0.424, 0.0226, 0.175, 0.0185, -0.0701, -0.251, 0.179, 0.0150,
            0.0134, 0.0296, 0.0752, 0.0192,
        ]
        .iter()
        .sum();
        assert_close(
            &RocketInjector.evaluate(&at(&[1.0; 4])),
            &[face, tip, length],
            1e-14,
        );
        // the ideal point: X_cc's minimum 0.153 − 0.322² / (4 · 0.175) at α = 0.322 / 0.35
        let ideal = RocketInjector.ideal_point().unwrap();
        assert_close(
            &ideal,
            &[
                0.008_893_413_911_060_3,
                -0.4315,
                0.153 - 0.322 * 0.322 / 0.7,
            ],
            1e-12,
        );
        check_random(&RocketInjector, 100_000);
    }

    #[test]
    fn vehicle_crashworthiness() {
        // at all thicknesses 1 mm: the sums of the coefficients
        let f = VehicleCrashworthiness.evaluate(&at(&[1.0; 5]));
        let mass = 1640.2823 + 2.357_328_5 + 2.322_003_5 + 4.568_876_8 + 7.721_363_3 + 4.455_950_4;
        let deceleration =
            6.5856 + 1.15 - 1.0427 + 0.9738 + 0.8364 - 0.3695 + 0.0861 + 0.3628 - 0.1106 - 0.3437
                + 0.1764;
        let intrusion =
            -0.0551 + 0.0181 + 0.1024 + 0.0421 - 0.0073 + 0.024 - 0.0118 - 0.0204 - 0.008 - 0.0241
                + 0.0109;
        assert_close(&f, &[mass, deceleration, intrusion], 1e-14);
        assert_eq!(VehicleCrashworthiness.ideal_point().unwrap()[0], f[0]);
        check_random(&VehicleCrashworthiness, 100_000);
    }

    #[test]
    fn marine_design() {
        // Parsons and Scott's designs, printed to their tables' digits, give their printed
        // criteria and deadweights: table 4's three single-criterion designs (Panamax case 2),
        // table 6's min-max design, and table 3's least transportation cost (case 1, where B and
        // T may exceed the Panamax limits); criteria (£/t, t, t a year) and deadweight (t)
        for (x, [cost, light, cargo, deadweight]) in [
            (
                [193.86, 32.31, 15.73, 11.71, 0.681, 14.0],
                [8.377, 9029.0, 551_265.0, 42_160.0],
            ),
            (
                [150.73, 25.12, 13.84, 10.39, 0.750, 14.0],
                [9.474, 5240.3, 386_500.0, 25_000.0],
            ),
            (
                [222.49, 32.31, 15.73, 11.71, 0.750, 18.0],
                [10.294, 12_436.0, 700_553.0, 52_277.0],
            ),
            (
                [169.07, 28.16, 15.36, 10.66, 0.655, 17.95],
                [10.696, 6920.3, 475_966.0, 27_137.0],
            ),
            (
                [220.02, 36.67, 19.76, 14.53, 0.726, 14.0],
                [7.973, 13_512.0, 746_121.0, 73_761.0],
            ),
        ] {
            let (f, _) = MarineDesign.evaluate(&at(&x));
            let g = MarineDesign.constraints(&at(&x));
            assert!((f[0] - cost).abs() < 0.0015, "{x:?}: {f:?}");
            assert_close(&[f[1], -f[2]], &[light, cargo], 5e-4);
            assert_close(&[25_000.0 - g.inequalities()[5]], &[deadweight], 1e-3);
        }
        // the ideal point, the same as the paper's table 4 to its digits, at feasible designs
        let ideal = MarineDesign.ideal_point().unwrap();
        assert_close(
            &ideal,
            &[8.376_894_177_775, 5_240.335_559_466, -700_552.764_631_57],
            1e-9,
        );
        for design in MarineDesign::EXTREMES {
            assert_eq!(MarineDesign.evaluate(&at(&design)).1, 0.0, "{design:?}");
        }
        // the lightest ship has the least deadweight, 25,000, and its draft at both limits
        let lightest = MarineDesign.constraints(&at(&MarineDesign::EXTREMES[1]));
        for i in [0, 3, 4, 5] {
            assert!(lightest.inequalities()[i] > -1e-5, "{i}: {lightest:?}");
        }
        assert_eq!(MarineDesign.constraint_count(), 9);
        check_random(&MarineDesign, 100_000);
    }

    #[test]
    fn water_resource_planning() {
        // at x = (0.1, 0.05, 0.05): x₁x₂ = 0.005
        let x = at(&[0.1, 0.05, 0.05]);
        let (f, violation) = WaterResourcePlanning.evaluate(&x);
        assert_close(
            &f,
            &[
                106_780.37 * 0.1 + 61_704.67,
                300.0,
                305_700.0 * 2289.0 * 0.05 / (0.06f64 * 2289.0).powf(0.65),
                250.0 * 2289.0 * (-39.75f64 * 0.05 + 9.9 * 0.05 + 2.74).exp(),
                25.0 * (1.39 / 0.005 + 4940.0 * 0.05 - 80.0),
            ],
            1e-12,
        );
        let g = WaterResourcePlanning.constraints(&x);
        let expected = [
            0.00139 / 0.005 + 4.94 * 0.05 - 0.08 - 1.0,
            0.000306 / 0.005 + 1.082 * 0.05 - 0.0986 - 1.0,
            12.307 / 0.005 + 49_408.24 * 0.05 + 4051.02 - 50_000.0,
            2.098 / 0.005 + 8046.33 * 0.05 - 696.71 - 16_000.0,
            2.138 / 0.005 + 7883.39 * 0.05 - 705.04 - 10_000.0,
            0.417 * 0.005 + 1721.26 * 0.05 - 136.54 - 2000.0,
            0.164 / 0.005 + 631.13 * 0.05 - 54.48 - 550.0,
        ];
        assert_close(g.inequalities(), &expected, 1e-12);
        assert_eq!(violation, 0.0);
        // the smallest x₁x₂, 10⁻⁴, breaks g₁: 13.9 + 0.0494 − 0.08 − 1
        let g = WaterResourcePlanning.constraints(&at(&[0.01, 0.01, 0.01]));
        assert_close(&[g.inequalities()[0]], &[13.9 + 0.0494 - 1.08], 1e-12);
        // g₆ holds everywhere, and so would 0.417 / (x₁x₂) wherever g₇ holds
        let real = WaterResourcePlanning.representation();
        let mut rng = StreamRng::seed_from_u64(9);
        for _ in 0..100_000 {
            let x = real.random_genome(&mut rng);
            let g = WaterResourcePlanning.constraints(&x);
            assert!(g.inequalities()[5] < 0.0);
            if g.inequalities()[6] <= 0.0 {
                let quotient = 0.417 / (x[0] * x[1]) + 1721.26 * x[2] - 136.54 - 2000.0;
                assert!(quotient < 0.0, "{x:?}");
            }
        }
        // the extreme designs are feasible, and on the least area only where they must be
        let problem = WaterResourcePlanning;
        for best in [true, false] {
            for design in problem.extremes(best) {
                assert_eq!(problem.evaluate(&at(&design)).1, 0.0, "{design:?}");
            }
        }
        let area = WaterResourcePlanning::least_area();
        assert_close(&[area], &[0.00139 / 1.0306], 1e-15);
        let x1 = WaterResourcePlanning::least_x1(0.1);
        assert!(x1 * 0.1 >= area * (1.0 - 1e-15) && x1 * 0.1 <= area * (1.0 + 1e-14));
        let (ideal, nadir) = (
            problem.ideal_point().unwrap(),
            problem.nadir_point().unwrap(),
        );
        assert_close(
            &ideal,
            &[
                106_780.37 * 0.02 + 61_704.67,
                3000.0 * area / 0.1,
                305_700.0 * 2289.0 * 0.01 / (0.06f64 * 2289.0).powf(0.65),
                250.0 * 2289.0 * (-3.975f64 + 0.099 + 2.74).exp(),
                25.0 * (1.39 / 0.045 + 49.4 - 80.0),
            ],
            1e-12,
        );
        assert_close(
            &nadir,
            &[
                106_780.37 * 0.11 + 61_704.67,
                1350.0,
                305_700.0 * 2289.0 * 0.1 / (0.06f64 * 2289.0).powf(0.65),
                250.0 * 2289.0 * (-0.3975f64 + 0.099 + 2.74).exp(),
                25_000.0,
            ],
            1e-12,
        );

        // the front: at least the points asked for, feasible, between the ideal and nadir points
        // and reaching them, mutually non-dominated
        let front = problem.optimal_front(500).unwrap();
        assert!(front.len() >= 500);
        for j in 0..5 {
            let low = front.iter().map(|p| p[j]).fold(f64::INFINITY, f64::min);
            let high = front.iter().map(|p| p[j]).fold(f64::NEG_INFINITY, f64::max);
            assert_close(&[low, high], &[ideal[j], nadir[j]], 1e-12);
        }
        let scores: Vec<crate::multi::Scores<5>> = front
            .iter()
            .map(|p| crate::multi::Scores::new(*p))
            .collect();
        let objectives = [crate::Objective::Minimize; 5];
        assert_eq!(
            crate::multi::non_dominated_sort(&scores, &objectives).len(),
            1
        );
        // random feasible genomes dominate no point of the front
        let mut rng = StreamRng::seed_from_u64(11);
        for _ in 0..20_000 {
            let genome = real.random_genome(&mut rng);
            let (f, violation) = problem.evaluate(&genome);
            if violation > 0.0 {
                continue;
            }
            for point in front.iter().step_by(7) {
                let better = (0..5).all(|j| f[j] <= point[j]) && (0..5).any(|j| f[j] < point[j]);
                assert!(!better, "{genome:?} dominates {point:?}");
            }
        }
        check_random(&WaterResourcePlanning, 100_000);
    }
}
