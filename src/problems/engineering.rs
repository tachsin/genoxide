//! Engineering design problems with one objective: the welded beam, pressure vessel, spring,
//! speed reducer, gear train, three-bar truss, cantilever beam and car side impact.
//!
//! Every problem is minimized. A constrained problem's fitness is `(f(x), violation)`, the
//! violation being `Σ max(0, gᵢ(x))` over its constraints written as `gᵢ(x) ≤ 0`, 0 when it's
//! feasible. The constraints keep the scale of the paper that states them: a violation of the
//! pressure vessel's volume is in cubic inches, of its thickness in inches.
//!
//! Most originals are journal papers, books or theses that aren't openly available. Each
//! problem's docs name its original and the later paper that restates the definition used here;
//! those definitions are still to be checked against the originals
//! ([#168](https://github.com/tachsin/genoxide/issues/168)).
//!
//! **Discrete variables.** A problem whose variables are all integers ([`GearTrain`]) has an
//! [`Integer`] genome. A problem that mixes discrete and continuous variables ([`PressureVessel`],
//! [`SpeedReducer`]) has a [`Real`] genome whose discrete genes are rounded to their grid when the
//! genome is evaluated; its `design` method gives the rounded design variables.

use super::{Constraints, Optimum, Problem};
use crate::engine::FitnessFunction;
use crate::genome::{Integer, Integers, Real, Reals};
use crate::math;
use std::f64::consts::{PI, SQRT_2};

// bounds that are valid by construction
fn bounds(ranges: &[(f64, f64)]) -> Real {
    Real::new(ranges.iter().map(|&(low, high)| low..=high)).expect("valid bounds")
}

fn reals(values: &[f64]) -> Reals {
    Reals::from(values.to_vec())
}

// the violation of constraints g(x) <= 0, as `Constraints::violation` adds it up
fn violation(constraints: &[f64]) -> f64 {
    Constraints::new(constraints.to_vec(), Vec::new()).violation(0.0)
}

// a constrained problem's fitness, from its `value` and its constraint `values`
macro_rules! constrained {
    ($name:ident) => {
        impl FitnessFunction<Reals> for $name {
            type Output = (f64, f64);

            /// The value of `x` and its constraint violation, 0 when it's feasible.
            ///
            /// # Panics
            ///
            /// If `x` has fewer genes than the problem's variables.
            fn evaluate(&self, x: &Reals) -> (f64, f64) {
                (self.value(x), violation(&self.values(x)))
            }
        }
    };
}

// the constraint values of a constrained problem, as `Problem::constraints`
macro_rules! constraint_values {
    () => {
        fn constraints(&self, x: &Reals) -> Constraints {
            Constraints::new(self.values(x).to_vec(), Vec::new())
        }
    };
}

// ---- welded beam ---------------------------------------------------------------------------------

// the load P (lb), the overhang L (in) and the moduli E and G (psi) of both welded beams
const LOAD: f64 = 6000.0;
const LENGTH: f64 = 14.0;
const YOUNG: f64 = 30e6;
const SHEAR: f64 = 12e6;

/// The welded beam: the cheapest beam welded to a support that carries a load of 6000 lb at
/// 14 in, in the form with seven constraints.
///
/// The genes are the weld's thickness h and length l and the bar's height t and thickness b, in
/// inches: x = (h, l, t, b). The cost is `1.10471 h² l + 0.04811 t b (14 + l)`. The constraints
/// are the weld's shear stress τ ≤ 13,600 psi, the bar's bending stress σ = 6PL / (b t²) ≤ 30,000
/// psi, h ≤ b, `0.10471 h² + 0.04811 t b (14 + l) ≤ 5`, h ≥ 0.125, the end deflection
/// δ = 4PL³ / (E t³ b) ≤ 0.25 in, and the buckling load
/// `P_c = 4.013 E √(t² b⁶ / 36) / L² (1 − t / (2L) √(E / (4G))) ≥ P`, with
/// `τ = √(τ'² + 2τ'τ'' l / (2R) + τ''²)`, `τ' = P / (√2 h l)`, `τ'' = MR / J`, `M = P (L + l/2)`,
/// `R = √(l²/4 + ((h + t)/2)²)` and `J = 2 √2 h l (l²/12 + ((h + t)/2)²)`.
///
/// Bounds h, b ∈ [0.1, 2], l, t ∈ [0.1, 10]; best known 1.724852 at (0.205730, 3.470489,
/// 9.036624, 0.205729), from Cagnina, Esquivel and Coello Coello (2008, *Informatica* 32:
/// 319-326, appendix), whose solution, printed to 6 digits, exceeds the bending limit by 0.09 psi
/// and the buckling limit by 0.06 lb, and has h above b by 1e-6. Not proven optimal. [`WeldedBeamRagsdell`] is the other
/// form in the literature.
///
/// [`constraints`](Problem::constraints) gives g₁…g₇ in this order, as `g(x) ≤ 0`.
///
/// This form isn't [`WeldedBeamRagsdell`]'s mechanics with other limits: its J, `2√2 h l (…)`,
/// is twice the weld throat's, and its buckling load has E where Ragsdell and Phillips's has
/// √(EG), 1.58 times larger. It's the benchmark as published, so it stays as it is.
///
/// Rao, S. S. (1996). *Engineering Optimization.* Wiley, third edition. Definition and bounds as
/// restated in Coello Coello, C. A. (2000). Use of a self-adaptive penalty approach for
/// engineering optimization problems. *Computers in Industry* 41(2): 113-127 (eqs. 22-37 and
/// section 6.2), and the same in Coello Coello and Mezura-Montes (2002, *Advanced Engineering
/// Informatics* 16: 193-203, eqs. 8-23), after Rao; Cagnina, Esquivel and Coello Coello (2008)
/// attribute it to Ragsdell and Phillips. Not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WeldedBeam;

impl WeldedBeam {
    fn value(&self, x: &Reals) -> f64 {
        let (h, l, t, b) = (x[0], x[1], x[2], x[3]);
        1.104_71 * h * h * l + 0.048_11 * t * b * (14.0 + l)
    }

    fn values(&self, x: &Reals) -> [f64; 7] {
        let (h, l, t, b) = (x[0], x[1], x[2], x[3]);
        let shear_1 = LOAD / (SQRT_2 * h * l);
        let moment = LOAD * (LENGTH + l / 2.0);
        let radius = (l * l / 4.0 + ((h + t) / 2.0).powi(2)).sqrt();
        let polar = 2.0 * (SQRT_2 * h * l * (l * l / 12.0 + ((h + t) / 2.0).powi(2)));
        let shear_2 = moment * radius / polar;
        let shear =
            (shear_1 * shear_1 + 2.0 * shear_1 * shear_2 * l / (2.0 * radius) + shear_2 * shear_2)
                .sqrt();
        let bending = 6.0 * LOAD * LENGTH / (b * t * t);
        let deflection = 4.0 * LOAD * LENGTH.powi(3) / (YOUNG * t.powi(3) * b);
        let buckling = 4.013 * YOUNG * (t * t * b.powi(6) / 36.0).sqrt() / (LENGTH * LENGTH)
            * (1.0 - t / (2.0 * LENGTH) * (YOUNG / (4.0 * SHEAR)).sqrt());
        [
            shear - 13_600.0,
            bending - 30_000.0,
            h - b,
            0.104_71 * h * h + 0.048_11 * t * b * (14.0 + l) - 5.0,
            0.125 - h,
            deflection - 0.25,
            LOAD - buckling,
        ]
    }
}

constrained!(WeldedBeam);

impl Problem for WeldedBeam {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "WeldedBeam"
    }

    fn representation(&self) -> Real {
        bounds(&[(0.1, 2.0), (0.1, 10.0), (0.1, 10.0), (0.1, 2.0)])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::best_known(
            1.724_852,
            vec![reals(&[0.205_730, 3.470_489, 9.036_624, 0.205_729])],
        ))
    }

    fn reference(&self) -> &'static str {
        "Rao, S. S. (1996). Engineering Optimization. Wiley, third edition."
    }

    constraint_values!();
}

/// The welded beam in the form with five constraints, after Ragsdell and Phillips: the cheapest
/// beam welded to a support that carries a load of 6000 lb at 14 in.
///
/// The genes are x = (h, l, t, b), in inches, as in [`WeldedBeam`], with the same cost. The
/// constraints are τ ≤ 13,600 psi, σ = 504,000 / (t² b) ≤ 30,000 psi, h ≤ b, the buckling load
/// `P_c = 64,746.022 (1 − 0.0282346 t) t b³ ≥ 6000` and the deflection δ = 2.1952 / (t³ b) ≤ 0.25
/// in, with `τ = √(τ'² + τ''² + l τ'τ'' / √(0.25 (l² + (h + t)²)))`, `τ' = 6000 / (√2 h l)` and
/// `τ'' = 6000 (14 + 0.5 l) √(0.25 (l² + (h + t)²)) / (2 · 0.707 h l (l²/12 + 0.25 (h + t)²))`.
///
/// Bounds h ∈ [0.125, 10], l, t, b ∈ [0.1, 10]; best known 2.3811341 at (0.24436895, 6.2186069,
/// 8.2914718, 0.24436895), feasible, found with genoxide's SHADE and checked by several runs,
/// which all end within 3e-9 of it. Not proven optimal. Reklaitis, Ravindran and Ragsdell (1983)
/// report 2.38116 at (0.2444, 6.2187, 8.2915, 0.2444); printed to 4 digits, that solution
/// evaluates to 2.38151. Deb (1991) reports 2.43 at (0.2489, 6.1730, 8.1789, 0.2533).
/// Ragsdell and Phillips's own solution, (0.2455, 6.1960, 8.2730, 0.2455) at 2.3859 as tabulated
/// by Coello Coello and Mezura-Montes (2002), exceeds the shear stress limit by 0.31 psi in this
/// form. The constants follow from the mechanics: P = 6000 lb at L = 14 in, E = 30·10⁶ psi,
/// G = 12·10⁶ psi.
///
/// [`constraints`](Problem::constraints) gives the five constraints in this order, as `g(x) ≤ 0`.
///
/// Ragsdell, K. M. and Phillips, D. T. (1976). Optimal design of a class of welded structures
/// using geometric programming. *Journal of Engineering for Industry* 98(3): 1021-1025.
/// Definition, bounds and best known solution as restated in Deb, K. (2000). An efficient
/// constraint handling method for genetic algorithms. *Computer Methods in Applied Mechanics and
/// Engineering* 186(2-4): 311-338 (eq. 3, in the author's version), after Reklaitis, G. V.,
/// Ravindran, A. and Ragsdell, K. M. (1983), *Engineering Optimization: Methods and
/// Applications*, Wiley; not yet checked against the originals
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WeldedBeamRagsdell;

impl WeldedBeamRagsdell {
    fn value(&self, x: &Reals) -> f64 {
        WeldedBeam.value(x)
    }

    fn values(&self, x: &Reals) -> [f64; 5] {
        let (h, l, t, b) = (x[0], x[1], x[2], x[3]);
        let half_diagonal = (0.25 * (l * l + (h + t).powi(2))).sqrt();
        let shear_1 = 6000.0 / (SQRT_2 * h * l);
        let shear_2 = 6000.0 * (14.0 + 0.5 * l) * half_diagonal
            / (2.0 * (0.707 * h * l * (l * l / 12.0 + 0.25 * (h + t).powi(2))));
        let shear =
            (shear_1 * shear_1 + shear_2 * shear_2 + l * shear_1 * shear_2 / half_diagonal).sqrt();
        let bending = 504_000.0 / (t * t * b);
        let buckling = 64_746.022 * (1.0 - 0.028_234_6 * t) * t * b.powi(3);
        let deflection = 2.1952 / (t.powi(3) * b);
        [
            shear - 13_600.0,
            bending - 30_000.0,
            h - b,
            6000.0 - buckling,
            deflection - 0.25,
        ]
    }
}

constrained!(WeldedBeamRagsdell);

impl Problem for WeldedBeamRagsdell {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "WeldedBeamRagsdell"
    }

    fn representation(&self) -> Real {
        bounds(&[(0.125, 10.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0)])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::best_known(
            2.381_134_118_152_728_4,
            vec![reals(&[
                0.244_368_953_265_364_54,
                6.218_606_921_256_014,
                8.291_471_775_474_83,
                0.244_368_953_453_794_95,
            ])],
        ))
    }

    fn reference(&self) -> &'static str {
        "Ragsdell, K. M. and Phillips, D. T. (1976). Optimal design of a class of welded \
         structures using geometric programming. Journal of Engineering for Industry 98(3): \
         1021-1025."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1115/1.3438995")
    }

    constraint_values!();
}

// ---- pressure vessel -----------------------------------------------------------------------------

// the thickness of a steel plate, in inches: the shell's and the heads' are multiples of it
const PLATE: f64 = 0.0625;

// the least volume of the pressure vessel, in cubic inches
const VOLUME: f64 = 1_296_000.0;

/// The pressure vessel: the cheapest cylindrical vessel with hemispherical heads that holds
/// 1,296,000 cubic inches, a mixed discrete-continuous problem.
///
/// The design variables are the shell's thickness T_s, the heads' thickness T_h, the inner radius
/// R and the length L of the cylinder, in inches. The thicknesses are multiples of 0.0625 in: the
/// genome is real, and [`evaluate`](FitnessFunction::evaluate) rounds genes 0 and 1 to the nearest
/// multiple; [`design`](PressureVessel::design) gives the rounded design. The cost of material,
/// forming and welding is `0.6224 T_s R L + 1.7781 T_h R² + 3.1661 T_s² L + 19.84 T_s² R`, subject
/// to `T_s ≥ 0.0193 R`, `T_h ≥ 0.00954 R`, `π R² L + 4/3 π R³ ≥ 1,296,000` and `L ≤ 240`.
///
/// Bounds T_s, T_h ∈ [0.0625, 6.1875] (1 to 99 plates), R, L ∈ [10, 200]; minimum
/// 6059.714335048436 at (0.8125, 0.4375, 42.0984455958549, 176.6365958424394), proven global by
/// Yang, Huyck, Karamanoglu and Khan (2013, *International Journal of Bio-Inspired Computation*
/// 5(6): 329-335, eqs. 3 and 21): R is at the first constraint's limit and the volume is exactly
/// 1,296,000, so the solution's length is a few units in the last place longer than printed, to be
/// feasible despite rounding.
///
/// [`constraints`](Problem::constraints) gives g₁…g₄ of the design in this order, as `g(x) ≤ 0`.
///
/// Sandgren, E. (1990). Nonlinear integer and discrete programming in mechanical design
/// optimization. *Journal of Mechanical Design* 112(2): 223-229; in the notation of Kannan, B. K.
/// and Kramer, S. N. (1994). *Journal of Mechanical Design* 116(2): 405-411. Definition as
/// restated in Coello Coello (2000, *Computers in Industry* 41(2): 113-127, eqs. 17-21), and the
/// same in Coello Coello and Mezura-Montes (2002) and Cagnina, Esquivel and Coello Coello (2008);
/// not yet checked against the originals ([#168](https://github.com/tachsin/genoxide/issues/168)).
/// The originals' own designs, as tabulated by Coello Coello and Mezura-Montes, suggest that the
/// originals also had minimum thicknesses, T_s ≥ 1.1 and T_h ≥ 0.6: Sandgren's (1.125, 0.625,
/// 47.70, 117.70), at 8129.10, has both walls thicker than g₁ and g₂ require, and Kannan and
/// Kramer's (1.125, 0.625, 58.291, 43.690), at 7198.04, has a head of 0.625 where 0.5625 meets g₂
/// and costs about 378 less. With those minimums the least cost is 7198.006, at (1.125, 0.625,
/// 58.29015, 43.69268); the minimum of 6059.714 is of the form without them. Some restatements
/// print 3.1611 for 3.1661, which gives 7197.729 there.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PressureVessel;

impl PressureVessel {
    /// The design of `genome`: the thicknesses of the shell and the heads, rounded to the nearest
    /// multiple of 0.0625 in, the radius and the length.
    ///
    /// # Panics
    ///
    /// If `genome` has fewer than 4 genes.
    pub fn design(&self, genome: &Reals) -> [f64; 4] {
        let plates = |thickness: f64| (thickness / PLATE).round() * PLATE;
        [plates(genome[0]), plates(genome[1]), genome[2], genome[3]]
    }

    fn value(&self, x: &Reals) -> f64 {
        let [shell, head, radius, length] = self.design(x);
        0.6224 * shell * radius * length
            + 1.7781 * head * radius * radius
            + 3.1661 * shell * shell * length
            + 19.84 * shell * shell * radius
    }

    fn values(&self, x: &Reals) -> [f64; 4] {
        let [shell, head, radius, length] = self.design(x);
        [
            -shell + 0.0193 * radius,
            -head + 0.009_54 * radius,
            -PI * radius * radius * length - 4.0 / 3.0 * PI * radius.powi(3) + VOLUME,
            length - 240.0,
        ]
    }
}

constrained!(PressureVessel);

impl Problem for PressureVessel {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "PressureVessel"
    }

    fn representation(&self) -> Real {
        bounds(&[
            (PLATE, 99.0 * PLATE),
            (PLATE, 99.0 * PLATE),
            (10.0, 200.0),
            (10.0, 200.0),
        ])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(
            6_059.714_335_048_436,
            // R is at the first constraint's limit, and a longer vessel holds more
            vec![feasible_by_growing(
                reals(&[0.8125, 0.4375, 42.098_445_595_854_9, 176.636_595_842_439_4]),
                &[3],
                |x| self.values(x),
            )],
        ))
    }

    fn reference(&self) -> &'static str {
        "Sandgren, E. (1990). Nonlinear integer and discrete programming in mechanical design \
         optimization. Journal of Mechanical Design 112(2): 223-229."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1115/1.2912596")
    }

    constraint_values!();
}

// ---- tension/compression spring ------------------------------------------------------------------

/// The tension/compression spring: the lightest coil spring under constraints on its deflection,
/// shear stress, surge frequency and outer diameter.
///
/// The genes are the wire diameter d, the mean coil diameter D and the number of active coils N:
/// x = (d, D, N). The weight is `(N + 2) D d²`, subject to `1 − D³N / (71785 d⁴) ≤ 0`,
/// `(4D² − dD) / (12566 (D d³ − d⁴)) + 1 / (5108 d²) − 1 ≤ 0`, `1 − 140.45 d / (D² N) ≤ 0` and
/// `(D + d) / 1.5 − 1 ≤ 0`.
///
/// Bounds d ∈ [0.05, 2], D ∈ [0.25, 1.3], N ∈ [2, 15]; best known 0.012665 at (0.051690,
/// 0.356750, 11.287126), from Cagnina, Esquivel and Coello Coello (2008, *Informatica* 32:
/// 319-326, appendix), whose solution, printed to 6 digits, exceeds g₂ by 2e-5. Not proven
/// optimal. N is continuous, as in the restatement.
///
/// [`constraints`](Problem::constraints) gives g₁…g₄ in this order.
///
/// Belegundu, A. D. (1982). *A Study of Mathematical Programming Methods for Structural
/// Optimization.* PhD thesis, University of Iowa; and Arora, J. S. (1989). *Introduction to
/// Optimum Design.* McGraw-Hill. Definition and bounds as restated in Coello Coello (2000,
/// *Computers in Industry* 41(2): 113-127, eqs. 38-42 and section 6.3); not yet checked against
/// the originals ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TensionCompressionSpring;

impl TensionCompressionSpring {
    fn value(&self, x: &Reals) -> f64 {
        let (d, coil, n) = (x[0], x[1], x[2]);
        (n + 2.0) * coil * d * d
    }

    fn values(&self, x: &Reals) -> [f64; 4] {
        let (d, coil, n) = (x[0], x[1], x[2]);
        [
            1.0 - coil.powi(3) * n / (71_785.0 * d.powi(4)),
            (4.0 * coil * coil - d * coil) / (12_566.0 * (coil * d.powi(3) - d.powi(4)))
                + 1.0 / (5108.0 * d * d)
                - 1.0,
            1.0 - 140.45 * d / (coil * coil * n),
            (coil + d) / 1.5 - 1.0,
        ]
    }
}

constrained!(TensionCompressionSpring);

impl Problem for TensionCompressionSpring {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "TensionCompressionSpring"
    }

    fn representation(&self) -> Real {
        bounds(&[(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::best_known(
            0.012_665,
            vec![reals(&[0.051_690, 0.356_750, 11.287_126])],
        ))
    }

    fn reference(&self) -> &'static str {
        "Belegundu, A. D. (1982). A Study of Mathematical Programming Methods for Structural \
         Optimization. PhD thesis, University of Iowa."
    }

    constraint_values!();
}

// ---- speed reducer -------------------------------------------------------------------------------

/// Golinski's speed reducer: the lightest gearbox under constraints on the gear teeth's bending
/// and surface stresses, the shafts' deflections and stresses, and its proportions.
///
/// The genes are the face width x₁, the module of the teeth x₂, the number of teeth on the pinion
/// x₃ (an integer: [`evaluate`](FitnessFunction::evaluate) rounds gene 2 to the nearest integer,
/// and [`design`](SpeedReducer::design) gives the rounded design), the lengths of the shafts
/// between bearings x₄ and x₅, and the diameters of the shafts x₆ and x₇. The weight is
/// `0.7854 x₁x₂² (3.3333 x₃² + 14.9334 x₃ − 43.0934) − 1.508 x₁ (x₆² + x₇²) + 7.4777 (x₆³ + x₇³)
/// + 0.7854 (x₄x₆² + x₅x₇²)`, subject to eleven constraints `gᵢ ≤ 0`: `27 / (x₁x₂²x₃) − 1`,
/// `397.5 / (x₁x₂²x₃²) − 1`, `1.93 x₄³ / (x₂x₃x₆⁴) − 1`, `1.93 x₅³ / (x₂x₃x₇⁴) − 1`,
/// `√((745 x₄ / (x₂x₃))² + 16.9·10⁶) / (110 x₆³) − 1`,
/// `√((745 x₅ / (x₂x₃))² + 157.5·10⁶) / (85 x₇³) − 1`, `x₂x₃ / 40 − 1`, `5x₂ / x₁ − 1`,
/// `x₁ / (12x₂) − 1`, `(1.5 x₆ + 1.9) / x₄ − 1` and `(1.1 x₇ + 1.9) / x₅ − 1`.
///
/// Bounds x₁ ∈ [2.6, 3.6], x₂ ∈ [0.7, 0.8], x₃ ∈ [17, 28], x₄ ∈ [7.3, 8.3], x₅ ∈ [7.8, 8.3],
/// x₆ ∈ [2.9, 3.9], x₇ ∈ [5.0, 5.5]; best known 2996.348165 at (3.5, 0.7, 17, 7.3, 7.8, 3.350214,
/// 5.286683), from the restatement, whose solution, printed to 6 digits, exceeds g₅ by 6.0e-7 and
/// g₆ by 1.3e-7, and evaluates to 2996.347849. Not proven optimal. The literature's formulations
/// differ (Ray, 2003, AIAA Journal 41(3): 556-558): some print 7.477 for 7.4777 and 1.5079 for
/// 1.508, which don't give this value, and some let x₅ go down to 7.3, where the minimum is
/// 2994.471 (Lin, Tsai, Hu and Chang, 2013, *Mathematical Problems in Engineering* 419043).
///
/// [`constraints`](Problem::constraints) gives g₁…g₁₁ of the design in this order.
///
/// Golinski, J. (1970). Optimal synthesis problems solved by means of nonlinear programming and
/// random methods. *Journal of Mechanisms* 5(3): 287-309, where the problem is first posed; and
/// Golinski, J. (1973). An adaptive optimization system applied to machine synthesis. *Mechanism
/// and Machine Theory* 8(4): 419-436. Definition, bounds and best known solution as restated in
/// Cagnina, L. C., Esquivel, S. C. and Coello Coello, C. A. (2008). Solving engineering
/// optimization problems with the simple constrained particle swarm optimizer. *Informatica* 32:
/// 319-326 (appendix, problem E03); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SpeedReducer;

impl SpeedReducer {
    /// The design of `genome`: its genes, with the number of teeth x₃ rounded to the nearest
    /// integer.
    ///
    /// # Panics
    ///
    /// If `genome` has fewer than 7 genes.
    pub fn design(&self, genome: &Reals) -> [f64; 7] {
        [
            genome[0],
            genome[1],
            genome[2].round(),
            genome[3],
            genome[4],
            genome[5],
            genome[6],
        ]
    }

    // 0.7854 is the paper's constant, which is close to π/4
    #[allow(clippy::approx_constant)]
    fn value(&self, x: &Reals) -> f64 {
        let [x1, x2, x3, x4, x5, x6, x7] = self.design(x);
        0.7854 * x1 * x2 * x2 * (3.3333 * x3 * x3 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6 * x6 + x7 * x7)
            + 7.4777 * (x6.powi(3) + x7.powi(3))
            + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)
    }

    fn values(&self, x: &Reals) -> [f64; 11] {
        let [x1, x2, x3, x4, x5, x6, x7] = self.design(x);
        [
            27.0 / (x1 * x2 * x2 * x3) - 1.0,
            397.5 / (x1 * x2 * x2 * x3 * x3) - 1.0,
            1.93 * x4.powi(3) / (x2 * x3 * x6.powi(4)) - 1.0,
            1.93 * x5.powi(3) / (x2 * x3 * x7.powi(4)) - 1.0,
            ((745.0 * x4 / (x2 * x3)).powi(2) + 16.9e6).sqrt() / (110.0 * x6.powi(3)) - 1.0,
            ((745.0 * x5 / (x2 * x3)).powi(2) + 157.5e6).sqrt() / (85.0 * x7.powi(3)) - 1.0,
            x2 * x3 / 40.0 - 1.0,
            5.0 * x2 / x1 - 1.0,
            x1 / (12.0 * x2) - 1.0,
            (1.5 * x6 + 1.9) / x4 - 1.0,
            (1.1 * x7 + 1.9) / x5 - 1.0,
        ]
    }
}

constrained!(SpeedReducer);

impl Problem for SpeedReducer {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "SpeedReducer"
    }

    fn representation(&self) -> Real {
        bounds(&[
            (2.6, 3.6),
            (0.7, 0.8),
            (17.0, 28.0),
            (7.3, 8.3),
            (7.8, 8.3),
            (2.9, 3.9),
            (5.0, 5.5),
        ])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::best_known(
            2_996.348_165,
            vec![reals(&[3.5, 0.7, 17.0, 7.3, 7.8, 3.350_214, 5.286_683])],
        ))
    }

    fn reference(&self) -> &'static str {
        "Golinski, J. (1973). An adaptive optimization system applied to machine synthesis. \
         Mechanism and Machine Theory 8(4): 419-436."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1016/0094-114X(73)90018-9")
    }

    constraint_values!();
}

// ---- gear train ----------------------------------------------------------------------------------

// the gear ratio to reach
const GEAR_RATIO: f64 = 1.0 / 6.931;

/// The gear train: the numbers of teeth of a compound gear train of four gears whose ratio is
/// closest to 1/6.931, an integer problem.
///
/// The genes are the numbers of teeth x = (T_d, T_b, T_a, T_f), integers from 12 to 60, and the
/// score is the squared error of the ratio, `(1/6.931 − T_d T_b / (T_a T_f))²`, with no
/// constraints.
///
/// Bounds [12, 60]⁴; minimum 2.7008571488865134e-12 at T_d T_b = 16 · 19 and T_a T_f = 43 · 49,
/// four genomes, as Deb and Goyal report (2.701e-12), and checked by evaluating all 49⁴ genomes.
///
/// Sandgren, E. (1990). Nonlinear integer and discrete programming in mechanical design
/// optimization. *Journal of Mechanical Design* 112(2): 223-229. Definition and bounds as
/// restated in Deb, K. and Goyal, M. (1996). A combined genetic adaptive search (GeneAS) for
/// engineering design. *Computer Science and Informatics* 26(4): 30-45 (section 3.1, in the
/// authors' version); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GearTrain;

impl FitnessFunction<Integers> for GearTrain {
    type Output = f64;

    /// The squared error of the gear ratio of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 4 genes.
    fn evaluate(&self, x: &Integers) -> f64 {
        let ratio = (x[0] * x[1]) as f64 / (x[2] * x[3]) as f64;
        (GEAR_RATIO - ratio).powi(2)
    }
}

impl Problem for GearTrain {
    type Representation = Integer;

    fn name(&self) -> &'static str {
        "GearTrain"
    }

    fn representation(&self) -> Integer {
        Integer::uniform(4, 12..=60).expect("valid bounds")
    }

    fn optimum(&self) -> Option<Optimum<Integers>> {
        let solution = |x: [i64; 4]| Integers::from(x.to_vec());
        Some(Optimum::proven(
            (GEAR_RATIO - 304.0 / 2107.0).powi(2),
            vec![
                solution([16, 19, 43, 49]),
                solution([16, 19, 49, 43]),
                solution([19, 16, 43, 49]),
                solution([19, 16, 49, 43]),
            ],
        ))
    }

    fn reference(&self) -> &'static str {
        "Sandgren, E. (1990). Nonlinear integer and discrete programming in mechanical design \
         optimization. Journal of Mechanical Design 112(2): 223-229."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1115/1.2912596")
    }
}

// ---- three-bar truss -----------------------------------------------------------------------------

// the length of the bars (cm), the load and the allowed stress (kN/cm²)
// A solution with an active constraint can evaluate a rounding error above it (1e-16, or 1e-9 on
// the pressure vessel's volume of 1,296,000) and so be infeasible, beaten under Deb's rules by any
// feasible point. When the constraints only loosen as the genes `grow` get larger, this scales
// them up by a unit in the last place at a time until no constraint is above 0; the value
// changes by as little.
fn feasible_by_growing<const N: usize>(
    mut solution: Reals,
    grow: &[usize],
    constraints: impl Fn(&Reals) -> [f64; N],
) -> Reals {
    for _ in 0..256 {
        if constraints(&solution).iter().all(|g| *g <= 0.0) {
            break;
        }
        solution = solution
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if grow.contains(&i) {
                    x * (1.0 + f64::EPSILON)
                } else {
                    *x
                }
            })
            .collect();
    }
    solution
}

const TRUSS_LENGTH: f64 = 100.0;
const TRUSS_LOAD: f64 = 2.0;
const TRUSS_STRESS: f64 = 2.0;

/// The three-bar truss: the least volume of a planar truss of three bars under a load, subject to
/// the bars' stresses.
///
/// The genes are the cross-sections x₁ (of the two outer bars) and x₂ (of the middle one), in
/// cm². The volume is `(2√2 x₁ + x₂) l`, subject to `(√2 x₁ + x₂) / (√2 x₁² + 2 x₁x₂) P ≤ σ`,
/// `x₂ / (√2 x₁² + 2 x₁x₂) P ≤ σ` and `P / (x₁ + √2 x₂) ≤ σ`, with l = 100 cm and
/// P = σ = 2 kN/cm². The stresses are undefined where a denominator is 0, at x₁ = 0: the fitness
/// there is invalid or its violation infinite.
///
/// Bounds [0, 1]²; minimum 100 (√2 + √6/2) ≈ 263.8958434 at ((3 + √3)/6, 1/√6), with the first
/// constraint active, derived from the definition: with g₁ active, x₂ = √2 x₁ (1 − x₁) / (2x₁ − 1)
/// and the volume is √2 l (3u/4 + 1 + 1/(4u)) with u = 2x₁ − 1, smallest at u = 1/√3.
///
/// [`constraints`](Problem::constraints) gives g₁…g₃ in this order.
///
/// Nowacki, H. (1974). Optimization in pre-contract ship design. In *Computer Applications in the
/// Automation of Shipyard Operation and Ship Design*, North-Holland: 327-338; and Ray, T. and
/// Saini, P. (2001). Engineering design optimization using a swarm with an intelligent
/// information sharing among individuals. *Engineering Optimization* 33(6): 735-748. Definition
/// and bounds as restated in Yang, X.-S. and Gandomi, A. H. (2012). Bat algorithm: a novel
/// approach for global engineering optimization. *Engineering Computations* 29(5): 464-483
/// (eqs. 14-17, arXiv:1211.6663); not yet checked against the originals
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ThreeBarTruss;

impl ThreeBarTruss {
    fn value(&self, x: &Reals) -> f64 {
        (2.0 * SQRT_2 * x[0] + x[1]) * TRUSS_LENGTH
    }

    fn values(&self, x: &Reals) -> [f64; 3] {
        let (x1, x2) = (x[0], x[1]);
        let denominator = SQRT_2 * x1 * x1 + 2.0 * x1 * x2;
        [
            (SQRT_2 * x1 + x2) / denominator * TRUSS_LOAD - TRUSS_STRESS,
            x2 / denominator * TRUSS_LOAD - TRUSS_STRESS,
            1.0 / (x1 + SQRT_2 * x2) * TRUSS_LOAD - TRUSS_STRESS,
        ]
    }
}

constrained!(ThreeBarTruss);

impl Problem for ThreeBarTruss {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "ThreeBarTruss"
    }

    fn representation(&self) -> Real {
        bounds(&[(0.0, 1.0), (0.0, 1.0)])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        let sqrt_3 = 3f64.sqrt();
        let solution = reals(&[(3.0 + sqrt_3) / 6.0, 1.0 / 6f64.sqrt()]);
        Some(Optimum::proven(
            TRUSS_LENGTH * (SQRT_2 + 6f64.sqrt() / 2.0),
            // larger cross-sections only lower the stresses
            vec![feasible_by_growing(solution, &[0, 1], |x| self.values(x))],
        ))
    }

    fn reference(&self) -> &'static str {
        "Nowacki, H. (1974). Optimization in pre-contract ship design. In Computer Applications \
         in the Automation of Shipyard Operation and Ship Design, North-Holland: 327-338."
    }

    constraint_values!();
}

// ---- cantilever beam -----------------------------------------------------------------------------

// the coefficients of the cantilever beam's constraint, from the fixed end to the free one
const CANTILEVER: [f64; 5] = [61.0, 37.0, 19.0, 7.0, 1.0];

/// The cantilever beam: the lightest beam of five hollow square segments with a load at its free
/// end, subject to its deflection.
///
/// The genes are the widths of the five segments, and the weight is `0.0624 Σ xᵢ`, subject to
/// `61/x₁³ + 37/x₂³ + 19/x₃³ + 7/x₄³ + 1/x₅³ ≤ 1`.
///
/// Bounds [0.01, 100]⁵; minimum 0.0624 s^(4/3) ≈ 1.339956361 at xᵢ = s^(1/3) aᵢ^(1/4), where
/// aᵢ are the constraint's coefficients and s = Σ aᵢ^(1/4): a convex problem, whose minimum Yang,
/// Huyck, Karamanoglu and Khan (2013) derive from the Lagrange conditions, 1.339956367 at
/// (6.0160159, 5.3091739, 4.4943296, 3.5014750, 2.15266533) (their eqs. 50-51), and here in
/// closed form.
///
/// [`constraints`](Problem::constraints) gives the one constraint, as `g(x) ≤ 0`.
///
/// Fleury, C. and Braibant, V. (1986). Structural optimization: a new dual method using mixed
/// variables. *International Journal for Numerical Methods in Engineering* 23(3): 409-428.
/// Definition and bounds as restated in Yang, X.-S., Huyck, C., Karamanoglu, M. and Khan, N.
/// (2013). True global optimality of the pressure vessel design problem: a benchmark for
/// bio-inspired optimisation algorithms. *International Journal of Bio-Inspired Computation* 5(6):
/// 329-335 (eqs. 35-37, arXiv:1403.7793); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CantileverBeam;

impl CantileverBeam {
    fn value(&self, x: &Reals) -> f64 {
        0.0624 * x[..5].iter().sum::<f64>()
    }

    fn values(&self, x: &Reals) -> [f64; 1] {
        let deflection: f64 = CANTILEVER
            .iter()
            .zip(&x[..5])
            .map(|(a, xi)| a / xi.powi(3))
            .sum();
        [deflection - 1.0]
    }
}

constrained!(CantileverBeam);

impl Problem for CantileverBeam {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "CantileverBeam"
    }

    fn representation(&self) -> Real {
        bounds(&[(0.01, 100.0); 5])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        let s: f64 = CANTILEVER.iter().map(|a| math::powf(*a, 0.25)).sum();
        let solution = CANTILEVER
            .iter()
            .map(|a| math::cbrt(s) * math::powf(*a, 0.25))
            .collect();
        Some(Optimum::proven(
            0.0624 * math::powf(s, 4.0 / 3.0),
            // wider segments only lower the deflection
            vec![feasible_by_growing(solution, &[0, 1, 2, 3, 4], |x| {
                self.values(x)
            })],
        ))
    }

    fn reference(&self) -> &'static str {
        "Fleury, C. and Braibant, V. (1986). Structural optimization: a new dual method using \
         mixed variables. International Journal for Numerical Methods in Engineering 23(3): \
         409-428."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1002/nme.1620230307")
    }

    constraint_values!();
}

// ---- car side impact -----------------------------------------------------------------------------

/// The car side impact: the lightest car body whose side withstands the European side-impact
/// test, through response surfaces fitted to crash simulations.
///
/// The genes are the thicknesses of the B-pillar inner and reinforcement, the floor side inner,
/// the cross members, the door beam, the door beltline reinforcement and the roof rail. The
/// weight is `1.98 + 4.9x₁ + 6.67x₂ + 6.98x₃ + 4.01x₄ + 1.78x₅ + 0.00001x₆ + 2.73x₇`, subject to
/// ten constraints on the abdomen load (≤ 1 kN), the dummy's upper, middle and lower chest
/// velocities (≤ 0.32 m/s), the upper, middle and lower rib deflections (≤ 32 mm), the pubic
/// force (≤ 4 kN), and the velocities of the B-pillar's middle point (≤ 9.9 mm/ms) and of the
/// front door (≤ 15.7 mm/ms), each a response surface as printed in the restatement's appendix.
///
/// Bounds x₁, x₃, x₄ ∈ [0.5, 1.5], x₂ ∈ [0.45, 1.35], x₅ ∈ [0.875, 2.625], x₆, x₇ ∈ [0.4, 1.2];
/// the restatement gives no optimum. Best known 23.585657980780084 at (0.5, 1.225732, 0.5,
/// 1.207111, 0.875, 0.884189, 0.4): x₁, x₃, x₅ and x₇ on their lower bounds, and the lower rib
/// deflection, the pubic force and the front door's velocity on their limits, which give x₂, x₄
/// and x₆ in turn. Stored to the last bit: each of x₂, x₄ and x₆ is the smallest `f64` whose
/// constraint holds when evaluated, so the violation is exactly 0. SLSQP from 2,000 random
/// starting points, and from the design of genoxide's L-SHADE, finds no other minimum, and the
/// seven active constraints' Lagrange multipliers there are positive: a strict local minimum.
/// The constraints aren't convex, so it isn't proven optimal. The weight is the first objective
/// of the restatement's three-objective problem; the original has four more variables (two
/// materials, the barrier's height and hitting position), fixed in these surfaces.
///
/// [`constraints`](Problem::constraints) gives g₁…g₁₀ in this order, each as the response
/// minus its limit.
///
/// Gu, L., Yang, R. J., Tho, C. H., Makowski, M., Faruque, O. and Li, Y. (2001). Optimisation and
/// robustness for crashworthiness of side impact. *International Journal of Vehicle Design*
/// 26(4): 348-360. Definition and bounds as restated in Jain, H. and Deb, K. (2014). An
/// evolutionary many-objective optimization algorithm using reference-point based nondominated
/// sorting approach, part II. *IEEE Transactions on Evolutionary Computation* 18(4): 602-622
/// (appendix, in the authors' version); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)). The abdomen load's `0.0092928 x₃`
/// is as published there and in the implementations that follow it; the eleven-variable form's
/// `0.484 x₃ x₉`, with x₉ fixed at 0.192, would give 0.092928, ten times as much. Likewise the
/// lower chest's `0.031296 x₃` is `0.163 x₃` times x₉'s 0.192, where the eleven-variable
/// restatements have x₈'s 0.345. Neither constraint is active at the best known design, so
/// neither changes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CarSideImpact;

impl CarSideImpact {
    fn value(&self, x: &Reals) -> f64 {
        let (x1, x2, x3, x4, x5, x6, x7) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
        1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.000_01 * x6 + 2.73 * x7
    }

    fn values(&self, x: &Reals) -> [f64; 10] {
        let (x1, x2, x3, x4, x5, x6, x7) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
        let abdomen = 1.16 - 0.3717 * x2 * x4 - 0.009_292_8 * x3;
        let upper_chest = 0.261 - 0.0159 * x1 * x2 - 0.064_86 * x1 - 0.019 * x2 * x7
            + 0.0144 * x3 * x5
            + 0.015_446_4 * x6;
        let middle_chest = 0.214 + 0.008_17 * x5 - 0.045_195 * x1 - 0.013_516_8 * x1
            + 0.030_99 * x2 * x6
            - 0.018 * x2 * x7
            + 0.007_176 * x3
            + 0.023_232 * x3
            - 0.003_64 * x5 * x6
            - 0.018 * x2 * x2;
        let lower_chest = 0.74 - 0.61 * x2 - 0.031_296 * x3 - 0.031_872 * x7 + 0.227 * x2 * x2;
        let upper_rib = 28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.272_96 * x6 - 2.680_65 * x7;
        let middle_rib = 33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.457_28;
        let lower_rib = 46.36 - 9.9 * x2 - 4.4505 * x1;
        let pubic = 4.72 - 0.5 * x4 - 0.19 * x2 * x3;
        let pillar = 10.58 - 0.674 * x1 * x2 - 0.672_75 * x2;
        let door = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6;
        [
            abdomen - 1.0,
            upper_chest - 0.32,
            middle_chest - 0.32,
            lower_chest - 0.32,
            upper_rib - 32.0,
            middle_rib - 32.0,
            lower_rib - 32.0,
            pubic - 4.0,
            pillar - 9.9,
            door - 15.7,
        ]
    }
}

constrained!(CarSideImpact);

impl Problem for CarSideImpact {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "CarSideImpact"
    }

    fn representation(&self) -> Real {
        bounds(&[
            (0.5, 1.5),
            (0.45, 1.35),
            (0.5, 1.5),
            (0.5, 1.5),
            (0.875, 2.625),
            (0.4, 1.2),
            (0.4, 1.2),
        ])
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::best_known(
            23.585_657_980_780_084,
            vec![reals(&[
                0.5,
                1.225_732_323_232_322_5,
                0.5,
                1.207_110_858_585_857_6,
                0.875,
                0.884_189_120_488_052_4,
                0.4,
            ])],
        ))
    }

    fn reference(&self) -> &'static str {
        "Gu, L., Yang, R. J., Tho, C. H., Makowski, M., Faruque, O. and Li, Y. (2001). \
         Optimisation and robustness for crashworthiness of side impact. International Journal \
         of Vehicle Design 26(4): 348-360."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1504/IJVD.2001.005210")
    }

    constraint_values!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Representation;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1e-300),
            "{actual} is not {expected}"
        );
    }

    // each solution of the optimum is in the bounds and has the optimum's value to `tolerance`
    // (relative); returns the violations
    fn check_optimum<P>(problem: &P, tolerance: f64) -> Vec<f64>
    where
        P: Problem<Representation = Real, Output = (f64, f64)>,
    {
        let optimum = problem.optimum().expect("known");
        optimum
            .solutions()
            .iter()
            .map(|solution| {
                problem
                    .representation()
                    .validate(solution)
                    .expect("in bounds");
                let (value, violation) = problem.evaluate(solution);
                assert_close(value, optimum.value(), tolerance);
                violation
            })
            .collect()
    }

    #[test]
    fn welded_beam() {
        // Cagnina et al.'s solution gives their 1.724852 to its digits, exceeding σ's limit by
        // 0.09 psi and P_c's by 0.06 lb
        let violations = check_optimum(&WeldedBeam, 3e-6);
        assert!(violations[0] < 0.2, "{violations:?}");
        // Coello's own solution and the constraint values of his table 3: f = 1.74830941,
        // g₁ = −0.337812, g₂ = −353.902604, g₇ = −363.232384
        let x = reals(&[0.2088, 3.4205, 8.9975, 0.2100]);
        let (value, violation) = WeldedBeam.evaluate(&x);
        assert_close(value, 1.748_309_41, 1e-8);
        assert_eq!(violation, 0.0);
        let g = WeldedBeam.constraints(&x);
        assert_close(g.inequalities()[0], -0.337_812, 1e-5);
        assert_close(g.inequalities()[1], -353.902_604, 1e-8);
        assert_close(g.inequalities()[6], -363.232_384, 1e-8);
        // the cost at h = l = t = b = 1: 1.10471 + 0.04811 · 15
        let (value, _) = WeldedBeam.evaluate(&reals(&[1.0, 1.0, 1.0, 1.0]));
        assert_close(value, 1.104_71 + 0.048_11 * 15.0, 1e-15);
        assert_eq!(WeldedBeam.constraints(&x).len(), 7);
    }

    #[test]
    fn welded_beam_ragsdell() {
        // the printed solution evaluates to 2.38151, 1.5e-4 above the printed value
        let violations = check_optimum(&WeldedBeamRagsdell, 2e-4);
        assert_eq!(violations, [0.0]);
        // Deb's (1991) solution, 2.433116 in Coello's table 3, to its digits
        let (value, violation) =
            WeldedBeamRagsdell.evaluate(&reals(&[0.2489, 6.1730, 8.1789, 0.2533]));
        assert_close(value, 2.433_116, 1e-7);
        assert_eq!(violation, 0.0);
        // at h = b = 1, t = 2: σ = 504,000 / 4 and P_c = 64,746.022 (1 − 0.0564692) 2
        let g = WeldedBeamRagsdell.constraints(&reals(&[1.0, 1.0, 2.0, 1.0]));
        assert_close(g.inequalities()[1], 126_000.0 - 30_000.0, 1e-15);
        assert_close(
            g.inequalities()[3],
            6000.0 - 64_746.022 * (1.0 - 0.056_469_2) * 2.0,
            1e-14,
        );
        assert_eq!(g.inequalities()[2], 0.0);
        assert_close(g.inequalities()[4], 2.1952 / 8.0 - 0.25, 1e-14);
    }

    #[test]
    fn pressure_vessel() {
        // the volume constraint is met to rounding, 1e-9 in^3 of 1,296,000
        let violations = check_optimum(&PressureVessel, 1e-15);
        assert!(violations[0] < 1e-8, "{violations:?}");
        // Yang et al.'s other end of the boundary, r = 40.31961872409872 at L = 200, 6288.677
        let (value, _) =
            PressureVessel.evaluate(&reals(&[0.8125, 0.4375, 40.319_618_724_098_72, 200.0]));
        assert_close(value, 6_288.677_045_653_44, 1e-10);
        // the genes are rounded to whole plates: 0.83 is 13 plates, 0.43 is 7
        assert_eq!(
            PressureVessel.design(&reals(&[0.83, 0.43, 50.0, 100.0])),
            [0.8125, 0.4375, 50.0, 100.0]
        );
        // the constraint values of 13 plates at R = 40: −0.8125 + 0.0193 · 40, and L − 240
        let g = PressureVessel.constraints(&reals(&[0.8125, 0.4375, 40.0, 200.0]));
        assert_close(g.inequalities()[0], -0.8125 + 0.772, 1e-14);
        assert_eq!(g.inequalities()[3], -40.0);
    }

    // the minimum cost over the radius and length for given thicknesses: the volume constraint
    // is active at the least cost, so L is a function of R, and a fine grid of R is searched
    fn least_cost(shell: f64, head: f64) -> f64 {
        let limit = (shell / 0.0193).min(head / 0.009_54).min(200.0);
        let mut best = f64::INFINITY;
        let steps = 500;
        for step in 0..=steps {
            let radius = 10.0 + (limit - 10.0) * step as f64 / steps as f64;
            let length = (VOLUME / (PI * radius * radius) - 4.0 / 3.0 * radius).max(10.0);
            if length > 200.0 || limit < 10.0 {
                continue;
            }
            let genome = reals(&[shell, head, radius, length]);
            let (value, violation) = PressureVessel.evaluate(&genome);
            if violation < 1e-6 {
                best = best.min(value);
            }
        }
        best
    }

    // no pair of thicknesses beats the optimum, on a grid of R for each: a check of Yang et al.'s
    // argument
    #[test]
    fn pressure_vessel_thicknesses() {
        let optimum = PressureVessel.optimum().expect("known").value();
        for shell in 1..=99 {
            for head in 1..=99 {
                let cost = least_cost(shell as f64 * PLATE, head as f64 * PLATE);
                assert!(cost >= optimum - 1e-6, "{shell}, {head}: {cost}");
            }
        }
    }

    #[test]
    fn tension_compression_spring() {
        let violations = check_optimum(&TensionCompressionSpring, 1e-5);
        assert!(violations[0] < 3e-5, "{violations:?}");
        // Coello's solution and its values (table 4): 0.0127047834, g₂ = −0.000110,
        // g₃ = −4.026318; his g₄ repeats g₃, a misprint: it's −0.731239
        let x = reals(&[0.051_480, 0.351_661, 11.632_201]);
        let (value, violation) = TensionCompressionSpring.evaluate(&x);
        assert_close(value, 0.012_704_783_4, 1e-8);
        assert_eq!(violation, 0.0);
        let g = TensionCompressionSpring.constraints(&x);
        assert_close(g.inequalities()[1], -0.000_110, 1e-2);
        assert_close(g.inequalities()[2], -4.026_318, 1e-6);
        assert_close(g.inequalities()[3], -0.731_239_333_333_333_4, 1e-12);
        // (N + 2) D d² at d = 0.1, D = 1, N = 8
        assert_close(
            TensionCompressionSpring
                .evaluate(&reals(&[0.1, 1.0, 8.0]))
                .0,
            0.1,
            1e-14,
        );
    }

    #[test]
    fn speed_reducer() {
        // the best known solution gives 2996.348165 to 1e-7, with g₅ and g₆ at their limits to
        // the printed digits
        let violations = check_optimum(&SpeedReducer, 2e-7);
        assert!(violations[0] < 1e-6, "{violations:?}");
        let x = SpeedReducer.optimum().expect("known").solutions()[0].clone();
        let g = SpeedReducer.constraints(&x);
        // the constraint values Cagnina et al. print (table 5)
        for (i, expected) in [
            (0, -0.073_915),
            (1, -0.197_998),
            (2, -0.499_172),
            (3, -0.901_471),
            (6, -0.702_500),
            (8, -0.583_333),
            (9, -0.051_325),
            (10, -0.010_852),
        ] {
            assert!((g.inequalities()[i] - expected).abs() < 1e-6, "g{}", i + 1);
        }
        // g₈ = 5 x₂ / x₁ − 1 is active
        assert!(g.inequalities()[7].abs() < 1e-15);
        // the number of teeth is rounded
        let design = SpeedReducer.design(&reals(&[3.0, 0.75, 20.4, 8.0, 8.0, 3.0, 5.0]));
        assert_eq!(design[2], 20.0);
    }

    #[test]
    fn gear_train() {
        let optimum = GearTrain.optimum().expect("known");
        for solution in optimum.solutions() {
            GearTrain
                .representation()
                .validate(solution)
                .expect("in bounds");
            assert_eq!(GearTrain.evaluate(solution), optimum.value());
        }
        // Deb and Goyal's 2.701e-12, to its digits
        assert_eq!((optimum.value() * 1e15).round(), 2701.0);
        // their GeneAS solution (17, 14, 33, 50): 1.362e-9
        let value = GearTrain.evaluate(&Integers::from(vec![17, 14, 33, 50]));
        assert_eq!((value * 1e12).round(), 1362.0);
        // a ratio of 1: (1/6.931 − 1)²
        let one = GearTrain.evaluate(&Integers::from(vec![12, 12, 12, 12]));
        assert_close(one, (1.0 / 6.931 - 1.0f64).powi(2), 1e-15);
    }

    // every one of the 49⁴ genomes, by the products of their pairs: none beats the optimum, and
    // only its four genomes reach it
    #[test]
    fn gear_train_enumeration() {
        let optimum = GearTrain.optimum().expect("known").value();
        let mut products: Vec<(i64, usize)> = Vec::new();
        for a in 12..=60i64 {
            for b in 12..=60i64 {
                products.push((a * b, 1));
            }
        }
        let mut best = f64::INFINITY;
        let mut count = 0;
        for &(top, _) in &products {
            for &(bottom, _) in &products {
                let value = (GEAR_RATIO - top as f64 / bottom as f64).powi(2);
                if value < best {
                    best = value;
                    count = 0;
                }
                if value == best {
                    count += 1;
                }
            }
        }
        assert_eq!(best, optimum);
        assert_eq!(count, 4);
    }

    #[test]
    fn three_bar_truss() {
        let violations = check_optimum(&ThreeBarTruss, 1e-15);
        assert!(violations[0] < 1e-14, "{violations:?}");
        // Yang and Gandomi's solution: 263.896248 to its digits, and feasible
        let (value, violation) = ThreeBarTruss.evaluate(&reals(&[0.788_63, 0.408_38]));
        assert_close(value, 263.896_248, 1e-8);
        assert_eq!(violation, 0.0);
        // at (1, 1): volume 100 (2√2 + 1); the first stress is (√2 + 1) / (√2 + 2) · 2
        let (value, violation) = ThreeBarTruss.evaluate(&reals(&[1.0, 1.0]));
        assert_close(value, 100.0 * (2.0 * SQRT_2 + 1.0), 1e-15);
        assert_eq!(violation, 0.0);
        // the volume with the first constraint active is smallest at u = 1/√3: its derivative
        // 3/4 − 1/(4u²) is 0 there
        let u = 1.0 / 3f64.sqrt();
        assert!((0.75 - 1.0 / (4.0 * u * u)).abs() < 1e-15);
    }

    #[test]
    fn cantilever_beam() {
        let violations = check_optimum(&CantileverBeam, 1e-15);
        assert!(violations[0] < 1e-14, "{violations:?}");
        // Yang et al.'s minimum and solution, to their digits
        let optimum = CantileverBeam.optimum().expect("known");
        assert_close(optimum.value(), 1.339_956_367, 1e-8);
        for (xi, expected) in optimum.solutions()[0].iter().zip([
            6.016_015_9,
            5.309_173_9,
            4.494_329_6,
            3.501_475_0,
            2.152_665_33,
        ]) {
            assert!((xi - expected).abs() < 1e-7, "{xi} is not {expected}");
        }
        // the Lagrange conditions: aᵢ / xᵢ⁴ is the same for every segment
        let x = &optimum.solutions()[0];
        let ratios: Vec<f64> = CANTILEVER
            .iter()
            .zip(x.iter())
            .map(|(a, xi)| a / xi.powi(4))
            .collect();
        assert!(ratios.iter().all(|r| (r - ratios[0]).abs() < 1e-15));
        // at xᵢ = 5: 0.0624 · 25, and 125 / 125 − 1 = 0
        assert_eq!(
            CantileverBeam.evaluate(&reals(&[5.0; 5])),
            (0.0624 * 25.0, 0.0)
        );
    }

    #[test]
    fn car_side_impact() {
        let optimum = CarSideImpact.optimum().expect("a best known design");
        let best = &optimum.solutions()[0];
        assert_eq!(CarSideImpact.evaluate(best), (optimum.value(), 0.0));
        // the lower rib deflection, the pubic force and the front door's velocity are on their
        // limits, to the last bit: x₂, x₄ or x₆ one step lower violates them
        let g = CarSideImpact.constraints(best);
        for (gene, constraint) in [(1, 6), (3, 7), (5, 9)] {
            assert_eq!(g.inequalities()[constraint], 0.0);
            let mut lower = best.clone();
            lower[gene] = lower[gene].next_down();
            let g = CarSideImpact.constraints(&lower);
            assert!(g.inequalities()[constraint] > 0.0, "x{}", gene + 1);
        }
        // the example's L-SHADE design, 23.58565798078049, is feasible and 1.7e-14 heavier,
        // relative to it
        let run = reals(&[
            0.500_000_000_000_000_2,
            1.225_732_323_232_359_4,
            0.500_000_000_000_001_3,
            1.207_110_858_585_885,
            0.875_000_000_000_000_2,
            0.884_189_121_233_704_5,
            0.400_000_000_000_011_9,
        ]);
        let (weight, violation) = CarSideImpact.evaluate(&run);
        assert_eq!(violation, 0.0);
        assert!(optimum.value() < weight && weight < optimum.value() * (1.0 + 2e-14));
        // at the lower bounds: the weight, and every response by hand
        let low = reals(&[0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4]);
        let weight = 1.98
            + 4.9 * 0.5
            + 6.67 * 0.45
            + 6.98 * 0.5
            + 4.01 * 0.5
            + 1.78 * 0.875
            + 0.000_004
            + 2.73 * 0.4;
        assert_close(CarSideImpact.evaluate(&low).0, weight, 1e-15);
        let g = CarSideImpact.constraints(&low);
        assert_eq!(g.len(), 10);
        // the lower rib deflection 46.36 − 9.9 · 0.45 − 4.4505 · 0.5 − 32
        assert_close(g.inequalities()[6], 46.36 - 4.455 - 2.225_25 - 32.0, 1e-13);
        // the pubic force 4.72 − 0.25 − 0.19 · 0.225 − 4
        assert_close(g.inequalities()[7], 0.47 - 0.042_75, 1e-13);
        // at the upper bounds, the lower rib and the B-pillar are within their limits
        let high = reals(&[1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2]);
        let g = CarSideImpact.constraints(&high);
        assert!(g.inequalities()[6] < 0.0 && g.inequalities()[8] < 0.0);
    }
}
