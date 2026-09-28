//! The classic continuous functions: scalable ones, which take the number of dimensions, and
//! two-dimensional ones.

use super::{Optimum, Problem};
use crate::engine::FitnessFunction;
use crate::genome::{Real, Reals};
use crate::math;
use std::f64::consts::{E, PI};

// the minimizer of Styblinski-Tang's term ½ (x⁴ − 16x² + 5x) on [−5, 5]: the root of its derivative
// 4x³ − 32x + 5 = 0 near −2.9035, computed to 40 digits by Newton's method and rounded
const STYBLINSKI_TANG_X: f64 = -2.903_534_027_771_177;
// the term's value there, ½ (x⁴ − 16x² + 5x), to 40 digits and rounded
const STYBLINSKI_TANG_MIN: f64 = -39.166_165_703_771_41;

// the minimizer of Schwefel 2.26's term −x sin √|x| on [−500, 500]: x = s², where s solves
// sin s + (s / 2) cos s = 0 (the derivative is zero) near 20.52, computed to 40 digits
const SCHWEFEL_2_26_X: f64 = 420.968_746_359_982_05;
// the term's value there, −x sin √x, to 40 digits and rounded
const SCHWEFEL_2_26_MIN: f64 = -418.982_887_272_433_7;

macro_rules! scalable {
    ($(#[$doc:meta])* $name:ident, $label:literal, $minimum:literal, $default:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name {
            dimensions: usize,
        }

        impl $name {
            #[doc = concat!("The function in `dimensions` dimensions, at least ", stringify!($minimum), ".")]
            ///
            /// # Panics
            ///
            #[doc = concat!("If `dimensions` is below ", stringify!($minimum), ".")]
            pub fn new(dimensions: usize) -> Self {
                assert!(
                    dimensions >= $minimum,
                    "{} needs at least {} dimensions, got {dimensions}",
                    $label,
                    $minimum
                );
                Self { dimensions }
            }

            /// The number of dimensions.
            pub fn dimensions(&self) -> usize {
                self.dimensions
            }
        }

        impl Default for $name {
            #[doc = concat!("The function in ", stringify!($default), " dimensions.")]
            fn default() -> Self {
                Self::new($default)
            }
        }
    };
}

// bounds that are valid by construction
fn uniform(dimensions: usize, low: f64, high: f64) -> Real {
    Real::uniform(dimensions, low..=high).expect("valid bounds")
}

// the same value in every dimension
fn repeated(dimensions: usize, value: f64) -> Reals {
    Reals::from(vec![value; dimensions])
}

fn reals(values: &[f64]) -> Reals {
    Reals::from(values.to_vec())
}

scalable!(
    /// The sphere, `Σ xᵢ²`, De Jong's F1: the simplest unimodal function.
    ///
    /// Bounds [−100, 100]ⁿ; minimum 0 at the origin; 30 dimensions by default.
    ///
    /// De Jong, K. A. (1975). *An Analysis of the Behavior of a Class of Genetic Adaptive
    /// Systems.* PhD thesis, University of Michigan. Definition, bounds and dimensions as restated
    /// in Yao, Liu and Lin (1999, f1); De Jong's F1 is the same function in 3 dimensions on
    /// [−5.12, 5.12], as restated by Pohlheim (GEATbx) and Laguna and Martí (2005), and Schwefel
    /// (1977, problem 1.1) states it in any dimension, unbounded. Not yet checked against De Jong's
    /// thesis ([#168](https://github.com/tachsin/genoxide/issues/168)).
    Sphere,
    "Sphere",
    1,
    30
);

scalable!(
    /// The axis-parallel hyper-ellipsoid, `Σ i xᵢ²` (i from 1): a sphere stretched along each
    /// axis, the more for the later genes. Not the rotated hyper-ellipsoid, and not
    /// [`Schwefel1_2`].
    ///
    /// Bounds [−5.12, 5.12]ⁿ; minimum 0 at the origin; 30 dimensions by default.
    ///
    /// Its origin is unknown. The earliest source found is Pohlheim's GEATbx documentation (function
    /// 1a, "the weighted sphere model"), which Molga and Smutnicki (2005, section 2.2) restate with
    /// the same bounds; it isn't among Schwefel's (1977) problems. Not yet checked against an
    /// original ([#168](https://github.com/tachsin/genoxide/issues/168)).
    AxisParallelEllipsoid,
    "AxisParallelEllipsoid",
    1,
    30
);

scalable!(
    /// Schwefel's problem 1.2, `Σᵢ (Σⱼ≤ᵢ xⱼ)²`: unimodal, with strongly interacting genes.
    ///
    /// Bounds [−100, 100]ⁿ; minimum 0 at the origin; 30 dimensions by default.
    ///
    /// Schwefel, H.-P. (1981). *Numerical Optimization of Computer Models.* Wiley, problem 1.2, the
    /// translation of *Numerische Optimierung von Computer-Modellen* (1977, Birkhäuser, p. 319),
    /// which states it in any dimension and unbounded, with its minimum 0 at the origin. The bounds
    /// are Yao, Liu and Lin's (1999, f3).
    Schwefel1_2,
    "Schwefel1_2",
    1,
    30
);

scalable!(
    /// Rastrigin's function, `10n + Σ (xᵢ² − 10 cos 2πxᵢ)`: a sphere plus a cosine, with a local
    /// minimum near every integer point.
    ///
    /// Bounds [−5.12, 5.12]ⁿ; minimum 0 at the origin; 30 dimensions by default.
    ///
    /// Rastrigin, L. A. (1974). *Systems of Extremal Control.* Nauka, Moscow, whose function is a
    /// related one in two dimensions with other constants, as Törn and Žilinskas (1989) restate
    /// it. The n-dimensional form is credited to Rudolph (1990), and was spread by Hoffmeister and
    /// Bäck (1991) and Mühlenbein, H., Schomisch, M. and Born, J. (1991). The parallel genetic
    /// algorithm as function optimizer. *Parallel Computing* 17(6-7): 619-632.
    /// doi:10.1016/S0167-8191(05)80052-3, whose F6 this is (as their 1993 paper restates it, with
    /// A = 10). Definition and bounds as in Yao, Liu and Lin (1999, f9); not yet checked against
    /// Rastrigin's book ([#168](https://github.com/tachsin/genoxide/issues/168)).
    Rastrigin,
    "Rastrigin",
    1,
    30
);

scalable!(
    /// Rosenbrock's function, chained: `Σᵢ₌₁ⁿ⁻¹ [100 (xᵢ₊₁ − xᵢ²)² + (xᵢ − 1)²]`, a narrow curved
    /// valley.
    ///
    /// Bounds [−30, 30]ⁿ; minimum 0 at (1, …, 1); at least 2 dimensions, 30 by default.
    ///
    /// Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least value of a
    /// function. *The Computer Journal* 3(3): 175-184, in two dimensions, with no bounds. This is
    /// the chained n-dimensional form and the bounds of Yao, Liu and Lin (1999, f5), not the
    /// pairwise "extended" form (a sum over the pairs x₁x₂, x₃x₄, …).
    Rosenbrock,
    "Rosenbrock",
    2,
    30
);

scalable!(
    /// Ackley's function, `−20 exp(−0.2 √(Σ xᵢ² / n)) − exp(Σ cos(2πxᵢ) / n) + 20 + e`: a nearly
    /// flat outer region around a deep hole, covered in local minima.
    ///
    /// Bounds [−32, 32]ⁿ; minimum 0 at the origin; 30 dimensions by default.
    ///
    /// Ackley, D. H. (1987). *A Connectionist Machine for Genetic Hillclimbing.* Kluwer (two
    /// dimensions); generalized to n dimensions by Bäck, T. (1996). *Evolutionary Algorithms in
    /// Theory and Practice.* Oxford University Press. Definition and bounds as restated in Yao,
    /// Liu and Lin (1999, f10), who use [−32, 32] (other papers use [−32.768, 32.768]); not yet
    /// checked against the originals ([#168](https://github.com/tachsin/genoxide/issues/168)).
    Ackley,
    "Ackley",
    1,
    30
);

scalable!(
    /// Griewank's function, `1 + Σ xᵢ² / 4000 − Π cos(xᵢ / √i)` (i from 1): a wide bowl with a
    /// product of cosines that couples the genes.
    ///
    /// Bounds [−600, 600]ⁿ; minimum 0 at the origin; 30 dimensions by default.
    ///
    /// Griewank, A. O. (1981). Generalized descent for global optimization. *Journal of
    /// Optimization Theory and Applications* 34(1): 11-39, whose function, as Bosse and Bücker
    /// (2024) restate it, is two-dimensional with the divisor 200:
    /// `1 + (x₁² + x₂²) / 200 − cos x₁ cos(x₂ / √2)`. The n-dimensional form with the divisor 4000
    /// and the bounds are those of Mühlenbein, Schomisch and Born (1991, F8) and Yao, Liu and Lin
    /// (1999, f11). Not yet checked against the original
    /// ([#168](https://github.com/tachsin/genoxide/issues/168)).
    Griewank,
    "Griewank",
    1,
    30
);

scalable!(
    /// Schwefel's problem 2.26, `−Σ xᵢ sin √|xᵢ|`: the best local minima are far apart, the
    /// global one near a corner of the box.
    ///
    /// Bounds [−500, 500]ⁿ; minimum −418.9828872724337 n at xᵢ = 420.96874635998205, both derived
    /// from the formula (each term is minimized alone); 30 dimensions by default.
    ///
    /// Schwefel, H.-P. (1981). *Numerical Optimization of Computer Models.* Wiley, problem 2.26, the
    /// translation of *Numerische Optimierung von Computer-Modellen* (1977, Birkhäuser, p. 335),
    /// where it has one variable, `−x₁ sin √|x₁|`, no bounds and so no finite minimum (its problem
    /// 2.44 bounds it to [−300, 300]). The sum over n variables on [−500, 500] is Mühlenbein,
    /// Schomisch and Born's (1991, F7), restated in Yao, Liu and Lin (1999, f8), whose minimum is
    /// −12569.5 for n = 30. This is the form without an offset: variants add 418.9829 n (a
    /// minimum near 0) or divide by n.
    Schwefel2_26,
    "Schwefel2_26",
    1,
    30
);

scalable!(
    /// Levy's function: with `wᵢ = 1 + (xᵢ − 1) / 4`,
    /// `sin²(πw₁) + Σᵢ₌₁ⁿ⁻¹ (wᵢ − 1)² [1 + 10 sin²(πwᵢ + 1)] + (wₙ − 1)² [1 + sin²(2πwₙ)]`.
    ///
    /// Bounds [−10, 10]ⁿ; minimum 0 at (1, …, 1); 30 dimensions by default.
    ///
    /// Usually credited to Levy, A. V. and Montalvo, A. (1985). The tunneling algorithm for the
    /// global minimization of functions. *SIAM Journal on Scientific and Statistical Computing*
    /// 6(1): 15-29, but several functions carry Levy's name. This is the form of Surjanovic and
    /// Bingham's Virtual Library of Simulation Experiments, with `sin²(πwᵢ + 1)` and bounds
    /// [−10, 10]; Laguna and Martí (2005, function 38) print xₙ instead of wₙ in the last sine
    /// (this uses wₙ, like the other terms), and the Levy-Montalvo functions as restated in Yao,
    /// Liu and Lin (1999, f12 and f13) have `sin²(πyᵢ₊₁)` instead, of which `πwᵢ + 1` may be a
    /// misreading. The minimum is 0 at (1, …, 1) in every form. Not yet checked against the
    /// original ([#168](https://github.com/tachsin/genoxide/issues/168)).
    Levy,
    "Levy",
    1,
    30
);

scalable!(
    /// Zakharov's function, `Σ xᵢ² + (Σ 0.5 i xᵢ)² + (Σ 0.5 i xᵢ)⁴` (i from 1): unimodal, with
    /// strongly interacting genes.
    ///
    /// Bounds [−5, 10]ⁿ; minimum 0 at the origin; 30 dimensions by default.
    ///
    /// Its origin is unknown: definition and bounds as restated in Laguna and Martí (2005,
    /// function 12); not yet checked against an original
    /// ([#168](https://github.com/tachsin/genoxide/issues/168)).
    Zakharov,
    "Zakharov",
    1,
    30
);

scalable!(
    /// The Styblinski-Tang function, `½ Σ (xᵢ⁴ − 16xᵢ² + 5xᵢ)`: separable, with a local minimum
    /// of each term near 2.75 and the global one near −2.90.
    ///
    /// Bounds [−5, 5]ⁿ; minimum −39.16616570377141 n at xᵢ = −2.903534027771177, both derived
    /// from the formula (the root of 4x³ − 32x + 5 = 0); 30 dimensions by default.
    ///
    /// Styblinski, M. A. and Tang, T.-S. (1990). Experiments in nonconvex optimization: stochastic
    /// approximation with function smoothing and simulated annealing. *Neural Networks* 3(4):
    /// 467-483. Definition and bounds as restated in Jamil and Yang (2013, function 144); not yet
    /// checked against the original ([#168](https://github.com/tachsin/genoxide/issues/168)).
    StyblinskiTang,
    "StyblinskiTang",
    1,
    30
);

scalable!(
    /// Michalewicz's function, `−Σ sin(xᵢ) sin²ᵐ(i xᵢ² / π)` with m = 10 (i from 1): steep
    /// narrow valleys on flat plateaus.
    ///
    /// Bounds [0, π]ⁿ; 10 dimensions by default. The minimum depends on n: the function is
    /// separable, so [`optimum`](Problem::optimum) minimizes each term on its own, which gives
    /// −1.8013034 for n = 2, −4.6876582 for n = 5 and −9.6601517 for n = 10 (Molga and Smutnicki
    /// give −4.687 and −9.66).
    ///
    /// Michalewicz, Z. (1992). *Genetic Algorithms + Data Structures = Evolution Programs.*
    /// Springer. Definition, m and bounds as restated in Molga and Smutnicki (2005, section
    /// 2.11); not yet checked against the original
    /// ([#168](https://github.com/tachsin/genoxide/issues/168)).
    Michalewicz,
    "Michalewicz",
    1,
    10
);

// ---- the scalable functions ----------------------------------------------------------------------

impl FitnessFunction<Reals> for Sphere {
    type Output = f64;

    fn evaluate(&self, x: &Reals) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }
}

impl Problem for Sphere {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Sphere"
    }

    fn representation(&self) -> Real {
        uniform(self.dimensions, -100.0, 100.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(0.0, vec![repeated(self.dimensions, 0.0)]))
    }

    fn reference(&self) -> &'static str {
        "De Jong, K. A. (1975). An Analysis of the Behavior of a Class of Genetic Adaptive \
         Systems. PhD thesis, University of Michigan."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://hdl.handle.net/2027.42/4507")
    }
}

impl FitnessFunction<Reals> for AxisParallelEllipsoid {
    type Output = f64;

    fn evaluate(&self, x: &Reals) -> f64 {
        x.iter()
            .enumerate()
            .map(|(i, xi)| (i + 1) as f64 * xi * xi)
            .sum()
    }
}

impl Problem for AxisParallelEllipsoid {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "AxisParallelEllipsoid"
    }

    fn representation(&self) -> Real {
        uniform(self.dimensions, -5.12, 5.12)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(0.0, vec![repeated(self.dimensions, 0.0)]))
    }

    fn reference(&self) -> &'static str {
        "Molga, M. and Smutnicki, C. (2005). Test functions for optimization needs."
    }
}

impl FitnessFunction<Reals> for Schwefel1_2 {
    type Output = f64;

    fn evaluate(&self, x: &Reals) -> f64 {
        let mut prefix = 0.0;
        let mut sum = 0.0;
        for xi in x.iter() {
            prefix += xi;
            sum += prefix * prefix;
        }
        sum
    }
}

impl Problem for Schwefel1_2 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Schwefel1_2"
    }

    fn representation(&self) -> Real {
        uniform(self.dimensions, -100.0, 100.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(0.0, vec![repeated(self.dimensions, 0.0)]))
    }

    fn reference(&self) -> &'static str {
        "Schwefel, H.-P. (1981). Numerical Optimization of Computer Models. Wiley. Problem 1.2."
    }
}

impl FitnessFunction<Reals> for Rastrigin {
    type Output = f64;

    fn evaluate(&self, x: &Reals) -> f64 {
        10.0 * x.len() as f64
            + x.iter()
                .map(|xi| xi * xi - 10.0 * math::cos(2.0 * PI * xi))
                .sum::<f64>()
    }
}

impl Problem for Rastrigin {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Rastrigin"
    }

    fn representation(&self) -> Real {
        uniform(self.dimensions, -5.12, 5.12)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(0.0, vec![repeated(self.dimensions, 0.0)]))
    }

    fn reference(&self) -> &'static str {
        "Rastrigin, L. A. (1974). Systems of Extremal Control. Nauka, Moscow. Generalized by \
         Mühlenbein, H., Schomisch, M. and Born, J. (1991). The parallel genetic algorithm as \
         function optimizer. Parallel Computing 17(6-7): 619-632."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1016/S0167-8191(05)80052-3")
    }
}

impl FitnessFunction<Reals> for Rosenbrock {
    type Output = f64;

    fn evaluate(&self, x: &Reals) -> f64 {
        x.windows(2)
            .map(|pair| {
                let (xi, next) = (pair[0], pair[1]);
                100.0 * math::powi(next - xi * xi, 2) + math::powi(xi - 1.0, 2)
            })
            .sum()
    }
}

impl Problem for Rosenbrock {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Rosenbrock"
    }

    fn representation(&self) -> Real {
        uniform(self.dimensions, -30.0, 30.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(0.0, vec![repeated(self.dimensions, 1.0)]))
    }

    fn reference(&self) -> &'static str {
        "Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least value \
         of a function. The Computer Journal 3(3): 175-184."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1093/comjnl/3.3.175")
    }
}

impl FitnessFunction<Reals> for Ackley {
    type Output = f64;

    fn evaluate(&self, x: &Reals) -> f64 {
        let n = x.len() as f64;
        let squares = x.iter().map(|xi| xi * xi).sum::<f64>() / n;
        let cosines = x.iter().map(|xi| math::cos(2.0 * PI * xi)).sum::<f64>() / n;
        // in this order, 0 at the origin: 20 − 20 and e − e cancel
        20.0 - 20.0 * math::exp(-0.2 * squares.sqrt()) + E - math::exp(cosines)
    }
}

impl Problem for Ackley {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Ackley"
    }

    fn representation(&self) -> Real {
        uniform(self.dimensions, -32.0, 32.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(0.0, vec![repeated(self.dimensions, 0.0)]))
    }

    fn reference(&self) -> &'static str {
        "Ackley, D. H. (1987). A Connectionist Machine for Genetic Hillclimbing. Kluwer. \
         Generalized by Bäck, T. (1996). Evolutionary Algorithms in Theory and Practice. Oxford \
         University Press."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1007/978-1-4613-1997-9")
    }
}

impl FitnessFunction<Reals> for Griewank {
    type Output = f64;

    fn evaluate(&self, x: &Reals) -> f64 {
        let squares = x.iter().map(|xi| xi * xi).sum::<f64>() / 4000.0;
        let product = x
            .iter()
            .enumerate()
            .map(|(i, xi)| math::cos(xi / ((i + 1) as f64).sqrt()))
            .product::<f64>();
        1.0 + squares - product
    }
}

impl Problem for Griewank {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Griewank"
    }

    fn representation(&self) -> Real {
        uniform(self.dimensions, -600.0, 600.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(0.0, vec![repeated(self.dimensions, 0.0)]))
    }

    fn reference(&self) -> &'static str {
        "Griewank, A. O. (1981). Generalized descent for global optimization. Journal of \
         Optimization Theory and Applications 34(1): 11-39."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1007/BF00933356")
    }
}

impl FitnessFunction<Reals> for Schwefel2_26 {
    type Output = f64;

    fn evaluate(&self, x: &Reals) -> f64 {
        -x.iter()
            .map(|xi| xi * math::sin(xi.abs().sqrt()))
            .sum::<f64>()
    }
}

impl Problem for Schwefel2_26 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Schwefel2_26"
    }

    fn representation(&self) -> Real {
        uniform(self.dimensions, -500.0, 500.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        let value = SCHWEFEL_2_26_MIN * self.dimensions as f64;
        Some(Optimum::proven(
            value,
            vec![repeated(self.dimensions, SCHWEFEL_2_26_X)],
        ))
    }

    fn reference(&self) -> &'static str {
        "Schwefel, H.-P. (1981). Numerical Optimization of Computer Models. Wiley. Problem 2.26."
    }
}

impl FitnessFunction<Reals> for Levy {
    type Output = f64;

    fn evaluate(&self, x: &Reals) -> f64 {
        let w = |xi: f64| 1.0 + (xi - 1.0) / 4.0;
        let (Some(&first), Some(&last)) = (x.first(), x.last()) else {
            return 0.0;
        };
        let (first, last) = (w(first), w(last));
        let middle: f64 = x[..x.len() - 1]
            .iter()
            .map(|&xi| {
                let wi = w(xi);
                math::powi(wi - 1.0, 2) * (1.0 + 10.0 * math::powi(math::sin(PI * wi + 1.0), 2))
            })
            .sum();
        math::powi(math::sin(PI * first), 2)
            + middle
            + math::powi(last - 1.0, 2) * (1.0 + math::powi(math::sin(2.0 * PI * last), 2))
    }
}

impl Problem for Levy {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Levy"
    }

    fn representation(&self) -> Real {
        uniform(self.dimensions, -10.0, 10.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(0.0, vec![repeated(self.dimensions, 1.0)]))
    }

    fn reference(&self) -> &'static str {
        "Levy, A. V. and Montalvo, A. (1985). The tunneling algorithm for the global \
         minimization of functions. SIAM Journal on Scientific and Statistical Computing 6(1): \
         15-29."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1137/0906002")
    }
}

impl FitnessFunction<Reals> for Zakharov {
    type Output = f64;

    fn evaluate(&self, x: &Reals) -> f64 {
        let squares: f64 = x.iter().map(|xi| xi * xi).sum();
        let weighted: f64 = x
            .iter()
            .enumerate()
            .map(|(i, xi)| 0.5 * (i + 1) as f64 * xi)
            .sum();
        squares + math::powi(weighted, 2) + math::powi(weighted, 4)
    }
}

impl Problem for Zakharov {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Zakharov"
    }

    fn representation(&self) -> Real {
        uniform(self.dimensions, -5.0, 10.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(0.0, vec![repeated(self.dimensions, 0.0)]))
    }

    fn reference(&self) -> &'static str {
        "Laguna, M. and Martí, R. (2005). Experimental testing of advanced scatter search designs \
         for global optimization of multimodal functions. Journal of Global Optimization 33(2): \
         235-255."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1007/s10898-004-1936-z")
    }
}

impl FitnessFunction<Reals> for StyblinskiTang {
    type Output = f64;

    fn evaluate(&self, x: &Reals) -> f64 {
        0.5 * x
            .iter()
            .map(|xi| math::powi(*xi, 4) - 16.0 * xi * xi + 5.0 * xi)
            .sum::<f64>()
    }
}

impl Problem for StyblinskiTang {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "StyblinskiTang"
    }

    fn representation(&self) -> Real {
        uniform(self.dimensions, -5.0, 5.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        let value = STYBLINSKI_TANG_MIN * self.dimensions as f64;
        Some(Optimum::proven(
            value,
            vec![repeated(self.dimensions, STYBLINSKI_TANG_X)],
        ))
    }

    fn reference(&self) -> &'static str {
        "Styblinski, M. A. and Tang, T.-S. (1990). Experiments in nonconvex optimization: \
         stochastic approximation with function smoothing and simulated annealing. Neural \
         Networks 3(4): 467-483."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1016/0893-6080(90)90029-K")
    }
}

// Michalewicz's steepness
const MICHALEWICZ_M: i32 = 10;

// the term of gene `i` (from 0)
fn michalewicz_term(i: usize, xi: f64) -> f64 {
    let index = (i + 1) as f64;
    -math::sin(xi) * math::powi(math::sin(index * xi * xi / PI), 2 * MICHALEWICZ_M)
}

// the gene that minimizes the term of gene `i` on [0, π]. The second sine is zero at
// x = π √(k / (i + 1)), for k = 0 … i + 1: between two such zeros, the term has one minimum,
// found by golden-section search. The term is at least −sin(x), so a bracket where sin(x) stays
// below the best term found can't hold the minimum: the search starts at the bracket around π/2
// (where sin is largest) and goes outwards until that bound rules the rest out.
fn michalewicz_minimizer(i: usize) -> f64 {
    let brackets = i + 1;
    let zero = |k: usize| PI * (k as f64 / brackets as f64).sqrt();
    let bracket_min = |k: usize| golden_section(|x| michalewicz_term(i, x), zero(k), zero(k + 1));
    // the largest sine in a bracket: 1 if it holds π/2, else at an end
    let largest_sine = |k: usize| {
        let (low, high) = (zero(k), zero(k + 1));
        if low <= PI / 2.0 && PI / 2.0 <= high {
            1.0
        } else {
            math::sin(low).max(math::sin(high))
        }
    };
    let middle = (0..brackets)
        .find(|&k| zero(k + 1) >= PI / 2.0)
        .unwrap_or(brackets - 1);
    let mut best = bracket_min(middle);
    let mut best_value = michalewicz_term(i, best);
    let consider = |k: usize, best: &mut f64, best_value: &mut f64| {
        if -largest_sine(k) >= *best_value {
            return false;
        }
        let x = bracket_min(k);
        let value = michalewicz_term(i, x);
        if value < *best_value {
            *best = x;
            *best_value = value;
        }
        true
    };
    for k in (0..middle).rev() {
        if !consider(k, &mut best, &mut best_value) {
            break;
        }
    }
    for k in middle + 1..brackets {
        if !consider(k, &mut best, &mut best_value) {
            break;
        }
    }
    best
}

// the minimum of a unimodal `f` on [low, high], by golden-section search to the precision of f64
fn golden_section(f: impl Fn(f64) -> f64, mut low: f64, mut high: f64) -> f64 {
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    let mut a = high - ratio * (high - low);
    let mut b = low + ratio * (high - low);
    let (mut fa, mut fb) = (f(a), f(b));
    for _ in 0..200 {
        if high - low <= f64::EPSILON * high.abs().max(1.0) {
            break;
        }
        if fa <= fb {
            high = b;
            b = a;
            fb = fa;
            a = high - ratio * (high - low);
            fa = f(a);
        } else {
            low = a;
            a = b;
            fa = fb;
            b = low + ratio * (high - low);
            fb = f(b);
        }
    }
    if fa <= fb { a } else { b }
}

impl FitnessFunction<Reals> for Michalewicz {
    type Output = f64;

    fn evaluate(&self, x: &Reals) -> f64 {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| michalewicz_term(i, xi))
            .sum()
    }
}

impl Problem for Michalewicz {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Michalewicz"
    }

    fn representation(&self) -> Real {
        uniform(self.dimensions, 0.0, PI)
    }

    /// The minimum for this `n`, computed a gene at a time, since the function is separable: the
    /// minimum of each term, to the precision of `f64`.
    fn optimum(&self) -> Option<Optimum<Reals>> {
        let solution: Reals = (0..self.dimensions).map(michalewicz_minimizer).collect();
        let value = self.evaluate(&solution);
        Some(Optimum::proven(value, vec![solution]))
    }

    fn reference(&self) -> &'static str {
        "Michalewicz, Z. (1992). Genetic Algorithms + Data Structures = Evolution Programs. \
         Springer."
    }
}

// ---- the two-dimensional functions ---------------------------------------------------------------

/// Himmelblau's function, `(x₁² + x₂ − 11)² + (x₁ + x₂² − 7)²`: four global minima.
///
/// Bounds [−5, 5]²; minimum 0 at (3, 2), (−2.805118086952745, 3.131312518250573),
/// (−3.779310253377747, −3.2831859912861696) and (3.5844283403304917, −1.8481265269644036): the
/// solutions of x₁² + x₂ = 11 and x₁ + x₂² = 7, computed to 40 digits by Newton's method and
/// rounded.
///
/// Himmelblau, D. M. (1972). *Applied Nonlinear Programming.* McGraw-Hill. Definition and bounds
/// as restated in Jamil and Yang (2013, function 65); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Himmelblau;

impl FitnessFunction<Reals> for Himmelblau {
    type Output = f64;

    /// The value at `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 2 genes.
    fn evaluate(&self, x: &Reals) -> f64 {
        let (x1, x2) = (x[0], x[1]);
        math::powi(x1 * x1 + x2 - 11.0, 2) + math::powi(x1 + x2 * x2 - 7.0, 2)
    }
}

impl Problem for Himmelblau {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Himmelblau"
    }

    fn representation(&self) -> Real {
        uniform(2, -5.0, 5.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(
            0.0,
            vec![
                reals(&[3.0, 2.0]),
                reals(&[-2.805_118_086_952_745, 3.131_312_518_250_573]),
                reals(&[-3.779_310_253_377_747, -3.283_185_991_286_169_6]),
                reals(&[3.584_428_340_330_491_7, -1.848_126_526_964_403_6]),
            ],
        ))
    }

    fn reference(&self) -> &'static str {
        "Himmelblau, D. M. (1972). Applied Nonlinear Programming. McGraw-Hill."
    }
}

/// Branin's function (RCOS), `(x₂ − 5.1 x₁² / (4π²) + 5 x₁ / π − 6)² + 10 (1 − 1 / (8π)) cos x₁ +
/// 10`: three global minima.
///
/// Bounds x₁ ∈ [−5, 10], x₂ ∈ [0, 15]; minimum 5 / (4π) ≈ 0.3978874 at (−π, 12.275), (π, 2.275)
/// and (3π, 2.475). The minimum and its solutions are derived from the formula: the value is at
/// least 10 / (8π), reached where cos x₁ = −1 and the square is 0. Yao, Liu and Lin (1999) and
/// Jamil and Yang (2013) print the third as (3π, 2.425).
///
/// Branin, F. H. (1972). Widely convergent method for finding multiple solutions of simultaneous
/// nonlinear equations. *IBM Journal of Research and Development* 16(5): 504-522. Definition and
/// bounds as restated in Yao, Liu and Lin (1999, f17); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Branin;

impl FitnessFunction<Reals> for Branin {
    type Output = f64;

    /// The value at `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 2 genes.
    fn evaluate(&self, x: &Reals) -> f64 {
        let (x1, x2) = (x[0], x[1]);
        let b = 5.1 / (4.0 * PI * PI);
        let c = 5.0 / PI;
        let t = 1.0 / (8.0 * PI);
        math::powi(x2 - b * x1 * x1 + c * x1 - 6.0, 2) + 10.0 * (1.0 - t) * math::cos(x1) + 10.0
    }
}

impl Problem for Branin {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Branin"
    }

    fn representation(&self) -> Real {
        Real::new([-5.0..=10.0, 0.0..=15.0]).expect("valid bounds")
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(
            5.0 / (4.0 * PI),
            vec![
                reals(&[-PI, 12.275]),
                reals(&[PI, 2.275]),
                reals(&[3.0 * PI, 2.475]),
            ],
        ))
    }

    fn reference(&self) -> &'static str {
        "Branin, F. H. (1972). Widely convergent method for finding multiple solutions of \
         simultaneous nonlinear equations. IBM Journal of Research and Development 16(5): 504-522."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1147/rd.165.0504")
    }
}

/// The Goldstein-Price function, `[1 + (x₁ + x₂ + 1)² (19 − 14x₁ + 3x₁² − 14x₂ + 6x₁x₂ + 3x₂²)] ×
/// [30 + (2x₁ − 3x₂)² (18 − 32x₁ + 12x₁² + 48x₂ − 36x₁x₂ + 27x₂²)]`.
///
/// Bounds [−2, 2]²; minimum 3 at (0, −1). The paper also lists the local minima (1.2, 0.8) with
/// 840, (1.8, 0.2) with 84 and (−0.6, −0.4) with 30, and found (0, −1) numerically. It's the
/// global minimum: with s = x₁ + x₂, the first factor is `1 + (s + 1)² (3s² − 14s + 19)` ≥ 1,
/// and with t = 2x₁ − 3x₂, the second is `30 + t² (3t² − 16t + 18)` ≥ 3; both bounds are met only
/// at s = −1 and t = 3, that is at (0, −1).
///
/// Goldstein, A. A. and Price, J. F. (1971). On descent from local minima. *Mathematics of
/// Computation* 25(115): 569-574. The paper gives no bounds: these are Dixon and Szegö's (1978),
/// as restated in Yao, Liu and Lin (1999, f18).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GoldsteinPrice;

impl FitnessFunction<Reals> for GoldsteinPrice {
    type Output = f64;

    /// The value at `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 2 genes.
    fn evaluate(&self, x: &Reals) -> f64 {
        let (x1, x2) = (x[0], x[1]);
        let a = 1.0
            + math::powi(x1 + x2 + 1.0, 2)
                * (19.0 - 14.0 * x1 + 3.0 * x1 * x1 - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2 * x2);
        let b = 30.0
            + math::powi(2.0 * x1 - 3.0 * x2, 2)
                * (18.0 - 32.0 * x1 + 12.0 * x1 * x1 + 48.0 * x2 - 36.0 * x1 * x2 + 27.0 * x2 * x2);
        a * b
    }
}

impl Problem for GoldsteinPrice {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "GoldsteinPrice"
    }

    fn representation(&self) -> Real {
        uniform(2, -2.0, 2.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(3.0, vec![reals(&[0.0, -1.0])]))
    }

    fn reference(&self) -> &'static str {
        "Goldstein, A. A. and Price, J. F. (1971). On descent from local minima. Mathematics of \
         Computation 25(115): 569-574."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1090/S0025-5718-1971-0312365-X")
    }
}

/// The six-hump camel-back function, `(4 − 2.1x₁² + x₁⁴ / 3) x₁² + x₁x₂ + (−4 + 4x₂²) x₂²`: two
/// global minima among six.
///
/// Bounds [−5, 5]²; minimum −1.0316284534898774 at ±(0.08984201310031806,
/// −0.7126564030207396): stationary points computed to 40 digits by Newton's method and rounded.
/// It's the global minimum: the function has 15 real stationary points (with x₁ = 8x₂ − 16x₂³,
/// a polynomial of degree 15 in x₂), this pair the lowest, the next −0.2154638 at ±(1.70361,
/// −0.79608), and it grows without bound. Yao, Liu and Lin's value agrees to 8 digits; their
/// x₁, 0.08983, is 0.089842 to 5.
///
/// Dixon, L. C. W. and Szegö, G. P. (eds.) (1978). *Towards Global Optimisation 2.*
/// North-Holland. Definition and bounds as restated in Yao, Liu and Lin (1999, f16), whose
/// bounds are wider than the x₁ ∈ [−3, 3], x₂ ∈ [−2, 2] of other papers; not yet checked against
/// the original ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SixHumpCamel;

impl FitnessFunction<Reals> for SixHumpCamel {
    type Output = f64;

    /// The value at `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 2 genes.
    fn evaluate(&self, x: &Reals) -> f64 {
        let (x1, x2) = (x[0], x[1]);
        let x1_squared = x1 * x1;
        (4.0 - 2.1 * x1_squared + x1_squared * x1_squared / 3.0) * x1_squared
            + x1 * x2
            + (-4.0 + 4.0 * x2 * x2) * x2 * x2
    }
}

impl Problem for SixHumpCamel {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "SixHumpCamel"
    }

    fn representation(&self) -> Real {
        uniform(2, -5.0, 5.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        let (x1, x2) = (0.089_842_013_100_318_06, -0.712_656_403_020_739_6);
        Some(Optimum::proven(
            -1.031_628_453_489_877_4,
            vec![reals(&[x1, x2]), reals(&[-x1, -x2])],
        ))
    }

    fn reference(&self) -> &'static str {
        "Dixon, L. C. W. and Szegö, G. P. (eds.) (1978). Towards Global Optimisation 2. \
         North-Holland."
    }
}

// ---- the functions with tables: Hartmann and Shekel ---------------------------------------------

// the paper that defines Hartmann's function, and the one whose constants these are
const HARTMANN_REFERENCE: &str = "Hartman, J. K. (1973). Some experiments in global \
    optimization. Naval Research Logistics Quarterly 20(3): 569-576. Constants as tabulated in \
    Dixon, L. C. W. and Szegö, G. P. (1978). The global optimisation problem: an introduction. In \
    Towards Global Optimisation 2, North-Holland: 1-15.";

// Hartmann's weights cᵢ, the same in 3 and 6 dimensions
const HARTMANN_C: [f64; 4] = [1.0, 1.2, 3.0, 3.2];

// Hartmann's function in 3 dimensions: the widths aᵢⱼ and the centers pᵢⱼ
const HARTMANN_3_A: [[f64; 3]; 4] = [
    [3.0, 10.0, 30.0],
    [0.1, 10.0, 35.0],
    [3.0, 10.0, 30.0],
    [0.1, 10.0, 35.0],
];
const HARTMANN_3_P: [[f64; 3]; 4] = [
    [0.3689, 0.1170, 0.2673],
    [0.4699, 0.4387, 0.7470],
    [0.1091, 0.8732, 0.5547],
    [0.03815, 0.5743, 0.8828],
];

// Hartmann's function in 6 dimensions: the widths aᵢⱼ and the centers pᵢⱼ
const HARTMANN_6_A: [[f64; 6]; 4] = [
    [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
    [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
    [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
    [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
];
const HARTMANN_6_P: [[f64; 6]; 4] = [
    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
];

// −Σᵢ cᵢ exp(−Σⱼ aᵢⱼ (xⱼ − pᵢⱼ)²)
fn hartmann<const N: usize>(a: &[[f64; N]; 4], p: &[[f64; N]; 4], x: &Reals) -> f64 {
    let x = &x[..N];
    -(0..4)
        .map(|i| {
            let distance: f64 = (0..N)
                .map(|j| a[i][j] * math::powi(x[j] - p[i][j], 2))
                .sum();
            HARTMANN_C[i] * math::exp(-distance)
        })
        .sum::<f64>()
}

/// Hartmann's function in 3 dimensions, `−Σᵢ₌₁⁴ cᵢ exp(−Σⱼ₌₁³ aᵢⱼ (xⱼ − pᵢⱼ)²)`: four
/// Gaussian wells of different widths and depths, the deepest near (0.11, 0.56, 0.85).
///
/// With c = (1, 1.2, 3, 3.2), and a and p the rows
///
/// | i | aᵢ₁, aᵢ₂, aᵢ₃ | pᵢ₁, pᵢ₂, pᵢ₃ |
/// |---|---|---|
/// | 1 | 3, 10, 30 | 0.3689, 0.1170, 0.2673 |
/// | 2 | 0.1, 10, 35 | 0.4699, 0.4387, 0.7470 |
/// | 3 | 3, 10, 30 | 0.1091, 0.8732, 0.5547 |
/// | 4 | 0.1, 10, 35 | 0.03815, 0.5743, 0.8828 |
///
/// Bounds [0, 1]³; minimum −3.862782147820755 at (0.11461433858967196, 0.5556488499718569,
/// 0.8525469535208658), where the gradient is 0, computed to 40 digits by Newton's method and
/// rounded; −3.86278 is the value later papers quote from Dixon and Szegö. It's the best of the
/// local minima that local searches from 2,000 random points find, with −3.0897641630922505 at
/// (0.10934, 0.86052, 0.56412) and −1.0008168635629282 at (0.36872, 0.11756, 0.26757); not
/// proven global.
///
/// The form is Hartman's: Hartman, J. K. (1972). *Some Experiments in Global Optimization.*
/// Report NPS-55HH72051A, Naval Postgraduate School (p. 10), published in *Naval Research
/// Logistics Quarterly* 20(3): 569-576 (1973). The report draws its constants at random, prints
/// none, and has no problem in 3 or 6 dimensions: these constants are those of Dixon, L. C. W.
/// and Szegö, G. P. (1978). The global optimisation problem: an introduction. In *Towards Global
/// Optimisation 2*, North-Holland: 1-15, which isn't online, as Yao, Liu and Lin (1999, f19,
/// table XII) reprint them (p₄₁ printed as 0.038150). Jamil and Yang (2013, function 62) print
/// p₂₂ = 0.4837 for 0.4387.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Hartmann3;

impl FitnessFunction<Reals> for Hartmann3 {
    type Output = f64;

    /// The value at `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 3 genes.
    fn evaluate(&self, x: &Reals) -> f64 {
        hartmann(&HARTMANN_3_A, &HARTMANN_3_P, x)
    }
}

impl Problem for Hartmann3 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Hartmann3"
    }

    fn representation(&self) -> Real {
        uniform(3, 0.0, 1.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::best_known(
            -3.862_782_147_820_755,
            vec![reals(&[
                0.114_614_338_589_671_96,
                0.555_648_849_971_856_9,
                0.852_546_953_520_865_8,
            ])],
        ))
    }

    fn reference(&self) -> &'static str {
        HARTMANN_REFERENCE
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1002/nav.3800200316")
    }
}

/// Hartmann's function in 6 dimensions, `−Σᵢ₌₁⁴ cᵢ exp(−Σⱼ₌₁⁶ aᵢⱼ (xⱼ − pᵢⱼ)²)`: four
/// Gaussian wells of different widths and depths, in two basins of nearly the same depth.
///
/// With c = (1, 1.2, 3, 3.2), and a and p the rows
///
/// | i | aᵢ₁ … aᵢ₆ | pᵢ₁ … pᵢ₆ |
/// |---|---|---|
/// | 1 | 10, 3, 17, 3.5, 1.7, 8 | 0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886 |
/// | 2 | 0.05, 10, 17, 0.1, 8, 14 | 0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991 |
/// | 3 | 3, 3.5, 1.7, 10, 17, 8 | 0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650 |
/// | 4 | 17, 8, 0.05, 10, 0.1, 14 | 0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381 |
///
/// Bounds [0, 1]⁶; minimum −3.3223680114155147 at (0.20168951100670543, 0.15001069182345797,
/// 0.476873974221897, 0.2753324304940561, 0.31165161660011326, 0.6573005340656204), where the
/// gradient is 0, computed to 40 digits by Newton's method and rounded; −3.32237 is the value
/// later papers quote from Dixon and Szegö. It's the best of the local minima that local
/// searches from 2,000 random points find; the other, −3.203161918396231 at (0.40465, 0.88244,
/// 0.84610, 0.57399, 0.13893, 0.03850), drew a third of them. Not proven global.
///
/// The form is Hartman's: Hartman, J. K. (1972). *Some Experiments in Global Optimization.*
/// Report NPS-55HH72051A, Naval Postgraduate School (p. 10), published in *Naval Research
/// Logistics Quarterly* 20(3): 569-576 (1973), whose constants are random and not printed. These
/// are those of Dixon, L. C. W. and Szegö, G. P. (1978). The global optimisation problem: an
/// introduction. In *Towards Global Optimisation 2*, North-Holland: 1-15, which isn't online, as
/// Yao, Liu and Lin (1999, f20, table XIII) reprint them, but for p₃₂: they print 0.1415, with
/// which the minimum is −3.3219952 at (0.2017, 0.1468, …), not their own minimizer; 0.1451 gives
/// the minimum that later papers quote, and Jamil and Yang (2013, function 63) print it, with
/// p₁₆ = 0.5586 for 0.5886.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Hartmann6;

impl FitnessFunction<Reals> for Hartmann6 {
    type Output = f64;

    /// The value at `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 6 genes.
    fn evaluate(&self, x: &Reals) -> f64 {
        hartmann(&HARTMANN_6_A, &HARTMANN_6_P, x)
    }
}

impl Problem for Hartmann6 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Hartmann6"
    }

    fn representation(&self) -> Real {
        uniform(6, 0.0, 1.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::best_known(
            -3.322_368_011_415_514_7,
            vec![reals(&[
                0.201_689_511_006_705_43,
                0.150_010_691_823_457_97,
                0.476_873_974_221_897,
                0.275_332_430_494_056_1,
                0.311_651_616_600_113_26,
                0.657_300_534_065_620_4,
            ])],
        ))
    }

    fn reference(&self) -> &'static str {
        HARTMANN_REFERENCE
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1002/nav.3800200316")
    }
}

// Shekel's centers aᵢ and widths cᵢ, of which Shekel m uses the first m
const SHEKEL_A: [[f64; 4]; 10] = [
    [4.0, 4.0, 4.0, 4.0],
    [1.0, 1.0, 1.0, 1.0],
    [8.0, 8.0, 8.0, 8.0],
    [6.0, 6.0, 6.0, 6.0],
    [3.0, 7.0, 3.0, 7.0],
    [2.0, 9.0, 2.0, 9.0],
    [5.0, 5.0, 3.0, 3.0],
    [8.0, 1.0, 8.0, 1.0],
    [6.0, 2.0, 6.0, 2.0],
    [7.0, 3.6, 7.0, 3.6],
];
const SHEKEL_C: [f64; 10] = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5];

// −Σᵢ₌₁ᵐ 1 / ((x − aᵢ)ᵀ(x − aᵢ) + cᵢ)
fn shekel(m: usize, x: &Reals) -> f64 {
    let x = &x[..4];
    -(0..m)
        .map(|i| {
            let distance: f64 = (0..4).map(|j| math::powi(x[j] - SHEKEL_A[i][j], 2)).sum();
            1.0 / (distance + SHEKEL_C[i])
        })
        .sum::<f64>()
}

// the docs of the three Shekel functions, which differ in m, the minimum and its solution
macro_rules! shekel {
    ($(#[$doc:meta])* $name:ident, $m:literal, $value:literal, $solution:expr) => {
        $(#[$doc])*
        ///
        /// The function is `−Σᵢ₌₁ᵐ 1 / ((x − aᵢ)ᵀ(x − aᵢ) + cᵢ)`, a well at each of m points aᵢ, as
        /// deep as 1 / cᵢ, with the first m rows of
        ///
        /// | i | aᵢ | cᵢ |
        /// |---|---|---|
        /// | 1 | 4, 4, 4, 4 | 0.1 |
        /// | 2 | 1, 1, 1, 1 | 0.2 |
        /// | 3 | 8, 8, 8, 8 | 0.2 |
        /// | 4 | 6, 6, 6, 6 | 0.4 |
        /// | 5 | 3, 7, 3, 7 | 0.4 |
        /// | 6 | 2, 9, 2, 9 | 0.6 |
        /// | 7 | 5, 5, 3, 3 | 0.3 |
        /// | 8 | 8, 1, 8, 1 | 0.7 |
        /// | 9 | 6, 2, 6, 2 | 0.5 |
        /// | 10 | 7, 3.6, 7, 3.6 | 0.5 |
        ///
        /// Bounds [0, 10]⁴. The minimum is near a₁ = (4, 4, 4, 4), but not at it: the other wells
        /// pull it aside, and the value at (4, 4, 4, 4) is a little higher. The minimum and its
        /// solution are where the gradient is 0, computed to 40 digits by Newton's method and
        /// rounded; the best of the local minima that local searches from 2,000 random points
        /// find, one per well, but not proven global. Jamil and Yang (2013, functions 130-132)
        /// put the minima at (4, 4, 4, 4), with −10.1499, −10.3999 and −10.5319: neither the
        /// minima nor the values there.
        ///
        /// Shekel, J. (1971). Test functions for multimodal search techniques. *Proceedings of
        /// the 5th Annual Princeton Conference on Information Sciences and Systems*, Princeton
        /// University, and Dixon, L. C. W. and Szegö, G. P. (1978). The global optimisation
        /// problem: an introduction. In *Towards Global Optimisation 2*, North-Holland: 1-15, who
        /// call these functions SQRIN5, SQRIN7 and SQRIN10. Neither is online: the constants and
        /// the bounds as Yao, Liu and Lin (1999, f21-f23, table XIV) reprint them, whose appendix
        /// gives the wells' values as 1/cᵢ, without the minus sign.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub struct $name;

        impl FitnessFunction<Reals> for $name {
            type Output = f64;

            /// The value at `x`.
            ///
            /// # Panics
            ///
            /// If `x` has fewer than 4 genes.
            fn evaluate(&self, x: &Reals) -> f64 {
                shekel($m, x)
            }
        }

        impl Problem for $name {
            type Representation = Real;

            fn name(&self) -> &'static str {
                stringify!($name)
            }

            fn representation(&self) -> Real {
                uniform(4, 0.0, 10.0)
            }

            fn optimum(&self) -> Option<Optimum<Reals>> {
                Some(Optimum::best_known($value, vec![reals(&$solution)]))
            }

            fn reference(&self) -> &'static str {
                "Shekel, J. (1971). Test functions for multimodal search techniques. Proceedings \
                 of the 5th Annual Princeton Conference on Information Sciences and Systems, \
                 Princeton University. Constants as tabulated in Dixon, L. C. W. and Szegö, G. P. \
                 (1978). The global optimisation problem: an introduction. In Towards Global \
                 Optimisation 2, North-Holland: 1-15."
            }
        }
    };
}

shekel!(
    /// Shekel's function with m = 5 wells, in 4 dimensions (Dixon and Szegö's SQRIN5).
    ///
    /// Minimum −10.153199679058227 at (4.000037152819676, 4.00013327659156, 4.000037152819676,
    /// 4.00013327659156); later papers quote −10.1532 from
    /// Dixon and Szegö.
    Shekel5,
    5,
    -10.153_199_679_058_227,
    [
        4.000_037_152_819_676,
        4.000_133_276_591_56,
        4.000_037_152_819_676,
        4.000_133_276_591_56
    ]
);

shekel!(
    /// Shekel's function with m = 7 wells, in 4 dimensions (Dixon and Szegö's SQRIN7).
    ///
    /// Minimum −10.40294056681866 at (4.000572916185823, 4.000689366185305, 3.9994897088591506,
    /// 3.9996061588586316); later papers quote −10.4029 from
    /// Dixon and Szegö.
    Shekel7,
    7,
    -10.402_940_566_818_66,
    [
        4.000_572_916_185_823,
        4.000_689_366_185_305,
        3.999_489_708_859_150_6,
        3.999_606_158_858_631_6
    ]
);

shekel!(
    /// Shekel's function with m = 10 wells, in 4 dimensions (Dixon and Szegö's SQRIN10).
    ///
    /// Minimum −10.536409816692043 at (4.000746531592046, 4.000592934138532, 3.9996633980403224,
    /// 3.9995098005868077); later papers quote −10.5364 from
    /// Dixon and Szegö.
    Shekel10,
    10,
    -10.536_409_816_692_043,
    [
        4.000_746_531_592_046,
        4.000_592_934_138_532,
        3.999_663_398_040_322_4,
        3.999_509_800_586_807_7
    ]
);

// ---- more two-dimensional functions -------------------------------------------------------------

/// Easom's function, `−cos x₁ cos x₂ exp(−((x₁ − π)² + (x₂ − π)²))`: a single narrow well in a
/// flat plane.
///
/// Bounds [−100, 100]²; minimum −1 at (π, π). It's the global minimum: both factors are at most 1
/// in absolute value, and the exponential is 1 only at (π, π). Away from it, the function is
/// nearly 0: below 1e-10 in absolute value farther than 4.8 from (π, π), which is all of the box
/// but 0.2%.
///
/// Easom, E. E. (1990). *A Survey of Global Optimization Techniques.* M.Eng. thesis, University of
/// Louisville; probably first in a journal in Stuckman, B. E. and Easom, E. E. (1992). A
/// comparison of Bayesian/sampling global optimization techniques. *IEEE Transactions on Systems,
/// Man, and Cybernetics* 22(5): 1024-1032. Neither could be read: the definition and the bounds
/// as restated in Jamil and Yang (2013, function 50); not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Easom;

impl FitnessFunction<Reals> for Easom {
    type Output = f64;

    /// The value at `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 2 genes.
    fn evaluate(&self, x: &Reals) -> f64 {
        let (x1, x2) = (x[0], x[1]);
        let distance = math::powi(x1 - PI, 2) + math::powi(x2 - PI, 2);
        -math::cos(x1) * math::cos(x2) * math::exp(-distance)
    }
}

impl Problem for Easom {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Easom"
    }

    fn representation(&self) -> Real {
        uniform(2, -100.0, 100.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(-1.0, vec![reals(&[PI, PI])]))
    }

    fn reference(&self) -> &'static str {
        "Easom, E. E. (1990). A Survey of Global Optimization Techniques. M.Eng. thesis, \
         University of Louisville."
    }
}

/// The eggholder function, `−(x₂ + 47) sin √|x₂ + x₁ / 2 + 47| − x₁ sin √|x₁ − (x₂ + 47)|`:
/// deep local minima all over, the deepest at the edge of the box.
///
/// Bounds [−512, 512]²; minimum −959.6406627208508 at (512, 404.2318051137578), on the bound
/// x₁ = 512, where the derivative in x₂ is 0, computed to 40 digits by Newton's method and
/// rounded. The best of the local minima that local searches from the 40 lowest points of a
/// 4097 × 4097 grid reach, the next −956.9182316246655 at (482.3533104647884,
/// 432.87899894548343); not proven global.
///
/// Whitley, D., Mathias, K., Rana, S. and Dzubera, J. (1996). Evaluating evolutionary
/// algorithms. *Artificial Intelligence* 85(1-2): 245-276, section 4.2 (read in the authors'
/// copy), where it is F101, this formula on [−512, 511] with 10 bits per variable, with no
/// minimum given in 2 dimensions: on [−512, 511]², x₁ = 512 is outside the box, and the minimum
/// is the next one above. The name and the bounds [−512, 512] are those of Mishra, S. K. (2006).
/// Some new test functions for global optimization and performance of repulsive particle swarm
/// method. MPRA paper 2718, who gives the minimum as 959.64 at (512, 404.2319): the sign is
/// lost, and x₂ is 404.2318 to 4 decimals; Jamil and Yang (2013, function 66) repeat both.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Eggholder;

impl FitnessFunction<Reals> for Eggholder {
    type Output = f64;

    /// The value at `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 2 genes.
    fn evaluate(&self, x: &Reals) -> f64 {
        let (x1, x2) = (x[0], x[1]);
        -(x2 + 47.0) * math::sin((x2 + x1 / 2.0 + 47.0).abs().sqrt())
            - x1 * math::sin((x1 - (x2 + 47.0)).abs().sqrt())
    }
}

impl Problem for Eggholder {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "Eggholder"
    }

    fn representation(&self) -> Real {
        uniform(2, -512.0, 512.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::best_known(
            -959.640_662_720_850_8,
            vec![reals(&[512.0, 404.231_805_113_757_8])],
        ))
    }

    fn reference(&self) -> &'static str {
        "Whitley, D., Mathias, K., Rana, S. and Dzubera, J. (1996). Evaluating evolutionary \
         algorithms. Artificial Intelligence 85(1-2): 245-276."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1016/0004-3702(95)00124-7")
    }
}

/// Schaffer's F6, `0.5 + (sin² √(x₁² + x₂²) − 0.5) / (1 + 0.001 (x₁² + x₂²))²`: rings of local
/// minima around the global one.
///
/// Bounds [−100, 100]²; minimum 0 at the origin. It's the global minimum: the numerator is at
/// least −0.5 and the denominator at least 1, so the value is at least 0, and 0 only where the
/// denominator is 1. The function depends on the distance r from the origin only: its local
/// minima are rings near r = kπ, the first at r = 3.1384848 with 0.0097159.
///
/// Schaffer, J. D., Caruana, R. A., Eshelman, L. J. and Das, R. (1989). A study of control
/// parameters affecting online performance of genetic algorithms for function optimization.
/// *Proceedings of the Third International Conference on Genetic Algorithms*, Morgan Kaufmann:
/// 51-60, which couldn't be read. Definition and bounds as Whitley, Mathias, Rana and Dzubera
/// (1996, table 1, F9, "the sine envelope sine wave") restate it, crediting Schaffer et al.; the
/// CEC 2005 report (Suganthan et al. 2005, section 2.3.2) has the same function, and expands it
/// to n dimensions as its function 14. Not yet checked against the original
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SchafferF6;

impl FitnessFunction<Reals> for SchafferF6 {
    type Output = f64;

    /// The value at `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 2 genes.
    fn evaluate(&self, x: &Reals) -> f64 {
        let squared = x[0] * x[0] + x[1] * x[1];
        let numerator = math::powi(math::sin(squared.sqrt()), 2) - 0.5;
        0.5 + numerator / math::powi(1.0 + 0.001 * squared, 2)
    }
}

impl Problem for SchafferF6 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "SchafferF6"
    }

    fn representation(&self) -> Real {
        uniform(2, -100.0, 100.0)
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        Some(Optimum::proven(0.0, vec![reals(&[0.0, 0.0])]))
    }

    fn reference(&self) -> &'static str {
        "Schaffer, J. D., Caruana, R. A., Eshelman, L. J. and Das, R. (1989). A study of control \
         parameters affecting online performance of genetic algorithms for function \
         optimization. Proceedings of the Third International Conference on Genetic Algorithms, \
         Morgan Kaufmann: 51-60."
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamRng;
    use crate::genome::Representation;

    fn at(values: &[f64]) -> Reals {
        reals(values)
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} is not {expected}"
        );
    }

    // every solution of the optimum has its value, to 1e-12 relative
    fn check_optimum<P: Problem<Representation = Real, Output = f64>>(problem: &P) {
        let optimum = problem.optimum().expect("known");
        assert!(!optimum.solutions().is_empty());
        for solution in optimum.solutions() {
            problem
                .representation()
                .validate(solution)
                .expect("in bounds");
            assert_close(problem.evaluate(solution), optimum.value(), 1e-12);
        }
    }

    #[test]
    fn sphere() {
        check_optimum(&Sphere::new(5));
        // 1 + 4 + 9
        assert_eq!(Sphere::new(3).evaluate(&at(&[1.0, -2.0, 3.0])), 14.0);
        assert_eq!(Sphere::default().dimensions(), 30);
        assert_eq!(
            Sphere::default().representation().bounds()[0],
            -100.0..=100.0
        );
    }

    #[test]
    fn axis_parallel_ellipsoid() {
        check_optimum(&AxisParallelEllipsoid::new(5));
        let problem = AxisParallelEllipsoid::new(3);
        // 1·1 + 2·1 + 3·1
        assert_eq!(problem.evaluate(&at(&[1.0, 1.0, 1.0])), 6.0);
        // 3·2²
        assert_eq!(problem.evaluate(&at(&[0.0, 0.0, 2.0])), 12.0);
        assert_eq!(problem.representation().bounds()[2], -5.12..=5.12);
    }

    #[test]
    fn schwefel_1_2() {
        check_optimum(&Schwefel1_2::new(5));
        let problem = Schwefel1_2::new(3);
        // prefix sums 1, 2, 3: 1 + 4 + 9
        assert_eq!(problem.evaluate(&at(&[1.0, 1.0, 1.0])), 14.0);
        // prefix sums 1, 0, 1: not the same as the ellipsoid's 1 + 2 + 3
        assert_eq!(problem.evaluate(&at(&[1.0, -1.0, 1.0])), 2.0);
    }

    #[test]
    fn rastrigin() {
        check_optimum(&Rastrigin::new(5));
        assert_eq!(Rastrigin::new(4).evaluate(&at(&[0.0; 4])), 0.0);
        // 1 − 10 cos 2π + 10 = 1 in the first dimension, 0 in the others
        assert_close(
            Rastrigin::new(3).evaluate(&at(&[1.0, 0.0, 0.0])),
            1.0,
            1e-12,
        );
        // 0.25 − 10 cos π + 10 = 20.25 per dimension
        assert_close(Rastrigin::new(2).evaluate(&at(&[0.5, 0.5])), 40.5, 1e-12);
    }

    #[test]
    fn rosenbrock() {
        check_optimum(&Rosenbrock::new(5));
        // Rosenbrock's starting point: 100 (1 − 1.44)² + (−2.2)² = 19.36 + 4.84
        assert_close(Rosenbrock::new(2).evaluate(&at(&[-1.2, 1.0])), 24.2, 1e-12);
        // at the origin, (0 − 1)² in each of the n − 1 terms
        assert_eq!(Rosenbrock::new(6).evaluate(&at(&[0.0; 6])), 5.0);
    }

    #[test]
    #[should_panic(expected = "Rosenbrock needs at least 2 dimensions")]
    fn rosenbrock_needs_two_dimensions() {
        let _ = Rosenbrock::new(1);
    }

    #[test]
    #[should_panic(expected = "Sphere needs at least 1 dimensions")]
    fn scalable_functions_need_a_dimension() {
        let _ = Sphere::new(0);
    }

    #[test]
    fn ackley() {
        check_optimum(&Ackley::new(5));
        assert!(Ackley::new(3).evaluate(&at(&[0.0; 3])).abs() < 1e-15);
        // at (1, 1): the mean square is 1 and cos 2π = 1, so −20 e^−0.2 − e + 20 + e
        let expected = 20.0 * (1.0 - math::exp(-0.2f64));
        assert_close(Ackley::new(2).evaluate(&at(&[1.0, 1.0])), expected, 1e-12);
        assert_eq!(Ackley::default().representation().bounds()[0], -32.0..=32.0);
    }

    #[test]
    fn griewank() {
        check_optimum(&Griewank::new(5));
        // 1 + π² / 4000 − cos π
        let expected = 2.0 + PI * PI / 4000.0;
        assert_close(Griewank::new(1).evaluate(&at(&[PI])), expected, 1e-12);
        // at (0, π√2), the second gene is divided by √2 in its cosine:
        // 1 + 2π² / 4000 − cos 0 · cos π = 2 + 2π² / 4000
        let x2 = PI * 2f64.sqrt();
        assert_close(
            Griewank::new(2).evaluate(&at(&[0.0, x2])),
            2.0 + 2.0 * PI * PI / 4000.0,
            1e-12,
        );
    }

    #[test]
    fn schwefel_2_26() {
        check_optimum(&Schwefel2_26::new(5));
        // √x = π/2: −x sin(π/2) = −π²/4, and the opposite for −x
        let x = PI * PI / 4.0;
        assert_close(Schwefel2_26::new(1).evaluate(&at(&[x])), -x, 1e-12);
        assert_close(Schwefel2_26::new(2).evaluate(&at(&[x, -x])), 0.0, 1e-12);
        // Yao, Liu and Lin's minimum for n = 30, −12569.5, to its digits
        let optimum = Schwefel2_26::new(30).optimum().expect("known").value();
        assert_eq!((optimum * 10.0).round() / 10.0, -12569.5);
        // the term's derivative, sin s + (s / 2) cos s with s = √x, is 0 at the minimizer
        let s = SCHWEFEL_2_26_X.sqrt();
        assert!((math::sin(s) + s / 2.0 * math::cos(s)).abs() < 1e-12);
        // and the ends of the bounds are worse: −500 sin √500 ≈ −180.6
        let ends = [-500.0, 500.0].map(|x| Schwefel2_26::new(1).evaluate(&at(&[x])));
        assert!(ends.iter().all(|&end| end > SCHWEFEL_2_26_MIN + 200.0));
    }

    #[test]
    fn levy() {
        check_optimum(&Levy::new(5));
        // x = 3, w = 1.5, in one dimension: sin²(1.5π) + 0.5² (1 + sin²(3π)) = 1 + 0.25
        assert_close(Levy::new(1).evaluate(&at(&[3.0])), 1.25, 1e-12);
        // x = (1, 5), w = (1, 2): sin² π + 0 + 1² (1 + sin² 4π) = 1
        assert_close(Levy::new(2).evaluate(&at(&[1.0, 5.0])), 1.0, 1e-12);
        // x = (5, 1), w = (2, 1): sin² 2π + 1² (1 + 10 sin²(2π + 1)) + 0
        let expected = 1.0 + 10.0 * math::powi(math::sin(1f64), 2);
        assert_close(Levy::new(2).evaluate(&at(&[5.0, 1.0])), expected, 1e-12);
    }

    #[test]
    fn zakharov() {
        check_optimum(&Zakharov::new(5));
        // Σ x² = 2 and Σ 0.5 i x = 0.5 + 1 = 1.5: 2 + 2.25 + 5.0625
        assert_eq!(Zakharov::new(2).evaluate(&at(&[1.0, 1.0])), 9.3125);
        assert_eq!(
            Zakharov::default().representation().bounds()[0],
            -5.0..=10.0
        );
    }

    #[test]
    fn styblinski_tang() {
        check_optimum(&StyblinskiTang::new(5));
        // ½ (1 − 16 + 5) and ½ (16 − 64 + 10)
        assert_eq!(StyblinskiTang::new(1).evaluate(&at(&[1.0])), -5.0);
        assert_eq!(StyblinskiTang::new(2).evaluate(&at(&[1.0, 2.0])), -24.0);
        // the minimizer is a root of the derivative 4x³ − 32x + 5
        let x = STYBLINSKI_TANG_X;
        assert!((4.0 * math::powi(x, 3) - 32.0 * x + 5.0).abs() < 1e-12);
        // the value that Jamil and Yang give for two dimensions, −78.332, to its digits
        let optimum = StyblinskiTang::new(2).optimum().expect("known").value();
        assert_eq!((optimum * 1000.0).round() / 1000.0, -78.332);
        // the other minimum of a term, near 2.75, is worse
        let other = StyblinskiTang::new(1).evaluate(&at(&[2.746_802_770_990_837]));
        assert!(other > STYBLINSKI_TANG_MIN + 14.0);
    }

    #[test]
    fn michalewicz() {
        check_optimum(&Michalewicz::new(5));
        // sin(π/2) sin²⁰(π/4) = (1/√2)²⁰ = 2⁻¹⁰
        let half_pi = PI / 2.0;
        assert_close(
            Michalewicz::new(1).evaluate(&at(&[half_pi])),
            -1.0 / 1024.0,
            1e-12,
        );
        // the second term: sin(π/2) sin²⁰(2 (π/2)² / π) = sin²⁰(π/2) = 1
        assert_close(
            Michalewicz::new(2).evaluate(&at(&[half_pi, half_pi])),
            -1.0 - 1.0 / 1024.0,
            1e-12,
        );
        // the minima that Molga and Smutnicki give, −4.687 (n = 5) and −9.66 (n = 10), and the
        // common −1.8013 at about (2.20, 1.57) for n = 2, to their digits
        let value = |n: usize| Michalewicz::new(n).optimum().expect("known").value();
        assert_eq!((value(2) * 1e4).round() / 1e4, -1.8013);
        assert_eq!((value(5) * 1e3).trunc() / 1e3, -4.687);
        assert_eq!((value(10) * 1e2).trunc() / 1e2, -9.66);
        let solution = Michalewicz::new(2).optimum().expect("known").solutions()[0].clone();
        assert_eq!((solution[0] * 100.0).round() / 100.0, 2.2);
        assert_eq!((solution[1] * 100.0).round() / 100.0, 1.57);
    }

    // the minimizer of each term beats a dense grid over [0, π]
    #[test]
    fn michalewicz_terms_are_minimized_globally() {
        for i in [0, 1, 4, 9, 29, 99] {
            let best = michalewicz_term(i, michalewicz_minimizer(i));
            let points = 200_000;
            let grid = (0..=points)
                .map(|k| michalewicz_term(i, PI * k as f64 / points as f64))
                .fold(f64::INFINITY, f64::min);
            assert!(best <= grid + 1e-12, "term {i}: {best} > {grid}");
        }
    }

    #[test]
    fn himmelblau() {
        check_optimum(&Himmelblau);
        // (0 + 0 − 11)² + (0 + 0 − 7)²
        assert_eq!(Himmelblau.evaluate(&at(&[0.0, 0.0])), 170.0);
        assert_eq!(Himmelblau.optimum().expect("known").solutions().len(), 4);
        // each minimum solves both equations
        for solution in Himmelblau.optimum().expect("known").solutions() {
            let (x1, x2) = (solution[0], solution[1]);
            assert!((x1 * x1 + x2 - 11.0).abs() < 1e-14);
            assert!((x1 + x2 * x2 - 7.0).abs() < 1e-14);
        }
    }

    #[test]
    fn branin() {
        check_optimum(&Branin);
        // (0 − 0 + 0 − 6)² + 10 (1 − 1/(8π)) + 10 = 56 − 10/(8π)
        assert_close(
            Branin.evaluate(&at(&[0.0, 0.0])),
            56.0 - 5.0 / (4.0 * PI),
            1e-12,
        );
        // the misprinted third minimum, (3π, 2.425), is 0.05² above the minimum
        let misprint = Branin.evaluate(&at(&[3.0 * PI, 2.425]));
        assert_close(misprint, 5.0 / (4.0 * PI) + 0.0025, 1e-12);
        let bounds = Branin.representation();
        assert_eq!(bounds.bounds(), [-5.0..=10.0, 0.0..=15.0]);
    }

    #[test]
    fn goldstein_price() {
        check_optimum(&GoldsteinPrice);
        // the paper's local minima
        for (x, value) in [
            ([1.2, 0.8], 840.0),
            ([1.8, 0.2], 84.0),
            ([-0.6, -0.4], 30.0),
        ] {
            assert_close(GoldsteinPrice.evaluate(&at(&x)), value, 1e-12);
        }
        // at the origin: (1 + 1 · 19) (30 + 0)
        assert_eq!(GoldsteinPrice.evaluate(&at(&[0.0, 0.0])), 600.0);
    }

    #[test]
    fn six_hump_camel() {
        check_optimum(&SixHumpCamel);
        assert_eq!(SixHumpCamel.evaluate(&at(&[0.0, 0.0])), 0.0);
        // (4 − 2.1 + 1/3) + 1 + 0 = 97/30
        assert_close(SixHumpCamel.evaluate(&at(&[1.0, 1.0])), 97.0 / 30.0, 1e-12);
        // Yao, Liu and Lin's minimum, −1.0316285, to its digits
        let optimum = SixHumpCamel.optimum().expect("known").value();
        assert_eq!((optimum * 1e7).round() / 1e7, -1.031_628_5);
        // the gradient is 0 at the minima
        for solution in SixHumpCamel.optimum().expect("known").solutions() {
            let (x1, x2) = (solution[0], solution[1]);
            let dx1 = 8.0 * x1 - 8.4 * math::powi(x1, 3) + 2.0 * math::powi(x1, 5) + x2;
            let dx2 = x1 - 8.0 * x2 + 16.0 * math::powi(x2, 3);
            assert!(dx1.abs() < 1e-14 && dx2.abs() < 1e-14);
        }
    }

    // the gradient at `x` by central differences, with steps of `h`
    fn gradient(f: impl Fn(&Reals) -> f64, x: &[f64], h: f64) -> Vec<f64> {
        (0..x.len())
            .map(|i| {
                let (mut up, mut down) = (x.to_vec(), x.to_vec());
                up[i] += h;
                down[i] -= h;
                (f(&at(&up)) - f(&at(&down))) / (2.0 * h)
            })
            .collect()
    }

    // `value` rounded to `decimals` decimals
    fn rounded(value: f64, decimals: i32) -> f64 {
        let scale = 10f64.powi(decimals);
        (value * scale).round() / scale
    }

    #[test]
    fn hartmann_3() {
        check_optimum(&Hartmann3);
        let optimum = Hartmann3.optimum().expect("known");
        assert!(!optimum.is_proven());
        // the minimum that later papers quote from Dixon and Szegö, −3.86278, and the point that
        // they quote, (0.114614, 0.555649, 0.852547), to their digits
        assert_eq!(rounded(optimum.value(), 5), -3.86278);
        let solution = &optimum.solutions()[0];
        for (x, reported) in solution.iter().zip([0.114_614, 0.555_649, 0.852_547]) {
            assert_eq!(rounded(*x, 6), reported);
        }
        // with p₄₁ = 0.0381, the minimum would be −3.8627798 at x₁ = 0.11459: not those digits
        let mut p = HARTMANN_3_P;
        p[3][0] = 0.0381;
        let shifted = hartmann(&HARTMANN_3_A, &p, &at(&[0.114_588_9, 0.555_649, 0.852_547]));
        assert!(rounded(shifted, 6) != rounded(optimum.value(), 6));
        // the gradient is 0 there, and at the other local minima, which are worse
        let f = |x: &Reals| Hartmann3.evaluate(x);
        let others = [
            (
                [
                    0.109_337_500_835_031_16,
                    0.860_524_221_769_077_3,
                    0.564_123_171_053_538,
                ],
                -3.089_764_163_092_250_5,
            ),
            (
                [
                    0.368_722_727_070_440_8,
                    0.117_561_628_825_735_62,
                    0.267_573_743_016_828_9,
                ],
                -1.000_816_863_562_928_2,
            ),
        ];
        for x in [solution.to_vec()]
            .into_iter()
            .chain(others.iter().map(|(x, _)| x.to_vec()))
        {
            assert!(gradient(f, &x, 1e-6).iter().all(|g| g.abs() < 1e-7));
        }
        for (x, value) in others {
            assert_close(f(&at(&x)), value, 1e-12);
        }
        // each term is positive and at most cᵢ: the value is between −Σ cᵢ = −8.4 and 0, and at
        // a center pᵢ, at most −cᵢ
        for (p, c) in HARTMANN_3_P.iter().zip(HARTMANN_C) {
            let value = f(&at(p));
            assert!(value <= -c && value > -8.4);
        }
        assert_eq!(Hartmann3.representation().bounds(), vec![0.0..=1.0; 3]);
    }

    #[test]
    fn hartmann_6() {
        check_optimum(&Hartmann6);
        let optimum = Hartmann6.optimum().expect("known");
        assert!(!optimum.is_proven());
        // the minimum that later papers quote from Dixon and Szegö, −3.32237, and the point that
        // Jamil and Yang give, (0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657301), to
        // their digits
        assert_eq!(rounded(optimum.value(), 5), -3.32237);
        let solution = &optimum.solutions()[0];
        let reported = [
            0.201_690, 0.150_011, 0.476_874, 0.275_332, 0.311_652, 0.657_301,
        ];
        for (x, reported) in solution.iter().zip(reported) {
            assert_eq!(rounded(*x, 6), reported);
        }
        // Yao, Liu and Lin's p₃₂ = 0.1415 moves the minimizer's x₂ to 0.1468, away from theirs
        let mut p = HARTMANN_6_P;
        p[2][1] = 0.1415;
        let misprint = |x: &Reals| hartmann(&HARTMANN_6_A, &p, x);
        let moved = at(&[
            0.201_708, 0.146_781, 0.476_745, 0.275_342, 0.311_652, 0.657_275,
        ]);
        assert!(
            gradient(misprint, &moved, 1e-6)
                .iter()
                .all(|g| g.abs() < 1e-4)
        );
        assert!(misprint(&moved) > optimum.value() + 3e-4);
        // the gradient is 0 there, and at the other local minimum, which is worse
        let f = |x: &Reals| Hartmann6.evaluate(x);
        let other = [
            0.404_653_127_706_171_3,
            0.882_444_923_972_383_2,
            0.846_101_570_478_180_8,
            0.573_989_692_430_365_8,
            0.138_926_603_667_243_88,
            0.038_495_892_478_237_27,
        ];
        assert_close(f(&at(&other)), -3.203_161_918_396_231, 1e-12);
        for x in [&solution[..], &other] {
            assert!(gradient(f, x, 1e-6).iter().all(|g| g.abs() < 1e-7));
        }
        for (p, c) in HARTMANN_6_P.iter().zip(HARTMANN_C) {
            let value = f(&at(p));
            assert!(value <= -c && value > -8.4);
        }
    }

    #[test]
    fn shekel_5_7_and_10() {
        check_optimum(&Shekel5);
        check_optimum(&Shekel7);
        check_optimum(&Shekel10);
        // at (4, 4, 4, 4), the squared distances to a₁ … a₁₀: 0, 4 · 9, 4 · 16, 4 · 4,
        // 1 + 9 + 1 + 9, 4 + 25 + 4 + 25, 1 + 1 + 1 + 1, 9 + 16 + 9 + 16, 4 · 4 and
        // 9 + 0.16 + 9 + 0.16
        let distances = [0.0, 36.0, 64.0, 16.0, 20.0, 58.0, 4.0, 50.0, 16.0, 18.32];
        let center = at(&[4.0; 4]);
        let values = [
            (5, Shekel5.evaluate(&center), Shekel5.optimum(), -10.1532),
            (7, Shekel7.evaluate(&center), Shekel7.optimum(), -10.4029),
            (10, Shekel10.evaluate(&center), Shekel10.optimum(), -10.5364),
        ];
        for (m, value, optimum, reported) in values {
            let expected: f64 = -(0..m)
                .map(|i| 1.0 / (distances[i] + SHEKEL_C[i]))
                .sum::<f64>();
            assert_close(value, expected, 1e-12);
            let optimum = optimum.expect("known");
            assert!(!optimum.is_proven());
            // the minimum is near (4, 4, 4, 4), and below the value there
            assert!(optimum.value() < value - 1e-6);
            let solution = &optimum.solutions()[0];
            assert!(solution.iter().all(|x| (x - 4.0).abs() < 1e-3));
            // the value that later papers quote from Dixon and Szegö, to its digits, and not the
            // values that Jamil and Yang give at (4, 4, 4, 4)
            assert!(![-10.1499, -10.3999, -10.5319].contains(&rounded(value, 4)));
            assert_eq!(rounded(optimum.value(), 4), reported);
            // the gradient is 0 there
            let f = |x: &Reals| shekel(m, x);
            assert!(gradient(f, solution, 1e-6).iter().all(|g| g.abs() < 1e-7));
        }
        // the wells at a₂ and a₃ are local minima, near −1/0.2 = −5
        assert!((Shekel5.evaluate(&at(&[1.0; 4])) + 5.0).abs() < 0.1);
        assert!((Shekel10.evaluate(&at(&[8.0; 4])) + 5.0).abs() < 0.2);
        assert_eq!(Shekel7.representation().bounds(), vec![0.0..=10.0; 4]);
    }

    #[test]
    fn easom() {
        check_optimum(&Easom);
        assert_eq!(Easom.evaluate(&at(&[PI, PI])), -1.0);
        // at the origin: −cos 0 cos 0 exp(−2π²)
        let origin = Easom.evaluate(&at(&[0.0, 0.0]));
        assert_close(origin, -math::exp(-2.0 * PI * PI), 1e-12);
        // at (π, 3π/2), cos x₂ is 0 (to rounding)
        assert!(Easom.evaluate(&at(&[PI, 1.5 * PI])).abs() < 1e-15);
        // far away, exp underflows to 0: the plane is flat
        assert_eq!(Easom.evaluate(&at(&[-100.0, 100.0])), 0.0);
        // 4.8 from (π, π), the value is below 1e-10
        assert!(Easom.evaluate(&at(&[PI + 4.8, PI])).abs() < 1e-10);
        assert_eq!(Easom.representation().bounds(), vec![-100.0..=100.0; 2]);
    }

    #[test]
    fn eggholder() {
        check_optimum(&Eggholder);
        let optimum = Eggholder.optimum().expect("known");
        assert!(!optimum.is_proven());
        // the value that Jamil and Yang give, with its sign corrected, to its digits; their x₂,
        // 404.2319, is 404.2318 to 4 decimals
        assert_eq!(rounded(optimum.value(), 4), -959.6407);
        let solution = &optimum.solutions()[0];
        assert_eq!(solution[0], 512.0);
        assert_eq!(rounded(solution[1], 4), 404.2318);
        // on the bound: the derivative in x₂ is 0, and the one in x₁ is negative (the value would
        // go on falling beyond x₁ = 512)
        let f = |x: &Reals| Eggholder.evaluate(x);
        let slope = gradient(f, solution, 1e-6);
        assert!(slope[1].abs() < 1e-6 && slope[0] < -1.0);
        // the next local minimum, inside the box
        let next = at(&[482.353_310_464_788_4, 432.878_998_945_483_43]);
        assert_close(f(&next), -956.918_231_624_665_5, 1e-12);
        assert!(gradient(f, &next, 1e-6).iter().all(|g| g.abs() < 1e-5));
        // at the origin: −47 sin √47 − 0
        let origin = f(&at(&[0.0, 0.0]));
        assert_close(origin, -47.0 * math::sin(47f64.sqrt()), 1e-12);
    }

    #[test]
    fn schaffer_f6() {
        check_optimum(&SchafferF6);
        assert_eq!(SchafferF6.evaluate(&at(&[0.0, 0.0])), 0.0);
        // at a distance of π/2, sin² = 1: 0.5 + 0.5 / (1 + 0.001 π² / 4)²
        let r = PI / 2.0;
        let expected = 0.5 + 0.5 / math::powi(1.0 + 0.001 * r * r, 2);
        assert_close(SchafferF6.evaluate(&at(&[r, 0.0])), expected, 1e-12);
        // it depends on the distance only: (3, 4) and (0, −5) have the same x₁² + x₂²
        assert_eq!(
            SchafferF6.evaluate(&at(&[3.0, 4.0])),
            SchafferF6.evaluate(&at(&[0.0, -5.0]))
        );
        // the first ring of local minima, at r = 3.1384848, is 0.0097159 above the minimum
        let ring = SchafferF6.evaluate(&at(&[3.138_484_821_601_593, 0.0]));
        assert_close(ring, 0.009_715_909_877_514_548, 1e-12);
        assert_eq!(
            SchafferF6.representation().bounds(),
            vec![-100.0..=100.0; 2]
        );
    }

    // sign changes and permutations of the genes don't change the functions that are symmetric
    #[test]
    fn symmetric_functions_are_symmetric() {
        let mut rng = StreamRng::seed_from_u64(7);
        let real = Real::uniform(6, -5.0..=5.0).expect("valid");
        for _ in 0..200 {
            let x = real.random_genome(&mut rng);
            let negated: Reals = x.iter().map(|xi| -xi).collect();
            let reversed: Reals = x.iter().rev().copied().collect();
            let n = x.len();
            let sign = |f: &dyn Fn(&Reals) -> f64| assert_close(f(&x), f(&negated), 1e-12);
            let order = |f: &dyn Fn(&Reals) -> f64| assert_close(f(&x), f(&reversed), 1e-12);
            sign(&|x| Sphere::new(n).evaluate(x));
            sign(&|x| Rastrigin::new(n).evaluate(x));
            sign(&|x| Ackley::new(n).evaluate(x));
            sign(&|x| Griewank::new(n).evaluate(x));
            sign(&|x| AxisParallelEllipsoid::new(n).evaluate(x));
            order(&|x| Sphere::new(n).evaluate(x));
            order(&|x| Rastrigin::new(n).evaluate(x));
            order(&|x| Ackley::new(n).evaluate(x));
            order(&|x| StyblinskiTang::new(n).evaluate(x));
            order(&|x| Schwefel2_26::new(n).evaluate(x));
            let pair = at(&[x[0], x[1]]);
            let opposite = at(&[-x[0], -x[1]]);
            assert_close(
                SixHumpCamel.evaluate(&pair),
                SixHumpCamel.evaluate(&opposite),
                1e-12,
            );
        }
    }
}
