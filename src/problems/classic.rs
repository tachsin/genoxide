//! The classic continuous functions: scalable ones, which take the number of dimensions, and
//! two-dimensional ones.

use super::{Optimum, Problem};
use crate::engine::FitnessFunction;
use crate::genome::{Real, Reals};
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
    /// in Yao, Liu and Lin (1999, f1); not yet checked against the original
    /// ([#168](https://github.com/tachsin/genoxide/issues/168)).
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
    /// Its origin is unknown: definition and bounds as restated in Molga and Smutnicki (2005,
    /// section 2.2); not yet checked against an original
    /// ([#168](https://github.com/tachsin/genoxide/issues/168)).
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
    /// Schwefel, H.-P. (1981). *Numerical Optimization of Computer Models.* Wiley, problem 1.2.
    /// Definition and bounds as restated in Yao, Liu and Lin (1999, f3); not yet checked against
    /// the original ([#168](https://github.com/tachsin/genoxide/issues/168)).
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
    /// Rastrigin, L. A. (1974). *Systems of Extremal Control.* Nauka, Moscow (two dimensions);
    /// generalized to n dimensions by Mühlenbein, H., Schomisch, M. and Born, J. (1991). The
    /// parallel genetic algorithm as function optimizer. *Parallel Computing* 17(6-7): 619-632.
    /// doi:10.1016/S0167-8191(05)80052-3. Definition and bounds as restated in Yao, Liu and Lin
    /// (1999, f9); not yet checked against the original
    /// ([#168](https://github.com/tachsin/genoxide/issues/168)).
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
    /// Optimization Theory and Applications* 34(1): 11-39. The divisor 4000 and the bounds are as
    /// restated in Yao, Liu and Lin (1999, f11); the original's divisor and domain are not yet
    /// checked ([#168](https://github.com/tachsin/genoxide/issues/168)).
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
    /// Schwefel, H.-P. (1981). *Numerical Optimization of Computer Models.* Wiley, problem 2.26.
    /// Definition and bounds as restated in Yao, Liu and Lin (1999, f8), whose minimum is −12569.5
    /// for n = 30; not yet checked against the original
    /// ([#168](https://github.com/tachsin/genoxide/issues/168)). This is the form without an
    /// offset: variants add 418.9829 n (a minimum near 0) or divide by n.
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
    /// 6(1): 15-29, but several functions carry Levy's name. Definition and bounds as restated
    /// in Laguna and Martí (2005, function 38), who print xₙ instead of wₙ in the last sine; this
    /// uses wₙ, like the other terms. Not yet checked against the original
    /// ([#168](https://github.com/tachsin/genoxide/issues/168)).
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
                .map(|xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
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
                100.0 * (next - xi * xi).powi(2) + (xi - 1.0).powi(2)
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
        let cosines = x.iter().map(|xi| (2.0 * PI * xi).cos()).sum::<f64>() / n;
        // in this order, 0 at the origin: 20 − 20 and e − e cancel
        20.0 - 20.0 * (-0.2 * squares.sqrt()).exp() + E - cosines.exp()
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
            .map(|(i, xi)| (xi / ((i + 1) as f64).sqrt()).cos())
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
        -x.iter().map(|xi| xi * xi.abs().sqrt().sin()).sum::<f64>()
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
                (wi - 1.0).powi(2) * (1.0 + 10.0 * (PI * wi + 1.0).sin().powi(2))
            })
            .sum();
        (PI * first).sin().powi(2)
            + middle
            + (last - 1.0).powi(2) * (1.0 + (2.0 * PI * last).sin().powi(2))
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
        squares + weighted.powi(2) + weighted.powi(4)
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
            .map(|xi| xi.powi(4) - 16.0 * xi * xi + 5.0 * xi)
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
    -xi.sin() * (index * xi * xi / PI).sin().powi(2 * MICHALEWICZ_M)
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
            low.sin().max(high.sin())
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
        (x1 * x1 + x2 - 11.0).powi(2) + (x1 + x2 * x2 - 7.0).powi(2)
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
        (x2 - b * x1 * x1 + c * x1 - 6.0).powi(2) + 10.0 * (1.0 - t) * x1.cos() + 10.0
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
/// 840, (1.8, 0.2) with 84 and (−0.6, −0.4) with 30.
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
            + (x1 + x2 + 1.0).powi(2)
                * (19.0 - 14.0 * x1 + 3.0 * x1 * x1 - 14.0 * x2 + 6.0 * x1 * x2 + 3.0 * x2 * x2);
        let b = 30.0
            + (2.0 * x1 - 3.0 * x2).powi(2)
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
/// −0.7126564030207396): stationary points computed to 40 digits by Newton's method and rounded,
/// which agree with the 8 digits Yao, Liu and Lin give.
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
        let expected = 20.0 * (1.0 - (-0.2f64).exp());
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
        assert!((s.sin() + s / 2.0 * s.cos()).abs() < 1e-12);
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
        let expected = 1.0 + 10.0 * 1f64.sin().powi(2);
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
        assert!((4.0 * x.powi(3) - 32.0 * x + 5.0).abs() < 1e-12);
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
            let dx1 = 8.0 * x1 - 8.4 * x1.powi(3) + 2.0 * x1.powi(5) + x2;
            let dx2 = x1 - 8.0 * x2 + 16.0 * x2.powi(3);
            assert!(dx1.abs() < 1e-14 && dx2.abs() < 1e-14);
        }
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
