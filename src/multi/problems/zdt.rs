//! ZDT1-4 and ZDT6: Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective
//! evolutionary algorithms: empirical results. *Evolutionary Computation* 8(2): 173-195,
//! definition 4 (eqs. 6-10 and 12).

use super::{MultiProblem, evenly};
use crate::genome::{Real, Reals};
use crate::math;
use crate::multi::MultiFitnessFunction;
use std::f64::consts::PI;

const REFERENCE: &str = "Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective \
                         evolutionary algorithms: empirical results. Evolutionary Computation \
                         8(2): 173-195.";
const REFERENCE_URL: &str = "https://doi.org/10.1162/106365600568202";

// the mean of the variables from the second on, ZDT's usual g - 1 over 9
fn tail_mean(x: &[f64]) -> f64 {
    x[1..].iter().sum::<f64>() / (x.len() - 1) as f64
}

// `points` points of a front f₂ = f2(f₁), with f₁ evenly spread from `start` to 1
fn zdt_front(points: usize, f2: impl Fn(f64) -> f64, start: f64) -> Vec<[f64; 2]> {
    (0..points)
        .map(|i| {
            let f1 = start + (1.0 - start) * evenly(i, points);
            [f1, f2(f1)]
        })
        .collect()
}

macro_rules! zdt {
    ($(#[$doc:meta])* $name:ident, $label:literal, $default:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name {
            variables: usize,
        }

        impl $name {
            #[doc = concat!("The problem with `variables` variables, at least 2; ", stringify!($default), " is standard.")]
            ///
            /// # Panics
            ///
            /// If `variables` is below 2.
            pub fn new(variables: usize) -> Self {
                assert!(variables >= 2, "{} needs at least 2 variables", $label);
                Self { variables }
            }

            /// The number of variables.
            pub fn variables(&self) -> usize {
                self.variables
            }
        }

        impl Default for $name {
            #[doc = concat!("The standard problem, with ", stringify!($default), " variables.")]
            fn default() -> Self {
                Self::new($default)
            }
        }
    };
}

zdt!(
    /// ZDT1: a convex front, `f₂ = 1 − √f₁` for f₁ in [0, 1].
    ///
    /// `f₁ = x₁`, `g = 1 + 9 Σᵢ₌₂ⁿ xᵢ / (n − 1)`, `f₂ = g (1 − √(f₁ / g))`, on [0, 1]ⁿ. The
    /// optimal solutions have x₂ = … = xₙ = 0.
    Zdt1,
    "ZDT1",
    30
);
zdt!(
    /// ZDT2: a concave front, `f₂ = 1 − f₁²` for f₁ in [0, 1].
    ///
    /// As [`Zdt1`], with `f₂ = g (1 − (f₁ / g)²)`.
    Zdt2,
    "ZDT2",
    30
);
zdt!(
    /// ZDT3: a front of five disconnected pieces of `f₂ = 1 − √f₁ − f₁ sin(10π f₁)`.
    ///
    /// As [`Zdt1`], with `f₂ = g (1 − √(f₁ / g) − (f₁ / g) sin(10π f₁))`.
    Zdt3,
    "ZDT3",
    30
);
zdt!(
    /// ZDT4: the convex front of ZDT1 behind 21⁹ local fronts.
    ///
    /// `f₁ = x₁` in [0, 1], `g = 1 + 10 (n − 1) + Σᵢ₌₂ⁿ (xᵢ² − 10 cos 4πxᵢ)` with the other
    /// variables in [−5, 5] (Rastrigin's function), `f₂ = g (1 − √(f₁ / g))`.
    Zdt4,
    "ZDT4",
    10
);
zdt!(
    /// ZDT6: a concave front, `f₂ = 1 − f₁²` for f₁ from 0.2807753188 to 1, with solutions
    /// dense near its upper end.
    ///
    /// `f₁ = 1 − exp(−4x₁) sin⁶(6πx₁)`, `g = 1 + 9 (Σᵢ₌₂ⁿ xᵢ / (n − 1))^0.25`,
    /// `f₂ = g (1 − (f₁ / g)²)`, on [0, 1]ⁿ.
    Zdt6,
    "ZDT6",
    10
);

macro_rules! zdt_problem {
    ($name:ident, $label:literal) => {
        impl MultiProblem<2> for $name {
            type Representation = Real;

            fn name(&self) -> &'static str {
                $label
            }

            fn representation(&self) -> Real {
                self.real()
            }

            fn reference(&self) -> &'static str {
                REFERENCE
            }

            fn reference_url(&self) -> Option<&'static str> {
                Some(REFERENCE_URL)
            }

            fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
                Some(self.front(points))
            }

            fn ideal_point(&self) -> Option<[f64; 2]> {
                Some(self.ideal())
            }

            fn nadir_point(&self) -> Option<[f64; 2]> {
                Some(self.nadir())
            }
        }
    };
}

zdt_problem!(Zdt1, "ZDT1");
zdt_problem!(Zdt2, "ZDT2");
zdt_problem!(Zdt3, "ZDT3");
zdt_problem!(Zdt4, "ZDT4");
zdt_problem!(Zdt6, "ZDT6");

// the bounds of ZDT1-3 and ZDT6
fn unit_box(variables: usize) -> Real {
    Real::uniform(variables, 0.0..=1.0).expect("valid bounds")
}

impl MultiFitnessFunction<Reals, 2> for Zdt1 {
    type Output = [f64; 2];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` is empty.
    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let g = 1.0 + 9.0 * tail_mean(x);
        [x[0], g * (1.0 - (x[0] / g).sqrt())]
    }
}

impl Zdt1 {
    fn real(&self) -> Real {
        unit_box(self.variables)
    }

    fn front(&self, points: usize) -> Vec<[f64; 2]> {
        zdt_front(points, |f1| 1.0 - f1.sqrt(), 0.0)
    }

    fn ideal(&self) -> [f64; 2] {
        [0.0, 0.0]
    }

    fn nadir(&self) -> [f64; 2] {
        [1.0, 1.0]
    }
}

impl MultiFitnessFunction<Reals, 2> for Zdt2 {
    type Output = [f64; 2];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` is empty.
    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let g = 1.0 + 9.0 * tail_mean(x);
        [x[0], g * (1.0 - (x[0] / g) * (x[0] / g))]
    }
}

impl Zdt2 {
    fn real(&self) -> Real {
        unit_box(self.variables)
    }

    fn front(&self, points: usize) -> Vec<[f64; 2]> {
        zdt_front(points, |f1| 1.0 - f1 * f1, 0.0)
    }

    fn ideal(&self) -> [f64; 2] {
        [0.0, 0.0]
    }

    fn nadir(&self) -> [f64; 2] {
        [1.0, 1.0]
    }
}

// the pieces of ZDT3's optimal front, as ranges of f1
pub(super) const ZDT3_PIECES: [(f64, f64); 5] = [
    (0.0, 0.083_001_534_9),
    (0.182_228_728_0, 0.257_762_363_4),
    (0.409_313_674_8, 0.453_882_104_1),
    (0.618_396_794_4, 0.652_511_703_8),
    (0.823_331_798_3, 0.851_832_865_4),
];

// ZDT3's front, f₂ as a function of f₁
fn zdt3_f2(f1: f64) -> f64 {
    1.0 - f1.sqrt() - f1 * math::sin(10.0 * PI * f1)
}

impl MultiFitnessFunction<Reals, 2> for Zdt3 {
    type Output = [f64; 2];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` is empty.
    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let g = 1.0 + 9.0 * tail_mean(x);
        let ratio = x[0] / g;
        [
            x[0],
            g * (1.0 - ratio.sqrt() - ratio * math::sin(10.0 * PI * x[0])),
        ]
    }
}

impl Zdt3 {
    fn real(&self) -> Real {
        unit_box(self.variables)
    }

    fn front(&self, points: usize) -> Vec<[f64; 2]> {
        // spread over the pieces in proportion to their widths
        let total: f64 = ZDT3_PIECES.iter().map(|(low, high)| high - low).sum();
        (0..points)
            .map(|i| {
                let mut position = evenly(i, points) * total;
                let mut f1 = ZDT3_PIECES[4].1;
                for (low, high) in ZDT3_PIECES {
                    if position <= high - low {
                        f1 = low + position;
                        break;
                    }
                    position -= high - low;
                }
                [f1, zdt3_f2(f1)]
            })
            .collect()
    }

    // each piece ends at a local minimum of f₂, the lowest at the end of the last
    fn ideal(&self) -> [f64; 2] {
        [0.0, zdt3_f2(ZDT3_PIECES[4].1)]
    }

    fn nadir(&self) -> [f64; 2] {
        [ZDT3_PIECES[4].1, 1.0]
    }
}

impl MultiFitnessFunction<Reals, 2> for Zdt4 {
    type Output = [f64; 2];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` is empty.
    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let g = 1.0
            + 10.0 * (x.len() - 1) as f64
            + x[1..]
                .iter()
                .map(|xi| xi * xi - 10.0 * math::cos(4.0 * PI * xi))
                .sum::<f64>();
        [x[0], g * (1.0 - (x[0] / g).sqrt())]
    }
}

impl Zdt4 {
    fn real(&self) -> Real {
        Real::new(std::iter::once(0.0..=1.0).chain((1..self.variables).map(|_| -5.0..=5.0)))
            .expect("valid bounds")
    }

    fn front(&self, points: usize) -> Vec<[f64; 2]> {
        zdt_front(points, |f1| 1.0 - f1.sqrt(), 0.0)
    }

    fn ideal(&self) -> [f64; 2] {
        [0.0, 0.0]
    }

    fn nadir(&self) -> [f64; 2] {
        [1.0, 1.0]
    }
}

// the smallest f1 of ZDT6's optimal front: 1 − exp(−4x) sin⁶(6πx) is smallest where its
// derivative, exp(−4x) sin⁵(6πx) (36π cos(6πx) − 4 sin(6πx)), is 0 with tan(6πx) = 9π, at
// x = atan(9π) / (6π) ≈ 0.08146; the value there, computed to 30 digits and rounded
pub(super) const ZDT6_START: f64 = 0.280_775_318_815_369_7;

impl MultiFitnessFunction<Reals, 2> for Zdt6 {
    type Output = [f64; 2];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` is empty.
    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let f1 = 1.0 - math::exp(-4.0 * x[0]) * math::sin(6.0 * PI * x[0]).powi(6);
        let g = 1.0 + 9.0 * math::powf(tail_mean(x), 0.25);
        [f1, g * (1.0 - (f1 / g) * (f1 / g))]
    }
}

impl Zdt6 {
    fn real(&self) -> Real {
        unit_box(self.variables)
    }

    fn front(&self, points: usize) -> Vec<[f64; 2]> {
        zdt_front(points, |f1| 1.0 - f1 * f1, ZDT6_START)
    }

    fn ideal(&self) -> [f64; 2] {
        [ZDT6_START, 0.0]
    }

    fn nadir(&self) -> [f64; 2] {
        [1.0, 1.0 - ZDT6_START * ZDT6_START]
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

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{actual} is not {expected}"
        );
    }

    // x = (a, b, …, b) in 30 variables: g = 1 + 9b
    fn point(a: f64, b: f64, variables: usize) -> Reals {
        let mut x = vec![b; variables];
        x[0] = a;
        at(&x)
    }

    // the formulas of Zitzler, Deb and Thiele (2000), at points chosen so that the values follow
    // by hand
    #[test]
    fn values_match_the_paper() {
        // a = 0.25 and b = 1/9: g = 2 and f₁ / g = 0.125, √0.125 = √2 / 4
        let x = point(0.25, 1.0 / 9.0, 30);
        // ZDT1: 2 (1 − √2 / 4) = 2 − √2 / 2
        let [f1, f2] = Zdt1::default().evaluate(&x);
        assert_eq!(f1, 0.25);
        assert_close(f2, 2.0 - 2f64.sqrt() / 2.0);
        // ZDT2: 2 (1 − 1/64)
        assert_close(Zdt2::default().evaluate(&x)[1], 1.968_75);
        // ZDT3: 2 (1 − √2 / 4 − 0.125 sin 2.5π) = 2 − √2 / 2 − 0.25
        assert_close(Zdt3::default().evaluate(&x)[1], 1.75 - 2f64.sqrt() / 2.0);
        // ZDT4 at (0.25, 1, 0, …, 0): g = 1 + 90 + (1 − 10 cos 4π) + 8 (0 − 10) = 2, so f₂ as
        // for ZDT1
        let mut x = vec![0.0; 10];
        x[0] = 0.25;
        x[1] = 1.0;
        assert_close(
            Zdt4::default().evaluate(&at(&x))[1],
            2.0 - 2f64.sqrt() / 2.0,
        );
        // ZDT6 at x₁ = 0 and the others 1/81: f₁ = 1 − e⁰ sin⁶ 0 = 1, g = 1 + 9 (1/81)^0.25 = 4,
        // f₂ = 4 (1 − 1/16)
        let [f1, f2] = Zdt6::default().evaluate(&point(0.0, 1.0 / 81.0, 10));
        assert_eq!(f1, 1.0);
        assert_close(f2, 3.75);
        // the worst corner of ZDT1: x = (1, 1, …, 1), g = 10, f₂ = 10 (1 − √0.1)
        assert_close(
            Zdt1::default().evaluate(&point(1.0, 1.0, 30))[1],
            10.0 - 10f64.sqrt(),
        );
    }

    #[test]
    fn optimal_solutions_lie_on_the_front() {
        // the optimal solutions have every variable but the first at 0
        for x1 in [0.0, 0.25, 0.5, 1.0] {
            let x = point(x1, 0.0, 30);
            let [f1, f2] = Zdt1::default().evaluate(&x);
            assert!((f2 - (1.0 - f1.sqrt())).abs() < 1e-12);
            let [f1, f2] = Zdt2::default().evaluate(&x);
            assert!((f2 - (1.0 - f1 * f1)).abs() < 1e-12);
            let [f1, f2] = Zdt3::default().evaluate(&x);
            assert!((f2 - zdt3_f2(f1)).abs() < 1e-12);
            let [f1, f2] = Zdt4::default().evaluate(&point(x1, 0.0, 10));
            assert!((f2 - (1.0 - f1.sqrt())).abs() < 1e-12);
            let [f1, f2] = Zdt6::default().evaluate(&point(x1, 0.0, 10));
            assert!((f2 - (1.0 - f1 * f1)).abs() < 1e-12);
        }
        let front = Zdt1::default().front(5);
        assert_eq!(front.len(), 5);
        assert_eq!((front[0], front[4]), ([0.0, 1.0], [1.0, 0.0]));
        // ZDT3 and ZDT6: the front points are mutually non-dominated
        for front in [Zdt3::default().front(200), Zdt6::default().front(200)] {
            let scores: Vec<Scores<2>> = front.iter().map(|p| Scores::new(*p)).collect();
            assert_eq!(non_dominated_sort(&scores, &[Minimize; 2]).len(), 1);
        }
        assert_eq!(Zdt6::default().front(2)[0][0], ZDT6_START);
        assert_eq!(Zdt4::default().real().bounds()[1], -5.0..=5.0);
    }

    // ZDT6's front starts where f₁ = 1 − exp(−4x) sin⁶(6πx) is smallest, at x = atan(9π) / (6π)
    #[test]
    fn zdt6_starts_at_the_minimum_of_f1() {
        let f1 = |x: f64| 1.0 - math::exp(-4.0 * x) * math::sin(6.0 * PI * x).powi(6);
        let x = math::atan(9.0 * PI) / (6.0 * PI);
        assert!((f1(x) - ZDT6_START).abs() < 1e-15);
        // a dense grid finds nothing lower
        let smallest = (0..=100_000)
            .map(|i| f1(i as f64 / 100_000.0))
            .fold(f64::INFINITY, f64::min);
        assert!(smallest >= ZDT6_START && smallest - ZDT6_START < 1e-8);
    }

    // each piece of ZDT3's front ends at a local minimum of f₂(f₁), which later pieces start
    // from
    #[test]
    fn zdt3_pieces_end_at_local_minima() {
        for (_, high) in ZDT3_PIECES {
            let step = 1e-4;
            assert!(zdt3_f2(high) < zdt3_f2(high - step));
            assert!(zdt3_f2(high) < zdt3_f2(high + step));
        }
        let [f1, f2] = Zdt3::default().ideal();
        assert_eq!(f1, 0.0);
        assert!((f2 + 0.773_369).abs() < 1e-6);
    }
}
