//! DTLZ1-4: Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable multi-objective
//! optimization test problems. *Proceedings of the 2002 Congress on Evolutionary Computation*:
//! 825-830; the numbering of their technical report (TIK-Report 112, ETH Zürich, 2001).

use super::{MultiProblem, das_dennis, divisions_for};
use crate::genome::{Real, Reals};
use crate::math;
use crate::multi::MultiFitnessFunction;
use std::f64::consts::PI;

const REFERENCE: &str = "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable \
                         multi-objective optimization test problems. Proceedings of the 2002 \
                         Congress on Evolutionary Computation: 825-830.";
const REFERENCE_URL: &str = "https://doi.org/10.1109/CEC.2002.1007032";

macro_rules! dtlz {
    ($(#[$doc:meta])* $name:ident, $label:literal, $k:literal) => {
        $(#[$doc])*
        ///
        #[doc = concat!("With `M` objectives and `n` variables in [0, 1]: the first `M − 1` place a solution on the front, the other `k = n − M + 1` its distance to it; ", stringify!($k), " is the standard `k`.")]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name<const M: usize> {
            variables: usize,
        }

        impl<const M: usize> $name<M> {
            /// The problem with `variables` variables, at least `M`.
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives, or fewer variables than objectives.
            pub fn new(variables: usize) -> Self {
                assert!(M >= 2, "{} needs at least 2 objectives", $label);
                assert!(variables >= M, "{} needs at least as many variables as objectives", $label);
                Self { variables }
            }

            /// The number of variables.
            pub fn variables(&self) -> usize {
                self.variables
            }
        }

        impl<const M: usize> Default for $name<M> {
            #[doc = concat!("The standard problem, with `M + ", stringify!($k), " − 1` variables.")]
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives.
            fn default() -> Self {
                Self::new(M + $k - 1)
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
                REFERENCE
            }

            fn reference_url(&self) -> Option<&'static str> {
                Some(REFERENCE_URL)
            }

            fn optimal_front(&self, points: usize) -> Option<Vec<[f64; M]>> {
                Some(Self::front(points))
            }

            fn ideal_point(&self) -> Option<[f64; M]> {
                Some([0.0; M])
            }

            fn nadir_point(&self) -> Option<[f64; M]> {
                Some([Self::NADIR; M])
            }
        }
    };
}

dtlz!(
    /// DTLZ1: a linear front, the objectives summing to 1/2, behind 11ᵏ − 1 local fronts.
    ///
    /// `g = 100 (k + Σ ((xᵢ − 0.5)² − cos(20π (xᵢ − 0.5))))` over the last k variables;
    /// `f₁ = ½ x₁ ⋯ x_{M−1} (1 + g)`, …, `f_M = ½ (1 − x₁) (1 + g)`. The optimal solutions have
    /// the last k variables at 0.5.
    Dtlz1,
    "DTLZ1",
    5
);
dtlz!(
    /// DTLZ2: a spherical front, the squared objectives summing to 1.
    ///
    /// `g = Σ (xᵢ − 0.5)²` over the last k variables; `f₁ = (1 + g) cos(x₁π/2) ⋯
    /// cos(x_{M−1}π/2)`, …, `f_M = (1 + g) sin(x₁π/2)`. The optimal solutions have the last k
    /// variables at 0.5.
    Dtlz2,
    "DTLZ2",
    10
);
dtlz!(
    /// DTLZ3: the spherical front of DTLZ2 behind 3ᵏ − 1 local fronts: DTLZ2 with the `g` of
    /// DTLZ1.
    Dtlz3,
    "DTLZ3",
    10
);
dtlz!(
    /// DTLZ4: the spherical front of DTLZ2, with solutions biased towards the `f_M`-`f₁` plane:
    /// DTLZ2 with each `xᵢ` of the angles raised to the power 100.
    Dtlz4,
    "DTLZ4",
    10
);

// DTLZ1 and DTLZ3's multimodal distance function
fn rastrigin_g(tail: &[f64]) -> f64 {
    100.0
        * (tail.len() as f64
            + tail
                .iter()
                .map(|x| (x - 0.5) * (x - 0.5) - math::cos(20.0 * PI * (x - 0.5)))
                .sum::<f64>())
}

// DTLZ2 and DTLZ4's distance function
fn sphere_g(tail: &[f64]) -> f64 {
    tail.iter().map(|x| (x - 0.5) * (x - 0.5)).sum()
}

// the objectives on a sphere of radius `radius`, with the angles from `x`
fn spherical<const M: usize>(x: &[f64], radius: f64, alpha: f64) -> [f64; M] {
    std::array::from_fn(|m| {
        let mut f = radius;
        let angle = |xi: f64| if alpha == 1.0 { xi } else { math::powf(xi, alpha) } * PI / 2.0;
        for &xi in &x[..M - 1 - m] {
            f *= math::cos(angle(xi));
        }
        if m > 0 {
            f *= math::sin(angle(x[M - 1 - m]));
        }
        f
    })
}

// Das and Dennis's points, as many as `MultiProblem::optimal_front` promises: none for 0, exactly
// `points` for 2 objectives (0 and 1 would otherwise give both ends), and the smallest set of
// at least `points` for more
fn simplex_points<const M: usize>(points: usize) -> Vec<[f64; M]> {
    if points == 0 {
        return Vec::new();
    }
    let mut simplex = das_dennis::<M>(divisions_for::<M>(points));
    if M == 2 {
        simplex.truncate(points);
    }
    simplex
}

fn spherical_front<const M: usize>(points: usize) -> Vec<[f64; M]> {
    simplex_points::<M>(points)
        .into_iter()
        .map(|p| {
            let norm = p.iter().map(|v| v * v).sum::<f64>().sqrt();
            p.map(|v| v / norm)
        })
        .collect()
}

impl<const M: usize> Dtlz1<M> {
    const NADIR: f64 = 0.5;

    // Das and Dennis's points, halved
    fn front(points: usize) -> Vec<[f64; M]> {
        simplex_points::<M>(points)
            .into_iter()
            .map(|p| p.map(|v| v / 2.0))
            .collect()
    }
}

macro_rules! spherical_front {
    ($($name:ident),*) => {$(
        impl<const M: usize> $name<M> {
            const NADIR: f64 = 1.0;

            // Das and Dennis's points, projected on the sphere
            fn front(points: usize) -> Vec<[f64; M]> {
                spherical_front(points)
            }
        }
    )*};
}

spherical_front!(Dtlz2, Dtlz3, Dtlz4);

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz1<M> {
    type Output = [f64; M];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than `M − 1` genes.
    fn evaluate(&self, x: &Reals) -> [f64; M] {
        let g = rastrigin_g(&x[M - 1..]);
        std::array::from_fn(|m| {
            let mut f = 0.5 * (1.0 + g);
            for xi in &x[..M - 1 - m] {
                f *= xi;
            }
            if m > 0 {
                f *= 1.0 - x[M - 1 - m];
            }
            f
        })
    }
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz2<M> {
    type Output = [f64; M];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than `M − 1` genes.
    fn evaluate(&self, x: &Reals) -> [f64; M] {
        spherical(x, 1.0 + sphere_g(&x[M - 1..]), 1.0)
    }
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz3<M> {
    type Output = [f64; M];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than `M − 1` genes.
    fn evaluate(&self, x: &Reals) -> [f64; M] {
        spherical(x, 1.0 + rastrigin_g(&x[M - 1..]), 1.0)
    }
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz4<M> {
    type Output = [f64; M];

    /// The objective values of `x`.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than `M − 1` genes.
    fn evaluate(&self, x: &Reals) -> [f64; M] {
        spherical(x, 1.0 + sphere_g(&x[M - 1..]), 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // the formulas of Deb, Thiele, Laumanns and Zitzler, at points chosen so that the values
    // follow by hand
    #[test]
    fn values_match_the_paper() {
        // DTLZ1 (M = 3, k = 5) at (0.5, …, 0.5): g = 100 (5 + 5 (0 − cos 0)) = 0, and
        // f = ½ (x₁x₂, x₁(1 − x₂), 1 − x₁) = (0.125, 0.125, 0.25)
        let problem = Dtlz1::<3>::default();
        assert_close(&problem.evaluate(&at(&[0.5; 7])), &[0.125, 0.125, 0.25]);
        // the distance variables at 1: each term is 0.25 − cos 10π = −0.75, so
        // g = 100 (5 − 3.75) = 125, and f is 126 times the above
        let x = at(&[0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_close(&problem.evaluate(&x), &[15.75, 15.75, 31.5]);
        // DTLZ2-4 (M = 3, k = 10) with the distance variables at 0.5: the corners of the front,
        // e.g. x₁ = 0, x₂ = 1: f = (cos 0 cos π/2, cos 0 sin π/2, sin 0) = (0, 1, 0)
        let mut x = vec![0.5; 12];
        for (angles, corner) in [
            ([0.0, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, 1.0], [0.0, 1.0, 0.0]),
            ([1.0, 0.0], [0.0, 0.0, 1.0]),
        ] {
            x[..2].copy_from_slice(&angles);
            for f in [
                Dtlz2::<3>::default().evaluate(&at(&x)),
                Dtlz3::<3>::default().evaluate(&at(&x)),
                Dtlz4::<3>::default().evaluate(&at(&x)),
            ] {
                assert!(f.iter().zip(corner).all(|(f, c)| (f - c).abs() < 1e-15));
            }
        }
        // the distance variables at 1 and the angles at 0: DTLZ2 g = 10 × 0.25, a radius of
        // 3.5; DTLZ3 g = 100 (10 + 10 (0.25 − cos 10π)) = 250, a radius of 251
        let mut x = vec![1.0; 12];
        x[..2].copy_from_slice(&[0.0, 0.0]);
        assert_close(&Dtlz2::<3>::default().evaluate(&at(&x)), &[3.5, 0.0, 0.0]);
        assert_close(&Dtlz3::<3>::default().evaluate(&at(&x)), &[251.0, 0.0, 0.0]);
        // DTLZ4 at x₁ = x₂ = 0.5 on the front: both angles are 0.5¹⁰⁰ π/2 ≈ 1.2e-30, so
        // f = (cos θ cos θ, cos θ sin θ, sin θ) = (1, θ, θ) to double precision: the bias
        // towards f₁ that the paper describes
        let theta = math::powi(0.5f64, 100) * PI / 2.0;
        let f = Dtlz4::<3>::default().evaluate(&at(&[0.5; 12]));
        assert_eq!(f[0], 1.0);
        assert_close(&f[1..], &[theta, theta]);
        assert!(theta > 1.239e-30 && theta < 1.24e-30);
    }

    #[test]
    fn optimal_solutions_lie_on_the_front() {
        // the distance variables at 0.5 put a solution on the front
        let mut x = vec![0.5; 7];
        x[0] = 0.3;
        x[1] = 0.8;
        let f = Dtlz1::<3>::default().evaluate(&at(&x));
        assert!((f.iter().sum::<f64>() - 0.5).abs() < 1e-12);
        let mut x = vec![0.5; 12];
        x[0] = 0.3;
        x[1] = 0.8;
        for f in [
            Dtlz2::<3>::default().evaluate(&at(&x)),
            Dtlz3::<3>::default().evaluate(&at(&x)),
            Dtlz4::<3>::default().evaluate(&at(&x)),
        ] {
            assert!((f.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let front = Dtlz2::<3>::front(91);
        assert_eq!(front.len(), 91);
        assert!(
            front
                .iter()
                .all(|p| (p.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12)
        );
        let front = Dtlz1::<4>::front(20);
        assert!(front.len() >= 20);
        assert!(
            front
                .iter()
                .all(|p| (p.iter().sum::<f64>() - 0.5).abs() < 1e-12)
        );
        assert_eq!(Dtlz1::<3>::default().representation().bounds().len(), 7);
        assert_eq!(Dtlz2::<5>::default().representation().bounds().len(), 14);
    }
}
