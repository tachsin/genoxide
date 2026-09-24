//! Constraint handling: measuring constraint violations, and penalty functions.
//!
//! A constrained problem is best described by a score and a constraint violation: 0 for a
//! feasible solution, and how far it is from feasible otherwise. A fitness function returns both
//! as `(score, violation)` (or a [`Fitness::constrained`]), and Deb's feasibility rules (see
//! [`Objective::compare`]) then guide the search: feasible solutions beat infeasible ones, and
//! infeasible ones are ranked by their violation. The functions here measure the violation of one
//! constraint; add them up for several.
//!
//! ```
//! use genoxide::prelude::*;
//! use genoxide::constraint::{at_least, at_most};
//!
//! // minimize (x - 2)² + (y - 1)² subject to x + y <= 1 and x >= 0: the optimum is x = 1, y = 0
//! let fitness = |genes: &Reals| {
//!     let (x, y) = (genes[0], genes[1]);
//!     let score = (x - 2.0).powi(2) + (y - 1.0).powi(2);
//!     (score, at_most(x + y, 1.0) + at_least(x, 0.0))
//! };
//! let ga = Ga::builder(Real::uniform(2, -5.0..=5.0)?)
//!     .population_size(50)
//!     .select(Tournament::new(3)?)
//!     .crossover(SimulatedBinaryCrossover::new(15.0)?)
//!     .mutate(PolynomialMutation::per_gene(0.5, 20.0)?)
//!     .minimize()
//!     .seed(1)
//!     .build()?;
//! let outcome = Engine::new(ga, fitness).stop_when(Stop::generations(300)).run()?;
//! assert!(outcome.best_fitness().is_feasible());
//! assert!((outcome.best_fitness().score().unwrap() - 2.0).abs() < 0.01);
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! [`Penalty`] is the alternative: a single score, worse by a multiple of the violation.

use crate::{Error, Fitness, Objective, Result};

/// The violation of `value <= limit`: how far `value` is above `limit`, or 0. NaN if a value is
/// NaN, so the fitness function's NaN is caught.
pub fn at_most(value: f64, limit: f64) -> f64 {
    let excess = value - limit;
    if excess > 0.0 {
        excess
    } else if excess.is_nan() {
        f64::NAN
    } else {
        0.0
    }
}

/// The violation of `value >= limit`: how far `value` is below `limit`, or 0.
pub fn at_least(value: f64, limit: f64) -> f64 {
    at_most(limit, value)
}

/// The violation of `value == target`, allowing a difference of `tolerance`: how far `value` is
/// outside `target ± tolerance`, or 0. Equality constraints of real values need a tolerance,
/// e.g. `1e-6`.
pub fn equal(value: f64, target: f64, tolerance: f64) -> f64 {
    at_most((value - target).abs(), tolerance)
}

/// A static penalty function: the score made worse by `weight` times the constraint violation.
///
/// Simpler than Deb's feasibility rules and common in the literature, but the weight needs
/// tuning: too low, and the search settles on infeasible solutions with a good score; too high,
/// and it can't cross infeasible regions. Prefer returning `(score, violation)` from the fitness
/// function when in doubt.
///
/// ```
/// use genoxide::constraint::{Penalty, at_most};
/// use genoxide::Objective;
///
/// let penalty = Penalty::new(100.0)?;
/// // maximizing: a violation of 0.5 costs 50
/// assert_eq!(penalty.score(Objective::Maximize, 80.0, at_most(10.5, 10.0)), 30.0);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Penalty {
    weight: f64,
}

impl Penalty {
    /// A penalty of `weight` (positive and finite) per unit of violation.
    pub fn new(weight: f64) -> Result<Self> {
        if weight > 0.0 && weight.is_finite() {
            Ok(Self { weight })
        } else {
            Err(Error::InvalidSetting {
                setting: "penalty_weight",
                reason: format!("must be positive and finite, got {weight}"),
            })
        }
    }

    /// The weight per unit of violation.
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// The penalized score: `score` made worse by `weight * violation` for the `objective`.
    pub fn score(&self, objective: Objective, score: f64, violation: f64) -> f64 {
        match objective {
            Objective::Maximize => score - self.weight * violation,
            Objective::Minimize => score + self.weight * violation,
        }
    }

    /// The penalized score as a (feasible) [`Fitness`], invalid if it's NaN.
    pub fn fitness(&self, objective: Objective, score: f64, violation: f64) -> Fitness {
        Fitness::new(self.score(objective, score, violation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn violations() {
        assert_eq!(at_most(3.0, 5.0), 0.0);
        assert_eq!(at_most(7.0, 5.0), 2.0);
        assert_eq!(at_least(3.0, 5.0), 2.0);
        assert_eq!(at_least(7.0, 5.0), 0.0);
        assert_eq!(equal(5.05, 5.0, 0.1), 0.0);
        assert!((equal(5.5, 5.0, 0.1) - 0.4).abs() < 1e-12);
        assert!((equal(4.5, 5.0, 0.1) - 0.4).abs() < 1e-12);
        assert!(at_most(f64::NAN, 1.0).is_nan());
        assert_eq!(at_most(f64::INFINITY, 1.0), f64::INFINITY);
    }

    #[test]
    fn penalty() {
        assert!(Penalty::new(0.0).is_err());
        assert!(Penalty::new(f64::NAN).is_err());
        let penalty = Penalty::new(10.0).unwrap();
        assert_eq!(penalty.score(Objective::Maximize, 5.0, 0.5), 0.0);
        assert_eq!(penalty.score(Objective::Minimize, 5.0, 0.5), 10.0);
        assert_eq!(
            penalty.fitness(Objective::Minimize, 5.0, 0.0),
            Fitness::new(5.0)
        );
        assert!(
            !penalty
                .fitness(Objective::Maximize, f64::NAN, 0.0)
                .is_valid()
        );
    }
}
