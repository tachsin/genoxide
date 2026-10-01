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
//!
//! [`Constrained`] gives the constraints' values one by one, `gᵢ(x) ≤ 0`, and optionally the
//! gradient and the constraint Jacobian, to the algorithms that use them, such as
//! [`Mma`](crate::algorithm::Mma); for every other algorithm it's a fitness function returning
//! `(score, violation)`.

use crate::engine::{Extras, FitnessFunction, Provided};
use crate::genome::Reals;
use crate::gradient::with_scratch;
use crate::{Error, Fitness, Objective, Result};

/// A fitness function on [`Real`](crate::genome::Real) genomes with `m` inequality constraints
/// `gᵢ(x) ≤ 0` whose values it gives one by one: for algorithms that use each constraint, such
/// as [`Mma`](crate::algorithm::Mma), which read them through [`Extras`].
///
/// - [`Constrained::new`] wraps a closure `|x: &Reals, g: &mut [f64]| score` that writes `gᵢ(x)`
///   into `g[i]`.
/// - [`Constrained::differentiable`] wraps a closure
///   `|x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| score` that also
///   writes the gradient of the score, `∂score / ∂xⱼ` into `gradient[j]`, and the constraint
///   Jacobian, `∂gᵢ / ∂xⱼ` into `jacobian[i * n + j]` for `n` genes (row-major, a row per
///   constraint).
///
/// The buffers are zeroed before each call. Either way, its
/// [`Output`](FitnessFunction::Output) is `(score, violation)`, the violation being
/// `Σ max(0, gᵢ(x))` ([`at_most`]), so any algorithm takes the same function and compares
/// solutions by Deb's rules: a plain [`evaluate`](FitnessFunction::evaluate) gives the closure
/// scratch buffers, kept per thread. Equality constraints aren't part of it yet.
///
/// ```
/// use genoxide::constraint::Constrained;
/// use genoxide::prelude::*;
///
/// // minimize (x₀ − 2)² + (x₁ − 1)² subject to x₀ + x₁ <= 1: the optimum is (1, 0)
/// let problem = Constrained::differentiable(
///     1,
///     |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
///         gradient[0] = 2.0 * (x[0] - 2.0);
///         gradient[1] = 2.0 * (x[1] - 1.0);
///         g[0] = x[0] + x[1] - 1.0;
///         jacobian.copy_from_slice(&[1.0, 1.0]);
///         (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2)
///     },
/// );
/// assert_eq!(problem.evaluate(&Reals::from(vec![1.0, 1.0])), (1.0, 1.0));
///
/// let mma = Mma::builder(Real::uniform(2, -5.0..=5.0)?)
///     .initial_genome(Reals::from(vec![-3.0, 4.0]))
///     .minimize()
///     .build()?;
/// let outcome = Engine::new(mma, problem).stop_when(Stop::evaluations(200)).run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Converged);
/// let x = outcome.best_genome();
/// assert!((x[0] - 1.0).abs() < 1e-6 && x[1].abs() < 1e-6);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Constrained<F> {
    inequalities: usize,
    function: F,
}

/// The closure of a [`Constrained::new`]: the score and the constraints' values.
#[derive(Clone, Copy, Debug)]
pub struct Values<F>(F);

/// The closure of a [`Constrained::differentiable`]: the score, the constraints' values, the
/// gradient and the constraint Jacobian.
#[derive(Clone, Copy, Debug)]
pub struct Derivatives<F>(F);

impl<F> Constrained<Values<F>>
where
    F: Fn(&Reals, &mut [f64]) -> f64,
{
    /// A function with `inequalities` constraints `gᵢ(x) ≤ 0`: `function(x, g)` writes `gᵢ(x)`
    /// into `g[i]` (zeroed before the call) and returns the score.
    pub fn new(inequalities: usize, function: F) -> Self {
        Self {
            inequalities,
            function: Values(function),
        }
    }
}

impl<F> Constrained<Derivatives<F>>
where
    F: Fn(&Reals, &mut [f64], &mut [f64], &mut [f64]) -> f64,
{
    /// A function with `inequalities` constraints `gᵢ(x) ≤ 0` and its derivatives:
    /// `function(x, gradient, g, jacobian)` writes the gradient of the score into `gradient`
    /// (a value per gene), `gᵢ(x)` into `g[i]`, and `∂gᵢ / ∂xⱼ` into `jacobian[i * n + j]`, for
    /// `n` genes; every buffer is zeroed before the call. It returns the score.
    pub fn differentiable(inequalities: usize, function: F) -> Self {
        Self {
            inequalities,
            function: Derivatives(function),
        }
    }
}

impl<C> Constrained<C> {
    /// The number of inequality constraints.
    pub fn inequalities(&self) -> usize {
        self.inequalities
    }
}

// Σ max(0, gᵢ), in order: NaN if a value is NaN
fn violation(values: &[f64]) -> f64 {
    values.iter().map(|&g| at_most(g, 0.0)).sum()
}

impl<F> FitnessFunction<Reals> for Constrained<Values<F>>
where
    F: Fn(&Reals, &mut [f64]) -> f64 + Sync,
{
    type Output = (f64, f64);

    /// The score at `x` and the violation of its constraints.
    fn evaluate(&self, x: &Reals) -> (f64, f64) {
        with_scratch(self.inequalities, |g| {
            let score = (self.function.0)(x, g);
            (score, violation(g))
        })
    }

    fn provides(&self) -> Provided {
        Provided::NOTHING.with_inequalities(self.inequalities)
    }

    fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> (f64, f64) {
        match extras.inequalities() {
            Some(g) => {
                let score = (self.function.0)(x, g);
                (score, violation(g))
            }
            None => self.evaluate(x),
        }
    }
}

impl<F> FitnessFunction<Reals> for Constrained<Derivatives<F>>
where
    F: Fn(&Reals, &mut [f64], &mut [f64], &mut [f64]) -> f64 + Sync,
{
    type Output = (f64, f64);

    /// The score at `x` and the violation of its constraints, with scratch buffers for the
    /// derivatives.
    fn evaluate(&self, x: &Reals) -> (f64, f64) {
        self.evaluate_with(x, &mut Extras::none())
    }

    fn provides(&self) -> Provided {
        Provided::GRADIENT
            .with_inequalities(self.inequalities)
            .with_constraint_jacobian()
    }

    fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> (f64, f64) {
        let (n, m) = (x.len(), self.inequalities);
        let (gradient, g, jacobian) = extras.buffers();
        // the buffers not wanted, from one scratch buffer
        let lengths = [
            if gradient.is_some() { 0 } else { n },
            if g.is_some() { 0 } else { m },
            if jacobian.is_some() { 0 } else { m * n },
        ];
        with_scratch(lengths.iter().sum(), |scratch| {
            let (gradient_scratch, rest) = scratch.split_at_mut(lengths[0]);
            let (g_scratch, jacobian_scratch) = rest.split_at_mut(lengths[1]);
            let gradient: &mut [f64] = gradient.unwrap_or(gradient_scratch);
            let g: &mut [f64] = g.unwrap_or(g_scratch);
            let jacobian: &mut [f64] = jacobian.unwrap_or(jacobian_scratch);
            let score = (self.function.0)(x, gradient, g, jacobian);
            (score, violation(g))
        })
    }
}

/// The violation of `value <= limit`: how far `value` is above `limit`, or 0. NaN if a value is
/// NaN, so the fitness function's NaN is caught.
pub fn at_most(value: f64, limit: f64) -> f64 {
    // compared first: an infinite value at an equal infinite limit meets it
    if value <= limit {
        0.0
    } else if value > limit {
        value - limit
    } else {
        f64::NAN
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
    // an infinite value at an equal infinite target is 0 apart, not NaN
    let distance = if value == target {
        0.0
    } else {
        (value - target).abs()
    };
    at_most(distance, tolerance)
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
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a weight that isn't positive and finite.
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
    fn equal_infinities_meet_their_constraints() {
        assert_eq!(at_most(f64::INFINITY, f64::INFINITY), 0.0);
        assert_eq!(at_least(f64::NEG_INFINITY, f64::NEG_INFINITY), 0.0);
        assert_eq!(equal(f64::INFINITY, f64::INFINITY, 0.0), 0.0);
        assert_eq!(at_most(f64::INFINITY, 1.0), f64::INFINITY);
        assert_eq!(equal(f64::INFINITY, 1.0, 0.5), f64::INFINITY);
        assert!(at_most(f64::NAN, 1.0).is_nan());
        assert!(at_most(1.0, f64::NAN).is_nan());
        assert!(equal(1.0, 1.0, f64::NAN).is_nan());
    }

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
