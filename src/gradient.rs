//! Gradients: supplied with the fitness, or estimated by finite differences.
//!
//! A gradient-based algorithm gets the gradient of the score in one of two ways, chosen by its
//! [`Gradients`] setting:
//!
//! - **Supplied** by the fitness function, which computes it along with the score:
//!   [`Differentiable`] wraps a closure `|x: &Reals, gradient: &mut [f64]| value`, the test
//!   problems of [`problems`](crate::problems) that are smooth give theirs, and any
//!   [`FitnessFunction`] can declare one with [`provides`](FitnessFunction::provides) and write it
//!   in [`evaluate_with`](FitnessFunction::evaluate_with). It costs about one evaluation.
//! - **Finite differences:** a [`Stencil`] of points around x, one gene moved in each, which the
//!   algorithm asks in the same round as x, so [`parallel`](crate::Engine::parallel) and
//!   [`Batch`] evaluation take them together; [`Stencil::gradient`] turns their values into the
//!   gradient. Forward differences cost n evaluations per gradient, central ones 2n, for n
//!   genes that aren't fixed.
//!
//! The gradient is of the score as the fitness function returns it, whatever the
//! [objective](crate::Objective): for a maximized score, the algorithm climbs it, and no sign
//! changes. [`check`] compares a supplied gradient with central differences, for tests.
//!
//! ```
//! use genoxide::gradient::{self, Differentiable};
//! use genoxide::prelude::*;
//!
//! // Rosenbrock's function in 2 dimensions, with its gradient
//! let rosenbrock = Differentiable(|x: &Reals, gradient: &mut [f64]| {
//!     let (a, b) = (x[1] - x[0] * x[0], 1.0 - x[0]);
//!     gradient[0] = -400.0 * x[0] * a - 2.0 * b;
//!     gradient[1] = 200.0 * a;
//!     100.0 * a * a + b * b
//! });
//! let error = gradient::check(&rosenbrock, &Reals::from(vec![-1.2, 1.0]))?;
//! assert!(error.largest() < 1e-8);
//!
//! // any algorithm takes it, as a plain fitness function
//! let cmaes = Cmaes::builder(Real::uniform(2, -5.0..=5.0)?).minimize().seed(1).build()?;
//! let outcome = Engine::new(cmaes, rosenbrock)
//!     .stop_when(Stop::target(1e-10).or(Stop::evaluations(10_000)))
//!     .run()?;
//! assert_eq!(outcome.stop_reason(), StopReason::Target);
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! # Finite differences
//!
//! The step of gene i is `h = s · max(|xᵢ|, 1)`, with the relative step s = √ε ≈ 1.5e-8
//! ([`FORWARD_STEP`]) for forward differences and s = ε^(1/3) ≈ 6.1e-6 ([`CENTRAL_STEP`]) for
//! central ones, ε being the machine epsilon of `f64`: the steps that balance the truncation error
//! of the formula (O(h) forward, O(h²) central) against the rounding error of the function values
//! (O(ε / h)), for a function whose value and derivatives are of the order of 1, as Gill, Murray,
//! Saunders and Wright (1983) derive and Nocedal and Wright (2006, section 8.1) restate. The step
//! is then rounded, `h = (xᵢ + h) − xᵢ`, so that the point is exactly `xᵢ + h`: the divisor is
//! the true distance between the points, not the step asked for.
//!
//! Every point lies in the bounds of the [`Real`] genome. Forward differences step backwards
//! where xᵢ + h is outside; central differences, which need both sides, take two steps inwards
//! from a bound instead, h and 2h, with the second-order one-sided formula
//! `(−3 f(x) + 4 f(x + h) − f(x + 2h)) / (2h)` (in its form for steps of any length), so they stay
//! O(h²) there. A gene whose range is narrower than the step moves within it. Fixed genes (low =
//! high) have no points and a gradient of 0.
//!
//! On the smooth test problems, the finite-difference gradient is accurate to about 1e-7
//! relative with forward differences and 1e-9 with central ones, less near a minimum, where the
//! gradient is small against the rounding error of the values: there, forward differences can't
//! bring the gradient's norm much below √ε · max(1, |f|), so a gradient tolerance must allow for
//! it, or central differences take over.
//!
//! # References
//!
//! - Gill, P. E., Murray, W., Saunders, M. A. and Wright, M. H. (1983). Computing
//!   forward-difference intervals for numerical optimization. *SIAM Journal on Scientific and
//!   Statistical Computing* 4(2): 310-321. doi:10.1137/0904025
//! - Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer, chapter 8
//!   (Calculating derivatives), section 8.1 (Finite-difference derivative approximations).
//!   doi:10.1007/978-0-387-40065-5
//!
//! Neither has been re-read for this implementation: the step sizes, the rounding of the step
//! and the one-sided formula are the textbook ones, as recorded in genoxide's optimization plan.

use crate::engine::{Batch, BatchExtras, Extras, FitnessFunction, IntoFitness, Provided};
use crate::genome::{Genome, Real, Reals};
use crate::{Error, Result};
use std::cell::Cell;

/// The relative step of forward differences: √ε = 2^−26 ≈ 1.49e-8, where ε is `f64::EPSILON`.
pub const FORWARD_STEP: f64 = 1.490_116_119_384_765_6e-8;

/// The relative step of central differences: ε^(1/3) = 2^(−52/3) ≈ 6.06e-6, where ε is
/// `f64::EPSILON`.
pub const CENTRAL_STEP: f64 = 6.055_454_452_393_339_5e-6;

/// The most genes that aren't fixed for which [`Gradients::Auto`] falls back to finite
/// differences when the fitness function supplies no gradient: 10⁴. Above it, a forward
/// difference costs more than 10⁴ evaluations per gradient, and
/// [`resolve`](Gradients::resolve) fails, pointing to [`Differentiable`].
pub const AUTO_LIMIT: usize = 10_000;

/// Where a gradient-based algorithm gets its gradients: a builder setting.
///
/// | Setting | Gradients |
/// |---|---|
/// | `Auto` (the default) | Supplied if the fitness function [provides](FitnessFunction::provides) them, forward differences otherwise (up to [`AUTO_LIMIT`] genes that aren't fixed) |
/// | `Supplied` | Supplied: a run fails at its start if the fitness function provides none |
/// | `Forward { step }`, `Central { step }` | Finite differences, even if the fitness function provides gradients, e.g. to compare them; `step` relative (`h = step · max(|xᵢ|, 1)`), `None` for [`FORWARD_STEP`] or [`CENTRAL_STEP`] |
///
/// The [`Engine`](crate::Engine) resolves `Auto` when a run starts, through the algorithm's
/// [`prepare`](crate::Algorithm::prepare). Driven by hand, with ask and tell and no engine, an
/// algorithm with `Auto` uses forward differences. Finite differences cost n (forward) or 2n
/// (central) evaluations per gradient, for n genes that aren't fixed, which count towards
/// [`Stop::evaluations`](crate::Stop::evaluations): see
/// [`extra_evaluations`](Gradients::extra_evaluations).
///
/// ```
/// use genoxide::engine::Provided;
/// use genoxide::gradient::Gradients;
/// use genoxide::genome::Real;
///
/// let real = Real::uniform(3, -1.0..=1.0)?;
/// let auto = Gradients::default();
/// assert_eq!(auto.resolve(Provided::GRADIENT, &real)?, Gradients::Supplied);
/// assert_eq!(auto.resolve(Provided::NOTHING, &real)?, Gradients::Forward { step: None });
/// assert!(Gradients::Supplied.resolve(Provided::NOTHING, &real).is_err());
/// # Ok::<(), genoxide::Error>(())
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Gradients {
    /// Supplied if the fitness function provides gradients, forward differences otherwise.
    #[default]
    Auto,
    /// Supplied by the fitness function, which must provide them.
    Supplied,
    /// Forward differences: n evaluations per gradient.
    Forward {
        /// The relative step, above 0; `None` for [`FORWARD_STEP`].
        step: Option<f64>,
    },
    /// Central differences: 2n evaluations per gradient, about the square of forward
    /// differences' accuracy.
    Central {
        /// The relative step, above 0; `None` for [`CENTRAL_STEP`].
        step: Option<f64>,
    },
}

impl Gradients {
    /// Checks the step of finite differences.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a step that isn't finite and above 0.
    pub fn validate(self) -> Result<()> {
        match self {
            Gradients::Forward { step: Some(step) } | Gradients::Central { step: Some(step) }
                if !(step.is_finite() && step > 0.0) =>
            {
                Err(Error::InvalidSetting {
                    setting: "gradients",
                    reason: format!(
                        "a finite-difference step must be finite and above 0, got {step}"
                    ),
                })
            }
            _ => Ok(()),
        }
    }

    /// The gradients for a run whose fitness function provides `provided`, on `real` genomes:
    /// never `Auto`. An algorithm calls it from its [`prepare`](crate::Algorithm::prepare).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for
    ///
    /// - `Supplied` when the fitness function provides no gradient;
    /// - `Auto` without a supplied gradient for more than [`AUTO_LIMIT`] genes that aren't fixed;
    /// - an invalid step, see [`validate`](Gradients::validate).
    pub fn resolve(self, provided: Provided, real: &Real) -> Result<Gradients> {
        self.validate()?;
        match self {
            Gradients::Auto if provided.gradient => Ok(Gradients::Supplied),
            Gradients::Auto => {
                let variables = real.variable_genes().len();
                if variables > AUTO_LIMIT {
                    return Err(Error::InvalidSetting {
                        setting: "gradients",
                        reason: format!(
                            "the fitness function supplies no gradient, and forward differences \
                             would cost {variables} evaluations per gradient, above the {AUTO_LIMIT} \
                             of `Gradients::Auto`: supply the gradient (`Differentiable`, or \
                             `FitnessFunction::provides` and `evaluate_with`), or ask for finite \
                             differences explicitly (`Gradients::Forward` or `Gradients::Central`)"
                        ),
                    });
                }
                Ok(Gradients::Forward { step: None })
            }
            Gradients::Supplied if !provided.gradient => Err(Error::InvalidSetting {
                setting: "gradients",
                reason: "`Gradients::Supplied`, but the fitness function provides no gradient: \
                         supply it (`Differentiable`, or `FitnessFunction::provides` and \
                         `evaluate_with`), or use `Gradients::Auto` or finite differences"
                    .to_string(),
            }),
            resolved => Ok(resolved),
        }
    }

    /// Whether the gradients come from the fitness function.
    pub fn is_supplied(self) -> bool {
        self == Gradients::Supplied
    }

    /// The evaluations a gradient costs besides the point's own, for `variables` genes that
    /// aren't fixed: 0 supplied, `variables` with forward differences (and `Auto` before it's
    /// resolved), twice that with central ones.
    pub fn extra_evaluations(self, variables: usize) -> usize {
        match self {
            Gradients::Supplied => 0,
            Gradients::Auto | Gradients::Forward { .. } => variables,
            Gradients::Central { .. } => 2 * variables,
        }
    }
}

// the step `h` (signed, non-zero) from `x`, rounded so that `(x + h) − x` is exactly the step
// returned; at least the gap to the next value of `f64` in its direction
fn rounded_step(x: f64, h: f64) -> f64 {
    let step = (x + h) - x;
    if step != 0.0 {
        return step;
    }
    let next = if h > 0.0 { x.next_up() } else { x.next_down() };
    next - x
}

// the gene's value a step `h` from `x`, within [low, high], and its distance from `x`
fn moved(x: f64, h: f64, low: f64, high: f64) -> [f64; 2] {
    let value = (x + rounded_step(x, h)).clamp(low, high);
    [value, value - x]
}

/// A finite-difference stencil on [`Real`] genomes: the points around x whose values give the
/// gradient at x, each x with one gene moved (see the [module docs](self#finite-differences)).
///
/// [`set_center`](Stencil::set_center) places it at x;
/// [`write_point`](Stencil::write_point) gives its [`len`](Stencil::len) points, which an algorithm
/// asks in one round, with x if it needs f(x) too; [`gradient`](Stencil::gradient) turns their
/// values into the gradient. Nothing is allocated after the first center.
///
/// ```
/// use genoxide::genome::{Real, Reals};
/// use genoxide::gradient::{Gradients, Stencil};
///
/// let f = |x: &[f64]| x[0] * x[0] + x[1] + 3.0 * x[2];
/// let real = Real::new([-1.0..=1.0, 0.0..=0.0, -1.0..=1.0])?; // the second gene is fixed
/// let mut stencil = Stencil::new(&real, Gradients::Central { step: None })?;
/// let x = [0.5, 0.0, 1.0]; // the third gene at its upper bound
/// stencil.set_center(&x)?;
/// assert_eq!(stencil.len(), 4); // 2 points for each of the 2 genes that aren't fixed
/// let mut point = Reals::from(vec![0.0; 3]);
/// let mut values = Vec::new();
/// for k in 0..stencil.len() {
///     stencil.write_point(k, &mut point)?;
///     assert!(real.bounds().iter().zip(point.iter()).all(|(range, v)| range.contains(v)));
///     values.push(f(&point));
/// }
/// let mut gradient = [0.0; 3];
/// stencil.gradient(f(&x), &values, &mut gradient)?;
/// assert!((gradient[0] - 1.0).abs() < 1e-9);
/// assert_eq!(gradient[1], 0.0); // fixed: 0, whatever the function
/// assert!((gradient[2] - 3.0).abs() < 1e-9); // one-sided, inwards from the bound
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Stencil {
    central: bool,
    step: f64,
    // [low, high] of every gene
    bounds: Vec<[f64; 2]>,
    // the genes that aren't fixed, in order
    genes: Vec<usize>,
    // x, once set
    center: Vec<f64>,
    // per gene of `genes`: its value at its points (one forward, two central), and their exact
    // distances from x
    values: Vec<[f64; 2]>,
    offsets: Vec<[f64; 2]>,
}

impl Stencil {
    /// A stencil for `real` genomes with the finite differences of `gradients`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] if `gradients` isn't [`Gradients::Forward`] or
    /// [`Gradients::Central`], or for an invalid step.
    pub fn new(real: &Real, gradients: Gradients) -> Result<Self> {
        gradients.validate()?;
        let (central, step) = match gradients {
            Gradients::Forward { step } => (false, step.unwrap_or(FORWARD_STEP)),
            Gradients::Central { step } => (true, step.unwrap_or(CENTRAL_STEP)),
            other => {
                return Err(Error::InvalidSetting {
                    setting: "gradients",
                    reason: format!(
                        "a stencil needs finite differences, `Gradients::Forward` or \
                         `Gradients::Central`, got {other:?}"
                    ),
                });
            }
        };
        let genes = real.variable_genes().to_vec();
        Ok(Self {
            central,
            step,
            bounds: real
                .bounds()
                .iter()
                .map(|range| [*range.start(), *range.end()])
                .collect(),
            values: vec![[0.0; 2]; genes.len()],
            offsets: vec![[0.0; 2]; genes.len()],
            genes,
            center: Vec::new(),
        })
    }

    /// The number of points: one per gene that isn't fixed with forward differences, two with
    /// central ones.
    pub fn len(&self) -> usize {
        if self.central {
            2 * self.genes.len()
        } else {
            self.genes.len()
        }
    }

    /// Whether there are no points: every gene is fixed.
    pub fn is_empty(&self) -> bool {
        self.genes.is_empty()
    }

    /// Whether the differences are central rather than forward.
    pub fn is_central(&self) -> bool {
        self.central
    }

    /// The relative step.
    pub fn step(&self) -> f64 {
        self.step
    }

    /// The center x, empty before the first [`set_center`](Stencil::set_center).
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    /// Places the stencil at `x`: the points are then around it.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidGenome`] if `x` doesn't have a gene per bound.
    pub fn set_center(&mut self, x: &[f64]) -> Result<()> {
        if x.len() != self.bounds.len() {
            return Err(Error::InvalidGenome {
                reason: format!(
                    "the stencil is for {} genes, the center has {}",
                    self.bounds.len(),
                    x.len()
                ),
            });
        }
        self.center.clear();
        self.center.extend_from_slice(x);
        for (k, &gene) in self.genes.iter().enumerate() {
            let x = x[gene];
            let [low, high] = self.bounds[gene];
            let h = self.step * x.abs().max(1.0);
            let (up, down) = ((high - x).max(0.0), (x - low).max(0.0));
            let [first, second] = if !self.central {
                // forward, backward from the upper bound, or as far as the range allows
                let h = if up >= h {
                    h
                } else if down >= h {
                    -h
                } else if up >= down {
                    up
                } else {
                    -down
                };
                let point = moved(x, h, low, high);
                [point, point]
            } else if up >= h && down >= h {
                [moved(x, h, low, high), moved(x, -h, low, high)]
            } else {
                // one-sided, two steps inwards from the bound
                let (direction, room) = if up >= down { (1.0, up) } else { (-1.0, down) };
                let h = h.min(room / 2.0);
                [
                    moved(x, direction * h, low, high),
                    moved(x, direction * 2.0 * h, low, high),
                ]
            };
            self.values[k] = [first[0], second[0]];
            self.offsets[k] = [first[1], second[1]];
        }
        Ok(())
    }

    /// The gene that point `k` moves, and its value there, or `None` beyond the points or before
    /// a center is set.
    pub fn point(&self, k: usize) -> Option<(usize, f64)> {
        if k >= self.len() || self.center.is_empty() {
            return None;
        }
        let (index, side) = if self.central { (k / 2, k % 2) } else { (k, 0) };
        Some((self.genes[index], self.values[index][side]))
    }

    /// Writes point `k` into `point`: the center, with one gene moved.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidGenome`] if `k` is beyond the points, no center is set, or `point` doesn't
    /// have the center's length.
    pub fn write_point(&self, k: usize, point: &mut [f64]) -> Result<()> {
        let Some((gene, value)) = self.point(k) else {
            return Err(Error::InvalidGenome {
                reason: format!(
                    "no point {k} of a stencil with {} points{}",
                    self.len(),
                    if self.center.is_empty() {
                        " and no center"
                    } else {
                        ""
                    }
                ),
            });
        };
        if point.len() != self.center.len() {
            return Err(Error::InvalidGenome {
                reason: format!(
                    "a point of the stencil has {} genes, got {}",
                    self.center.len(),
                    point.len()
                ),
            });
        }
        point.copy_from_slice(&self.center);
        point[gene] = value;
        Ok(())
    }

    /// The gradient at the center into `gradient` (a value per gene), from the score there,
    /// `center_value`, and at the points, `values`, in their order. A fixed gene's is 0. An
    /// invalid score should be passed as NaN: the genes whose differences use it get NaN.
    ///
    /// Forward differences use `(f(x + h) − f(x)) / h`; central ones the derivative at x of the
    /// parabola through x and its two points: `(f(x + h) − f(x − h)) / (2h)` for equal steps on
    /// both sides, where the center's score drops out, and the one-sided second-order formula at
    /// a bound.
    ///
    /// # Errors
    ///
    /// [`Error::FitnessCount`] if there isn't a value per point, and [`Error::InvalidGenome`] if
    /// `gradient` doesn't have a value per gene or no center is set.
    pub fn gradient(&self, center_value: f64, values: &[f64], gradient: &mut [f64]) -> Result<()> {
        if values.len() != self.len() {
            return Err(Error::FitnessCount {
                expected: self.len(),
                got: values.len(),
            });
        }
        if gradient.len() != self.bounds.len() || self.center.is_empty() {
            return Err(Error::InvalidGenome {
                reason: format!(
                    "the gradient needs a value per gene, {}, got {}{}",
                    self.bounds.len(),
                    gradient.len(),
                    if self.center.is_empty() {
                        ", and the stencil has no center"
                    } else {
                        ""
                    }
                ),
            });
        }
        gradient.fill(0.0);
        for (k, &gene) in self.genes.iter().enumerate() {
            let [a, b] = self.offsets[k];
            gradient[gene] = if !self.central {
                (values[k] - center_value) / a
            } else {
                let (fa, fb) = (values[2 * k], values[2 * k + 1]);
                parabola_slope(center_value, a, fa, b, fb)
            };
        }
        Ok(())
    }
}

// the slope at 0 of the parabola through (0, f0), (a, fa) and (b, fb): Lagrange's interpolation,
// differentiated. For b = −a, (fa − fb) / (2a); for a and b of the same sign, the one-sided
// second-order formula; for a = b (a range too narrow for two steps), the forward difference
fn parabola_slope(f0: f64, a: f64, fa: f64, b: f64, fb: f64) -> f64 {
    if a == b {
        return (fa - f0) / a;
    }
    let sum = a + b;
    let slope = fa * (b / (a * (b - a))) - fb * (a / (b * (b - a)));
    if sum == 0.0 {
        slope
    } else {
        slope - f0 * (sum / (a * b))
    }
}

/// A supplied gradient compared with central differences, by [`check`].
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    supplied: Vec<f64>,
    estimated: Vec<f64>,
    errors: Vec<f64>,
}

impl Check {
    /// The gradient the fitness function supplied.
    pub fn supplied(&self) -> &[f64] {
        &self.supplied
    }

    /// The gradient by central differences.
    pub fn estimated(&self) -> &[f64] {
        &self.estimated
    }

    /// The error of each gene: `|supplied − estimated| / max(|supplied|, |estimated|, 1)`,
    /// relative to the larger of the two, or absolute where both are below 1. NaN if either is.
    pub fn errors(&self) -> &[f64] {
        &self.errors
    }

    /// The largest error of a gene, NaN if any is NaN, and 0 for no genes.
    pub fn largest(&self) -> f64 {
        self.errors.iter().fold(0.0, |largest: f64, &error| {
            if error.is_nan() || largest.is_nan() {
                f64::NAN
            } else {
                largest.max(error)
            }
        })
    }

    /// The gene with the largest error (the first, for ties or NaN), or `None` for no genes.
    pub fn worst_gene(&self) -> Option<usize> {
        let largest = self.largest();
        self.errors
            .iter()
            .position(|&error| error == largest || (largest.is_nan() && error.is_nan()))
    }
}

/// Compares the gradient a fitness function supplies at `x` with central differences: the error
/// of each gene, and the [largest](Check::largest), for a test of a hand-written gradient. A
/// correct gradient has errors of about 1e-10 or below on a well-scaled function; a wrong one,
/// of the order of 1 in the genes it gets wrong.
///
/// The points are `x` with one gene moved by `h = ε^(1/3) · max(|xᵢ|, 1)` each way, without
/// bounds. An invalid score counts as NaN.
///
/// ```
/// use genoxide::gradient::{self, Differentiable};
/// use genoxide::genome::Reals;
///
/// // a wrong gradient: the derivative of x₀³ is 3x₀², not 3x₀
/// let cube = Differentiable(|x: &Reals, gradient: &mut [f64]| {
///     gradient[0] = 3.0 * x[0];
///     x[0] * x[0] * x[0]
/// });
/// let check = gradient::check(&cube, &Reals::from(vec![2.0]))?;
/// assert!(check.largest() > 0.4);
/// assert_eq!(check.worst_gene(), Some(0));
/// # Ok::<(), genoxide::Error>(())
/// ```
///
/// # Errors
///
/// [`Error::InvalidSetting`] if the fitness function doesn't
/// [provide](FitnessFunction::provides) a gradient, and the errors of its
/// [`IntoFitness`] values, such as [`Error::InvalidFitness`] for a negative violation.
pub fn check<F>(function: &F, x: &Reals) -> Result<Check>
where
    F: FitnessFunction<Reals> + ?Sized,
{
    if !function.provides().gradient {
        return Err(Error::InvalidSetting {
            setting: "fitness",
            reason: "the fitness function provides no gradient to check".to_string(),
        });
    }
    let score = |genome: &Reals| -> Result<f64> {
        Ok(function
            .evaluate(genome)
            .into_fitness()?
            .score()
            .unwrap_or(f64::NAN))
    };
    let mut supplied = vec![0.0; x.len()];
    let value = function
        .evaluate_with(x, &mut Extras::with_gradient(&mut supplied))
        .into_fitness()?
        .score()
        .unwrap_or(f64::NAN);
    let mut point = x.clone();
    let mut estimated = Vec::with_capacity(x.len());
    for gene in 0..x.len() {
        let xi = x[gene];
        let h = CENTRAL_STEP * xi.abs().max(1.0);
        let [above, a] = moved(xi, h, f64::NEG_INFINITY, f64::INFINITY);
        let [below, b] = moved(xi, -h, f64::NEG_INFINITY, f64::INFINITY);
        point[gene] = above;
        let fa = score(&point)?;
        point[gene] = below;
        let fb = score(&point)?;
        point[gene] = xi;
        estimated.push(parabola_slope(value, a, fa, b, fb));
    }
    let errors = supplied
        .iter()
        .zip(&estimated)
        .map(|(&g, &e)| (g - e).abs() / g.abs().max(e.abs()).max(1.0))
        .collect();
    Ok(Check {
        supplied,
        estimated,
        errors,
    })
}

thread_local! {
    // the gradient that `Differentiable` writes when none is wanted, one per thread, reused
    static SCRATCH: Cell<Vec<f64>> = const { Cell::new(Vec::new()) };
}

// `f` with a zeroed scratch buffer of `len` values
fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f64]) -> R) -> R {
    // taken out of its cell, so that a fitness function that evaluates another gets one of its own
    let mut scratch = SCRATCH.take();
    scratch.clear();
    scratch.resize(len, 0.0);
    let result = f(&mut scratch);
    SCRATCH.set(scratch);
    result
}

/// A fitness function with its gradient: a closure `|x: &Reals, gradient: &mut [f64]| value`
/// that writes `∂value / ∂xᵢ` into `gradient[i]` (one value per gene, zeroed before the call) and
/// returns the value: `f64`, or any other [`IntoFitness`] value.
///
/// It [provides](FitnessFunction::provides) the gradient to algorithms that want it, and is a
/// plain fitness function for the others: its [`evaluate`](FitnessFunction::evaluate) gives the
/// closure a scratch buffer, kept per thread. The gradient is of the value as returned, whatever
/// the [objective](crate::Objective). [`check`] tests it against central differences.
///
/// `Batch(Differentiable(|xs: &[&Reals], gradients: &mut [f64]| values))` is the batch form:
/// the gradients of all genomes in one flat buffer, row-major, a row of `x.len()` values per
/// genome (genomes of the same length), e.g. for a GPU.
///
/// ```
/// use genoxide::gradient::Differentiable;
/// use genoxide::prelude::*;
///
/// let sphere = Differentiable(|x: &Reals, gradient: &mut [f64]| {
///     for (g, xi) in gradient.iter_mut().zip(x.iter()) {
///         *g = 2.0 * xi;
///     }
///     x.iter().map(|xi| xi * xi).sum::<f64>()
/// });
/// assert!(sphere.provides().gradient);
/// assert_eq!(sphere.evaluate(&Reals::from(vec![3.0, 4.0])), 25.0);
///
/// let batch = Batch(Differentiable(|xs: &[&Reals], gradients: &mut [f64]| {
///     let n = xs.first().map_or(0, |x| x.len());
///     let mut values = Vec::with_capacity(xs.len());
///     for (x, row) in xs.iter().zip(gradients.chunks_mut(n.max(1))) {
///         for (g, xi) in row.iter_mut().zip(x.iter()) {
///             *g = 2.0 * xi;
///         }
///         values.push(x.iter().map(|xi| xi * xi).sum::<f64>());
///     }
///     values
/// }));
/// assert!(batch.provides().gradient && batch.is_batch());
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Differentiable<F>(pub F);

impl<F, T> FitnessFunction<Reals> for Differentiable<F>
where
    F: Fn(&Reals, &mut [f64]) -> T + Sync,
    T: IntoFitness,
{
    type Output = T;

    /// The value at `x`, with a scratch gradient.
    fn evaluate(&self, x: &Reals) -> T {
        with_scratch(x.len(), |gradient| (self.0)(x, gradient))
    }

    fn provides(&self) -> Provided {
        Provided::GRADIENT
    }

    fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> T {
        match extras.gradient() {
            Some(gradient) => (self.0)(x, gradient),
            None => self.evaluate(x),
        }
    }
}

impl<F, T> FitnessFunction<Reals> for Batch<Differentiable<F>>
where
    F: Fn(&[&Reals], &mut [f64]) -> Vec<T> + Sync,
    T: IntoFitness,
{
    type Output = T;

    /// The value at `x`: a batch of one, with a scratch gradient.
    ///
    /// # Panics
    ///
    /// If the batch function returns no value for it.
    fn evaluate(&self, x: &Reals) -> T {
        first(with_scratch(x.len(), |gradient| (self.0.0)(&[x], gradient)))
    }

    fn is_batch(&self) -> bool {
        true
    }

    /// The values at `xs`, with a scratch buffer for the gradients.
    fn evaluate_batch(&self, xs: &[&Reals]) -> Vec<T> {
        let dimensions = xs.first().map_or(0, |x| x.len());
        with_scratch(xs.len() * dimensions, |gradients| (self.0.0)(xs, gradients))
    }

    fn provides(&self) -> Provided {
        Provided::GRADIENT
    }

    /// The value at `x`, and its gradient if it's wanted: a batch of one.
    ///
    /// # Panics
    ///
    /// If the batch function returns no value for it.
    fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> T {
        match extras.gradient() {
            Some(gradient) => first((self.0.0)(&[x], gradient)),
            None => self.evaluate(x),
        }
    }

    fn evaluate_batch_with(&self, xs: &[&Reals], extras: &mut BatchExtras<'_>) -> Vec<T> {
        match extras.gradients() {
            Some(gradients) => (self.0.0)(xs, gradients),
            None => self.evaluate_batch(xs),
        }
    }
}

// the only value of a batch of one
fn first<T>(values: Vec<T>) -> T {
    values
        .into_iter()
        .next()
        .expect("the batch function returns a value per genome")
}

#[cfg(test)]
mod tests;
