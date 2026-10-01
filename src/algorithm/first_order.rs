//! First-order methods: steps along the gradient with a step rule of their own, no line search.
//! Gradient descent, Polyak's momentum, Nesterov's accelerated gradient, Adam and AdamW, for
//! smooth problems with up to millions of variables where a line search costs too much.

use super::local::Restarts;
use super::{Algorithm, Candidates, Reevaluate};
use crate::engine::{Evaluations, Provided, Wanted};
use crate::genome::{Real, Reals, Representation};
use crate::gradient::{Gradients, Stencil};
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;
use std::ops::RangeInclusive;

/// The step rule of a [`FirstOrder`] method: how the point moves along the gradient.
///
/// `g` is the gradient of the score where it was evaluated, its sign turned for a maximized
/// score so that every rule descends; `η` is the [schedule multiplier](FirstOrder::set_multiplier),
/// 1 unless changed; every operation is per gene, and each new point is projected onto the
/// bounds (see [`FirstOrder`]).
///
/// | Rule | Step | Source |
/// |---|---|---|
/// | `Gradient` | `x ← x − η α g` | |
/// | `Momentum` | `v ← μ v − η α g`, `x ← x + v` | Polyak (1964), as Sutskever et al. (2013) state it, eqs. 1-2 |
/// | `Nesterov` | `v ← μ v − η α ∇f(x + μ v)`, `x ← x + v` | Nesterov (1983), as Sutskever et al. (2013) state it, eqs. 3-4 |
/// | `Adam` | `m ← β₁ m + (1 − β₁) g`, `v ← β₂ v + (1 − β₂) g²`, `m̂ = m / (1 − β₁ᵗ)`, `v̂ = v / (1 − β₂ᵗ)`, `x ← x − η α m̂ / (√v̂ + ε)` | Kingma and Ba (2015), Algorithm 1 |
/// | `AdamW` | `m`, `v`, `m̂` and `v̂` as Adam's, `x ← x − η (α m̂ / (√v̂ + ε) + λ x)` | Loshchilov and Hutter (2019), Algorithm 2 |
///
/// `α` is the learning rate, in the units of the genes: the rules are sensitive to the scale of
/// the genes and of the score, so scale the genes alike. Adam's steps are about `α` per gene
/// whatever the size of the gradient; the others' are `α` times the gradient.
///
/// ```
/// use genoxide::algorithm::first_order::Step;
///
/// // Kingma and Ba's defaults
/// assert_eq!(
///     Step::adam(0.001),
///     Step::Adam { learning_rate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }
/// );
/// assert_eq!(Step::nesterov(0.01, 0.9).learning_rate(), 0.01);
/// assert!(Step::momentum(0.01, 1.0).validate().is_err());
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Step {
    /// Gradient descent with a fixed step: `x ← x − η α g`.
    Gradient {
        /// α, greater than 0 and finite.
        learning_rate: f64,
    },
    /// Polyak's heavy ball, classical momentum: `v ← μ v − η α g`, `x ← x + v`. The velocity
    /// keeps the direction of persistent descent, and speeds up long, shallow valleys.
    Momentum {
        /// α, greater than 0 and finite.
        learning_rate: f64,
        /// μ, how much of the velocity is kept each step, in `0..1`: 0.9 is common, 0 is
        /// gradient descent.
        momentum: f64,
    },
    /// Nesterov's accelerated gradient, in Sutskever et al.'s form: classical momentum with
    /// the gradient at the look-ahead point `x + μ v`, which is the point evaluated. It corrects
    /// a velocity that overshoots sooner than classical momentum does, so it tolerates larger
    /// `μ`.
    Nesterov {
        /// α, greater than 0 and finite.
        learning_rate: f64,
        /// μ, in `0..1`.
        momentum: f64,
    },
    /// Adam (Kingma and Ba, 2015): steps of about `α` per gene, from averages of the gradient
    /// and of its square, corrected for their start at 0.
    Adam {
        /// α, greater than 0 and finite: 0.001 is Kingma and Ba's.
        learning_rate: f64,
        /// β₁, the decay of the average of the gradient, in `0..1`: 0.9 is Kingma and Ba's.
        beta1: f64,
        /// β₂, the decay of the average of its square, in `0..1`: 0.999 is Kingma and Ba's.
        beta2: f64,
        /// ε, added to `√v̂` against a division by 0, greater than 0 and finite: 1e-8 is Kingma
        /// and Ba's.
        epsilon: f64,
    },
    /// AdamW (Loshchilov and Hutter, 2019): Adam with decoupled weight decay, `λ x` subtracted
    /// from the point beside Adam's step rather than added to the gradient. Adam with `λ x` in
    /// the gradient (L2 regularization) scales the decay down where the gradient is large;
    /// AdamW decays every gene alike.
    AdamW {
        /// α, greater than 0 and finite.
        learning_rate: f64,
        /// β₁, in `0..1`.
        beta1: f64,
        /// β₂, in `0..1`.
        beta2: f64,
        /// ε, greater than 0 and finite.
        epsilon: f64,
        /// λ, the decay toward 0 per step, 0 or more and finite: multiplied by the schedule
        /// multiplier η, not by α.
        weight_decay: f64,
    },
}

impl Step {
    /// Gradient descent with a fixed step.
    pub fn gradient(learning_rate: f64) -> Self {
        Step::Gradient { learning_rate }
    }

    /// Classical momentum.
    pub fn momentum(learning_rate: f64, momentum: f64) -> Self {
        Step::Momentum {
            learning_rate,
            momentum,
        }
    }

    /// Nesterov's accelerated gradient.
    pub fn nesterov(learning_rate: f64, momentum: f64) -> Self {
        Step::Nesterov {
            learning_rate,
            momentum,
        }
    }

    /// Adam with Kingma and Ba's β₁ = 0.9, β₂ = 0.999 and ε = 1e-8.
    pub fn adam(learning_rate: f64) -> Self {
        Step::Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// AdamW with Loshchilov and Hutter's β₁ = 0.9, β₂ = 0.999 and ε = 1e-8 (Algorithm 2,
    /// line 1, Kingma and Ba's), and the weight decay λ.
    pub fn adamw(learning_rate: f64, weight_decay: f64) -> Self {
        Step::AdamW {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
        }
    }

    /// The learning rate α.
    pub fn learning_rate(self) -> f64 {
        match self {
            Step::Gradient { learning_rate }
            | Step::Momentum { learning_rate, .. }
            | Step::Nesterov { learning_rate, .. }
            | Step::Adam { learning_rate, .. }
            | Step::AdamW { learning_rate, .. } => learning_rate,
        }
    }

    /// The same rule with another learning rate.
    #[must_use]
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        match &mut self {
            Step::Gradient { learning_rate }
            | Step::Momentum { learning_rate, .. }
            | Step::Nesterov { learning_rate, .. }
            | Step::Adam { learning_rate, .. }
            | Step::AdamW { learning_rate, .. } => *learning_rate = rate,
        }
        self
    }

    /// Whether the rule keeps Adam's two averages.
    pub fn is_adam(self) -> bool {
        matches!(self, Step::Adam { .. } | Step::AdamW { .. })
    }

    /// Checks the settings.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a learning rate that isn't greater than 0 and finite, a
    /// momentum or decay outside `0..1`, an ε that isn't greater than 0 and finite, or a weight
    /// decay that isn't 0 or more and finite.
    pub fn validate(self) -> Result<()> {
        let invalid = |reason: String| {
            Err(Error::InvalidSetting {
                setting: "step",
                reason,
            })
        };
        let learning_rate = self.learning_rate();
        if !(learning_rate > 0.0 && learning_rate.is_finite()) {
            return invalid(format!(
                "the learning rate must be greater than 0 and finite, got {learning_rate}"
            ));
        }
        match self {
            Step::Gradient { .. } => Ok(()),
            Step::Momentum { momentum, .. } | Step::Nesterov { momentum, .. } => {
                if (0.0..1.0).contains(&momentum) {
                    Ok(())
                } else {
                    invalid(format!("momentum must be in 0..1, got {momentum}"))
                }
            }
            Step::Adam {
                beta1,
                beta2,
                epsilon,
                ..
            } => check_adam(beta1, beta2, epsilon, 0.0),
            Step::AdamW {
                beta1,
                beta2,
                epsilon,
                weight_decay,
                ..
            } => check_adam(beta1, beta2, epsilon, weight_decay),
        }
    }
}

// Adam's and AdamW's settings besides the learning rate
fn check_adam(beta1: f64, beta2: f64, epsilon: f64, weight_decay: f64) -> Result<()> {
    let invalid = |reason: String| {
        Err(Error::InvalidSetting {
            setting: "step",
            reason,
        })
    };
    for (name, decay) in [("beta1", beta1), ("beta2", beta2)] {
        if !(0.0..1.0).contains(&decay) {
            return invalid(format!("{name} must be in 0..1, got {decay}"));
        }
    }
    if !(epsilon > 0.0 && epsilon.is_finite()) {
        return invalid(format!(
            "epsilon must be greater than 0 and finite, got {epsilon}"
        ));
    }
    if !(weight_decay >= 0.0 && weight_decay.is_finite()) {
        return invalid(format!(
            "the weight decay must be 0 or more and finite, got {weight_decay}"
        ));
    }
    Ok(())
}

/// Why a [`FirstOrder`] run has converged: [`FirstOrder::converged`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Convergence {
    /// The projected gradient at the point is within the
    /// [gradient tolerance](FirstOrderBuilder::gradient_tolerance): a stationary point, or a
    /// point on the bounds where the gradient points outwards.
    Gradient,
    /// The last step, or the step back from an invalid point, is within the
    /// [step tolerance](FirstOrderBuilder::step_tolerance): the point no longer moves, e.g.
    /// when a schedule has brought the learning rate near 0.
    Step,
    /// The run's point is invalid (its fitness, or a gradient that isn't finite), with no
    /// valid point to step back to: the start, or the point re-evaluated on a changed fitness
    /// function.
    Invalid,
}

// Adam's averages, shared with `OpenEs`: the average gradient (also the velocity of momentum),
// the average of its square, and β₁ᵗ and β₂ᵗ, as products of the decays
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Moments {
    pub(crate) first: Vec<f64>,
    pub(crate) second: Vec<f64>,
    pub(crate) beta1_power: f64,
    pub(crate) beta2_power: f64,
}

impl Moments {
    // both averages of `genes` values at 0
    pub(crate) fn new(genes: usize) -> Self {
        Self::with(genes, genes)
    }

    // `first` and `second` values at 0
    fn with(first: usize, second: usize) -> Self {
        Self {
            first: vec![0.0; first],
            second: vec![0.0; second],
            beta1_power: 1.0,
            beta2_power: 1.0,
        }
    }

    // back to the start: the averages at 0
    fn reset(&mut self) {
        self.first.fill(0.0);
        self.second.fill(0.0);
        self.beta1_power = 1.0;
        self.beta2_power = 1.0;
    }

    // β₁ᵗ and β₂ᵗ for the next step t
    #[inline]
    pub(crate) fn advance(&mut self, beta1: f64, beta2: f64) {
        self.beta1_power *= beta1;
        self.beta2_power *= beta2;
    }
}

// Kingma and Ba's updates of one gene's averages with its gradient `g` (Algorithm 1)
#[inline]
pub(crate) fn update_moments(m: &mut f64, v: &mut f64, g: f64, beta1: f64, beta2: f64) {
    *m = beta1 * *m + (1.0 - beta1) * g;
    *v = beta2 * *v + (1.0 - beta2) * g * g;
}

// The step rule's memory, apart from the point: the velocity of momentum, or Adam's averages,
// and the number of steps t since the run (re)started. A continuation that keeps the optimizer's
// state keeps it; one that keeps only the point resets it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Memory {
    moments: Moments,
    steps: u64,
}

impl Memory {
    fn reset(&mut self) {
        self.moments.reset();
        self.steps = 0;
    }
}

// where the best is: none before the first tell, the population's point, or the buffer kept for
// it (allocated when the method is built, so that no step allocates)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum BestAt {
    Nowhere,
    Point,
    Kept,
}

/// A first-order method on [`Real`] genomes, as an ask / tell [`Algorithm`]: steps along the
/// gradient by a [`Step`] rule (gradient descent, momentum, Nesterov, Adam or AdamW), with the
/// step's size a setting rather than found by a line search. For smooth problems with many
/// variables, up to millions: memory and work are linear in the number of genes, and a step
/// allocates nothing.
///
/// - **Gradients.** One gradient per generation, from the [`Gradients`] setting: supplied by the
///   fitness function ([`Differentiable`](crate::gradient::Differentiable), or a test problem's
///   own), one evaluation each; or by finite differences, whose points are asked in the same
///   round as the point (n more evaluations forward, 2n central), up to
///   [`AUTO_LIMIT`](crate::gradient::AUTO_LIMIT) genes with `Gradients::Auto`. An ask of finite
///   differences holds its n or 2n points, each of n genes: at most 2^28 values (2 GiB), beyond
///   which `build` or the run's start fails. The gradient is of the score as returned: a
///   maximized score is climbed.
/// - **Stochastic gradients.** For mini-batches, the fitness function reads the current batch
///   from state that [`Engine::control`](crate::Engine::control) advances each generation, as the
///   penalty example of `control` shares its weight. [`best`](Algorithm::best) is then by the
///   values of different batches, and the tolerances apply to a mini-batch's gradient: a stop
///   condition such as [`Stop::generations`](crate::Stop::generations) ends such runs. Full-batch
///   runs are what the tests check.
/// - **Bounds.** Each new point is projected onto the bounds, gene by gene: the projected
///   gradient method (unlike [`NelderMead`](super::NelderMead), which mirrors trial points, a
///   gradient method lands on an active bound and stays there while the gradient points out).
///   The velocity of momentum is the step taken after the projection; Adam's averages are of the
///   gradients, which the projection doesn't change.
/// - **Convergence.** A run has converged when the projected gradient's largest component is
///   within the [gradient tolerance](FirstOrderBuilder::gradient_tolerance) (a component counts
///   as 0 at a bound it points out of), or when the last step moved no gene by more than the
///   [step tolerance](FirstOrderBuilder::step_tolerance), relative to `max(1, |xᵢ|)`: see
///   [`Convergence`]. Without [`Restarts`] left the method has then
///   [finished](Algorithm::is_finished), and the [`Engine`](crate::Engine) stops with
///   [`StopReason::Converged`](crate::StopReason::Converged).
/// - **Invalid points.** A point whose fitness is invalid, or whose gradient isn't finite,
///   doesn't move the method: the next point is halfway back to the last valid one, with the
///   velocity of momentum halved and Adam's averages kept, until a point is valid, or the step
///   back is within the step tolerance (converged at the valid point).
/// - **Generations.** A generation is one round of evaluations: the point, and the points of a
///   finite-difference stencil. [`iterations`](FirstOrder::iterations) counts the steps.
/// - **Schedules.** [`set_learning_rate`](FirstOrder::set_learning_rate) changes α, and
///   [`set_multiplier`](FirstOrder::set_multiplier) the schedule multiplier η of Loshchilov and
///   Hutter, which also scales AdamW's decay, in [`Engine::control`](crate::Engine::control):
///   from the next step on, with the memory kept.
/// - **The optimizer's state** (the velocity, Adam's averages and its step count t) is kept
///   apart from the point, and reset by a restart.
///
/// Every operation is a sum, product, quotient or square root in a fixed order, so a seed gives
/// the same run on every platform.
///
/// Built with [`FirstOrder::builder`], run with an [`Engine`](crate::Engine).
///
/// ```
/// use genoxide::algorithm::first_order::Step;
/// use genoxide::prelude::*;
/// use genoxide::problems::{AxisParallelEllipsoid, Problem};
///
/// // the ellipsoid Σ i xᵢ² in 1000 dimensions, with its analytic gradient
/// let problem = AxisParallelEllipsoid::new(1_000);
/// let adam = FirstOrder::builder(problem.representation())
///     .step(Step::adam(0.05))
///     .minimize()
///     .seed(1)
///     .build()?;
/// let outcome = Engine::new(adam, problem)
///     .stop_when(Stop::target(1e-10).or(Stop::generations(10_000)))
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Target);
/// # Ok::<(), genoxide::Error>(())
/// ```
///
/// References: Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration
/// methods. *USSR Computational Mathematics and Mathematical Physics* 4(5): 1-17. Nesterov, Y.
/// (1983). A method for solving the convex programming problem with convergence rate O(1/k²).
/// *Soviet Mathematics Doklady* 27: 372-376. Sutskever, I., Martens, J., Dahl, G. and Hinton, G.
/// (2013). On the importance of initialization and momentum in deep learning. *ICML 2013*,
/// PMLR 28(3): 1139-1147, eqs. 1-4. Kingma, D. P. and Ba, J. (2015). Adam: a method for
/// stochastic optimization. *ICLR 2015*, arXiv:1412.6980, Algorithm 1. Loshchilov, I. and Hutter,
/// F. (2019). Decoupled weight decay regularization. *ICLR 2019*, arXiv:1711.05101, Algorithm 2.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FirstOrder {
    real: Real,
    step: Step,
    multiplier: f64,
    // the setting, and what the run uses: never `Auto`
    setting: Gradients,
    gradients: Gradients,
    stencil: Option<Stencil>,
    gradient_tolerance: f64,
    step_tolerance: f64,
    restarts: Restarts,
    objective: Objective,
    seed: u64,
    rng: StreamRng,
    // the iterate the rule steps from: the evaluated point, but for Nesterov's, which evaluates
    // the look-ahead point
    x: Vec<f64>,
    memory: Memory,
    // the last valid point, evaluated, and its gradient
    population: Population<Reals>,
    gradient: Vec<f64>,
    gradient_norm: f64,
    // whether the last step moved no gene by more than the step tolerance (false before the
    // first step of a run)
    small_step: bool,
    // the genomes of an ask: the point, then the stencil's points
    asked: Vec<Individual<Reals>>,
    pending: Vec<usize>,
    // the stencil's scores
    values: Vec<f64>,
    // whether the population's point is valid, and the next ask steps from it
    valid: bool,
    ready: bool,
    converged: Option<Convergence>,
    restart_count: u64,
    iterations: u64,
    reevaluating: bool,
    started: bool,
    asking: bool,
    generation: u64,
    evaluations: u64,
    // the best, unless it's the population's point
    best: Individual<Reals>,
    best_at: BestAt,
    best_generation: u64,
}

impl FirstOrder {
    /// A builder for a first-order method on `real`.
    pub fn builder(real: Real) -> FirstOrderBuilder {
        FirstOrderBuilder {
            real,
            step: Step::adam(0.001),
            gradients: Gradients::Auto,
            gradient_tolerance: 1e-6,
            step_tolerance: 1e-12,
            restarts: Restarts::Never,
            initial_genome: None,
            objective: Objective::default(),
            seed: None,
        }
    }

    /// The representation.
    pub fn real(&self) -> &Real {
        &self.real
    }

    /// The step rule, with the learning rate as it is now.
    pub fn step(&self) -> Step {
        self.step
    }

    /// The schedule multiplier η, 1 unless [changed](FirstOrder::set_multiplier).
    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }

    /// Where the gradients come from: the [setting](FirstOrderBuilder::gradients), resolved by
    /// the engine at the start of a run, never [`Gradients::Auto`]. Before a run, and driven by
    /// hand, `Auto` is forward differences up to [`AUTO_LIMIT`](crate::gradient::AUTO_LIMIT)
    /// genes that aren't fixed, and supplied gradients above it.
    pub fn gradients(&self) -> Gradients {
        self.gradients
    }

    /// The gradient of the last tell: at the [point](Algorithm::population), or at an invalid
    /// point that the method steps back from. All zeros before the first tell.
    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }

    /// The largest component of the projected gradient at the point: the measure of the
    /// [gradient tolerance](FirstOrderBuilder::gradient_tolerance). Infinite before the first
    /// tell.
    pub fn gradient_norm(&self) -> f64 {
        self.gradient_norm
    }

    /// Whether the current run has converged, and why. With [`Restarts`] left, the next
    /// [`ask`](Algorithm::ask) starts a new run.
    pub fn converged(&self) -> Option<Convergence> {
        self.converged
    }

    /// The number of steps of every run: a step back from an invalid point isn't one.
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// The steps t of the current run, the exponent of Adam's corrections: reset by a restart.
    pub fn steps(&self) -> u64 {
        self.memory.steps
    }

    /// The number of restarts so far.
    pub fn restart_count(&self) -> u64 {
        self.restart_count
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Changes the learning rate α during a run, e.g. for a schedule that lowers it, from
    /// [`Engine::control`](crate::Engine::control). It applies from the next step; the velocity
    /// and Adam's averages are kept. AdamW's decay doesn't change with it (it's decoupled from
    /// α): scale both with [`set_multiplier`](FirstOrder::set_multiplier).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a learning rate that isn't greater than 0 and finite.
    /// Nothing changes on errors.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        let step = self.step.with_learning_rate(learning_rate);
        step.validate()?;
        self.step = step;
        Ok(())
    }

    /// Changes the schedule multiplier η of Loshchilov and Hutter (2019, Algorithm 2), which
    /// multiplies every step, AdamW's weight decay included: `x ← x − η (α m̂ / (√v̂ + ε) + λ x)`.
    /// 1 by default. It applies from the next step.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a multiplier that isn't greater than 0 and finite. Nothing
    /// changes on errors.
    pub fn set_multiplier(&mut self, multiplier: f64) -> Result<()> {
        if !(multiplier > 0.0 && multiplier.is_finite()) {
            return Err(Error::InvalidSetting {
                setting: "multiplier",
                reason: format!("must be greater than 0 and finite, got {multiplier}"),
            });
        }
        self.multiplier = multiplier;
        Ok(())
    }

    /// Marks the point as not evaluated, for a fitness function that changed during the run. The
    /// next [`ask`](Algorithm::ask) gives the point (with its finite-difference stencil), and its
    /// tell sets its fitness and gradient without a step.
    ///
    /// - It isn't a generation: [`generation`](Algorithm::generation) doesn't change. The
    ///   evaluations are counted.
    /// - The velocity, Adam's averages and the step count t are kept: the next step uses the new
    ///   gradient with the memory of the old ones, as a continuation of the run.
    /// - A step back from an invalid point under way is dropped: the next step is from the
    ///   point. Convergence is decided again, by the new gradient only.
    /// - [`best`](Algorithm::best) is then the best of the re-evaluated points, found in the
    ///   current generation: old and new values are never compared.
    /// - No random number is drawn: a seeded run that re-evaluates at the same points gives the
    ///   same results.
    /// - Before the run has a valid point, it changes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::ReevaluationOutOfTurn`] between an ask and its tell. Nothing changes on errors.
    pub fn reevaluate(&mut self) -> Result<()> {
        if self.asking {
            return Err(Error::ReevaluationOutOfTurn);
        }
        self.reevaluating = self.started && self.valid;
        Ok(())
    }

    // the sign that makes every rule descend: 1 minimizing, −1 maximizing
    fn sign(&self) -> f64 {
        match self.objective {
            Objective::Minimize => 1.0,
            Objective::Maximize => -1.0,
        }
    }

    // the step rule's move from the population's point, along its gradient: the iterate, the
    // memory and the next point, written into the first asked genome
    fn take_step(&mut self) {
        self.memory.steps += 1;
        self.iterations += 1;
        let sign = self.sign();
        let eta = self.multiplier;
        let bounds = self.real.bounds();
        let gradient = &self.gradient;
        let x = &mut self.x;
        let Moments {
            first,
            second,
            beta1_power,
            beta2_power,
        } = &mut self.memory.moments;
        let point = self.asked[0].genome_mut();
        // whether the iterate moved no gene by more than the step tolerance: for Nesterov's, the
        // iterate, not the look-ahead point, which can stay on a bound while the iterate moves
        let tolerance = self.step_tolerance;
        let mut small = true;
        match self.step {
            Step::Gradient { learning_rate } => {
                let rate = eta * learning_rate;
                let genes = x.iter_mut().zip(point.iter_mut()).zip(gradient);
                for (((xi, yi), &g), range) in genes.zip(bounds) {
                    let moved = project(*xi - rate * (sign * g), range.start(), range.end());
                    small &= within(*xi, moved, tolerance);
                    *xi = moved;
                    *yi = moved;
                }
            }
            Step::Momentum {
                learning_rate,
                momentum,
            }
            | Step::Nesterov {
                learning_rate,
                momentum,
            } => {
                let rate = eta * learning_rate;
                let nesterov = matches!(self.step, Step::Nesterov { .. });
                let genes = x.iter_mut().zip(first.iter_mut()).zip(gradient);
                for (((xi, v), &g), (range, yi)) in genes.zip(bounds.iter().zip(point.iter_mut())) {
                    let (low, high) = (range.start(), range.end());
                    *v = momentum * *v - rate * (sign * g);
                    let moved = *xi + *v;
                    let projected = project(moved, low, high);
                    if projected != moved {
                        // the velocity is the step taken
                        *v = projected - *xi;
                    }
                    small &= within(*xi, projected, tolerance);
                    *xi = projected;
                    // Nesterov's next gradient is at the look-ahead point
                    *yi = if nesterov {
                        project(projected + momentum * *v, low, high)
                    } else {
                        projected
                    };
                }
            }
            Step::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            }
            | Step::AdamW {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                ..
            } => {
                let weight_decay = match self.step {
                    Step::AdamW { weight_decay, .. } => weight_decay,
                    _ => 0.0,
                };
                *beta1_power *= beta1;
                *beta2_power *= beta2;
                let (correction1, correction2) = (1.0 - *beta1_power, 1.0 - *beta2_power);
                let genes = x.iter_mut().zip(point.iter_mut()).zip(first.iter_mut());
                let rest = second.iter_mut().zip(gradient).zip(bounds);
                for (((xi, yi), m), ((v, &g), range)) in genes.zip(rest) {
                    update_moments(m, v, sign * g, beta1, beta2);
                    let m_hat = *m / correction1;
                    let v_hat = *v / correction2;
                    let mut update = learning_rate * m_hat / (v_hat.sqrt() + epsilon);
                    if weight_decay != 0.0 {
                        update += weight_decay * *xi;
                    }
                    let moved = project(*xi - eta * update, range.start(), range.end());
                    small &= within(*xi, moved, tolerance);
                    *xi = moved;
                    *yi = moved;
                }
            }
        }
        self.small_step = small;
    }

    // halfway back from the asked point, which was invalid, to the population's valid point
    fn step_back(&mut self) {
        let bounds = self.real.bounds();
        let valid = self.population[0].genome();
        let point = self.asked[0].genome_mut();
        for ((yi, &p), range) in point.iter_mut().zip(valid.iter()).zip(bounds) {
            *yi = project(p + 0.5 * (*yi - p), range.start(), range.end());
        }
        self.x.copy_from_slice(point);
        if matches!(self.step, Step::Momentum { .. } | Step::Nesterov { .. }) {
            for v in &mut self.memory.moments.first {
                *v *= 0.5;
            }
        }
        self.small_step = self.within_tolerance(1.0);
    }

    // whether `fraction` of the move from the population's point to the asked one changes no
    // gene by more than the step tolerance
    fn within_tolerance(&self, fraction: f64) -> bool {
        let from = self.population[0].genome();
        let to = self.asked[0].genome();
        let tolerance = self.step_tolerance;
        from.iter()
            .zip(to.iter())
            .all(|(&a, &b)| within(a, a + fraction * (b - a), tolerance))
    }

    // the gradient at the asked point, `supplied` or already in `self.gradient` (from the
    // stencil), measured in the same pass: whether it's finite, and the largest component of the
    // projected gradient, 0 for a gene at a bound that the descent direction points out of
    fn absorb(&mut self, supplied: Option<&[f64]>) -> (bool, f64) {
        let sign = self.sign();
        let point = self.asked[0].genome();
        let bounds = self.real.bounds();
        let mut finite = true;
        let mut norm = 0.0f64;
        let mut measure = |x: f64, g: f64, range: &RangeInclusive<f64>| {
            finite &= g.is_finite();
            let descent = -sign * g;
            let blocked =
                (x <= *range.start() && descent <= 0.0) || (x >= *range.end() && descent >= 0.0);
            if !blocked {
                norm = norm.max(g.abs());
            }
        };
        match supplied {
            Some(supplied) => {
                let genes = self.gradient.iter_mut().zip(supplied).zip(point.iter());
                for (((g, &value), &x), range) in genes.zip(bounds) {
                    *g = value;
                    measure(x, value, range);
                }
            }
            None => {
                for ((&g, &x), range) in self.gradient.iter().zip(point.iter()).zip(bounds) {
                    measure(x, g, range);
                }
            }
        }
        (finite, norm)
    }

    // a new run from a random point, with the memory reset
    fn restart(&mut self) {
        self.restart_count += 1;
        let start = self.real.random_genome(&mut self.rng);
        self.x.copy_from_slice(&start);
        self.asked[0].genome_mut().copy_from_slice(&start);
        self.memory.reset();
        self.converged = None;
        self.valid = false;
        self.ready = false;
        self.small_step = false;
    }

    // whether a restart is due at the next ask
    fn restart_due(&self) -> bool {
        self.converged.is_some() && self.restart_count < self.restarts.times()
    }

    // the best so far, or `None` for a re-evaluation, whose points are compared only with each
    // other
    fn best_fitness(&self, reevaluation: bool) -> Option<Fitness> {
        if reevaluation {
            return None;
        }
        self.best_individual()
            .map(|best| best.fitness().unwrap_or(Fitness::invalid()))
    }

    // the asked genome that is the new best, if any: the first of the best, if better than the
    // best so far
    fn new_best(&self, reevaluation: bool) -> Option<usize> {
        let objective = self.objective;
        let mut best = self.best_fitness(reevaluation);
        let mut found = None;
        for (k, individual) in self.asked.iter().enumerate() {
            let fitness = individual.fitness().unwrap_or(Fitness::invalid());
            if best.is_none_or(|best| objective.is_better(fitness, best)) {
                best = Some(fitness);
                found = Some(k);
            }
        }
        found
    }

    fn best_individual(&self) -> Option<&Individual<Reals>> {
        match self.best_at {
            BestAt::Nowhere => None,
            BestAt::Point => Some(&self.population[0]),
            BestAt::Kept => Some(&self.best),
        }
    }

    // the asked genome at `k` copied as the best
    fn copy_best(&mut self, k: usize) {
        self.best.clone_from(&self.asked[k]);
        self.best_at = BestAt::Kept;
    }

    // the evaluated asked genomes, the point valid or not: the best kept, and the point moved
    // into the population if valid. The best is the population's point until a later point is
    // worse, so that a descent copies no genome: the first worse point moves it out of the
    // population, without a copy.
    fn settle(&mut self, valid: bool, reevaluation: bool) {
        let found = self.new_best(reevaluation);
        if found.is_some() {
            self.best_generation = self.generation;
        }
        match (valid, found) {
            (true, Some(0)) => {
                std::mem::swap(&mut self.population[0], &mut self.asked[0]);
                self.best_at = BestAt::Point;
            }
            (true, found) => {
                if self.best_at == BestAt::Point {
                    // the population's point, the best, is kept apart
                    std::mem::swap(&mut self.best, &mut self.population[0]);
                    self.best_at = BestAt::Kept;
                }
                if let Some(k) = found {
                    self.copy_best(k);
                }
                std::mem::swap(&mut self.population[0], &mut self.asked[0]);
            }
            (false, Some(k)) => self.copy_best(k),
            (false, None) => {}
        }
    }
}

// the most genes of all the genomes of an ask with finite differences: n + 1 or 2n + 1 genomes of
// n genes, 2 GiB
const STENCIL_GENES: usize = 1 << 28;

// the gradients before the engine resolves them: `Auto` is forward differences up to
// `AUTO_LIMIT` genes that aren't fixed, as the engine resolves it without a supplied gradient,
// and supplied gradients above it
fn unprepared(gradients: Gradients, real: &Real) -> Gradients {
    match gradients {
        Gradients::Auto => gradients
            .resolve(Provided::NOTHING, real)
            .unwrap_or(Gradients::Supplied),
        gradients => gradients,
    }
}

// the stencil of finite differences, none for supplied gradients
fn stencil(real: &Real, gradients: Gradients) -> Result<Option<Stencil>> {
    if gradients.is_supplied() {
        return Ok(None);
    }
    let stencil = Stencil::new(real, gradients)?;
    let genes = real.bounds().len();
    let total = (stencil.len() + 1).saturating_mul(genes);
    if total > STENCIL_GENES {
        return Err(Error::InvalidSetting {
            setting: "gradients",
            reason: format!(
                "finite differences in {genes} genes would ask {} genomes of {genes} genes each \
                 round, {total} values, above the {STENCIL_GENES} allowed: supply the gradient \
                 (`Differentiable`, or `FitnessFunction::provides` and `evaluate_with`)",
                stencil.len() + 1
            ),
        });
    }
    Ok(Some(stencil))
}

// whether a gene's change from `from` to `to` is within `tolerance`, relative to max(1, |from|)
#[inline]
fn within(from: f64, to: f64, tolerance: f64) -> bool {
    (to - from).abs() <= tolerance * from.abs().max(1.0)
}

// `x` within [low, high]; NaN, from a step that overflowed, onto the lower bound
#[inline]
fn project(x: f64, low: &f64, high: &f64) -> f64 {
    if x > *high {
        *high
    } else if x >= *low {
        x
    } else {
        *low
    }
}

impl Reevaluate for FirstOrder {
    /// As [`FirstOrder::reevaluate`]: the next ask gives the point.
    fn reevaluate(&mut self) -> Result<()> {
        FirstOrder::reevaluate(self)
    }
}

impl Algorithm for FirstOrder {
    type Genome = Reals;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, Reals> {
        let mut moved = false;
        if !self.asking {
            if self.reevaluating {
                let point = self.population[0].genome();
                self.asked[0].genome_mut().copy_from_slice(point);
                self.x.copy_from_slice(point);
            } else if self.started {
                if self.restart_due() {
                    self.restart();
                } else if self.ready {
                    self.take_step();
                } else if self.valid {
                    self.step_back();
                } else {
                    // no valid point: the start again, as it was
                    let start = &self.x;
                    self.asked[0].genome_mut().copy_from_slice(start);
                }
            }
            self.asking = true;
            moved = true;
        }
        if self.asked.len() != self.pending.len() {
            // the genomes of the stencil's points: at the first ask, and when the engine's
            // `prepare` changed the gradients
            let genome = self.asked[0].genome().clone();
            self.asked
                .resize_with(self.pending.len(), || Individual::new(genome.clone()));
            self.asked.truncate(self.pending.len());
            moved = true;
        }
        if moved && let Some(stencil) = &mut self.stencil {
            let (point, points) = self.asked.split_at_mut(1);
            stencil
                .set_center(point[0].genome())
                .expect("the point has a gene per bound");
            for (k, individual) in points.iter_mut().enumerate() {
                stencil
                    .write_point(k, individual.genome_mut())
                    .expect("a point of the stencil");
            }
        }
        Candidates::new(&self.asked, &self.pending)
    }

    fn tell(&mut self, fitness: &[Fitness]) -> Result<()> {
        self.tell_evaluations(&Evaluations::new(fitness))
    }

    fn tell_evaluations(&mut self, evaluations: &Evaluations<'_>) -> Result<()> {
        if !self.asking {
            return Err(Error::TellWithoutAsk);
        }
        if evaluations.len() != self.pending.len() {
            return Err(Error::FitnessCount {
                expected: self.pending.len(),
                got: evaluations.len(),
            });
        }
        let fitness = evaluations.fitness();
        let score = |fitness: Fitness| fitness.score().unwrap_or(f64::NAN);
        let (finite, norm) = match &self.stencil {
            None => {
                let genes = self.gradient.len();
                let gradient = evaluations
                    .gradient(0)
                    .filter(|gradient| gradient.len() == genes)
                    .ok_or_else(|| Error::InvalidFitness {
                        reason: format!(
                            "the method uses supplied gradients and needs one of {genes} values \
                             with the fitness: tell it with `tell_evaluations`, or use finite \
                             differences (`Gradients::Forward`)"
                        ),
                    })?;
                self.absorb(Some(gradient))
            }
            Some(stencil) => {
                self.values.clear();
                self.values.extend(fitness[1..].iter().map(|&f| score(f)));
                stencil.gradient(score(fitness[0]), &self.values, &mut self.gradient)?;
                self.absorb(None)
            }
        };
        self.asking = false;
        self.evaluations += fitness.len() as u64;
        for (individual, &fitness) in self.asked.iter_mut().zip(fitness) {
            individual.set_fitness(fitness);
        }
        let reevaluation = self.reevaluating;
        self.reevaluating = false;
        if !reevaluation && self.started {
            self.generation += 1;
        }
        self.started = true;
        let valid = fitness[0].is_valid() && finite;
        // the step back from an invalid point halves the distance to the valid one
        let small_back = !valid && self.valid && self.within_tolerance(0.5);
        self.settle(valid, reevaluation);
        if valid {
            self.valid = true;
            self.ready = true;
            self.gradient_norm = norm;
            if reevaluation {
                self.small_step = false;
            }
            self.converged = if self.gradient_norm <= self.gradient_tolerance {
                Some(Convergence::Gradient)
            } else if self.small_step {
                Some(Convergence::Step)
            } else {
                None
            };
        } else {
            self.ready = false;
            if self.valid && !reevaluation {
                self.converged = small_back.then_some(Convergence::Step);
            } else {
                self.valid = false;
                self.converged = Some(Convergence::Invalid);
            }
        }
        Ok(())
    }

    fn population(&self) -> &Population<Reals> {
        &self.population
    }

    fn best(&self) -> Option<&Individual<Reals>> {
        self.best_individual()
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn evaluations(&self) -> u64 {
        self.evaluations
    }

    fn best_generation(&self) -> u64 {
        self.best_generation
    }

    /// Whether the run has converged with no restart left.
    fn is_finished(&self) -> bool {
        self.converged.is_some() && !self.restart_due()
    }

    /// Resolves the [`Gradients`] setting against what the fitness function provides.
    fn prepare(&mut self, provided: Provided) -> Result<()> {
        let resolved = self.setting.resolve(provided, &self.real)?;
        if resolved != self.gradients {
            let stencil = stencil(&self.real, resolved)?;
            self.gradients = resolved;
            self.stencil = stencil;
            let points = self.stencil.as_ref().map_or(0, Stencil::len);
            self.pending.clear();
            self.pending.extend(0..1 + points);
        }
        Ok(())
    }

    /// The gradient at the point, when it's supplied; nothing with finite differences.
    fn wants(&self) -> Wanted {
        if self.stencil.is_none() {
            Wanted::GRADIENT
        } else {
            Wanted::NOTHING
        }
    }
}

/// A builder for a [`FirstOrder`] method, from [`FirstOrder::builder`].
///
/// Defaults: maximize, [`Step::adam`] with Kingma and Ba's learning rate 0.001,
/// [`Gradients::Auto`], a gradient tolerance of 1e-6 and a step tolerance of 1e-12, no restarts, a
/// random initial genome and a random seed.
#[derive(Clone, Debug)]
pub struct FirstOrderBuilder {
    real: Real,
    step: Step,
    gradients: Gradients,
    gradient_tolerance: f64,
    step_tolerance: f64,
    restarts: Restarts,
    initial_genome: Option<Reals>,
    objective: Objective,
    seed: Option<u64>,
}

impl FirstOrderBuilder {
    /// The step rule. [`Step::adam`]`(0.001)` by default.
    pub fn step(mut self, step: Step) -> Self {
        self.step = step;
        self
    }

    /// Where the gradients come from. [`Gradients::Auto`] by default: supplied if the fitness
    /// function provides them, forward differences otherwise.
    pub fn gradients(mut self, gradients: Gradients) -> Self {
        self.gradients = gradients;
        self
    }

    /// The largest component of the projected gradient at which a run has converged, 0 or more:
    /// 1e-6 by default. In the units of the score per unit of a gene. Forward differences can't
    /// bring it much below `√ε · max(1, |f|)` (about 1e-8 times the score), so a smaller one
    /// needs supplied gradients or central differences.
    pub fn gradient_tolerance(mut self, tolerance: f64) -> Self {
        self.gradient_tolerance = tolerance;
        self
    }

    /// The step at which a run has converged, 0 or more: the largest change of a gene in the last
    /// step, relative to `max(1, |xᵢ|)`. 1e-12 by default.
    pub fn step_tolerance(mut self, tolerance: f64) -> Self {
        self.step_tolerance = tolerance;
        self
    }

    /// Whether to start again from random points when a run has converged, with the memory
    /// reset. [`Restarts::Never`] by default.
    pub fn restarts(mut self, restarts: Restarts) -> Self {
        self.restarts = restarts;
        self
    }

    /// The genome to start from. Random by default. Restarts start from random points.
    pub fn initial_genome(mut self, genome: Reals) -> Self {
        self.initial_genome = Some(genome);
        self
    }

    /// Whether higher or lower fitness is better. Maximize by default.
    pub fn objective(mut self, objective: Objective) -> Self {
        self.objective = objective;
        self
    }

    /// Higher fitness is better (the default): the method climbs the score.
    pub fn maximize(self) -> Self {
        self.objective(Objective::Maximize)
    }

    /// Lower fitness is better: the method descends the score.
    pub fn minimize(self) -> Self {
        self.objective(Objective::Minimize)
    }

    /// The seed of the random numbers, for a reproducible run: the initial genome, unless it's
    /// given, and the starts of restarts. Random by default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the settings and creates the method.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidSetting`] for an invalid [`Step`] or [`Gradients`] setting, a
    ///   tolerance that isn't 0 or more and finite, or random restarts 0 times.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<FirstOrder> {
        self.step.validate()?;
        self.gradients.validate()?;
        self.restarts.validate()?;
        for (setting, tolerance) in [
            ("gradient_tolerance", self.gradient_tolerance),
            ("step_tolerance", self.step_tolerance),
        ] {
            if !(tolerance >= 0.0 && tolerance.is_finite()) {
                return Err(Error::InvalidSetting {
                    setting,
                    reason: format!("must be 0 or more and finite, got {tolerance}"),
                });
            }
        }
        if let Some(genome) = &self.initial_genome {
            self.real.validate(genome)?;
        }
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let mut rng = StreamRng::seed_from_u64(seed);
        let start = match self.initial_genome {
            Some(genome) => genome,
            None => self.real.random_genome(&mut rng),
        };
        let genes = start.len();
        let resolved = unprepared(self.gradients, &self.real);
        // the velocity or Adam's first average, and Adam's second
        let first = if matches!(self.step, Step::Gradient { .. }) {
            0
        } else {
            genes
        };
        let averages = if self.step.is_adam() { genes } else { 0 };
        let mut first_order = FirstOrder {
            real: self.real,
            step: self.step,
            multiplier: 1.0,
            setting: self.gradients,
            gradients: resolved,
            stencil: None,
            gradient_tolerance: self.gradient_tolerance,
            step_tolerance: self.step_tolerance,
            restarts: self.restarts,
            objective: self.objective,
            seed,
            rng,
            x: start.to_vec(),
            memory: Memory {
                moments: Moments::with(first, averages),
                steps: 0,
            },
            population: Population::new(vec![Individual::new(start.clone())]),
            gradient: vec![0.0; genes],
            gradient_norm: f64::INFINITY,
            small_step: false,
            asked: vec![Individual::new(start.clone())],
            pending: Vec::new(),
            values: Vec::new(),
            valid: false,
            ready: false,
            converged: None,
            restart_count: 0,
            iterations: 0,
            reevaluating: false,
            started: false,
            asking: false,
            generation: 0,
            evaluations: 0,
            best: Individual::new(start),
            best_at: BestAt::Nowhere,
            best_generation: 0,
        };
        first_order.stencil = stencil(&first_order.real, resolved)?;
        let points = first_order.stencil.as_ref().map_or(0, Stencil::len);
        first_order.pending.extend(0..1 + points);
        Ok(first_order)
    }
}
