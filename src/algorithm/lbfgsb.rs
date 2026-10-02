//! L-BFGS-B: the limited-memory BFGS method with bounds, for smooth functions with gradients.
//!
//! See [`Lbfgsb`].

mod model;

use super::continuation::{Continue, Keep};
use super::line_search::{MoreThuente, Settings, Status};
use super::local::Restarts;
use super::{Algorithm, Candidates, Reevaluate};
use crate::engine::{Evaluations, Provided, Wanted};
use crate::genome::{Real, Reals, Representation};
use crate::gradient::{Gradients, Stencil};
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use model::{Memory, Workspace};
use rand::Rng;

/// Why a run of an [`Lbfgsb`] has converged: [`Lbfgsb::converged`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Criterion {
    /// The projected gradient's largest component is within the
    /// [gradient tolerance](LbfgsbBuilder::gradient_tolerance): a minimum in the box, to that
    /// accuracy (Zhu et al., 1997, eq. 2).
    ProjectedGradient,
    /// The last step lowered the function by at most the
    /// [function tolerance](LbfgsbBuilder::function_tolerance), relative to its value (Zhu et
    /// al., 1997, eq. 1).
    RelativeDecrease,
    /// No step lowers the function: the line search found none along the steepest descent
    /// direction, or there's no descent direction left. Usually rounding near a minimum, where
    /// the tolerances ask for more digits than the function or its gradient has (Zhu et al.,
    /// 1997, section 4); or an inaccurate gradient.
    LineSearch,
    /// The function or its gradient isn't finite at the start point (or, after a re-evaluation,
    /// at the current point): there's no direction to search.
    NotFinite,
}

// what the next ask evaluates
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum Phase {
    // the start point of a run: the initial genome, or a random one after a restart
    Start,
    // a trial step of the line search
    Search,
    // nothing: the run has converged and has no restart left
    Finished,
}

/// L-BFGS-B, the limited-memory BFGS method for bound-constrained problems, on [`Real`]
/// genomes, as an ask / tell [`Algorithm`]: a local method for smooth functions with a gradient,
/// supplied by the fitness function or by finite differences, from a few genes to millions.
///
/// Each iteration builds a quadratic model of the function from the gradient and a
/// limited-memory BFGS matrix, the curvature of the last [`memory`](LbfgsbBuilder::memory) steps
/// in compact form (Byrd, Lu, Nocedal and Zhu, 1995, section 3), minimizes it in the box of the
/// [`Real`] genome in two stages, then searches along the step:
///
/// - **The generalized Cauchy point** (section 4, Algorithm CP): the first minimizer of the model
///   along the path of steepest descent bent at the bounds. The genes it holds at a bound are the
///   active set.
/// - **The subspace step** (section 5.1, the direct primal method): the minimizer of the model
///   over the other genes, projected into the box (Morales and Nocedal, 2011), or, if the
///   projection isn't a descent direction, cut back at the box.
/// - **The line search** of Moré and Thuente (1994) along the step, with the sufficient decrease
///   constant 10⁻⁴ and the curvature constant 0.9 of the paper (section 6), never beyond the
///   bounds.
/// - **The update.** The new pair s = xₖ₊₁ − xₖ, y = gₖ₊₁ − gₖ replaces the oldest one, unless
///   sᵀy ≤ ε‖y‖² (eq. 3.9, ε the machine epsilon): then it's skipped, which keeps the matrix
///   positive definite. The scaling θ = yᵀy / sᵀy of the newest pair (Liu and Nocedal, 1989,
///   eq. 4.1) sets the model's scale.
///
/// The first iteration of a run, with no pairs yet, tries a step of length 1 (`min(1, 1/‖d‖)`)
/// towards the Cauchy point and no further, as the authors' implementation does for problems not
/// bounded in every variable: a wide box then gives the run a narrow one gives, wherever no bound
/// is active.
///
/// **Gradients** ([`gradients`](LbfgsbBuilder::gradients)): supplied by the fitness function
/// when it [provides](crate::engine::FitnessFunction::provides) them, as
/// [`Differentiable`](crate::gradient::Differentiable) and the smooth test problems of
/// [`problems`](crate::problems) do, and forward differences otherwise
/// ([`Gradients::Auto`]); [`Lbfgsb::gradients`] says which, once a run has started. With finite
/// differences each trial point is asked together with its [stencil](crate::gradient::Stencil),
/// n (forward) or 2n (central) points more, in one round, so
/// [`parallel`](crate::Engine::parallel) evaluation takes them together.
///
/// **Convergence** ([`converged`](Lbfgsb::converged)): when the largest component of the
/// projected gradient is within the [gradient tolerance](LbfgsbBuilder::gradient_tolerance), when
/// a step lowers the function by less than the
/// [function tolerance](LbfgsbBuilder::function_tolerance), relative to its value, or when no step
/// lowers it (the tests of Zhu, Byrd, Lu and Nocedal, 1997). The method has then
/// [finished](Algorithm::is_finished), unless [`Restarts`] are left: the
/// [`Engine`](crate::Engine) stops with [`StopReason::Converged`](crate::StopReason::Converged).
///
/// - **Generations.** A generation is one round of evaluations: the start point or a trial step,
///   each with its stencil. An iteration takes one round when the first trial is accepted, as it
///   is near a minimum. [`iterations`](Lbfgsb::iterations) counts the iterations.
/// - **Failures.** If the line search finds no lower value, or the model's matrices are singular
///   from rounding, the pairs are dropped and the iteration starts again from steepest descent
///   (Zhu et al., 1997, section 4); if that fails too, the run has converged
///   ([`Criterion::LineSearch`]). Points where the fitness is invalid, or the function or its
///   gradient isn't finite, are failed trials: the line search steps back from them.
/// - **Best.** [`best`](Algorithm::best) is the best point evaluated, stencil points included.
///   The search uses the score only: a constraint violation is ignored by it, though `best`
///   compares by Deb's rules, as everywhere in genoxide.
/// - **Scale.** It keeps 2m vectors of n genes and works on 2m × 2m matrices, never an n × n one:
///   O(m · n) memory and operations per iteration, plus O(m² · k) for the fewer of the k free or
///   bound genes, and no allocation after the first iteration.
///
/// Every operation is a sum, product, quotient or square root in a fixed order, so a seed gives
/// the same run on every platform.
///
/// Built with [`Lbfgsb::builder`], run with an [`Engine`](crate::Engine).
///
/// ```
/// use genoxide::prelude::*;
/// use genoxide::problems::{Problem, Rosenbrock};
///
/// // Rosenbrock's function in 10 dimensions, with its analytic gradient
/// let problem = Rosenbrock::new(10);
/// let lbfgsb = Lbfgsb::builder(problem.representation())
///     .initial_genome(Reals::from(vec![-1.2; 10]))
///     .minimize()
///     .build()?;
/// let outcome = Engine::new(lbfgsb, problem)
///     .stop_when(Stop::evaluations(1_000))
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Converged);
/// assert!(outcome.best_fitness().score().unwrap() < 1e-10);
/// # Ok::<(), genoxide::Error>(())
/// ```
///
/// References: Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). A limited memory algorithm
/// for bound constrained optimization. *SIAM Journal on Scientific Computing* 16(5): 1190-1208.
/// Zhu, C., Byrd, R. H., Lu, P. and Nocedal, J. (1997). Algorithm 778: L-BFGS-B, Fortran
/// subroutines for large-scale bound-constrained optimization. *ACM Transactions on Mathematical
/// Software* 23(4): 550-560. Morales, J. L. and Nocedal, J. (2011). Remark on "Algorithm 778".
/// *ACM Transactions on Mathematical Software* 38(1): 7. Liu, D. C. and Nocedal, J. (1989). On
/// the limited memory BFGS method for large scale optimization. *Mathematical Programming* 45:
/// 503-528. Moré, J. J. and Thuente, D. J. (1994). Line search algorithms with guaranteed
/// sufficient decrease. *ACM Transactions on Mathematical Software* 20(3): 286-307.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Lbfgsb {
    real: Real,
    lower: Vec<f64>,
    upper: Vec<f64>,
    // genes with a single value: never moved, gradient 0
    fixed: Vec<bool>,
    // the setting, and where the gradients come from in the current run
    gradients: Gradients,
    resolved: Gradients,
    gradient_tolerance: f64,
    function_tolerance: f64,
    max_line_search: usize,
    restarts: Restarts,
    // whether a continuation's next stage that keeps the state keeps the pairs
    #[cfg_attr(feature = "serde", serde(default))]
    keep_pairs: bool,
    objective: Objective,
    seed: u64,
    rng: StreamRng,
    memory: Memory,
    // the current point (its genome and fitness), its value to minimize and its gradient
    current: Population<Reals>,
    value: f64,
    gradient: Vec<f64>,
    // the search direction d = x̄ − x, and x̄, the point at step 1
    direction: Vec<f64>,
    target: Vec<f64>,
    // the Cauchy point
    cauchy: Vec<f64>,
    search: Option<MoreThuente>,
    // the step of the next trial, and whether the run is in its first iteration
    step: f64,
    first_iteration: bool,
    // the best trial of the line search so far: its step, point, value, gradient and fitness
    best_step: f64,
    best_point: Vec<f64>,
    best_value: f64,
    best_gradient: Vec<f64>,
    best_fitness: Fitness,
    // the gradient at the last trial
    trial_gradient: Vec<f64>,
    // finite differences around the trial, when the gradients aren't supplied
    stencil: Option<Stencil>,
    phase: Phase,
    converged: Option<Criterion>,
    restart_count: u64,
    iterations: u64,
    skipped: u64,
    resets: u64,
    gradient_evaluations: u64,
    stencil_evaluations: u64,
    reevaluating: bool,
    started: bool,
    asked: bool,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<Reals>>,
    best_generation: u64,
    // the points of the current ask (the trial, then its stencil), built by `ask`
    #[cfg_attr(feature = "serde", serde(skip))]
    points: Vec<Individual<Reals>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pending: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    built: bool,
    // the points of the last tell that aren't the current point
    #[cfg_attr(feature = "serde", serde(skip))]
    discarded: std::ops::Range<usize>,
    #[cfg_attr(feature = "serde", serde(skip))]
    stencil_values: Vec<f64>,
    #[cfg_attr(feature = "serde", serde(skip))]
    workspace: Workspace,
}

// the sufficient decrease and curvature constants of the line search (Byrd et al., section 6)
const LINE_SEARCH: Settings = Settings::QUASI_NEWTON;

impl Lbfgsb {
    /// A builder for an L-BFGS-B search on `real`.
    pub fn builder(real: Real) -> LbfgsbBuilder {
        LbfgsbBuilder {
            real,
            memory: 10,
            gradients: Gradients::Auto,
            gradient_tolerance: 1e-5,
            function_tolerance: 1e7 * f64::EPSILON,
            max_line_search: 20,
            restarts: Restarts::Never,
            keep_pairs: false,
            initial_genome: None,
            objective: Objective::default(),
            seed: None,
        }
    }

    /// The representation.
    pub fn real(&self) -> &Real {
        &self.real
    }

    /// The number of correction pairs the matrix keeps at most: m.
    pub fn memory(&self) -> usize {
        self.memory.capacity()
    }

    /// Changes the number of correction pairs kept, from the next iteration on, keeping the
    /// newest ones that fit: e.g. from [`Engine::control`](crate::Engine::control).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for 0. Nothing changes on errors.
    pub fn set_memory(&mut self, memory: usize) -> Result<()> {
        validate_memory(memory)?;
        if memory != self.memory.capacity() {
            self.memory = self.memory.with_capacity(memory);
        }
        Ok(())
    }

    /// The number of correction pairs stored now, at most [`memory`](Lbfgsb::memory).
    pub fn pairs(&self) -> usize {
        self.memory.len()
    }

    /// Where the gradients come from: the [setting](LbfgsbBuilder::gradients) as resolved for the
    /// current run, never [`Gradients::Auto`]. Before a run, as an ask without an engine would
    /// resolve it: forward differences for `Auto`.
    // the setting resolved, not the setting itself
    #[allow(clippy::misnamed_getters)]
    pub fn gradients(&self) -> Gradients {
        self.resolved
    }

    /// The number of gradients computed: by the fitness function, or each from a stencil of
    /// finite differences. Every point the line search tries has one.
    pub fn gradient_evaluations(&self) -> u64 {
        self.gradient_evaluations
    }

    /// The evaluations of finite-difference stencil points: part of
    /// [`evaluations`](Algorithm::evaluations), 0 with supplied gradients. The cost of the
    /// gradients, n or 2n per gradient for n genes that aren't fixed.
    pub fn stencil_evaluations(&self) -> u64 {
        self.stencil_evaluations
    }

    /// Why the current run has converged, or `None` while it's still searching. With
    /// [`Restarts`] left, the next [`ask`](Algorithm::ask) starts a new run.
    pub fn converged(&self) -> Option<Criterion> {
        self.converged
    }

    /// The largest component of the projected gradient at the current point,
    /// `‖P(x − g) − x‖∞` with P the projection into the box (Byrd et al., eq. 6.1): 0 at a
    /// minimum in the box. NaN before the first tell.
    pub fn projected_gradient(&self) -> f64 {
        if !self.started {
            return f64::NAN;
        }
        projected_gradient(
            self.current[0].genome(),
            &self.gradient,
            &self.lower,
            &self.upper,
        )
    }

    /// The number of completed iterations, of every run: the steps taken.
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// The number of correction pairs skipped because sᵀy ≤ ε‖y‖² (Byrd et al., eq. 3.9), e.g.
    /// after a step cut short by a bound.
    pub fn skipped_pairs(&self) -> u64 {
        self.skipped
    }

    /// The number of times the pairs were dropped to search along steepest descent again: after
    /// a failed line search, a matrix singular from rounding, a direction that isn't one of
    /// descent, or a re-evaluation.
    pub fn memory_resets(&self) -> u64 {
        self.resets
    }

    /// The number of restarts so far.
    pub fn restart_count(&self) -> u64 {
        self.restart_count
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Marks the current point as not evaluated, for a fitness function that changed during the
    /// run, and drops the correction pairs, which describe the old function. The next
    /// [`ask`](Algorithm::ask) gives the current point, and its [`tell`](Algorithm::tell) sets
    /// its value and gradient; the search goes on from there along steepest descent, as at a
    /// start. [`reevaluate_keeping_pairs`](Lbfgsb::reevaluate_keeping_pairs) keeps them.
    ///
    /// - It isn't a generation: [`generation`](Algorithm::generation) doesn't change. The
    ///   evaluations are counted.
    /// - A line search under way is dropped. A run that had converged is checked again at the
    ///   new values: it goes on if its projected gradient is no longer within the tolerance. A
    ///   restart that was due still happens at the next ask after the re-evaluation.
    /// - [`best`](Algorithm::best) is then the current point, found in the current generation:
    ///   old and new values are never compared.
    /// - No random number is drawn. Before the first tell nothing is evaluated yet, and it
    ///   changes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::ReevaluationOutOfTurn`] between an ask and its tell. Nothing changes on errors.
    pub fn reevaluate(&mut self) -> Result<()> {
        self.mark_reevaluation()?;
        if self.reevaluating && self.memory.len() > 0 {
            self.memory.clear();
            self.resets += 1;
        }
        Ok(())
    }

    /// As [`reevaluate`](Lbfgsb::reevaluate), keeping the correction pairs: for a function that
    /// changed a little, whose curvature the old pairs still describe.
    ///
    /// # Errors
    ///
    /// [`Error::ReevaluationOutOfTurn`] between an ask and its tell. Nothing changes on errors.
    pub fn reevaluate_keeping_pairs(&mut self) -> Result<()> {
        self.mark_reevaluation()
    }

    fn mark_reevaluation(&mut self) -> Result<()> {
        if self.asked {
            return Err(Error::ReevaluationOutOfTurn);
        }
        if self.started {
            self.reevaluating = true;
            self.built = false;
        }
        Ok(())
    }

    // the value to minimize: the score, negated when it's maximized; NaN for invalid fitness
    fn value_of(&self, fitness: Fitness) -> f64 {
        let score = fitness.score().unwrap_or(f64::NAN);
        match self.objective {
            Objective::Minimize => score,
            Objective::Maximize => -score,
        }
    }

    // the points of the next ask: the start, current or trial point, then its stencil
    fn build_points(&mut self) {
        let n = self.lower.len();
        let stencil_len = self.stencil.as_ref().map_or(0, Stencil::len);
        let count = if self.phase == Phase::Finished && !self.reevaluating {
            0
        } else {
            1 + stencil_len
        };
        while self.points.len() < count {
            self.points.push(Individual::new(Reals::from(vec![0.0; n])));
        }
        self.pending.clear();
        self.pending.extend(0..count);
        self.built = true;
        if count == 0 {
            return;
        }
        {
            let point = self.points[0].genome_mut();
            if self.reevaluating || self.phase == Phase::Start {
                point.copy_from_slice(self.current[0].genome());
            } else {
                // x + α d in the box; x̄ itself at α = 1, without rounding
                let x = self.current[0].genome();
                let alpha = self.step;
                if alpha == 1.0 {
                    point.copy_from_slice(&self.target);
                } else {
                    // resliced to n: no bounds checks in the loop
                    let (point, x, direction) = (&mut point[..n], &x[..n], &self.direction[..n]);
                    let (lower, upper) = (&self.lower[..n], &self.upper[..n]);
                    for i in 0..n {
                        point[i] = (x[i] + alpha * direction[i]).clamp(lower[i], upper[i]);
                    }
                }
            }
        }
        if let Some(stencil) = &mut self.stencil {
            let (first, rest) = self.points.split_at_mut(1);
            // a center of the right length: no error
            let _ = stencil.set_center(first[0].genome());
            for (k, point) in rest[..stencil_len].iter_mut().enumerate() {
                let _ = stencil.write_point(k, point.genome_mut());
            }
        }
    }

    // a new run from a random point
    fn restart(&mut self) {
        self.restart_count += 1;
        let start = self.real.random_genome(&mut self.rng);
        self.current = Population::new(vec![Individual::new(start)]);
        self.memory.clear();
        self.converged = None;
        self.first_iteration = true;
        self.search = None;
    }

    // the value and gradient at the asked trial point into `trial_gradient`, from the told
    // fitness and the supplied gradient or the stencil's values; false if either isn't finite
    fn trial_value(&mut self, fitness: &[Fitness], gradient: Option<&[f64]>) -> (f64, bool) {
        let value = self.value_of(fitness[0]);
        let sign = match self.objective {
            Objective::Minimize => 1.0,
            Objective::Maximize => -1.0,
        };
        match (&self.stencil, gradient) {
            (Some(stencil), _) => {
                self.stencil_values.clear();
                self.stencil_values.extend(
                    fitness[1..]
                        .iter()
                        .map(|fitness| fitness.score().unwrap_or(f64::NAN)),
                );
                let center = fitness[0].score().unwrap_or(f64::NAN);
                // the lengths are right
                let _ = stencil.gradient(center, &self.stencil_values, &mut self.trial_gradient);
                for g in &mut self.trial_gradient {
                    *g *= sign;
                }
            }
            (None, Some(gradient)) => {
                for (g, &supplied) in self.trial_gradient.iter_mut().zip(gradient) {
                    *g = sign * supplied;
                }
            }
            (None, None) => self.trial_gradient.fill(f64::NAN),
        }
        for (g, &fixed) in self.trial_gradient.iter_mut().zip(&self.fixed) {
            if fixed {
                *g = 0.0;
            }
        }
        let finite = value.is_finite() && self.trial_gradient.iter().all(|g| g.is_finite());
        (value, finite)
    }

    // the run has converged: finished, unless a restart is due
    fn finish(&mut self, criterion: Criterion) {
        self.converged = Some(criterion);
        self.search = None;
        self.phase = if self.restart_count < self.restarts.times() {
            Phase::Start
        } else {
            Phase::Finished
        };
    }

    // the current point is new (a start, a step, a re-evaluation): converged, or the next
    // direction
    fn at_new_point(&mut self, previous_value: Option<f64>) {
        let x = self.current[0].genome();
        let projected = projected_gradient(x, &self.gradient, &self.lower, &self.upper);
        if projected <= self.gradient_tolerance {
            return self.finish(Criterion::ProjectedGradient);
        }
        if let Some(previous) = previous_value {
            let scale = previous.abs().max(self.value.abs()).max(1.0);
            if previous - self.value <= self.function_tolerance * scale {
                return self.finish(Criterion::RelativeDecrease);
            }
        }
        self.converged = None;
        self.next_direction();
    }

    // drops the pairs: the model is steepest descent again
    fn reset_memory(&mut self) {
        self.memory.clear();
        self.resets += 1;
        self.first_iteration = true;
    }

    // the search direction from the current point, and the line search along it
    fn next_direction(&mut self) {
        loop {
            match self.compute_direction() {
                Some(slope) => {
                    self.start_line_search(slope);
                    return;
                }
                None if self.memory.len() > 0 => self.reset_memory(),
                None => return self.finish(Criterion::LineSearch),
            }
        }
    }

    // the Cauchy point, the subspace step and d = x̄ − x; the slope gᵀd if it's a descent
    // direction
    fn compute_direction(&mut self) -> Option<f64> {
        let x = self.current[0].genome();
        let workspace = &mut self.workspace;
        workspace.reserve(x.len(), self.memory.capacity());
        if self.memory.len() > 0 && workspace.factor_middle(&self.memory).is_err() {
            return None;
        }
        workspace.cauchy_point(
            &self.memory,
            x,
            &self.gradient,
            &self.lower,
            &self.upper,
            &self.fixed,
            &mut self.cauchy,
        );
        workspace
            .subspace_step(
                &self.memory,
                x,
                &self.gradient,
                &self.lower,
                &self.upper,
                &self.fixed,
                &self.cauchy,
                &mut self.target,
            )
            .ok()?;
        let n = x.len();
        let (target, gradient) = (&self.target[..n], &self.gradient[..n]);
        let direction = &mut self.direction[..n];
        let mut slope = 0.0;
        for i in 0..n {
            let d = target[i] - x[i];
            direction[i] = d;
            slope += gradient[i] * d;
        }
        (slope < 0.0 && slope.is_finite()).then_some(slope)
    }

    fn start_line_search(&mut self, slope: f64) {
        let x = self.current[0].genome();
        // the largest step in the box: 1 in the first iteration, as the authors' implementation
        let max_step = if self.first_iteration {
            1.0
        } else {
            let n = x.len();
            let (direction, lower, upper) =
                (&self.direction[..n], &self.lower[..n], &self.upper[..n]);
            let mut max_step = f64::MAX;
            for i in 0..n {
                let d = direction[i];
                let room = if d > 0.0 {
                    (upper[i] - x[i]) / d
                } else if d < 0.0 {
                    (lower[i] - x[i]) / d
                } else {
                    continue;
                };
                max_step = max_step.min(room);
            }
            max_step.max(f64::MIN_POSITIVE)
        };
        let initial = if self.first_iteration {
            let norm = self.direction.iter().map(|d| d * d).sum::<f64>().sqrt();
            (1.0 / norm).min(1.0)
        } else {
            1.0
        }
        .min(max_step);
        let settings = Settings {
            max_step,
            max_trials: self.max_line_search,
            ..LINE_SEARCH
        };
        match MoreThuente::new(self.value, slope, initial, settings) {
            Ok(search) => {
                self.search = Some(search);
                self.step = initial;
                self.best_step = 0.0;
                self.phase = Phase::Search;
            }
            // a step too small to represent: no step lowers the function
            Err(_) => self.finish(Criterion::LineSearch),
        }
    }

    // the trial at `self.step` was evaluated: the line search's next step, or the end of the
    // iteration
    fn line_search_told(&mut self, value: f64, finite: bool) {
        let Some(search) = &mut self.search else {
            return;
        };
        let step = self.step;
        let derivative = if finite {
            let mut slope = 0.0;
            for (g, d) in self.trial_gradient.iter().zip(&self.direction) {
                slope += g * d;
            }
            slope
        } else {
            f64::NAN
        };
        let (value, derivative) = if finite {
            (value, derivative)
        } else {
            (f64::NAN, f64::NAN)
        };
        let status = search.tell(value, derivative);
        // the step the search ends at, if it ends
        let end = match status {
            Status::Evaluate(_) => None,
            Status::Converged(end) | Status::Stopped { step: end, .. } => Some(end),
        };
        // the trial is kept if it's the search's best point (α_l) or the step it ends at
        if finite && (search.best().step == step || end == Some(step)) {
            self.best_step = step;
            self.best_value = value;
            self.best_fitness = self.points[0].fitness().unwrap_or(Fitness::invalid());
            self.best_point.copy_from_slice(self.points[0].genome());
            std::mem::swap(&mut self.best_gradient, &mut self.trial_gradient);
        }
        match (status, end) {
            (Status::Evaluate(next), _) => self.step = next,
            (_, Some(end))
                if end > 0.0 && end == self.best_step && self.best_value < self.value =>
            {
                self.accept(end == step);
            }
            _ if self.memory.len() > 0 => {
                // no lower point: the authors' implementation searches along steepest descent
                // again
                self.reset_memory();
                self.search = None;
                self.next_direction();
            }
            _ => self.finish(Criterion::LineSearch),
        }
    }

    // the best trial becomes the current point (`last`: the trial just told): the pair, the
    // convergence tests, the next direction
    fn accept(&mut self, last: bool) {
        self.iterations += 1;
        self.search = None;
        let x = self.current[0].genome();
        if !self.memory.update(
            &self.best_point,
            x,
            &self.best_gradient,
            &self.gradient,
            f64::EPSILON,
        ) {
            self.skipped += 1;
        }
        let previous = self.value;
        let current = &mut self.current[0];
        current.genome_mut().copy_from_slice(&self.best_point);
        current.set_fitness(self.best_fitness);
        std::mem::swap(&mut self.gradient, &mut self.best_gradient);
        self.value = self.best_value;
        if last {
            // the trial is the current point now, not a discarded one
            self.discarded.start = 1;
        }
        self.first_iteration = false;
        self.at_new_point(Some(previous));
    }

    // the fitness of the asked points, with the supplied gradient of the trial
    fn receive(&mut self, fitness: &[Fitness], gradient: Option<&[f64]>) -> Result<()> {
        if !self.asked {
            return Err(Error::TellWithoutAsk);
        }
        if fitness.len() != self.pending.len() {
            return Err(Error::FitnessCount {
                expected: self.pending.len(),
                got: fitness.len(),
            });
        }
        if !self.built {
            self.build_points();
        }
        self.asked = false;
        self.built = false;
        self.evaluations += fitness.len() as u64;
        if fitness.is_empty() {
            // a finished run, asked again
            self.generation += 1;
            self.discarded = 0..0;
            return Ok(());
        }
        self.gradient_evaluations += 1;
        self.stencil_evaluations += (fitness.len() - 1) as u64;
        for (point, &fitness) in self.points.iter_mut().zip(fitness) {
            point.set_fitness(fitness);
        }
        if !self.reevaluating && self.started {
            self.generation += 1;
        }
        self.discarded = 0..fitness.len();
        let (value, finite) = self.trial_value(fitness, gradient);
        if self.reevaluating || self.phase == Phase::Start {
            // the current point: scored, and the best of this generation on a re-evaluation
            let current = &mut self.current[0];
            current.set_fitness(fitness[0]);
            self.discarded.start = 1;
            if self.reevaluating {
                self.best = None;
                self.best_generation = self.generation;
            }
            update_best(
                &mut self.best,
                &mut self.best_generation,
                self.generation,
                self.objective,
                &self.points[..fitness.len()],
            );
            self.started = true;
            self.value = value;
            std::mem::swap(&mut self.gradient, &mut self.trial_gradient);
            let restart_due =
                self.reevaluating && self.phase == Phase::Start && self.converged.is_some();
            self.reevaluating = false;
            self.search = None;
            if restart_due {
                return Ok(());
            }
            self.first_iteration = true;
            if !finite {
                self.finish(Criterion::NotFinite);
            } else {
                self.at_new_point(None);
            }
            return Ok(());
        }
        update_best(
            &mut self.best,
            &mut self.best_generation,
            self.generation,
            self.objective,
            &self.points[..fitness.len()],
        );
        self.line_search_told(value, finite);
        Ok(())
    }
}

// ‖P(x − g) − x‖∞: the largest move of a projected steepest descent step of length 1
fn projected_gradient(x: &[f64], g: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    let n = x.len();
    let (g, lower, upper) = (&g[..n], &lower[..n], &upper[..n]);
    let mut largest = 0.0f64;
    for i in 0..n {
        let moved = (x[i] - g[i]).clamp(lower[i], upper[i]) - x[i];
        if moved.is_nan() {
            std::hint::cold_path();
            return f64::NAN;
        }
        largest = largest.max(moved.abs());
    }
    largest
}

// the best so far, from the evaluated `individuals` in order: the first on ties; in the memory
// of the old best, without allocating
fn update_best(
    best: &mut Option<Individual<Reals>>,
    best_generation: &mut u64,
    generation: u64,
    objective: Objective,
    individuals: &[Individual<Reals>],
) {
    for individual in individuals {
        let fitness = individual.fitness().unwrap_or(Fitness::invalid());
        match best {
            Some(best) => {
                if objective.is_better(fitness, best.fitness().unwrap_or(Fitness::invalid())) {
                    best.clone_from(individual);
                    *best_generation = generation;
                }
            }
            None => {
                *best = Some(individual.clone());
                *best_generation = generation;
            }
        }
    }
}

fn validate_memory(memory: usize) -> Result<()> {
    if memory == 0 {
        return Err(Error::InvalidSetting {
            setting: "memory",
            reason: "must be at least 1 correction pair, got 0".to_string(),
        });
    }
    Ok(())
}

impl Reevaluate for Lbfgsb {
    /// As [`Lbfgsb::reevaluate`]: the next ask gives the current point, and the pairs are
    /// dropped.
    fn reevaluate(&mut self) -> Result<()> {
        Lbfgsb::reevaluate(self)
    }
}

impl Continue for Lbfgsb {
    /// The current point evaluated again, as [`Lbfgsb::reevaluate`] does, and the search goes on
    /// from it along steepest descent, as at a start: the curvature pairs are dropped, unless the
    /// method was built with [`keep_pairs`](LbfgsbBuilder::keep_pairs) and `keep` is
    /// [`Keep::State`] ([`Lbfgsb::reevaluate_keeping_pairs`]). The convergence is decided again at
    /// the new values, and a restart that was due is dropped.
    fn next_stage(&mut self, keep: Keep) -> Result<()> {
        if keep == Keep::State && self.keep_pairs {
            self.reevaluate_keeping_pairs()?;
        } else {
            self.reevaluate()?;
        }
        if self.reevaluating {
            self.converged = None;
        }
        Ok(())
    }
}

impl Algorithm for Lbfgsb {
    type Genome = Reals;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, Reals> {
        if !self.asked {
            if self.phase == Phase::Start && self.started && !self.reevaluating {
                self.restart();
            }
            self.build_points();
            self.asked = true;
        } else if !self.built {
            // after loading a checkpoint saved between an ask and its tell
            self.build_points();
        }
        Candidates::new(&self.points, &self.pending)
    }

    fn tell(&mut self, fitness: &[Fitness]) -> Result<()> {
        if self.asked && self.resolved.is_supplied() && !fitness.is_empty() {
            return Err(Error::InvalidSetting {
                setting: "gradients",
                reason: "the gradients are supplied: tell them with `tell_evaluations`".to_string(),
            });
        }
        self.receive(fitness, None)
    }

    fn population(&self) -> &Population<Reals> {
        &self.current
    }

    fn best(&self) -> Option<&Individual<Reals>> {
        self.best.as_ref()
    }

    fn discarded(&self) -> &[Individual<Reals>] {
        self.points.get(self.discarded.clone()).unwrap_or(&[])
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
        self.phase == Phase::Finished && !self.reevaluating
    }

    /// Resolves [`Gradients::Auto`]: supplied if the fitness function provides gradients,
    /// forward differences otherwise.
    fn prepare(&mut self, provided: Provided) -> Result<()> {
        let resolved = self.gradients.resolve(provided, &self.real)?;
        if resolved != self.resolved {
            self.set_resolved(resolved)?;
        }
        Ok(())
    }

    fn wants(&self) -> Wanted {
        if self.resolved.is_supplied() {
            Wanted::GRADIENT
        } else {
            Wanted::NOTHING
        }
    }

    fn tell_evaluations(&mut self, evaluations: &Evaluations<'_>) -> Result<()> {
        if !self.resolved.is_supplied() {
            return self.receive(evaluations.fitness(), None);
        }
        let fitness = evaluations.fitness();
        if fitness.is_empty() {
            return self.receive(fitness, None);
        }
        let n = self.lower.len();
        match evaluations.gradient(0) {
            Some(gradient) if gradient.len() == n => self.receive(fitness, Some(gradient)),
            _ => Err(Error::InvalidSetting {
                setting: "gradients",
                reason: format!(
                    "the gradients are supplied, but the evaluations have no gradient of {n} \
                     values"
                ),
            }),
        }
    }
}

impl Lbfgsb {
    /// Minimizes `f` in the box of `real` from `start` with at most `max_evaluations` evaluations,
    /// driving the search by hand with the gradient `f` writes: the inner solver of the Gaussian
    /// processes' hyperparameters and of Bayesian optimization's acquisition functions. `f` writes
    /// the gradient into its second argument (zeroed) and returns the value; a value or gradient
    /// that isn't finite is a failed trial, which the line search steps back from.
    ///
    /// Returns the best point and its value, or `None` if no point had a finite value. No random
    /// number is drawn, so the result depends only on `start` and `f`.
    pub(crate) fn minimize_with(
        real: Real,
        start: Reals,
        max_evaluations: u64,
        mut f: impl FnMut(&[f64], &mut [f64]) -> f64,
    ) -> Option<(Reals, f64)> {
        let n = start.len();
        let mut lbfgsb = Lbfgsb::builder(real)
            .gradients(Gradients::Supplied)
            .initial_genome(start)
            .minimize()
            .seed(0)
            .build()
            .ok()?;
        let mut gradient = vec![0.0; n];
        while lbfgsb.evaluations < max_evaluations && !lbfgsb.is_finished() {
            let candidates = lbfgsb.ask();
            let Some(x) = candidates.get(0) else { break };
            gradient.fill(0.0);
            let value = f(x, &mut gradient);
            let fitness = if value.is_finite() {
                Fitness::new(value)
            } else {
                Fitness::invalid()
            };
            lbfgsb.receive(&[fitness], Some(&gradient)).ok()?;
        }
        let best = lbfgsb.best.take()?;
        let value = best.fitness()?.score()?;
        value.is_finite().then(|| (best.into_genome(), value))
    }

    // the gradients of this run: a stencil for finite differences; an ask under way is asked
    // again
    fn set_resolved(&mut self, resolved: Gradients) -> Result<()> {
        self.stencil = match resolved {
            Gradients::Forward { .. } | Gradients::Central { .. } => {
                Some(Stencil::new(&self.real, resolved)?)
            }
            _ => None,
        };
        self.resolved = resolved;
        self.asked = false;
        self.built = false;
        Ok(())
    }
}

/// A builder for an [`Lbfgsb`], from [`Lbfgsb::builder`].
///
/// Defaults: maximize, a memory of 10 pairs, [`Gradients::Auto`], a gradient tolerance of 1e-5
/// and a function tolerance of 10⁷ ε ≈ 2.2e-9, at most 20 trials per line search, no restarts, a
/// random initial genome and a random seed.
#[derive(Clone, Debug)]
pub struct LbfgsbBuilder {
    real: Real,
    memory: usize,
    gradients: Gradients,
    gradient_tolerance: f64,
    function_tolerance: f64,
    max_line_search: usize,
    restarts: Restarts,
    keep_pairs: bool,
    initial_genome: Option<Reals>,
    objective: Objective,
    seed: Option<u64>,
}

impl LbfgsbBuilder {
    /// The number of correction pairs (sᵢ, yᵢ) the limited-memory matrix keeps, m: 10 by default.
    /// The authors recommend 3 to 20 (Zhu et al., 1997, section 1): more pairs model the
    /// curvature better, for fewer iterations on hard problems, at O(m · n) memory and work per
    /// iteration.
    pub fn memory(mut self, memory: usize) -> Self {
        self.memory = memory;
        self
    }

    /// Where the gradients come from: [`Gradients::Auto`] by default, the fitness function's if
    /// it provides them and forward differences otherwise.
    pub fn gradients(mut self, gradients: Gradients) -> Self {
        self.gradients = gradients;
        self
    }

    /// A run has converged when the largest component of the projected gradient,
    /// [`Lbfgsb::projected_gradient`], is at most this (`pgtol` of Zhu et al., 1997, eq. 2): 1e-5
    /// by default, as in the authors' tests. Absolute, so it depends on the scale of the function
    /// and of the genes. At least 0; 0 turns the test off but at a stationary point.
    ///
    /// With forward differences, the gradient's error near a minimum is about √ε · max(1, |f|),
    /// so a tolerance much below 1e-7 of the function's scale is rarely met: the run then ends
    /// by the relative decrease or a failed line search instead.
    pub fn gradient_tolerance(mut self, tolerance: f64) -> Self {
        self.gradient_tolerance = tolerance;
        self
    }

    /// A run has converged when a step lowers the function by at most this times
    /// `max(|fₖ|, |fₖ₊₁|, 1)` (`factr · ε` of Zhu et al., 1997, eq. 1): 10⁷ ε ≈ 2.2e-9 by
    /// default, the authors' "moderate accuracy". 10¹² ε for low accuracy, 10 ε for extremely
    /// high; 0 stops only when a step doesn't lower the function at all. At least 0.
    pub fn function_tolerance(mut self, tolerance: f64) -> Self {
        self.function_tolerance = tolerance;
        self
    }

    /// The most trial steps of a line search before it gives up: 20 by default, as the authors'
    /// implementation (Zhu et al., 1997, section 4). At least 1.
    pub fn max_line_search(mut self, trials: usize) -> Self {
        self.max_line_search = trials;
        self
    }

    /// Whether to start again from random points when a run has converged.
    /// [`Restarts::Never`] by default.
    pub fn restarts(mut self, restarts: Restarts) -> Self {
        self.restarts = restarts;
        self
    }

    /// Whether the next stage of a [`Continuation`](super::Continuation) that keeps the state
    /// ([`Keep::State`]) keeps the correction pairs: false by default, as the pairs describe the
    /// old function's curvature. Keep them for stages whose functions differ little, where the
    /// old curvature still serves.
    pub fn keep_pairs(mut self, keep: bool) -> Self {
        self.keep_pairs = keep;
        self
    }

    /// The genome to start from, e.g. the best of a global method, to polish it. Random by
    /// default. Restarts start from random points.
    pub fn initial_genome(mut self, genome: Reals) -> Self {
        self.initial_genome = Some(genome);
        self
    }

    /// Whether higher or lower fitness is better. Maximize by default.
    pub fn objective(mut self, objective: Objective) -> Self {
        self.objective = objective;
        self
    }

    /// Higher fitness is better (the default).
    pub fn maximize(self) -> Self {
        self.objective(Objective::Maximize)
    }

    /// Lower fitness is better.
    pub fn minimize(self) -> Self {
        self.objective(Objective::Minimize)
    }

    /// The seed of the random numbers, for a reproducible run: the initial genome, unless it's
    /// given, and the starts of restarts. Random by default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the settings and creates the search.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidSetting`] for a representation without a gene that has more than one
    ///   value, a memory of 0, an invalid finite-difference step, tolerances that aren't finite
    ///   and at least 0, a line search of 0 trials, or random restarts 0 times.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<Lbfgsb> {
        let invalid =
            |setting: &'static str, reason: String| Err(Error::InvalidSetting { setting, reason });
        if self.real.variable_genes().is_empty() {
            return invalid(
                "real",
                "L-BFGS-B needs a gene with more than one value".to_string(),
            );
        }
        validate_memory(self.memory)?;
        self.gradients.validate()?;
        for (setting, tolerance) in [
            ("gradient_tolerance", self.gradient_tolerance),
            ("function_tolerance", self.function_tolerance),
        ] {
            if !(tolerance >= 0.0 && tolerance.is_finite()) {
                return invalid(
                    setting,
                    format!("must be finite and at least 0, got {tolerance}"),
                );
            }
        }
        if self.max_line_search == 0 {
            return invalid(
                "max_line_search",
                "must be at least 1 trial, got 0".to_string(),
            );
        }
        self.restarts.validate()?;
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
        let n = start.len();
        let bounds = self.real.bounds();
        let lower: Vec<f64> = bounds.iter().map(|range| *range.start()).collect();
        let upper: Vec<f64> = bounds.iter().map(|range| *range.end()).collect();
        let fixed = lower.iter().zip(&upper).map(|(l, u)| l == u).collect();
        // driven by hand, `Auto` is forward differences
        let resolved = match self.gradients {
            Gradients::Auto => Gradients::Forward { step: None },
            other => other,
        };
        let mut lbfgsb = Lbfgsb {
            lower,
            upper,
            fixed,
            gradients: self.gradients,
            resolved,
            gradient_tolerance: self.gradient_tolerance,
            function_tolerance: self.function_tolerance,
            max_line_search: self.max_line_search,
            keep_pairs: self.keep_pairs,
            restarts: self.restarts,
            objective: self.objective,
            seed,
            rng,
            memory: Memory::new(n, self.memory),
            current: Population::new(vec![Individual::new(start)]),
            value: f64::NAN,
            gradient: vec![0.0; n],
            direction: vec![0.0; n],
            target: vec![0.0; n],
            cauchy: vec![0.0; n],
            search: None,
            step: 1.0,
            first_iteration: true,
            best_step: 0.0,
            best_point: vec![0.0; n],
            best_value: f64::NAN,
            best_gradient: vec![0.0; n],
            best_fitness: Fitness::invalid(),
            trial_gradient: vec![0.0; n],
            stencil: None,
            phase: Phase::Start,
            converged: None,
            restart_count: 0,
            iterations: 0,
            skipped: 0,
            resets: 0,
            gradient_evaluations: 0,
            stencil_evaluations: 0,
            reevaluating: false,
            started: false,
            asked: false,
            generation: 0,
            evaluations: 0,
            best: None,
            best_generation: 0,
            points: Vec::new(),
            pending: Vec::new(),
            built: false,
            discarded: 0..0,
            stencil_values: Vec::new(),
            workspace: Workspace::default(),
            real: self.real,
        };
        lbfgsb.set_resolved(resolved)?;
        Ok(lbfgsb)
    }
}
