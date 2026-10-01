//! Continuation: one problem solved in stages, a parameter of the fitness function changing
//! between them, with the optimizer's state kept.
//!
//! Some smooth problems are solved best in stages: a smooth version first, then sharper ones, each
//! started from the last stage's result. Between stages a parameter of the fitness function
//! changes (the sharpness of a smoothed maximum or absolute value, the steepness of a projection,
//! a penalty weight, a filter radius), and the optimizer carries on where it was, its state kept:
//! Adam's moment estimates, MMA's asymptotes, a Nelder-Mead simplex, a CMA-ES distribution.
//!
//! [`Continuation`] runs an algorithm that can go on to a next stage, one that implements
//! [`Continue`], through a number of stages. See [`Continuation`].

use super::{Algorithm, Candidates, Reevaluate};
use crate::engine::{Evaluations, Provided, Wanted};
use crate::{Error, Fitness, Individual, Objective, Population, Result};
use std::fmt;
use std::sync::Arc;

/// What an algorithm keeps of its state when a [`Continuation`] goes on to the next stage.
///
/// Either way the current point (and whatever else the algorithm compares with, such as a
/// simplex or a population) is evaluated again on the changed fitness function, and the
/// algorithm's convergence is decided again, by the new function only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Keep {
    /// The optimizer's state (the default): Adam's averages and step count, the velocity of
    /// momentum, MMA's asymptotes and the iterates their rule reads, a Nelder-Mead simplex, a
    /// CMA-ES distribution. L-BFGS-B drops its curvature pairs unless it was built to keep them
    /// ([`LbfgsbBuilder::keep_pairs`](super::LbfgsbBuilder::keep_pairs)). See each algorithm's
    /// [`Continue`] implementation.
    #[default]
    State,
    /// Only the point: the state starts again as at the start of a run, from the point.
    Point,
}

/// An algorithm that can go on to the next stage of a [`Continuation`], on a changed fitness
/// function.
///
/// [`next_stage`](Continue::next_stage) marks what the algorithm keeps for evaluation by the next
/// [`ask`](Algorithm::ask), as [`Reevaluate::reevaluate`] does, keeps or resets its state as
/// [`Keep`] says, and resets its convergence, so that a run that has
/// [finished](Algorithm::is_finished) goes on. What carries over is each algorithm's to define:
///
/// | Algorithm | [`Keep::State`] | [`Keep::Point`] |
/// |---|---|---|
/// | [`FirstOrder`](super::FirstOrder) | the velocity of momentum, Adam's averages and the step count t of its corrections | all three reset: t starts again from 0 |
/// | [`Mma`](super::Mma) | the asymptotes and the last iterates their rule reads (Svanberg 2007, eqs. 3.11-3.14), and the move limit's fraction | the asymptotes at their initial distance from the point, as at a start |
/// | [`Lbfgsb`](super::Lbfgsb) | the curvature pairs dropped (they describe the old function), unless built with [`keep_pairs`](super::LbfgsbBuilder::keep_pairs) | the pairs dropped |
/// | [`NelderMead`](super::NelderMead) | the simplex, evaluated again and reordered | a new simplex of the initial steps around the best vertex |
/// | [`Cmaes`](super::Cmaes) | the distribution: its mean, step size, covariance matrix and evolution paths | a new distribution around the mean, of the run's initial step size |
///
/// A restart that was due when the stage ended is dropped: the next stage goes on from the
/// point. The counts (generations, evaluations, iterations, restarts) carry on, and no random
/// number is drawn.
pub trait Continue: Reevaluate {
    /// Goes on to the next stage of a [`Continuation`], on a changed fitness function: the next
    /// [`ask`](Algorithm::ask) gives the point to evaluate again (with the rest of what the
    /// algorithm keeps), its tell scores it without a generation, and the run goes on from there.
    /// [`best`](Algorithm::best) is then by the new function only.
    ///
    /// # Errors
    ///
    /// [`Error::ReevaluationOutOfTurn`] between an ask and its tell. Nothing changes on errors.
    fn next_stage(&mut self, keep: Keep) -> Result<()>;
}

/// Why a stage of a [`Continuation`] ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StageEnd {
    /// The wrapped algorithm [finished](Algorithm::is_finished): it has converged, with no
    /// restart left.
    Finished,
    /// The stage's [budget of generations](ContinuationBuilder::generations) ran out.
    Generations,
}

/// What a finished stage of a [`Continuation`] did: [`Continuation::stages`], and the argument of
/// [`on_stage_finished`](ContinuationBuilder::on_stage_finished).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Stage {
    index: usize,
    generations: u64,
    evaluations: u64,
    best: Fitness,
    end: StageEnd,
}

impl Stage {
    /// The stage's index, from 0.
    pub fn index(&self) -> usize {
        self.index
    }

    /// The generations of the wrapped algorithm in the stage: for a local method, its
    /// iterations or rounds of evaluations. The re-evaluation that starts a stage isn't one.
    pub fn generations(&self) -> u64 {
        self.generations
    }

    /// The evaluations of the stage, the re-evaluation that starts it included.
    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }

    /// The fitness of the best individual at the end of the stage, by the stage's fitness
    /// function: the wrapped algorithm's [`best`](Algorithm::best), which compares only the
    /// values of the stage.
    pub fn best(&self) -> Fitness {
        self.best
    }

    /// Why the stage ended.
    pub fn end(&self) -> StageEnd {
        self.end
    }
}

// sets the fitness function's parameters, and the algorithm's settings, for a stage
type OnStage<A> = Arc<dyn Fn(usize, &mut A) -> Result<()> + Send + Sync>;
// called with a finished stage
type OnStageFinished<A> = Arc<dyn Fn(&Stage, &A) + Send + Sync>;

/// Continuation: an algorithm run through stages of one problem, a parameter of the fitness
/// function changing between them, with the algorithm's state kept. For smooth problems solved
/// best in stages, a smooth version first and sharper ones after it, each from the last stage's
/// result.
///
/// - **Stages.** A stage ends when the wrapped algorithm has [finished](Algorithm::is_finished)
///   (converged, with no restart left), or after its own budget of
///   [generations](ContinuationBuilder::generations). Then the
///   [`on_stage`](ContinuationBuilder::on_stage) closure sets the next stage's parameters, shared
///   with the fitness function (e.g. an `Arc<AtomicU64>` holding an `f64`'s bits, as the penalty
///   example of [`Engine::control`](crate::Engine::control) shares its weight), and the algorithm
///   goes on to the next stage with [`Continue::next_stage`]: its point is evaluated again on the
///   changed function, and it goes on from there with the state [`Keep`] says.
/// - **Finishing.** The continuation has [finished](Algorithm::is_finished) only after its last
///   stage, so the [`Engine`](crate::Engine) stops with
///   [`StopReason::Converged`](crate::StopReason::Converged) then, not when a stage converges. A
///   stage's convergence can't end the run: that's why it's a wrapper, and not something done in
///   [`Engine::control`](crate::Engine::control), which runs after the engine has checked
///   convergence. The engine's stop conditions still apply, to the whole run.
/// - **The closure** is called with the index of a stage, from 0, and the algorithm: it sets
///   everything that stage needs from its index alone. It's called for the current stage when a
///   run starts ([`prepare`](Algorithm::prepare), which the engine calls at the start of every
///   run), and for each next stage when it begins. So a run resumed from a checkpoint sets the
///   parameters of the stage it resumes in, and a run continued by another engine sets them
///   again, to the same values. Driven by hand, call `prepare` before the first ask.
/// - **Reporting.** Each stage's generations, evaluations, best fitness and end are kept
///   ([`stages`](Continuation::stages)) and given to the
///   [`on_stage_finished`](ContinuationBuilder::on_stage_finished) closure, if any.
/// - **Generations and best.** [`generation`](Algorithm::generation),
///   [`evaluations`](Algorithm::evaluations), [`best`](Algorithm::best) and the population are the
///   wrapped algorithm's. Each stage starts with a re-evaluation, which isn't a generation, so
///   `best` is by the current stage's function: at the end, the last stage's best. The engine's
///   [`Stop::target`](crate::Stop::target) compares that best, which in an earlier stage is by an
///   earlier stage's function.
/// - **Checkpoints** (the `serde` feature) hold the stage index, the finished stages and the
///   wrapped algorithm, so a resumed run continues in its stage, with the results it would have
///   had without the interruption. The closures aren't part of a checkpoint, like an engine's
///   observers and controls: give them again to the loaded continuation with
///   [`set_on_stage`](Continuation::set_on_stage) and
///   [`set_on_stage_finished`](Continuation::set_on_stage_finished). A run without an `on_stage`
///   closure fails at its start with [`Error::MissingSetting`].
/// - **Scale.** Nothing is allocated per generation: the stages' records have their room from the
///   start.
///
/// ```
/// use genoxide::algorithm::continuation::Continuation;
/// use genoxide::prelude::*;
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicU64, Ordering};
///
/// // Σ √((xᵢ − cᵢ)² + ε²): a smoothed Σ |xᵢ − cᵢ|, sharper as ε shrinks
/// const EPSILON: [f64; 3] = [1.0, 0.1, 0.01];
/// let c = [0.3, -1.2, 2.0];
/// let epsilon = Arc::new(AtomicU64::new(EPSILON[0].to_bits()));
/// let shared = Arc::clone(&epsilon);
/// let smoothed = Differentiable(move |x: &Reals, gradient: &mut [f64]| {
///     let e = f64::from_bits(shared.load(Ordering::Relaxed));
///     let mut value = 0.0;
///     for i in 0..x.len() {
///         let root = ((x[i] - c[i]) * (x[i] - c[i]) + e * e).sqrt();
///         gradient[i] = (x[i] - c[i]) / root;
///         value += root;
///     }
///     value
/// });
/// let lbfgsb = Lbfgsb::builder(Real::uniform(3, -5.0..=5.0)?)
///     .gradient_tolerance(1e-10)
///     .minimize()
///     .seed(1)
///     .build()?;
/// let continuation = Continuation::builder(lbfgsb)
///     .stages(EPSILON.len())
///     // the stage's ε, shared with the fitness function
///     .on_stage(move |stage, _| {
///         epsilon.store(EPSILON[stage].to_bits(), Ordering::Relaxed);
///         Ok(())
///     })
///     .build()?;
/// let mut engine = Engine::new(continuation, smoothed).stop_when(Stop::evaluations(10_000));
/// let outcome = engine.run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Converged);
/// assert_eq!(engine.algorithm().stages().len(), 3);
/// for (x, c) in outcome.best_genome().iter().zip(c) {
///     assert!((x - c).abs() < 1e-8);
/// }
/// # Ok::<(), genoxide::Error>(())
/// ```
///
/// Built with [`Continuation::builder`], run with an [`Engine`](crate::Engine).
#[derive(Clone)]
// deserialized through `Deserialize for Continuation`, which checks what the builder checks
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self")
)]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "A: serde::Serialize",
        deserialize = "A: serde::Deserialize<'de>"
    ))
)]
pub struct Continuation<A> {
    algorithm: A,
    stage_count: usize,
    generations: Option<u64>,
    keep: Keep,
    // the current stage, and the wrapped algorithm's generation and evaluations at its start
    stage: usize,
    stage_generation: u64,
    stage_evaluations: u64,
    // the finished stages, with room for all of them
    stages: Vec<Stage>,
    // whether the last stage has ended
    finished: bool,
    #[cfg_attr(feature = "serde", serde(skip))]
    on_stage: Option<OnStage<A>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    on_stage_finished: Option<OnStageFinished<A>>,
}

#[cfg(feature = "serde")]
impl<A: serde::Serialize> serde::Serialize for Continuation<A> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        Self::serialize(self, serializer)
    }
}

// the builder's checks, for a hand-edited or damaged checkpoint: at least one stage, the current
// one among them, and no more finished ones
#[cfg(feature = "serde")]
impl<'de, A: serde::Deserialize<'de>> serde::Deserialize<'de> for Continuation<A> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let mut continuation = Self::deserialize(deserializer)?;
        let count = continuation.stage_count;
        if count == 0
            || continuation.stage >= count
            || continuation.stages.len() > count
            || continuation.generations == Some(0)
        {
            return Err(serde::de::Error::custom(format!(
                "a continuation of {count} stages in stage {}, with {} finished",
                continuation.stage,
                continuation.stages.len()
            )));
        }
        let room = count - continuation.stages.len();
        continuation.stages.reserve_exact(room);
        Ok(continuation)
    }
}

impl<A> fmt::Debug for Continuation<A>
where
    A: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Continuation")
            .field("algorithm", &self.algorithm)
            .field("stage_count", &self.stage_count)
            .field("generations", &self.generations)
            .field("keep", &self.keep)
            .field("stage", &self.stage)
            .field("stages", &self.stages)
            .field("finished", &self.finished)
            .field("on_stage", &self.on_stage.as_ref().map(|_| ".."))
            .field(
                "on_stage_finished",
                &self.on_stage_finished.as_ref().map(|_| ".."),
            )
            .finish_non_exhaustive()
    }
}

impl<A: Continue> Continuation<A> {
    /// A builder for a continuation of `algorithm`, which starts the first stage as it is.
    pub fn builder(algorithm: A) -> ContinuationBuilder<A> {
        ContinuationBuilder {
            algorithm,
            stages: None,
            generations: None,
            keep: Keep::State,
            on_stage: None,
            on_stage_finished: None,
        }
    }

    /// The wrapped algorithm.
    pub fn algorithm(&self) -> &A {
        &self.algorithm
    }

    /// The wrapped algorithm, mutably: e.g. to change its settings from
    /// [`Engine::control`](crate::Engine::control).
    pub fn algorithm_mut(&mut self) -> &mut A {
        &mut self.algorithm
    }

    /// The wrapped algorithm, consuming the continuation.
    pub fn into_algorithm(self) -> A {
        self.algorithm
    }

    /// The current stage, from 0: the last one once the continuation has finished.
    pub fn stage(&self) -> usize {
        self.stage
    }

    /// The number of stages.
    pub fn stage_count(&self) -> usize {
        self.stage_count
    }

    /// The budget of generations of each stage, if any.
    pub fn generations(&self) -> Option<u64> {
        self.generations
    }

    /// What the algorithm keeps between stages.
    pub fn keep(&self) -> Keep {
        self.keep
    }

    /// The finished stages, in order: what each did.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Sets the closure that sets a stage's parameters, as
    /// [`on_stage`](ContinuationBuilder::on_stage) does: for a continuation loaded from a
    /// checkpoint, which has none.
    pub fn set_on_stage<F>(&mut self, on_stage: F)
    where
        F: Fn(usize, &mut A) -> Result<()> + Send + Sync + 'static,
    {
        self.on_stage = Some(Arc::new(on_stage));
    }

    /// Sets the closure called with each finished stage, as
    /// [`on_stage_finished`](ContinuationBuilder::on_stage_finished) does: for a continuation
    /// loaded from a checkpoint, which has none.
    pub fn set_on_stage_finished<F>(&mut self, on_stage_finished: F)
    where
        F: Fn(&Stage, &A) + Send + Sync + 'static,
    {
        self.on_stage_finished = Some(Arc::new(on_stage_finished));
    }

    /// Gives this continuation the closures of `other`, e.g. the continuation a loaded one was
    /// saved from, built again with them.
    pub fn set_closures_of(&mut self, other: &Self) {
        self.on_stage.clone_from(&other.on_stage);
        self.on_stage_finished.clone_from(&other.on_stage_finished);
    }

    // after a tell: the end of the stage, if it has ended, and the start of the next one
    fn after_tell(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        let algorithm = &self.algorithm;
        let generations = algorithm.generation() - self.stage_generation;
        let end = if algorithm.is_finished() {
            StageEnd::Finished
        } else if self.generations.is_some_and(|budget| generations >= budget) {
            StageEnd::Generations
        } else {
            return Ok(());
        };
        let stage = Stage {
            index: self.stage,
            generations,
            evaluations: algorithm.evaluations() - self.stage_evaluations,
            best: algorithm
                .best()
                .and_then(Individual::fitness)
                .unwrap_or(Fitness::invalid()),
            end,
        };
        self.stages.push(stage);
        if let Some(on_stage_finished) = &self.on_stage_finished {
            on_stage_finished(&stage, &self.algorithm);
        }
        if self.stage + 1 == self.stage_count {
            self.finished = true;
            return Ok(());
        }
        self.stage += 1;
        self.stage_generation = self.algorithm.generation();
        self.stage_evaluations = self.algorithm.evaluations();
        self.algorithm.next_stage(self.keep)?;
        self.set_stage()
    }

    // the current stage's parameters, from the closure
    fn set_stage(&mut self) -> Result<()> {
        let on_stage = self.on_stage.as_ref().ok_or(Error::MissingSetting {
            setting: "on_stage",
        })?;
        on_stage(self.stage, &mut self.algorithm)
    }
}

impl<A: Continue> Reevaluate for Continuation<A> {
    /// The wrapped algorithm's [`reevaluate`](Reevaluate::reevaluate): within a stage, for a
    /// fitness function that changed otherwise.
    fn reevaluate(&mut self) -> Result<()> {
        self.algorithm.reevaluate()
    }
}

impl<A: Continue> Algorithm for Continuation<A> {
    type Genome = A::Genome;

    fn objective(&self) -> Objective {
        self.algorithm.objective()
    }

    fn ask(&mut self) -> Candidates<'_, A::Genome> {
        self.algorithm.ask()
    }

    /// The wrapped algorithm's tell; then, if the stage has ended, its record, and the next
    /// stage's start.
    ///
    /// # Errors
    ///
    /// The wrapped algorithm's, which change nothing, and those of the
    /// [`on_stage`](ContinuationBuilder::on_stage) closure, after which the continuation is in
    /// the next stage, with its parameters set as far as the closure got: the closure sets them
    /// again when a run starts.
    fn tell(&mut self, fitness: &[Fitness]) -> Result<()> {
        self.algorithm.tell(fitness)?;
        self.after_tell()
    }

    /// As [`tell`](Continuation::tell), with the wrapped algorithm's `tell_evaluations`.
    ///
    /// # Errors
    ///
    /// As [`tell`](Continuation::tell).
    fn tell_evaluations(&mut self, evaluations: &Evaluations<'_>) -> Result<()> {
        self.algorithm.tell_evaluations(evaluations)?;
        self.after_tell()
    }

    fn population(&self) -> &Population<A::Genome> {
        self.algorithm.population()
    }

    fn best(&self) -> Option<&Individual<A::Genome>> {
        self.algorithm.best()
    }

    fn discarded(&self) -> &[Individual<A::Genome>] {
        self.algorithm.discarded()
    }

    fn generation(&self) -> u64 {
        self.algorithm.generation()
    }

    fn evaluations(&self) -> u64 {
        self.algorithm.evaluations()
    }

    fn best_generation(&self) -> u64 {
        self.algorithm.best_generation()
    }

    /// Whether the last stage has ended.
    fn is_finished(&self) -> bool {
        self.finished
    }

    /// The wrapped algorithm's `prepare`, then the [`on_stage`](ContinuationBuilder::on_stage)
    /// closure with the current stage: the parameters of the stage the run starts or resumes in.
    ///
    /// # Errors
    ///
    /// [`Error::MissingSetting`] without an `on_stage` closure (after loading a checkpoint, see
    /// [`set_on_stage`](Continuation::set_on_stage)), and the errors of the wrapped algorithm's
    /// `prepare` and of the closure.
    fn prepare(&mut self, provided: Provided) -> Result<()> {
        if self.on_stage.is_none() {
            return Err(Error::MissingSetting {
                setting: "on_stage",
            });
        }
        self.algorithm.prepare(provided)?;
        self.set_stage()
    }

    fn wants(&self) -> Wanted {
        self.algorithm.wants()
    }
}

/// A builder for a [`Continuation`], from [`Continuation::builder`].
///
/// Required: the number of [`stages`](ContinuationBuilder::stages) and the
/// [`on_stage`](ContinuationBuilder::on_stage) closure. Defaults: no budget of generations (each
/// stage runs until the algorithm has finished), [`Keep::State`], and no `on_stage_finished`
/// closure.
pub struct ContinuationBuilder<A> {
    algorithm: A,
    stages: Option<usize>,
    generations: Option<u64>,
    keep: Keep,
    on_stage: Option<OnStage<A>>,
    on_stage_finished: Option<OnStageFinished<A>>,
}

impl<A: fmt::Debug> fmt::Debug for ContinuationBuilder<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContinuationBuilder")
            .field("algorithm", &self.algorithm)
            .field("stages", &self.stages)
            .field("generations", &self.generations)
            .field("keep", &self.keep)
            .finish_non_exhaustive()
    }
}

impl<A: Continue> ContinuationBuilder<A> {
    /// The number of stages, at least 1. Required.
    pub fn stages(mut self, stages: usize) -> Self {
        self.stages = Some(stages);
        self
    }

    /// The closure that sets the parameters of a stage, from its index (from 0): typically a
    /// parameter of the fitness function, in state shared with it such as an `Arc<AtomicU64>`,
    /// and maybe a setting of the algorithm, which it gets mutably. Required.
    ///
    /// It's called with the current stage at the start of every run (the first stage, or the
    /// stage a resumed run is in), and with each next stage when it begins, before its first
    /// evaluation: it must set everything from the index alone, so that calling it again for the
    /// same stage changes nothing. An error stops the run with that error.
    pub fn on_stage<F>(mut self, on_stage: F) -> Self
    where
        F: Fn(usize, &mut A) -> Result<()> + Send + Sync + 'static,
    {
        self.on_stage = Some(Arc::new(on_stage));
        self
    }

    /// The most generations of the wrapped algorithm in each stage, at least 1; the re-evaluation
    /// that starts a stage isn't one. A stage also ends when the algorithm has finished, and the
    /// last stage's end, by either, finishes the continuation. None by default: each stage runs
    /// until the algorithm has finished, or the run until its stop conditions.
    pub fn generations(mut self, generations: u64) -> Self {
        self.generations = Some(generations);
        self
    }

    /// What the algorithm keeps between stages: [`Keep::State`] by default.
    pub fn keep(mut self, keep: Keep) -> Self {
        self.keep = keep;
        self
    }

    /// A closure called with each finished stage (its index, generations, evaluations, best
    /// fitness and end) and the algorithm, before the next stage's parameters are set: e.g. to
    /// print or plot the stages as the run goes. None by default.
    pub fn on_stage_finished<F>(mut self, on_stage_finished: F) -> Self
    where
        F: Fn(&Stage, &A) + Send + Sync + 'static,
    {
        self.on_stage_finished = Some(Arc::new(on_stage_finished));
        self
    }

    /// Validates the settings and creates the continuation, in its first stage.
    ///
    /// # Errors
    ///
    /// - [`Error::MissingSetting`] without the number of stages or the `on_stage` closure.
    /// - [`Error::InvalidSetting`] for 0 stages or a budget of 0 generations.
    pub fn build(self) -> Result<Continuation<A>> {
        let stage_count = self
            .stages
            .ok_or(Error::MissingSetting { setting: "stages" })?;
        if stage_count == 0 {
            return Err(Error::InvalidSetting {
                setting: "stages",
                reason: "must be at least 1, got 0".to_string(),
            });
        }
        if self.generations == Some(0) {
            return Err(Error::InvalidSetting {
                setting: "generations",
                reason: "a stage's budget must be at least 1 generation, got 0".to_string(),
            });
        }
        let on_stage = self.on_stage.ok_or(Error::MissingSetting {
            setting: "on_stage",
        })?;
        let stage_generation = self.algorithm.generation();
        let stage_evaluations = self.algorithm.evaluations();
        Ok(Continuation {
            algorithm: self.algorithm,
            stage_count,
            generations: self.generations,
            keep: self.keep,
            stage: 0,
            stage_generation,
            stage_evaluations,
            stages: Vec::with_capacity(stage_count),
            finished: false,
            on_stage: Some(on_stage),
            on_stage_finished: self.on_stage_finished,
        })
    }
}
