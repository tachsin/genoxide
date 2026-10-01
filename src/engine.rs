//! Running an algorithm: fitness evaluation, stop conditions, observers and cancellation.
//!
//! - [`Engine`] runs a single-objective [`Algorithm`] one generation at a time.
//! - [`AsyncEngine`] runs an [`Incremental`](crate::algorithm::Incremental) algorithm such as a
//!   [`SteadyGa`](crate::algorithm::SteadyGa) on worker threads, each starting a new evaluation as
//!   soon as it's done: for expensive fitness functions whose time varies.
//! - [`MultiEngine`](crate::multi::MultiEngine) runs a
//!   [`MultiObjectiveAlgorithm`](crate::multi::MultiObjectiveAlgorithm).
//!
//! `Engine` and `MultiEngine` evaluate the genomes of a generation one after the other by
//! default. `parallel(true)` (the `parallel` feature) evaluates them on rayon's threads, with the
//! same results: it pays off when the fitness function is expensive. A [`Batch`] fitness function
//! gets the whole generation in one call, e.g. for a GPU or a remote service.

pub mod asynchronous;
mod extras;
mod info;
pub mod stop;
pub(crate) mod trace;

pub use asynchronous::AsyncEngine;
pub use extras::{BatchExtras, Evaluations, Extras, Provided, Wanted};
pub use info::Evaluated;
#[doc(hidden)]
pub use info::Info;
pub(crate) use info::{GenomeHashing, InfoStore};
pub use stop::{STALL_GENERATIONS, Stop, StopReason};

use crate::algorithm::{Algorithm, Candidates};
use crate::genome::Genome;
use crate::observer::{Observer, Snapshot};
use crate::{Error, Fitness, Individual, Objective, Result};
use std::any::Any;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// A fitness function: scores a genome.
///
/// Closures `|genome: &G| -> T` are fitness functions, where `T` is `f64`, [`Fitness`],
/// `Option<f64>` (`None` for an invalid solution) or `(f64, f64)` (a score and a constraint
/// violation, see [`Fitness::constrained`]), see [`IntoFitness`], or an [`Evaluated`] of one of
/// them with extras the engine keeps alongside. Fitness functions must be deterministic: the same
/// genome always gets the same fitness.
///
/// Implement it for a fitness function with data of its own:
///
/// ```
/// use genoxide::prelude::*;
///
/// // the squared distance to a point
/// struct Distance {
///     point: Vec<f64>,
/// }
///
/// impl FitnessFunction<Reals> for Distance {
///     type Output = f64;
///
///     fn evaluate(&self, genome: &Reals) -> f64 {
///         genome.iter().zip(&self.point).map(|(x, p)| (x - p) * (x - p)).sum()
///     }
/// }
///
/// let ga = Ga::builder(Real::uniform(2, -5.0..=5.0)?)
///     .population_size(50)
///     .select(Tournament::new(3)?)
///     .crossover(SimulatedBinaryCrossover::new(15.0)?)
///     .mutate(PolynomialMutation::per_gene(0.5, 20.0)?)
///     .minimize()
///     .seed(1)
///     .build()?;
/// let distance = Distance { point: vec![1.0, -2.0] };
/// let outcome = Engine::new(ga, distance).stop_when(Stop::generations(100)).run()?;
/// assert!(outcome.best_fitness().score().unwrap() < 0.1);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a fitness function for `{G}`",
    label = "not a fitness function for `{G}`",
    note = "a fitness function is a closure `|genome: &{G}| ...` returning `f64`, `Fitness`, `Option<f64>` or `(f64, f64)` (a score and a constraint violation), a `Batch` of a closure `|genomes: &[&{G}]| ...` returning a `Vec` of those, or a type implementing `FitnessFunction<{G}>`"
)]
pub trait FitnessFunction<G>: Sync {
    /// The type of a score.
    type Output: IntoFitness;

    /// The score of `genome`.
    fn evaluate(&self, genome: &G) -> Self::Output;

    /// Whether the engine evaluates a generation with one call of
    /// [`evaluate_batch`](FitnessFunction::evaluate_batch) (`true`) rather than one call of
    /// [`evaluate`](FitnessFunction::evaluate) per genome, in parallel if asked (`false`, the
    /// default). See [`Batch`].
    fn is_batch(&self) -> bool {
        false
    }

    /// The scores of `genomes`, in their order: one [`evaluate`](FitnessFunction::evaluate)
    /// each by default. Override it, and [`is_batch`](FitnessFunction::is_batch), to evaluate a
    /// generation at once, e.g. on a GPU.
    fn evaluate_batch(&self, genomes: &[&G]) -> Vec<Self::Output> {
        genomes.iter().map(|genome| self.evaluate(genome)).collect()
    }

    /// What the function gives besides the fitness, such as the gradient of the score: nothing by
    /// default. An algorithm that needs an extra checks it once per run in
    /// [`Algorithm::prepare`]. Declare an extra here, and write it in
    /// [`evaluate_with`](FitnessFunction::evaluate_with).
    #[inline]
    fn provides(&self) -> Provided {
        Provided::NOTHING
    }

    /// The score of `genome`, with the extras that `extras` has buffers for (only ones that
    /// [`provides`](FitnessFunction::provides) declares). The engine calls it instead of
    /// [`evaluate`](FitnessFunction::evaluate) when the algorithm [wants](Algorithm::wants)
    /// extras. The score must be the same as `evaluate`'s, to the bit. By default,
    /// `evaluate(genome)`.
    ///
    /// ```
    /// use genoxide::engine::{Extras, FitnessFunction, Provided};
    /// use genoxide::genome::Reals;
    ///
    /// // the squared distance to a point, and its gradient
    /// struct Distance {
    ///     point: Vec<f64>,
    /// }
    ///
    /// impl FitnessFunction<Reals> for Distance {
    ///     type Output = f64;
    ///
    ///     fn evaluate(&self, x: &Reals) -> f64 {
    ///         x.iter().zip(&self.point).map(|(x, p)| (x - p) * (x - p)).sum()
    ///     }
    ///
    ///     fn provides(&self) -> Provided {
    ///         Provided::GRADIENT
    ///     }
    ///
    ///     fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> f64 {
    ///         if let Some(gradient) = extras.gradient() {
    ///             for ((g, x), p) in gradient.iter_mut().zip(x.iter()).zip(&self.point) {
    ///                 *g = 2.0 * (x - p);
    ///             }
    ///         }
    ///         self.evaluate(x)
    ///     }
    /// }
    ///
    /// let distance = Distance { point: vec![1.0, -2.0] };
    /// let error = genoxide::gradient::check(&distance, &Reals::from(vec![0.5, 3.0]))?;
    /// assert!(error.largest() < 1e-8);
    /// # Ok::<(), genoxide::Error>(())
    /// ```
    ///
    /// [`Differentiable`](crate::gradient::Differentiable) does this for a closure.
    fn evaluate_with(&self, genome: &G, extras: &mut Extras<'_>) -> Self::Output {
        let _ = extras;
        self.evaluate(genome)
    }

    /// The scores of `genomes`, in their order, with the extras that `extras` has buffers for,
    /// a row per genome. The engine calls it instead of
    /// [`evaluate_batch`](FitnessFunction::evaluate_batch) for a batch function when the
    /// algorithm wants extras. By default, `evaluate_batch(genomes)` without extras, and one
    /// [`evaluate_with`](FitnessFunction::evaluate_with) each otherwise.
    fn evaluate_batch_with(
        &self,
        genomes: &[&G],
        extras: &mut BatchExtras<'_>,
    ) -> Vec<Self::Output> {
        if extras.wanted().is_empty() {
            return self.evaluate_batch(genomes);
        }
        genomes
            .iter()
            .enumerate()
            .map(|(position, genome)| self.evaluate_with(genome, &mut extras.get(position)))
            .collect()
    }
}

/// A fitness function that takes all the genomes of a generation at once and returns their
/// scores, in their order: for SIMD or GPU evaluation, a remote service, or any function with a
/// large cost per call. [`Engine`] takes a `Batch` of a closure returning a `Vec` of `f64`,
/// [`Fitness`], `Option<f64>` or `(f64, f64)`, and [`MultiEngine`](crate::multi::MultiEngine) a
/// `Batch` returning a `Vec` of `[f64; M]` or the other
/// [`IntoScores`](crate::multi::IntoScores) types.
///
/// It's called once per generation, with the genomes to evaluate (none when every child is a
/// copy that inherits its fitness). It decides how to evaluate them, so `Engine::parallel`
/// doesn't apply. Returning a different number of scores than genomes stops the run with
/// [`Error::FitnessCount`]. The repository's `examples/gpu` evaluates a generation of neural
/// networks in one GPU dispatch with wgpu.
///
/// ```
/// use genoxide::prelude::*;
///
/// let ga = Ga::builder(Binary::new(64)?)
///     .population_size(30)
///     .select(Tournament::new(3)?)
///     .crossover(UniformCrossover::new())
///     .mutate(BitFlip::per_gene(1.0 / 64.0)?)
///     .seed(1)
///     .build()?;
/// // one call per generation, e.g. one GPU dispatch or one request
/// let onemax = Batch(|genomes: &[&Bits]| {
///     genomes.iter().map(|genome| genome.count_ones() as f64).collect::<Vec<_>>()
/// });
/// let outcome = Engine::new(ga, onemax)
///     .stop_when(Stop::target(64.0).or(Stop::generations(1_000)))
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Target);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Batch<F>(pub F);

impl<G, F, T> FitnessFunction<G> for Batch<F>
where
    F: Fn(&[&G]) -> Vec<T> + Sync,
    T: IntoFitness,
{
    type Output = T;

    /// The score of one genome: a batch of one.
    ///
    /// # Panics
    ///
    /// If the batch function returns no score for it.
    fn evaluate(&self, genome: &G) -> T {
        (self.0)(&[genome])
            .into_iter()
            .next()
            .expect("the batch function returns a score per genome")
    }

    fn is_batch(&self) -> bool {
        true
    }

    fn evaluate_batch(&self, genomes: &[&G]) -> Vec<T> {
        (self.0)(genomes)
    }
}

impl<G, F, T> FitnessFunction<G> for F
where
    F: Fn(&G) -> T + Sync,
    T: IntoFitness,
{
    type Output = T;

    fn evaluate(&self, genome: &G) -> T {
        self(genome)
    }
}

/// A value that converts to a [`Fitness`]: the result of a [`FitnessFunction`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a fitness value",
    label = "a fitness function must return `f64`, `Fitness`, `Option<f64>` or `(f64, f64)`",
    note = "convert other numbers with `as f64`"
)]
pub trait IntoFitness {
    /// The fitness.
    ///
    /// # Errors
    ///
    /// [`Error::NanFitness`] for NaN, and [`Error::InvalidFitness`] for a negative constraint
    /// violation.
    fn into_fitness(self) -> Result<Fitness>;

    /// The fitness, and the info of an [`Evaluated`] result (`None` by default).
    #[doc(hidden)]
    #[inline]
    fn into_evaluation(self) -> (Result<Fitness>, Option<Info>)
    where
        Self: Sized,
    {
        (self.into_fitness(), None)
    }
}

impl IntoFitness for Fitness {
    fn into_fitness(self) -> Result<Fitness> {
        Ok(self)
    }
}

impl IntoFitness for f64 {
    fn into_fitness(self) -> Result<Fitness> {
        Fitness::try_new(self)
    }
}

impl IntoFitness for (f64, f64) {
    /// A score and a constraint violation, see [`Fitness::try_constrained`].
    fn into_fitness(self) -> Result<Fitness> {
        Fitness::try_constrained(self.0, self.1)
    }
}

impl IntoFitness for Option<f64> {
    /// `None` is [`Fitness::invalid`].
    fn into_fitness(self) -> Result<Fitness> {
        self.map_or(Ok(Fitness::invalid()), Fitness::try_new)
    }
}

/// What to do when a fitness function returns NaN.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NanPolicy {
    /// The solution gets [`Fitness::invalid`] (the default).
    #[default]
    Invalid,
    /// The run stops with [`Error::NanFitness`] (also for a NaN constraint violation).
    ///
    /// Either way, a negative constraint violation stops the run with [`Error::InvalidFitness`].
    Error,
}

/// The state of a run, for stop conditions and observers.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Progress {
    pub(crate) generation: u64,
    pub(crate) evaluations: u64,
    pub(crate) elapsed: Duration,
    pub(crate) best: Option<Fitness>,
    pub(crate) objective: Objective,
    pub(crate) best_generation: u64,
}

impl Progress {
    /// The number of completed generations, 0 for the initial population.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The number of fitness evaluations so far.
    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }

    /// The time since this run started.
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// The fitness of the best individual found so far. Always `None` in a multi-objective run
    /// ([`MultiEngine`](crate::multi::MultiEngine)), which has a front of trade-offs instead.
    pub fn best(&self) -> Option<Fitness> {
        self.best
    }

    /// Whether higher or lower fitness is better. In a multi-objective run, the default
    /// ([`Objective::Maximize`]): each objective's direction is a setting of the algorithm.
    pub fn objective(&self) -> Objective {
        self.objective
    }

    /// The generation in which the best individual so far was found. In a multi-objective run,
    /// the last generation in which the front gained a solution that no member of the previous
    /// generation's front dominated or equaled.
    pub fn best_generation(&self) -> u64 {
        self.best_generation
    }

    /// The number of generations since the best fitness (or the front) last improved.
    pub fn stagnant_generations(&self) -> u64 {
        self.generation.saturating_sub(self.best_generation)
    }

    // the progress of a multi-objective run, which has no best fitness
    pub(crate) fn multi_objective(
        generation: u64,
        evaluations: u64,
        elapsed: Duration,
        best_generation: u64,
    ) -> Self {
        Self {
            generation,
            evaluations,
            elapsed,
            best: None,
            objective: Objective::default(),
            best_generation,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(generation: u64, objective: Objective) -> Self {
        Self {
            generation,
            evaluations: 0,
            elapsed: Duration::ZERO,
            best: None,
            objective,
            best_generation: 0,
        }
    }
}

/// The result of a run.
///
/// Two outcomes are equal when their best individuals, counts, durations and stop reasons are:
/// the [info](Outcome::best_info) isn't compared, as a deterministic fitness function gives the
/// same info for the same genome. With the `serde` feature, the info isn't serialized: a
/// deserialized outcome has none.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Outcome<G: Genome> {
    best: Individual<G>,
    generations: u64,
    evaluations: u64,
    elapsed: Duration,
    stop_reason: StopReason,
    #[cfg_attr(feature = "serde", serde(skip))]
    best_info: Option<Info>,
}

impl<G: Genome> PartialEq for Outcome<G> {
    fn eq(&self, other: &Self) -> bool {
        self.best == other.best
            && self.generations == other.generations
            && self.evaluations == other.evaluations
            && self.elapsed == other.elapsed
            && self.stop_reason == other.stop_reason
    }
}

impl<G: Genome> Eq for Outcome<G> {}

impl<G: Genome> Outcome<G> {
    /// The info the fitness function returned with the best individual's fitness in an
    /// [`Evaluated`], as a `T`: `None` if it returned none or another type, or if the best was
    /// evaluated by another engine, e.g. before the run resumed from a checkpoint.
    pub fn best_info<T: Any>(&self) -> Option<&T> {
        self.best_info.as_ref()?.downcast_ref()
    }

    /// The best individual found.
    pub fn best(&self) -> &Individual<G> {
        &self.best
    }

    /// The genome of the best individual found.
    pub fn best_genome(&self) -> &G {
        self.best.genome()
    }

    /// The fitness of the best individual found.
    pub fn best_fitness(&self) -> Fitness {
        // the best individual comes from the algorithm, which only keeps evaluated ones
        self.best.fitness().unwrap_or(Fitness::invalid())
    }

    /// The best individual found, consuming the outcome.
    pub fn into_best(self) -> Individual<G> {
        self.best
    }

    /// The number of completed generations of the algorithm.
    pub fn generations(&self) -> u64 {
        self.generations
    }

    /// The number of fitness evaluations of the algorithm.
    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }

    /// The duration of the run.
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Why the run stopped.
    pub fn stop_reason(&self) -> StopReason {
        self.stop_reason
    }
}

/// Runs an [`Algorithm`] with a fitness function until a stop condition is met.
///
/// Every generation, the engine asks the algorithm for genomes, evaluates them, tells the
/// algorithm their fitness, notifies the observers and checks the stop conditions and the abort
/// flag. With parallel evaluation, the results are identical to sequential evaluation, for any
/// number of threads.
///
/// [`run`](Engine::run) can be called again, e.g. after an abort. To continue with another stop
/// condition, run the algorithm in a new engine: `Engine::new(engine.into_algorithm(), fitness)`.
/// Stop conditions count from the algorithm's start, so a run whose condition is already met
/// returns at once.
///
/// A run whose stop conditions can only be met with new evaluations (a target or an evaluation
/// limit) stops with [`StopReason::Stalled`] once the algorithm has asked for no genome to
/// evaluate in [`STALL_GENERATIONS`] generations in a row, e.g. a genetic algorithm whose children
/// are all copies of their parents.
///
/// A run whose algorithm has converged with nothing more to do, such as a
/// [`NelderMead`](crate::algorithm::NelderMead) without restarts left or a
/// [`Cmaes`](crate::algorithm::Cmaes) with
/// [`Restarts::Stop`](crate::algorithm::cmaes::Restarts::Stop), stops with
/// [`StopReason::Converged`] (see [`Algorithm::is_finished`]).
///
/// ```
/// use genoxide::prelude::*;
///
/// let ga = Ga::builder(Binary::new(64)?)
///     .population_size(100)
///     .select(Tournament::new(3)?)
///     .crossover(UniformCrossover::new())
///     .mutate(BitFlip::per_gene(1.0 / 64.0)?)
///     .seed(7)
///     .build()?;
/// let mut statistics = Statistics::new();
/// let outcome = Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
///     .stop_when(Stop::target(64.0).or(Stop::generations(1_000)))
///     .observe(&mut statistics)
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Target);
/// assert_eq!(statistics.records().len() as u64, outcome.generations() + 1);
/// # Ok::<(), genoxide::Error>(())
/// ```
pub struct Engine<'o, A: Algorithm, F> {
    algorithm: A,
    fitness: F,
    stop: Option<Stop>,
    observers: Vec<Box<dyn Observer<A::Genome> + 'o>>,
    abort: Option<Arc<AtomicBool>>,
    nan_policy: NanPolicy,
    parallel: bool,
    checkpoint: Option<Checkpoint<'o, A>>,
    controls: Vec<Control<'o, A>>,
    results: Vec<Result<Fitness>>,
    scores: Vec<Fitness>,
    // the gradients of the last evaluations, row-major, when the algorithm wants them
    gradients: Vec<f64>,
    // the info returned with the last evaluations, with their genomes
    evaluated_infos: Vec<(A::Genome, Info)>,
    // the info of the population, the discarded individuals and the best
    infos: InfoStore<A::Genome>,
    // the generations in a row in which the algorithm asked for no genome to evaluate
    idle: u64,
}

// a closure that changes the algorithm between generations
type Control<'o, A> = Box<dyn FnMut(&mut A, &Progress) -> Result<()> + 'o>;

// every how many generations to call a closure with the algorithm, and the closure
pub(crate) type Checkpoint<'o, A> = (u64, Box<dyn FnMut(&A) -> Result<()> + 'o>);

// calls the checkpoint closure if it's due after `generation`, or the run stops
pub(crate) fn checkpoint<A>(
    checkpoint: &mut Option<Checkpoint<'_, A>>,
    algorithm: &A,
    generation: u64,
    stopping: bool,
) -> Result<()> {
    match checkpoint {
        Some((every, save)) if stopping || generation.is_multiple_of(*every) => save(algorithm),
        _ => Ok(()),
    }
}

// `StopReason::Stalled` after `idle` generations in a row without a genome to evaluate, if they're
// at least `STALL_GENERATIONS` and the stop condition can only be met with new evaluations
pub(crate) fn stalled(stop: Option<&Stop>, idle: u64) -> Option<StopReason> {
    (idle >= STALL_GENERATIONS && stop.is_some_and(Stop::needs_evaluations))
        .then_some(StopReason::Stalled)
}

// `StopReason::Converged` once the algorithm has nothing more to do
fn converged<A: Algorithm>(algorithm: &A) -> Option<StopReason> {
    algorithm.is_finished().then_some(StopReason::Converged)
}

// the error for checkpoints every 0 generations
pub(crate) fn validate_checkpoint<A>(checkpoint: &Option<Checkpoint<'_, A>>) -> Result<()> {
    match checkpoint {
        Some((0, _)) => Err(Error::InvalidSetting {
            setting: "checkpoint_every",
            reason: "must be at least 1 generation".to_string(),
        }),
        _ => Ok(()),
    }
}

impl<'o, A, F> Engine<'o, A, F>
where
    A: Algorithm,
    F: FitnessFunction<A::Genome>,
{
    /// An engine running `algorithm` with the `fitness` function.
    pub fn new(algorithm: A, fitness: F) -> Self {
        Self {
            algorithm,
            fitness,
            stop: None,
            observers: Vec::new(),
            abort: None,
            nan_policy: NanPolicy::default(),
            parallel: false,
            checkpoint: None,
            controls: Vec::new(),
            results: Vec::new(),
            scores: Vec::new(),
            gradients: Vec::new(),
            evaluated_infos: Vec::new(),
            infos: InfoStore::default(),
            idle: 0,
        }
    }

    /// Adds a stop condition: the run stops when any of them is met. At least one stop condition
    /// or an abort flag is required.
    pub fn stop_when(mut self, stop: Stop) -> Self {
        self.stop = Some(match self.stop {
            Some(existing) => existing.or(stop),
            None => stop,
        });
        self
    }

    /// Stops the run after the current generation once `flag` is set, e.g. from another thread.
    pub fn abort_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.abort = Some(flag);
        self
    }

    /// What to do when the fitness function returns NaN. Invalid fitness by default.
    pub fn nan_policy(mut self, policy: NanPolicy) -> Self {
        self.nan_policy = policy;
        self
    }

    /// Adds an observer, notified after every generation, including the initial population. Pass
    /// `&mut observer` to keep access to it after the run.
    pub fn observe<O: Observer<A::Genome> + 'o>(mut self, observer: O) -> Self {
        self.observers.push(Box::new(observer));
        self
    }

    /// Adds a closure called after every generation, including the initial population.
    ///
    /// ```
    /// # use genoxide::prelude::*;
    /// # let ga = Ga::builder(Binary::new(8)?).population_size(4).select(Tournament::new(2)?)
    /// #     .crossover(UniformCrossover::new()).mutate(BitFlip::count(1)?).seed(0).build()?;
    /// Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
    ///     .stop_when(Stop::generations(3))
    ///     .on_generation(|snapshot| {
    ///         let progress = snapshot.progress();
    ///         println!("{}: {:?}", progress.generation(), progress.best());
    ///     })
    ///     .run()?;
    /// # Ok::<(), genoxide::Error>(())
    /// ```
    pub fn on_generation<C>(self, callback: C) -> Self
    where
        C: FnMut(&Snapshot<'_, A::Genome>) + 'o,
    {
        self.observe(FnObserver(callback))
    }

    /// Adds a closure that gets the algorithm mutably after every generation, including the
    /// initial population, to change it for the next one: parameter control, such as a mutation
    /// step annealed over the run, a rate raised while the best stagnates, or a new operator; or
    /// [`Reevaluate::reevaluate`](crate::algorithm::Reevaluate::reevaluate) after the fitness
    /// function changed, e.g. a raised penalty weight.
    ///
    /// It runs once per generation: not after the tell of a re-evaluation, which scores the
    /// population again without starting a new generation. It runs after the observers and the
    /// stop conditions, and before a checkpoint, so a checkpoint holds the changed settings and a
    /// resumed run follows the same schedule. It also runs after the last generation: the
    /// algorithm the engine returns has the settings for the generation after it, as it would in
    /// a longer run. Closures added by several calls run in their order. An error stops the run
    /// with that error.
    ///
    /// ```
    /// use genoxide::prelude::*;
    ///
    /// let ga = Ga::builder(Real::uniform(10, -5.0..=5.0)?)
    ///     .population_size(40)
    ///     .select(Tournament::new(3)?)
    ///     .crossover(UniformCrossover::new())
    ///     .mutate(GaussianMutation::per_gene(0.2, 0.1)?)
    ///     .minimize()
    ///     .seed(1)
    ///     .build()?;
    /// let sphere = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
    /// let outcome = Engine::new(ga, sphere)
    ///     .stop_when(Stop::generations(300))
    ///     // the step shrinks from 10% to 0.1% of each gene's range over the run
    ///     .control(|ga, progress| {
    ///         let done = progress.generation() as f64 / 300.0;
    ///         *ga.mutate_mut() = GaussianMutation::per_gene(0.2, 0.1 * 0.01_f64.powf(done))?;
    ///         Ok(())
    ///     })
    ///     .run()?;
    /// assert!(outcome.best_fitness().score().unwrap() < 1e-3);
    /// # Ok::<(), genoxide::Error>(())
    /// ```
    ///
    /// A fitness function that changes: a penalty weight shared with the fitness function, raised
    /// while the best is infeasible, after which the population is scored again.
    ///
    /// ```
    /// use genoxide::prelude::*;
    /// use std::sync::atomic::{AtomicU64, Ordering};
    ///
    /// // maximize the ones, with at most 10 of them allowed
    /// let weight = AtomicU64::new(0.1_f64.to_bits());
    /// let penalized = |bits: &Bits| {
    ///     let ones = bits.count_ones() as f64;
    ///     ones - f64::from_bits(weight.load(Ordering::Relaxed)) * (ones - 10.0).max(0.0)
    /// };
    /// let ga = Ga::builder(Binary::new(32)?)
    ///     .population_size(30)
    ///     .select(Tournament::new(3)?)
    ///     .crossover(UniformCrossover::new())
    ///     .mutate(BitFlip::per_gene(1.0 / 32.0)?)
    ///     .seed(2)
    ///     .build()?;
    /// let outcome = Engine::new(ga, &penalized)
    ///     .stop_when(Stop::generations(200))
    ///     .control(|ga, progress| {
    ///         let best = ga.best().expect("evaluated");
    ///         if progress.generation() % 20 == 19 && best.genome().count_ones() > 10 {
    ///             let raised = f64::from_bits(weight.load(Ordering::Relaxed)) * 4.0;
    ///             weight.store(raised.to_bits(), Ordering::Relaxed);
    ///             ga.reevaluate()?;
    ///         }
    ///         Ok(())
    ///     })
    ///     .run()?;
    /// assert_eq!(outcome.best_genome().count_ones(), 10);
    /// # Ok::<(), genoxide::Error>(())
    /// ```
    pub fn control<C>(mut self, control: C) -> Self
    where
        C: FnMut(&mut A, &Progress) -> Result<()> + 'o,
    {
        self.controls.push(Box::new(control));
        self
    }

    /// Calls `save` with the algorithm every `generations` generations (when the algorithm's
    /// generation is a multiple of it, so a resumed run keeps the schedule) and when the run
    /// stops, e.g. to write a checkpoint with `genoxide::checkpoint::save_file` (the `serde`
    /// feature). An error from `save` stops the run with that error. `generations` must be at
    /// least 1.
    pub fn checkpoint_every<C>(mut self, generations: u64, save: C) -> Self
    where
        C: FnMut(&A) -> Result<()> + 'o,
    {
        self.checkpoint = Some((generations, Box::new(save)));
        self
    }

    /// Evaluates the genomes of each generation in parallel with rayon. Off by default: it pays
    /// off when the fitness function is expensive.
    #[cfg(feature = "parallel")]
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// The algorithm.
    pub fn algorithm(&self) -> &A {
        &self.algorithm
    }

    /// The algorithm, consuming the engine.
    pub fn into_algorithm(self) -> A {
        self.algorithm
    }

    /// Runs until a stop condition is met or the abort flag is set.
    ///
    /// # Errors
    ///
    /// - [`Error::MissingSetting`] without a stop condition or abort flag.
    /// - [`Error::InvalidSetting`] for an invalid stop condition, or checkpoints every 0
    ///   generations.
    /// - [`Error::NanFitness`] if the fitness function returns NaN (a score or a constraint
    ///   violation) with [`NanPolicy::Error`].
    /// - [`Error::InvalidFitness`] if the fitness function returns a negative constraint violation,
    ///   with any [`NanPolicy`]: that's a bug in the fitness function, not a result.
    /// - [`Error::FitnessCount`] if a [`Batch`] returns a different number of scores than genomes.
    /// - [`Error::InvalidSetting`] if the algorithm wants an extra, such as a gradient, that the
    ///   fitness function doesn't [provide](FitnessFunction::provides), and
    ///   [`Error::InvalidGenome`] if it wants the gradients of genomes of different lengths in one
    ///   ask.
    /// - The errors of the algorithm's [`prepare`](Algorithm::prepare), its
    ///   [`tell`](Algorithm::tell) or [`tell_evaluations`](Algorithm::tell_evaluations), of the
    ///   [`control`](Engine::control) closures and of the checkpoint closure.
    ///
    /// If the algorithm has run before and a stop condition is already met, it returns that
    /// outcome without another generation. A run whose stop conditions need new evaluations stops
    /// with [`StopReason::Stalled`] after [`STALL_GENERATIONS`] generations in a row without a
    /// genome to evaluate, and a run whose algorithm [has finished](Algorithm::is_finished) with
    /// [`StopReason::Converged`].
    ///
    /// # Panics
    ///
    /// A panic in the fitness function propagates to the caller. It also panics if the algorithm
    /// has no [`best`](Algorithm::best) individual after a [`tell`](Algorithm::tell).
    pub fn run(&mut self) -> Result<Outcome<A::Genome>> {
        if self.stop.is_none() && self.abort.is_none() {
            return Err(Error::MissingSetting {
                setting: "stop_when",
            });
        }
        if let Some(stop) = &self.stop {
            stop.validate()?;
        }
        validate_checkpoint(&self.checkpoint)?;
        self.algorithm.prepare(self.fitness.provides())?;
        let _span = trace::run::<A>();
        if let Some(best) = self.algorithm.best() {
            // a run that continues: its stop condition may already be met
            let progress = Progress {
                generation: self.algorithm.generation(),
                evaluations: self.algorithm.evaluations(),
                elapsed: Duration::ZERO,
                best: best.fitness(),
                objective: self.algorithm.objective(),
                best_generation: self.algorithm.best_generation(),
            };
            let aborted = self
                .abort
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed));
            let reason = if aborted {
                Some(StopReason::Aborted)
            } else {
                self.stop.as_ref().and_then(|stop| stop.check(&progress))
            }
            .or_else(|| converged(&self.algorithm))
            .or_else(|| stalled(self.stop.as_ref(), self.idle));
            if let Some(stop_reason) = reason {
                return Ok(Outcome {
                    best: best.clone(),
                    generations: progress.generation,
                    evaluations: progress.evaluations,
                    elapsed: Duration::ZERO,
                    stop_reason,
                    best_info: self.infos.get(best.genome()).cloned(),
                });
            }
        }
        let start = Instant::now();
        // whether the algorithm has a best: from its first tell on, a tell that doesn't advance
        // the generation scores again what the algorithm keeps (a re-evaluation)
        let mut evaluated = self.algorithm.best().is_some();
        loop {
            let generation = self.algorithm.generation();
            let wanted = self.algorithm.wants();
            if wanted.is_empty() {
                // the path of every algorithm that wants no extras, as it always was
                self.evaluate()?;
                self.algorithm.tell(&self.scores)?;
            } else {
                let dimensions = self.evaluate_with_extras(wanted)?;
                let gradients = wanted.gradient.then_some(self.gradients.as_slice());
                let evaluations = Evaluations::from_parts(&self.scores, gradients, dimensions);
                self.algorithm.tell_evaluations(&evaluations)?;
            }
            let reevaluated = evaluated && self.algorithm.generation() == generation;
            evaluated = true;
            self.idle = if self.scores.is_empty() {
                self.idle + 1
            } else {
                0
            };
            if !self.evaluated_infos.is_empty() || !self.infos.is_empty() {
                let algorithm = &self.algorithm;
                let kept = algorithm.population().iter().chain(algorithm.discarded());
                let best = algorithm.best().map(Individual::genome);
                self.infos.update(
                    &mut self.evaluated_infos,
                    kept.map(Individual::genome).chain(best),
                );
            }

            let best = self
                .algorithm
                .best()
                .expect("an algorithm has a best individual after a tell");
            let progress = Progress {
                generation: self.algorithm.generation(),
                evaluations: self.algorithm.evaluations(),
                elapsed: start.elapsed(),
                best: best.fitness(),
                objective: self.algorithm.objective(),
                best_generation: self.algorithm.best_generation(),
            };
            trace::generation(&progress, None);
            if !self.observers.is_empty() {
                let snapshot = Snapshot::new(
                    self.algorithm.population(),
                    self.algorithm.discarded(),
                    best,
                    &progress,
                    &self.infos,
                );
                for observer in &mut self.observers {
                    observer.observe(&snapshot);
                }
            }

            let aborted = self
                .abort
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed));
            let reason = if aborted {
                Some(StopReason::Aborted)
            } else {
                self.stop.as_ref().and_then(|stop| stop.check(&progress))
            }
            .or_else(|| converged(&self.algorithm))
            .or_else(|| stalled(self.stop.as_ref(), self.idle));
            let outcome = reason.map(|stop_reason| Outcome {
                best: best.clone(),
                generations: progress.generation,
                evaluations: progress.evaluations,
                elapsed: progress.elapsed,
                stop_reason,
                best_info: self.infos.get(best.genome()).cloned(),
            });
            if !reevaluated {
                for control in &mut self.controls {
                    control(&mut self.algorithm, &progress)?;
                }
            }
            checkpoint(
                &mut self.checkpoint,
                &self.algorithm,
                progress.generation,
                reason.is_some(),
            )?;
            if let Some(outcome) = outcome {
                trace::finished(&progress, outcome.stop_reason, None);
                return Ok(outcome);
            }
        }
    }

    // evaluates the asked genomes into `self.scores`
    fn evaluate(&mut self) -> Result<()> {
        let candidates = self.algorithm.ask();
        let fitness = &self.fitness;
        if fitness.is_batch() {
            evaluate_batch(
                candidates,
                |genomes| fitness.evaluate_batch(genomes),
                &mut self.results,
                &mut self.evaluated_infos,
                IntoFitness::into_evaluation,
            )?;
        } else if !self.parallel {
            // straight into the scores, without a vector of results
            self.evaluated_infos.clear();
            self.scores.clear();
            return evaluate_scores(
                candidates,
                &|genome: &A::Genome| fitness.evaluate(genome).into_evaluation(),
                self.nan_policy,
                &mut self.scores,
                &mut self.evaluated_infos,
            );
        } else {
            evaluate_all(
                candidates,
                self.parallel,
                &|genome: &A::Genome| fitness.evaluate(genome).into_evaluation(),
                &mut self.results,
                &mut self.evaluated_infos,
            );
        }
        self.scores.clear();
        for result in self.results.drain(..) {
            self.scores.push(match (result, self.nan_policy) {
                (Ok(fitness), _) => fitness,
                (Err(Error::NanFitness), NanPolicy::Invalid) => Fitness::invalid(),
                (Err(error), _) => return Err(error),
            });
        }
        Ok(())
    }

    // evaluates the asked genomes with the `wanted` extras: the fitness into `self.scores`, the
    // gradients into `self.gradients`, a row per genome; returns the length of a row. Each genome
    // writes its own row, so parallel and batch evaluation give the same bits as sequential.
    #[inline(never)]
    fn evaluate_with_extras(&mut self, wanted: Wanted) -> Result<usize> {
        let fitness = &self.fitness;
        if let Some(missing) = wanted.missing_from(fitness.provides()) {
            return Err(Error::InvalidSetting {
                setting: "fitness",
                reason: format!(
                    "the algorithm wants the {missing} of the score, which the fitness function \
                     doesn't provide: supply it, e.g. with `Differentiable`, or let the algorithm \
                     compute it by finite differences"
                ),
            });
        }
        let candidates = self.algorithm.ask();
        let dimensions = candidates.get(0).map_or(0, Genome::len);
        if candidates.iter().any(|genome| genome.len() != dimensions) {
            return Err(Error::InvalidGenome {
                reason: "the genomes of an ask that wants gradients must have the same length"
                    .to_string(),
            });
        }
        let width = if wanted.gradient { dimensions } else { 0 };
        self.gradients.clear();
        self.gradients.resize(candidates.len() * width, 0.0);
        let gradients = &mut self.gradients;
        if fitness.is_batch() {
            let mut batch = if wanted.gradient {
                BatchExtras::with_gradients(gradients, width)
            } else {
                BatchExtras::none()
            };
            evaluate_batch(
                candidates,
                |genomes| fitness.evaluate_batch_with(genomes, &mut batch),
                &mut self.results,
                &mut self.evaluated_infos,
                IntoFitness::into_evaluation,
            )?;
        } else if self.parallel && width > 0 {
            evaluate_rows_parallel(
                candidates,
                gradients,
                width,
                &|genome: &A::Genome, row: &mut [f64]| {
                    fitness
                        .evaluate_with(genome, &mut extras_for(wanted, row))
                        .into_evaluation()
                },
                &mut self.results,
                &mut self.evaluated_infos,
            );
        } else {
            self.results.clear();
            self.evaluated_infos.clear();
            for (position, genome) in candidates.iter().enumerate() {
                let row = &mut gradients[position * width..(position + 1) * width];
                let (result, info) = fitness
                    .evaluate_with(genome, &mut extras_for(wanted, row))
                    .into_evaluation();
                if let Some(info) = info {
                    self.evaluated_infos.push((genome.clone(), info));
                }
                self.results.push(result);
            }
        }
        // NaN in an extra follows the NaN policy, as in the score: the whole evaluation is invalid
        // or an error. The first error in order is returned, after every genome is evaluated.
        self.scores.clear();
        let mut first_error = None;
        for (position, result) in self.results.drain(..).enumerate() {
            let row = &self.gradients[position * width..(position + 1) * width];
            let result = match result {
                Ok(fitness) if fitness.is_valid() && row.iter().any(|value| value.is_nan()) => {
                    Err(Error::NanFitness)
                }
                result => result,
            };
            self.scores.push(match (result, self.nan_policy) {
                (Ok(fitness), _) => fitness,
                (Err(Error::NanFitness), NanPolicy::Invalid) => Fitness::invalid(),
                (Err(error), _) => {
                    first_error.get_or_insert(error);
                    Fitness::invalid()
                }
            });
        }
        first_error.map_or(Ok(dimensions), Err)
    }
}

// the buffers of one evaluation for the `wanted` extras: `row` for the gradient
fn extras_for(wanted: Wanted, row: &mut [f64]) -> Extras<'_> {
    if wanted.gradient {
        Extras::with_gradient(row)
    } else {
        Extras::none()
    }
}

// evaluates every candidate in parallel with its own row of `rows` (`width` values each), into
// `results` in order, and the info returned into `infos`, with a copy of its genome
#[cfg(feature = "parallel")]
fn evaluate_rows_parallel<G, T, E>(
    candidates: Candidates<'_, G>,
    rows: &mut [f64],
    width: usize,
    evaluate: &E,
    results: &mut Vec<T>,
    infos: &mut Vec<(G, Info)>,
) where
    G: Genome,
    T: Send,
    E: Fn(&G, &mut [f64]) -> (T, Option<Info>) + Sync,
{
    use rayon::prelude::*;
    let mut found = Vec::new();
    // unzipping an indexed parallel iterator keeps the order, whatever the thread count
    rows.par_chunks_mut(width)
        .enumerate()
        .map(|(position, row)| evaluate(candidates.get(position).expect("in bounds"), row))
        .unzip_into_vecs(results, &mut found);
    infos.clear();
    for (genome, info) in candidates.iter().zip(found) {
        if let Some(info) = info {
            infos.push((genome.clone(), info));
        }
    }
}

#[cfg(not(feature = "parallel"))]
fn evaluate_rows_parallel<G, T, E>(
    _: Candidates<'_, G>,
    _: &mut [f64],
    _: usize,
    _: &E,
    _: &mut Vec<T>,
    _: &mut Vec<(G, Info)>,
) where
    G: Genome,
{
    unreachable!("parallel evaluation can't be enabled without the `parallel` feature")
}

// evaluates every candidate with one call of a batch function into `results`, in order, and the
// info returned into `infos`, with a copy of its genome
pub(crate) fn evaluate_batch<G, X, T, R>(
    candidates: Candidates<'_, G, X>,
    batch: impl FnOnce(&[&G]) -> Vec<T>,
    results: &mut Vec<R>,
    infos: &mut Vec<(G, Info)>,
    convert: impl Fn(T) -> (R, Option<Info>),
) -> Result<()>
where
    G: Genome,
{
    infos.clear();
    let genomes: Vec<&G> = candidates.iter().collect();
    let values = batch(&genomes);
    if values.len() != genomes.len() {
        return Err(Error::FitnessCount {
            expected: genomes.len(),
            got: values.len(),
        });
    }
    results.clear();
    results.extend(values.into_iter().zip(&genomes).map(|(value, genome)| {
        let (result, info) = convert(value);
        if let Some(info) = info {
            infos.push(((*genome).clone(), info));
        }
        result
    }));
    Ok(())
}

// evaluates every candidate into `results`, in order, in parallel if asked, and the info returned
// into `infos`, with a copy of its genome; the results are the same either way
pub(crate) fn evaluate_all<G, F, T, E>(
    candidates: Candidates<'_, G, F>,
    parallel: bool,
    evaluate: &E,
    results: &mut Vec<T>,
    infos: &mut Vec<(G, Info)>,
) where
    G: Genome,
    F: Sync,
    T: Send,
    E: Fn(&G) -> (T, Option<Info>) + Sync,
{
    results.clear();
    infos.clear();
    if parallel {
        evaluate_parallel(candidates, evaluate, results, infos);
    } else {
        evaluate_sequential(candidates, evaluate, results, infos);
    }
}

// evaluates every candidate into `results` one after the other. Without info (the default
// `into_evaluation`), `info` is always `None` and this compiles to the evaluations alone. Kept out
// of the engine's loop: inlined there, it compiles to a few more instructions per evaluation.
#[inline(never)]
fn evaluate_sequential<G, F, T, E>(
    candidates: Candidates<'_, G, F>,
    evaluate: &E,
    results: &mut Vec<T>,
    infos: &mut Vec<(G, Info)>,
) where
    G: Genome,
    E: Fn(&G) -> (T, Option<Info>),
{
    results.extend(candidates.iter().map(|genome| {
        let (result, info) = evaluate(genome);
        if let Some(info) = info {
            infos.push((genome.clone(), info));
        }
        result
    }));
}

// evaluates every candidate one after the other into `scores`, as `Engine::evaluate` converts
// the results of `evaluate_all`: NaN is invalid or an error by `nan_policy`, and the first error
// in order is returned, after every candidate is evaluated. Out of line, as `evaluate_sequential`.
#[inline(never)]
fn evaluate_scores<G, E>(
    candidates: Candidates<'_, G>,
    evaluate: &E,
    nan_policy: NanPolicy,
    scores: &mut Vec<Fitness>,
    infos: &mut Vec<(G, Info)>,
) -> Result<()>
where
    G: Genome,
    E: Fn(&G) -> (Result<Fitness>, Option<Info>),
{
    let mut first_error = None;
    scores.extend(candidates.iter().map(|genome| {
        let (result, info) = evaluate(genome);
        if let Some(info) = info {
            infos.push((genome.clone(), info));
        }
        match (result, nan_policy) {
            (Ok(fitness), _) => fitness,
            (Err(Error::NanFitness), NanPolicy::Invalid) => Fitness::invalid(),
            (Err(error), _) => {
                first_error.get_or_insert(error);
                Fitness::invalid()
            }
        }
    }));
    first_error.map_or(Ok(()), Err)
}

#[cfg(feature = "parallel")]
fn evaluate_parallel<G, F, T, E>(
    candidates: Candidates<'_, G, F>,
    evaluate: &E,
    results: &mut Vec<T>,
    infos: &mut Vec<(G, Info)>,
) where
    G: Genome,
    F: Sync,
    T: Send,
    E: Fn(&G) -> (T, Option<Info>) + Sync,
{
    use rayon::prelude::*;
    let mut found = Vec::new();
    // unzipping an indexed parallel iterator keeps the order, whatever the thread count
    (0..candidates.len())
        .into_par_iter()
        .map(|position| evaluate(candidates.get(position).expect("position in bounds")))
        .unzip_into_vecs(results, &mut found);
    for (genome, info) in candidates.iter().zip(found) {
        if let Some(info) = info {
            infos.push((genome.clone(), info));
        }
    }
}

#[cfg(not(feature = "parallel"))]
fn evaluate_parallel<G, F, T, E>(
    _: Candidates<'_, G, F>,
    _: &E,
    _: &mut Vec<T>,
    _: &mut Vec<(G, Info)>,
) where
    G: Genome,
{
    unreachable!("parallel evaluation can't be enabled without the `parallel` feature")
}

impl<A: Algorithm + fmt::Debug, F> fmt::Debug for Engine<'_, A, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Engine")
            .field("algorithm", &self.algorithm)
            .field("stop", &self.stop)
            .field("observers", &self.observers.len())
            .field("abort", &self.abort)
            .field("nan_policy", &self.nan_policy)
            .field("parallel", &self.parallel)
            .field("controls", &self.controls.len())
            .field(
                "checkpoint_every",
                &self.checkpoint.as_ref().map(|(every, _)| every),
            )
            .finish_non_exhaustive()
    }
}

struct FnObserver<C>(C);

impl<G: Genome, C: FnMut(&Snapshot<'_, G>)> Observer<G> for FnObserver<C> {
    fn observe(&mut self, snapshot: &Snapshot<'_, G>) {
        (self.0)(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::Scheme;
    use crate::genome::{Binary, Bits};
    use crate::operator::{BitFlip, Tournament, UniformCrossover};

    fn ga(scheme: Scheme) -> crate::Ga<Binary, Tournament, UniformCrossover, BitFlip> {
        crate::Ga::builder(Binary::new(30).unwrap())
            .population_size(20)
            .select(Tournament::new(3).unwrap())
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::per_gene(0.05).unwrap())
            .scheme(scheme)
            .seed(1)
            .build()
            .unwrap()
    }

    fn one_max(genome: &Bits) -> f64 {
        genome.count_ones() as f64
    }

    #[test]
    fn no_info_no_store() {
        let mut engine =
            Engine::new(ga(Scheme::default()), one_max).stop_when(Stop::generations(20));
        engine.run().unwrap();
        assert!(engine.infos.is_empty());
        assert!(!engine.infos.is_allocated());
        assert_eq!(engine.evaluated_infos.capacity(), 0);
    }

    #[test]
    fn the_store_keeps_the_population_the_discarded_and_the_best() {
        for scheme in [
            Scheme::Generational { elitism: 1 },
            Scheme::MuPlusLambda { lambda: 30 },
            Scheme::MuCommaLambda { lambda: 30 },
        ] {
            let fitness = |genome: &Bits| Evaluated::new(one_max(genome), genome.to_string());
            let mut engine = Engine::new(ga(scheme), fitness);
            // one generation at a time, over a long run
            for generation in 0..500 {
                engine.stop = Some(Stop::generations(generation));
                engine.run().unwrap();
                let algorithm = &engine.algorithm;
                let kept = algorithm.population().iter().chain(algorithm.discarded());
                let kept: std::collections::HashSet<&Bits> = kept
                    .chain(algorithm.best())
                    .map(Individual::genome)
                    .collect();
                assert_eq!(engine.infos.len(), kept.len(), "{scheme:?}");
                for genome in kept {
                    assert_eq!(
                        engine.infos.info::<String>(genome),
                        Some(&genome.to_string())
                    );
                }
            }
        }
    }
}
