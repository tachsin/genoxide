//! Running an algorithm: fitness evaluation, stop conditions, observers and cancellation.

pub mod stop;

pub use stop::{Stop, StopReason};

use crate::algorithm::{Algorithm, Candidates};
use crate::genome::Genome;
use crate::observer::{Observer, Snapshot};
use crate::{Error, Fitness, Individual, Objective, Result};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// A fitness function: scores a genome.
///
/// Closures `|genome: &G| -> T` are fitness functions, where `T` is `f64`, [`Fitness`] or
/// `Option<f64>` (`None` for an invalid solution), see [`IntoFitness`]. Fitness functions must be
/// deterministic: the same genome always gets the same fitness.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a fitness function for `{G}`",
    label = "not a fitness function for `{G}`",
    note = "a fitness function is a closure `|genome: &{G}| ...` returning `f64`, `Fitness` or `Option<f64>`, or a type implementing `FitnessFunction<{G}>`"
)]
pub trait FitnessFunction<G>: Sync {
    /// The type of a score.
    type Output: IntoFitness;

    /// The score of `genome`.
    fn evaluate(&self, genome: &G) -> Self::Output;
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
pub trait IntoFitness {
    /// The fitness, or [`Error::NanFitness`] for NaN.
    fn into_fitness(self) -> Result<Fitness>;
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

impl IntoFitness for Option<f64> {
    /// `None` is [`Fitness::invalid`].
    fn into_fitness(self) -> Result<Fitness> {
        self.map_or(Ok(Fitness::invalid()), Fitness::try_new)
    }
}

/// What to do when a fitness function returns NaN.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum NanPolicy {
    /// The solution gets [`Fitness::invalid`] (the default).
    #[default]
    Invalid,
    /// The run stops with [`Error::NanFitness`].
    Error,
}

/// The state of a run, for stop conditions and observers.
#[derive(Clone, Debug, PartialEq)]
pub struct Progress {
    generation: u64,
    evaluations: u64,
    elapsed: Duration,
    best: Option<Fitness>,
    objective: Objective,
    best_generation: u64,
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

    /// The fitness of the best individual found so far.
    pub fn best(&self) -> Option<Fitness> {
        self.best
    }

    /// Whether higher or lower fitness is better.
    pub fn objective(&self) -> Objective {
        self.objective
    }

    /// The generation in which the best individual so far was found.
    pub fn best_generation(&self) -> u64 {
        self.best_generation
    }

    /// The number of generations since the best fitness last improved.
    pub fn stagnant_generations(&self) -> u64 {
        self.generation.saturating_sub(self.best_generation)
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome<G: Genome> {
    best: Individual<G>,
    generations: u64,
    evaluations: u64,
    elapsed: Duration,
    stop_reason: StopReason,
}

impl<G: Genome> Outcome<G> {
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
/// [`run`](Engine::run) can be called again to continue, e.g. with a new stop condition.
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
    results: Vec<Result<Fitness>>,
    scores: Vec<Fitness>,
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
            results: Vec::new(),
            scores: Vec::new(),
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
    /// - [`Error::InvalidSetting`] for an invalid stop condition.
    /// - [`Error::NanFitness`] if the fitness function returns NaN with [`NanPolicy::Error`].
    pub fn run(&mut self) -> Result<Outcome<A::Genome>> {
        if self.stop.is_none() && self.abort.is_none() {
            return Err(Error::MissingSetting {
                setting: "stop_when",
            });
        }
        if let Some(stop) = &self.stop {
            stop.validate()?;
        }
        let start = Instant::now();
        loop {
            self.evaluate()?;
            self.algorithm.tell(&self.scores)?;

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
            if !self.observers.is_empty() {
                let snapshot = Snapshot::new(
                    self.algorithm.population(),
                    self.algorithm.discarded(),
                    best,
                    &progress,
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
            };
            if let Some(stop_reason) = reason {
                return Ok(Outcome {
                    best: best.clone(),
                    generations: progress.generation,
                    evaluations: progress.evaluations,
                    elapsed: progress.elapsed,
                    stop_reason,
                });
            }
        }
    }

    // evaluates the asked genomes into `self.scores`
    fn evaluate(&mut self) -> Result<()> {
        let candidates = self.algorithm.ask();
        self.results.clear();
        if self.parallel {
            evaluate_parallel(&self.fitness, candidates, &mut self.results);
        } else {
            self.results.extend(
                candidates
                    .iter()
                    .map(|genome| self.fitness.evaluate(genome).into_fitness()),
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
}

#[cfg(feature = "parallel")]
fn evaluate_parallel<G, F>(
    fitness: &F,
    candidates: Candidates<'_, G>,
    results: &mut Vec<Result<Fitness>>,
) where
    G: Genome,
    F: FitnessFunction<G>,
{
    use rayon::prelude::*;
    // collecting an indexed parallel iterator keeps the order, whatever the thread count
    (0..candidates.len())
        .into_par_iter()
        .map(|position| {
            let genome = candidates.get(position).expect("position in bounds");
            fitness.evaluate(genome).into_fitness()
        })
        .collect_into_vec(results);
}

#[cfg(not(feature = "parallel"))]
fn evaluate_parallel<G, F>(_: &F, _: Candidates<'_, G>, _: &mut Vec<Result<Fitness>>)
where
    G: Genome,
    F: FitnessFunction<G>,
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
            .finish_non_exhaustive()
    }
}

struct FnObserver<C>(C);

impl<G: Genome, C: FnMut(&Snapshot<'_, G>)> Observer<G> for FnObserver<C> {
    fn observe(&mut self, snapshot: &Snapshot<'_, G>) {
        (self.0)(snapshot)
    }
}
