//! Running a multi-objective algorithm.

use super::{MultiObjectiveAlgorithm, Scores};
use crate::engine::{NanPolicy, Progress, evaluate_all};
use crate::genome::Genome;
use crate::{Error, Individual, Population, Result, Stop, StopReason};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// A multi-objective fitness function: scores a genome on `M` objectives.
///
/// Implemented for every `Fn(&G) -> T + Sync` closure returning a type that converts into
/// [`Scores`], see [`IntoScores`]. Implement it for your own type to keep state, such as a
/// simulator.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a fitness function with {M} objectives for `{G}`",
    label = "not a fitness function with {M} objectives for `{G}`",
    note = "a multi-objective fitness function is a closure `|genome: &{G}| ...` returning `[f64; {M}]`, `([f64; {M}], f64)` (the objective values and a constraint violation), `Option<[f64; {M}]>` or `Scores<{M}>`: as many values as the algorithm's objectives"
)]
pub trait MultiFitnessFunction<G, const M: usize>: Sync {
    /// The type of the result.
    type Output: IntoScores<M>;

    /// Scores a genome.
    fn evaluate(&self, genome: &G) -> Self::Output;
}

impl<G, F, T, const M: usize> MultiFitnessFunction<G, M> for F
where
    F: Fn(&G) -> T + Sync,
    T: IntoScores<M>,
{
    type Output = T;

    fn evaluate(&self, genome: &G) -> T {
        self(genome)
    }
}

/// A value that converts into [`Scores`]: the result of a multi-objective fitness function.
///
/// - `[f64; M]`: the objective values.
/// - `([f64; M], f64)`: the objective values and a constraint violation.
/// - `Option<[f64; M]>`: `None` for a solution that can't be scored.
/// - [`Scores<M>`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a result with {M} objective values",
    label = "a multi-objective fitness function must return `[f64; {M}]`, `([f64; {M}], f64)`, `Option<[f64; {M}]>` or `Scores<{M}>`"
)]
pub trait IntoScores<const M: usize> {
    /// The scores, or an error for a NaN (see [`NanPolicy`]) or a negative constraint violation.
    ///
    /// # Errors
    ///
    /// [`Error::NanFitness`] and [`Error::InvalidFitness`].
    fn into_scores(self) -> Result<Scores<M>>;
}

impl<const M: usize> IntoScores<M> for Scores<M> {
    fn into_scores(self) -> Result<Scores<M>> {
        Ok(self)
    }
}

impl<const M: usize> IntoScores<M> for [f64; M] {
    fn into_scores(self) -> Result<Scores<M>> {
        Scores::try_new(self)
    }
}

impl<const M: usize> IntoScores<M> for ([f64; M], f64) {
    fn into_scores(self) -> Result<Scores<M>> {
        Scores::try_constrained(self.0, self.1)
    }
}

impl<const M: usize> IntoScores<M> for Option<[f64; M]> {
    fn into_scores(self) -> Result<Scores<M>> {
        self.map_or(Ok(Scores::invalid()), Scores::try_new)
    }
}

/// The state of a multi-objective run after a generation, for the
/// [`on_generation`](MultiEngine::on_generation) callback.
#[derive(Debug)]
pub struct MultiSnapshot<'a, G: Genome, const M: usize> {
    population: &'a Population<G, Scores<M>>,
    front: &'a [Individual<G, Scores<M>>],
    discarded: &'a [Individual<G, Scores<M>>],
    progress: &'a Progress,
}

impl<'a, G: Genome, const M: usize> MultiSnapshot<'a, G, M> {
    /// The current population.
    pub fn population(&self) -> &'a Population<G, Scores<M>> {
        self.population
    }

    /// The non-dominated individuals of the current population.
    pub fn front(&self) -> &'a [Individual<G, Scores<M>>] {
        self.front
    }

    /// The individuals evaluated in this generation that didn't survive.
    pub fn discarded(&self) -> &'a [Individual<G, Scores<M>>] {
        self.discarded
    }

    /// The generation, evaluations, elapsed time and the last generation the front improved.
    pub fn progress(&self) -> &'a Progress {
        self.progress
    }
}

/// The result of a multi-objective run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiOutcome<G: Genome, const M: usize> {
    front: Vec<Individual<G, Scores<M>>>,
    generations: u64,
    evaluations: u64,
    elapsed: Duration,
    stop_reason: StopReason,
}

impl<G: Genome, const M: usize> MultiOutcome<G, M> {
    /// The non-dominated individuals of the final population: the best trade-offs found.
    pub fn front(&self) -> &[Individual<G, Scores<M>>] {
        &self.front
    }

    /// The objective values of the front, for plotting or further analysis; invalid scores are
    /// left out.
    pub fn front_values(&self) -> Vec<[f64; M]> {
        self.front
            .iter()
            .filter_map(|individual| individual.fitness().and_then(|scores| scores.values()))
            .collect()
    }

    /// The front, consuming the outcome.
    pub fn into_front(self) -> Vec<Individual<G, Scores<M>>> {
        self.front
    }

    /// The number of completed generations after the initial one.
    pub fn generations(&self) -> u64 {
        self.generations
    }

    /// The number of evaluations.
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

type Callback<'o, G, const M: usize> = Box<dyn FnMut(&MultiSnapshot<'_, G, M>) + 'o>;

/// Runs a [`MultiObjectiveAlgorithm`] with a fitness function: the multi-objective counterpart of
/// [`Engine`](crate::Engine), with the same stop conditions, parallel evaluation (with the same
/// results as sequential evaluation), abort flag and NaN policy.
///
/// - [`Stop::stagnation`] counts the generations since the front last gained a solution that no
///   earlier front member dominated or equaled.
/// - [`Stop::target`] needs a single objective: running with it is an error.
///
/// ```
/// use genoxide::prelude::*;
/// use genoxide::Objective::Minimize;
///
/// // Schaffer's problem: minimize x² and (x − 2)², a trade-off for x between 0 and 2
/// let nsga2 = Nsga2::builder(Real::uniform(1, -10.0..=10.0)?, [Minimize, Minimize])
///     .population_size(20)
///     .crossover(SimulatedBinaryCrossover::new(15.0)?)
///     .mutate(PolynomialMutation::per_gene(1.0, 20.0)?)
///     .seed(1)
///     .build()?;
/// let outcome = MultiEngine::new(nsga2, |x: &Reals| [x[0] * x[0], (x[0] - 2.0) * (x[0] - 2.0)])
///     .stop_when(Stop::generations(50))
///     .run()?;
/// assert_eq!(outcome.front().len(), 20);
/// assert!(outcome.front().iter().all(|individual| (-0.01..=2.01).contains(&individual.genome()[0])));
/// # Ok::<(), genoxide::Error>(())
/// ```
pub struct MultiEngine<'o, A, F, const M: usize>
where
    A: MultiObjectiveAlgorithm<M>,
{
    algorithm: A,
    fitness: F,
    stop: Option<Stop>,
    callbacks: Vec<Callback<'o, A::Genome, M>>,
    abort: Option<Arc<AtomicBool>>,
    nan_policy: NanPolicy,
    parallel: bool,
    results: Vec<Result<Scores<M>>>,
    scores: Vec<Scores<M>>,
}

impl<'o, A, F, const M: usize> MultiEngine<'o, A, F, M>
where
    A: MultiObjectiveAlgorithm<M>,
    F: MultiFitnessFunction<A::Genome, M>,
{
    /// An engine for `algorithm` with `fitness`. Set a stop condition before running.
    pub fn new(algorithm: A, fitness: F) -> Self {
        Self {
            algorithm,
            fitness,
            stop: None,
            callbacks: Vec::new(),
            abort: None,
            nan_policy: NanPolicy::default(),
            parallel: false,
            results: Vec::new(),
            scores: Vec::new(),
        }
    }

    /// When to stop, checked after every generation. Calling it again adds a condition: the run
    /// stops when any of them is met.
    pub fn stop_when(mut self, stop: Stop) -> Self {
        self.stop = Some(match self.stop {
            Some(existing) => existing.or(stop),
            None => stop,
        });
        self
    }

    /// A flag that stops the run after the current generation when set, e.g. from another thread
    /// or a Ctrl+C handler.
    pub fn abort_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.abort = Some(flag);
        self
    }

    /// What a NaN objective value or violation means: invalid scores (the default), or an
    /// error. A fitness function that returns [`Scores`] made with [`Scores::new`] has already
    /// turned a NaN into invalid scores; return `[f64; M]`, or build them with
    /// [`Scores::try_new`], for the error.
    pub fn nan_policy(mut self, policy: NanPolicy) -> Self {
        self.nan_policy = policy;
        self
    }

    /// Calls `callback` after every generation, e.g. to report progress or record the front.
    pub fn on_generation<C>(mut self, callback: C) -> Self
    where
        C: FnMut(&MultiSnapshot<'_, A::Genome, M>) + 'o,
    {
        self.callbacks.push(Box::new(callback));
        self
    }

    /// Evaluates each generation's genomes in parallel (with rayon), with the same results as
    /// sequentially. Worth it for expensive fitness functions. Off by default.
    #[cfg(feature = "parallel")]
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// The algorithm.
    pub fn algorithm(&self) -> &A {
        &self.algorithm
    }

    /// The algorithm, consuming the engine, e.g. to continue it by hand.
    pub fn into_algorithm(self) -> A {
        self.algorithm
    }

    /// Runs the algorithm until a stop condition is met or the abort flag is set.
    ///
    /// # Errors
    ///
    /// - [`Error::MissingSetting`] without a stop condition or abort flag.
    /// - [`Error::InvalidSetting`] for an invalid stop condition, or [`Stop::target`].
    /// - [`Error::NanFitness`] for a NaN with [`NanPolicy::Error`], and
    ///   [`Error::InvalidFitness`] for a negative constraint violation.
    /// - The errors of the algorithm's [`tell`](MultiObjectiveAlgorithm::tell).
    pub fn run(&mut self) -> Result<MultiOutcome<A::Genome, M>> {
        if self.stop.is_none() && self.abort.is_none() {
            return Err(Error::MissingSetting {
                setting: "stop_when",
            });
        }
        if let Some(stop) = &self.stop {
            stop.validate()?;
            if stop.has_target() {
                return Err(Error::InvalidSetting {
                    setting: "stop_when",
                    reason: "a target needs a single objective; stop a multi-objective run by \
                             generations, evaluations, time, stagnation or a custom condition"
                        .to_string(),
                });
            }
        }
        let start = Instant::now();
        loop {
            self.evaluate()?;
            self.algorithm.tell(&self.scores)?;
            let progress = Progress::multi_objective(
                self.algorithm.generation(),
                self.algorithm.evaluations(),
                start.elapsed(),
                self.algorithm.front_generation(),
            );
            if !self.callbacks.is_empty() {
                let snapshot = MultiSnapshot {
                    population: self.algorithm.population(),
                    front: self.algorithm.front(),
                    discarded: self.algorithm.discarded(),
                    progress: &progress,
                };
                for callback in &mut self.callbacks {
                    callback(&snapshot);
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
                return Ok(MultiOutcome {
                    front: self.algorithm.front().to_vec(),
                    generations: progress.generation(),
                    evaluations: progress.evaluations(),
                    elapsed: progress.elapsed(),
                    stop_reason,
                });
            }
        }
    }

    // evaluates the asked genomes into `self.scores`
    fn evaluate(&mut self) -> Result<()> {
        let candidates = self.algorithm.ask();
        let fitness = &self.fitness;
        evaluate_all(
            candidates,
            self.parallel,
            &|genome: &A::Genome| fitness.evaluate(genome).into_scores(),
            &mut self.results,
        );
        self.scores.clear();
        for result in self.results.drain(..) {
            self.scores.push(match (result, self.nan_policy) {
                (Ok(scores), _) => scores,
                (Err(Error::NanFitness), NanPolicy::Invalid) => Scores::invalid(),
                (Err(error), _) => return Err(error),
            });
        }
        Ok(())
    }
}

impl<A, F, const M: usize> fmt::Debug for MultiEngine<'_, A, F, M>
where
    A: MultiObjectiveAlgorithm<M> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiEngine")
            .field("algorithm", &self.algorithm)
            .field("stop", &self.stop)
            .field("callbacks", &self.callbacks.len())
            .field("abort", &self.abort)
            .field("nan_policy", &self.nan_policy)
            .field("parallel", &self.parallel)
            .finish_non_exhaustive()
    }
}
