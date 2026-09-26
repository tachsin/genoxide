//! Asynchronous evaluation: every worker gets a new genome as soon as it's done.

use super::{
    Checkpoint, FitnessFunction, IntoFitness, NanPolicy, Outcome, Progress, Stop, StopReason,
    checkpoint, trace, validate_checkpoint,
};
use crate::algorithm::Incremental;
use crate::observer::{Observer, Snapshot};
use crate::{Error, Fitness, Individual, Result};
use std::any::Any;
use std::fmt;
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// The most workers an [`AsyncEngine`] starts: far more than the CPUs of one machine, for fitness
/// functions that mostly wait, and few enough that starting them doesn't exhaust the system.
pub const MAX_WORKERS: usize = 4096;

/// Runs an [`Incremental`] algorithm, such as a [`SteadyGa`](crate::algorithm::SteadyGa), with
/// asynchronous evaluation on worker threads: each worker gets a new genome as soon as it's done,
/// so none waits for the slowest evaluation of a generation. For expensive fitness functions
/// whose time varies, e.g. simulations or training runs.
///
/// - The workers are threads of their own (not rayon's), one per available CPU by default.
/// - A generation is counted for every `population_size` evaluations: observers, checkpoints and
///   [`Stop::generations`] use them. Observers are also notified once at the end, if evaluations
///   happened since the last generation.
/// - Stop conditions are checked after every result, once the initial population
///   (`population_size` results) is evaluated, like with [`Engine`](super::Engine). Then no new
///   genome is proposed, and the run returns once the evaluations in flight are done; their
///   results count.
///   [`Stop::evaluations`] and [`Stop::generations`] stop proposing once the evaluations done and
///   in flight reach the limit, so the run ends on it exactly. The generations that the results in
///   flight complete are observed and checkpointed too.
/// - A panic in the fitness function stops the run once the other evaluations in flight are done,
///   and then continues on the calling thread.
///
/// With one worker, a seed gives the same run every time. With more, the order of the results
/// depends on how long each evaluation takes, so runs differ.
///
/// ```
/// use genoxide::prelude::*;
///
/// let ga = Ga::builder(Binary::new(32)?)
///     .population_size(20)
///     .select(Tournament::new(3)?)
///     .crossover(UniformCrossover::new())
///     .mutate(BitFlip::per_gene(1.0 / 32.0)?)
///     .seed(1)
///     .build_steady()?;
/// let outcome = AsyncEngine::new(ga, |genome: &Bits| genome.count_ones() as f64)
///     .workers(2)
///     .stop_when(Stop::evaluations(1_000))
///     .run()?;
/// assert_eq!(outcome.evaluations(), 1_000);
/// # Ok::<(), genoxide::Error>(())
/// ```
pub struct AsyncEngine<'o, A: Incremental, F> {
    algorithm: A,
    fitness: F,
    stop: Option<Stop>,
    observers: Vec<Box<dyn Observer<A::Genome> + 'o>>,
    abort: Option<Arc<AtomicBool>>,
    nan_policy: NanPolicy,
    workers: usize,
    checkpoint: Option<Checkpoint<'o, A>>,
}

// a finished evaluation: the genome and its fitness, or the panic of the fitness function
type Done<G> = (G, std::result::Result<Result<Fitness>, Box<dyn Any + Send>>);

impl<'o, A, F> AsyncEngine<'o, A, F>
where
    A: Incremental,
    A::Genome: Send,
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
            workers: thread::available_parallelism().map_or(1, NonZeroUsize::get),
            checkpoint: None,
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

    /// Stops the run once `flag` is set, e.g. from another thread, after the evaluations in
    /// flight.
    pub fn abort_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.abort = Some(flag);
        self
    }

    /// What to do when the fitness function returns NaN. Invalid fitness by default.
    pub fn nan_policy(mut self, policy: NanPolicy) -> Self {
        self.nan_policy = policy;
        self
    }

    /// Adds an observer, notified after every `population_size` evaluations (a generation), and
    /// at the end. Pass `&mut observer` to keep access to it after the run.
    pub fn observe<O: Observer<A::Genome> + 'o>(mut self, observer: O) -> Self {
        self.observers.push(Box::new(observer));
        self
    }

    /// The number of evaluations at a time, between 1 and [`MAX_WORKERS`]. The number of
    /// available CPUs by default; more for fitness functions that mostly wait, e.g. on other
    /// processes.
    pub fn workers(mut self, workers: usize) -> Self {
        self.workers = workers;
        self
    }

    /// Calls `save` with the algorithm every `generations` generations and when the run stops,
    /// e.g. to write a checkpoint with `genoxide::checkpoint::save_file` (the `serde` feature).
    /// The evaluations in flight are not in it. An error from `save` stops the run with that
    /// error. `generations` must be at least 1.
    pub fn checkpoint_every<C>(mut self, generations: u64, save: C) -> Self
    where
        C: FnMut(&A) -> Result<()> + 'o,
    {
        self.checkpoint = Some((generations, Box::new(save)));
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

    /// Runs the algorithm until a stop condition is met or the abort flag is set.
    ///
    /// # Errors
    ///
    /// - [`Error::MissingSetting`] without a stop condition or abort flag.
    /// - [`Error::InvalidSetting`] for an invalid stop condition, 0 workers or more than
    ///   [`MAX_WORKERS`], workers the system can't start, or checkpoints every 0 generations.
    /// - [`Error::NanFitness`] for a NaN with [`NanPolicy::Error`], and
    ///   [`Error::InvalidFitness`] for a negative constraint violation.
    /// - [`Error::FitnessCount`] if a [`Batch`](super::Batch) doesn't return one score for one
    ///   genome.
    /// - The errors of the algorithm's [`receive`](Incremental::receive) and of the checkpoint
    ///   closure.
    ///
    /// The run stops at the first error, once the evaluations in flight are done. If the algorithm
    /// has run before and a stop condition is already met, or its limit of evaluations was
    /// reached before the initial population was complete, it returns that outcome at once,
    /// without notifying the observers or calling the checkpoint closure again.
    ///
    /// # Panics
    ///
    /// A panic in the fitness function propagates to the caller, on the calling thread, once the
    /// other evaluations in flight are done.
    pub fn run(&mut self) -> Result<Outcome<A::Genome>>
    where
        F: Sync,
    {
        if self.stop.is_none() && self.abort.is_none() {
            return Err(Error::MissingSetting {
                setting: "stop_when",
            });
        }
        if let Some(stop) = &self.stop {
            stop.validate()?;
        }
        if self.workers == 0 || self.workers > MAX_WORKERS {
            return Err(Error::InvalidSetting {
                setting: "workers",
                reason: format!("must be between 1 and {MAX_WORKERS}, got {}", self.workers),
            });
        }
        validate_checkpoint(&self.checkpoint)?;
        let _span = trace::run::<A>();
        let Self {
            algorithm,
            fitness,
            stop,
            observers,
            abort,
            nan_policy,
            workers,
            checkpoint: save,
        } = self;
        let workers = *workers;
        let fitness = &*fitness;
        let mut driver = Driver {
            size: algorithm.population_size().max(1) as u64,
            algorithm,
            stop: stop.as_ref(),
            observers,
            abort: abort.as_deref(),
            nan_policy: *nan_policy,
            save,
            start: Instant::now(),
            notified: None,
            discarded: Vec::new(),
        };
        // a run that continues: its stop condition may already be met, or its budget of
        // evaluations spent before the initial population was complete
        let evaluations = driver.algorithm.evaluations();
        if evaluations > 0 {
            let progress = driver.progress();
            let aborted = driver
                .abort
                .is_some_and(|flag| flag.load(Ordering::Relaxed));
            let reason = if evaluations < driver.size {
                // like `finish`, a spent budget is the reason even if no condition says so
                driver.budget_reached(0).then_some(StopReason::Evaluations)
            } else if aborted {
                Some(StopReason::Aborted)
            } else {
                driver.stop.and_then(|stop| stop.check(&progress))
            };
            if let (Some(stop_reason), Some(best)) = (reason, driver.algorithm.best()) {
                return Ok(Outcome {
                    best: best.clone(),
                    generations: progress.generation,
                    evaluations: progress.evaluations,
                    elapsed: Duration::ZERO,
                    stop_reason,
                });
            }
        }
        let (jobs, job_queue) = mpsc::channel::<A::Genome>();
        let job_queue = Mutex::new(job_queue);
        let (done, results) = mpsc::channel::<Done<A::Genome>>();
        let mut stop_reason = None;
        let mut failure = None;
        let mut panicked = None;
        thread::scope(|scope| {
            for _ in 0..workers {
                let done = done.clone();
                let job_queue = &job_queue;
                let spawned = thread::Builder::new().spawn_scoped(scope, move || {
                    loop {
                        // the lock is held only while waiting for a job
                        let job = match job_queue.lock() {
                            Ok(queue) => queue.recv(),
                            Err(_) => return,
                        };
                        let Ok(genome) = job else { return };
                        let evaluated = panic::catch_unwind(AssertUnwindSafe(|| {
                            if !fitness.is_batch() {
                                return fitness.evaluate(&genome).into_fitness();
                            }
                            // a batch of one, which must give one score
                            let mut scores = fitness.evaluate_batch(&[&genome]);
                            match (scores.pop(), scores.len()) {
                                (Some(score), 0) => score.into_fitness(),
                                (score, rest) => Err(Error::FitnessCount {
                                    expected: 1,
                                    got: rest + usize::from(score.is_some()),
                                }),
                            }
                        }));
                        if done.send((genome, evaluated)).is_err() {
                            return;
                        }
                    }
                });
                if let Err(error) = spawned {
                    failure = Some(Error::InvalidSetting {
                        setting: "workers",
                        reason: format!("can't start {workers} threads: {error}"),
                    });
                    break;
                }
            }
            drop(done);
            if failure.is_some() {
                // the workers that started stop, without a job
                drop(jobs);
                return;
            }

            let mut in_flight = 0;
            while in_flight < workers && !driver.budget_reached(in_flight) {
                jobs.send(driver.algorithm.propose())
                    .expect("the workers wait for jobs");
                in_flight += 1;
            }
            while in_flight > 0 {
                let (genome, evaluated) = results.recv().expect("a worker sends every result");
                in_flight -= 1;
                // after a panic or an error, the results in flight are ignored
                if panicked.is_some() || failure.is_some() {
                    continue;
                }
                let fitness = match evaluated {
                    Ok(fitness) => fitness,
                    Err(payload) => {
                        panicked = Some(payload);
                        continue;
                    }
                };
                if let Err(error) = driver.accept(genome, fitness) {
                    failure = Some(error);
                    continue;
                }
                // while stopping, the results in flight count, and nothing more is proposed
                if stop_reason.is_some() {
                    if let Err(error) = driver.after_late_result() {
                        failure = Some(error);
                    }
                    continue;
                }
                match driver.after_result() {
                    Err(error) => failure = Some(error),
                    Ok(Some(reason)) => stop_reason = Some(reason),
                    Ok(None) => {
                        if !driver.budget_reached(in_flight) {
                            jobs.send(driver.algorithm.propose())
                                .expect("the workers wait for jobs");
                            in_flight += 1;
                        }
                    }
                }
            }
            drop(jobs);
        });
        if let Some(payload) = panicked {
            panic::resume_unwind(payload);
        }
        if let Some(error) = failure {
            return Err(error);
        }
        // the budget of evaluations is reached, even if the stop condition didn't say so
        driver.finish(stop_reason.unwrap_or(StopReason::Evaluations))
    }
}

// the state of a run, besides the fitness function and the workers
struct Driver<'a, 'o, A: Incremental> {
    algorithm: &'a mut A,
    stop: Option<&'a Stop>,
    observers: &'a mut Vec<Box<dyn Observer<A::Genome> + 'o>>,
    abort: Option<&'a AtomicBool>,
    nan_policy: NanPolicy,
    save: &'a mut Option<Checkpoint<'o, A>>,
    start: Instant,
    // the population size, at least 1: evaluations per generation
    size: u64,
    // the evaluations at the last notification of the observers
    notified: Option<u64>,
    // the individuals that left the population or didn't enter it, since the last notification
    discarded: Vec<Individual<A::Genome>>,
}

impl<A: Incremental> Driver<'_, '_, A> {
    fn progress(&self) -> Progress {
        // generation g is complete after (g + 1) * size evaluations; the best was found in the
        // generation its evaluation belongs to
        let evaluations = self.algorithm.evaluations();
        Progress {
            generation: (evaluations / self.size).saturating_sub(1),
            evaluations,
            elapsed: self.start.elapsed(),
            best: self.algorithm.best().and_then(Individual::fitness),
            objective: self.algorithm.objective(),
            best_generation: self.algorithm.best_evaluation().saturating_sub(1) / self.size,
        }
    }

    // gives a result to the algorithm, after the NaN policy
    fn accept(&mut self, genome: A::Genome, fitness: Result<Fitness>) -> Result<()> {
        let fitness = match fitness {
            Ok(fitness) => fitness,
            Err(Error::NanFitness) if self.nan_policy == NanPolicy::Invalid => Fitness::invalid(),
            Err(error) => return Err(error),
        };
        if let Some(individual) = self.algorithm.receive(genome, fitness)? {
            if !self.observers.is_empty() {
                self.discarded.push(individual);
            }
        }
        Ok(())
    }

    // after a result while running: a generation's observers and checkpoint, and why to stop
    fn after_result(&mut self) -> Result<Option<StopReason>> {
        let progress = self.progress();
        if progress.evaluations % self.size == 0 {
            trace::generation(&progress, None);
            self.notify(&progress);
            checkpoint(self.save, &*self.algorithm, progress.generation, false)?;
        }
        if self.abort.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return Ok(Some(StopReason::Aborted));
        }
        // like `Engine`, stop conditions apply once the initial population is evaluated
        if progress.evaluations < self.size {
            return Ok(None);
        }
        Ok(self.stop.and_then(|stop| stop.check(&progress)))
    }

    // the end: the last notification and checkpoint include the results that arrived while
    // stopping
    fn finish(&mut self, stop_reason: StopReason) -> Result<Outcome<A::Genome>> {
        let progress = self.progress();
        if self.notified != Some(progress.evaluations) {
            self.notify(&progress);
        }
        checkpoint(self.save, &*self.algorithm, progress.generation, true)?;
        trace::finished(&progress, stop_reason, None);
        Ok(Outcome {
            best: self
                .algorithm
                .best()
                .expect("an algorithm has a best individual after a result")
                .clone(),
            generations: progress.generation,
            evaluations: progress.evaluations,
            elapsed: progress.elapsed,
            stop_reason,
        })
    }

    // notifies the observers
    fn notify(&mut self, progress: &Progress) {
        self.notified = Some(progress.evaluations);
        let Some(best) = self.algorithm.best() else {
            return;
        };
        let snapshot = Snapshot::new(self.algorithm.population(), &self.discarded, best, progress);
        for observer in self.observers.iter_mut() {
            observer.observe(&snapshot);
        }
        self.discarded.clear();
    }

    // whether the evaluations done and in flight reach a limit of evaluations; the first genome
    // is always evaluated
    fn budget_reached(&self, in_flight: usize) -> bool {
        let Some(stop) = self.stop else {
            return false;
        };
        let evaluations = self.algorithm.evaluations() + in_flight as u64;
        // generation g is complete after (g + 1) * size evaluations
        let generation = (evaluations / self.size).checked_sub(1);
        evaluations > 0 && stop.limit_reached(evaluations, generation)
    }

    // a generation's observers, trace and checkpoint for a result that arrives while stopping
    fn after_late_result(&mut self) -> Result<()> {
        let progress = self.progress();
        if progress.evaluations % self.size == 0 {
            trace::generation(&progress, None);
            self.notify(&progress);
            checkpoint(self.save, &*self.algorithm, progress.generation, false)?;
        }
        Ok(())
    }
}

impl<A: Incremental + fmt::Debug, F> fmt::Debug for AsyncEngine<'_, A, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncEngine")
            .field("algorithm", &self.algorithm)
            .field("stop", &self.stop)
            .field("observers", &self.observers.len())
            .field("abort", &self.abort)
            .field("nan_policy", &self.nan_policy)
            .field("workers", &self.workers)
            .field(
                "checkpoint_every",
                &self.checkpoint.as_ref().map(|(every, _)| every),
            )
            .finish_non_exhaustive()
    }
}
