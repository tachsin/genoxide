//! Observers: statistics, hall of fame and custom callbacks, notified after every generation.
//!
//! For a callback, pass a closure to [`Engine::on_generation`](crate::Engine::on_generation).
//! For an observer that keeps state and can be reused, implement [`Observer`].

pub mod hall_of_fame;
pub mod report;
pub mod statistics;

pub use hall_of_fame::HallOfFame;
pub use report::Report;
pub use statistics::{GenerationStatistics, Statistics};

use crate::engine::{InfoStore, Progress};
use crate::genome::Genome;
use crate::{Individual, Population};
use std::any::Any;

/// Notified by the [`Engine`](crate::Engine) after every generation, including the initial
/// population.
///
/// `&mut O` is an observer too, so an observer can be lent to an engine and read after the run.
///
/// ```
/// use genoxide::observer::{Observer, Snapshot};
/// use genoxide::prelude::*;
///
/// // the best score after every generation
/// #[derive(Default)]
/// struct BestScores(Vec<f64>);
///
/// impl Observer<Bits> for BestScores {
///     fn observe(&mut self, snapshot: &Snapshot<'_, Bits>) {
///         if let Some(score) = snapshot.best().fitness().and_then(|fitness| fitness.score()) {
///             self.0.push(score);
///         }
///     }
/// }
///
/// let ga = Ga::builder(Binary::new(16)?)
///     .population_size(10)
///     .select(Tournament::new(2)?)
///     .crossover(UniformCrossover::new())
///     .mutate(BitFlip::count(1)?)
///     .seed(1)
///     .build()?;
/// let mut best = BestScores::default();
/// let outcome = Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
///     .stop_when(Stop::generations(10))
///     .observe(&mut best)
///     .run()?;
/// assert_eq!(best.0.len() as u64, outcome.generations() + 1);
/// # Ok::<(), genoxide::Error>(())
/// ```
pub trait Observer<G: Genome> {
    /// Called after a generation.
    fn observe(&mut self, snapshot: &Snapshot<'_, G>);
}

impl<G: Genome, O: Observer<G> + ?Sized> Observer<G> for &mut O {
    fn observe(&mut self, snapshot: &Snapshot<'_, G>) {
        (**self).observe(snapshot);
    }
}

/// The state of a run after a generation.
#[derive(Debug)]
pub struct Snapshot<'a, G: Genome> {
    population: &'a Population<G>,
    discarded: &'a [Individual<G>],
    best: &'a Individual<G>,
    progress: &'a Progress,
    infos: &'a InfoStore<G>,
}

impl<'a, G: Genome> Snapshot<'a, G> {
    pub(crate) fn new(
        population: &'a Population<G>,
        discarded: &'a [Individual<G>],
        best: &'a Individual<G>,
        progress: &'a Progress,
        infos: &'a InfoStore<G>,
    ) -> Self {
        Self {
            population,
            discarded,
            best,
            progress,
            infos,
        }
    }

    /// The population after the generation.
    pub fn population(&self) -> &'a Population<G> {
        self.population
    }

    /// The individuals evaluated in this generation that didn't survive into the population, e.g.
    /// the offspring rejected by (μ,λ) selection.
    pub fn discarded(&self) -> &'a [Individual<G>] {
        self.discarded
    }

    /// The best individual found so far.
    pub fn best(&self) -> &'a Individual<G> {
        self.best
    }

    /// The generation, evaluations, time and best fitness so far.
    pub fn progress(&self) -> &'a Progress {
        self.progress
    }

    /// The info the fitness function returned with the fitness of `genome` in an
    /// [`Evaluated`](crate::engine::Evaluated), as a `T`, for a genome of the population, of the
    /// discarded individuals or of the best. `None` for another type, or a genome without info:
    /// the fitness function returned none, or it was evaluated by another engine (e.g. before
    /// resuming from a checkpoint).
    ///
    /// A copy of a parent and a survivor have the info of their genome: the fitness function is
    /// deterministic. See [`Evaluated`](crate::engine::Evaluated) for an example.
    pub fn info<T: Any>(&self, genome: &G) -> Option<&'a T> {
        self.infos.info(genome)
    }

    /// The info of the best individual found so far, see [`info`](Snapshot::info).
    pub fn best_info<T: Any>(&self) -> Option<&'a T> {
        self.info(self.best.genome())
    }
}
