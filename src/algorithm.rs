//! Algorithms as ask / tell state machines.
//!
//! An [`Algorithm`] asks for genomes to evaluate and is told their fitness. The
//! [`Engine`](crate::Engine) runs that loop with a fitness function, stop conditions and
//! observers, but the loop can also be driven by hand, e.g. to evaluate genomes in another process
//! or asynchronously:
//!
//! ```
//! use genoxide::prelude::*;
//!
//! let mut ga = Ga::builder(Binary::new(8)?)
//!     .population_size(10)
//!     .select(Tournament::new(2)?)
//!     .crossover(UniformCrossover::new())
//!     .mutate(BitFlip::count(1)?)
//!     .seed(1)
//!     .build()?;
//! while ga.generation() < 20 {
//!     let fitness: Vec<Fitness> = ga
//!         .ask()
//!         .iter()
//!         .map(|genome| Fitness::new(genome.count_ones() as f64))
//!         .collect();
//!     ga.tell(&fitness)?;
//! }
//! assert!(ga.best().unwrap().fitness().unwrap().score().unwrap() >= 6.0);
//! # Ok::<(), genoxide::Error>(())
//! ```

pub mod cmaes;
pub mod de;
pub mod es;
pub mod ga;
pub mod islands;
pub(crate) mod line_search;
pub mod local;
pub mod local_search;
pub mod nelder_mead;
pub mod open_es;
pub mod pso;
pub mod steady;

pub use cmaes::{Cmaes, CmaesBuilder, Covariance, Restarts};
pub use de::{De, DeBuilder};
pub use es::{Es, EsBuilder};
pub use ga::{Ga, GaBuilder, Scheme, Unset};
pub use islands::{Islands, IslandsBuilder, Migrate};
pub use local_search::{Acceptance, LocalSearch, LocalSearchBuilder};
pub use nelder_mead::{NelderMead, NelderMeadBuilder};
pub use open_es::{OpenEs, OpenEsBuilder};
pub use pso::{Pso, PsoBuilder, Topology};
pub use steady::{Incremental, SteadyGa};

use crate::engine::{Evaluations, Provided, Wanted};
use crate::genome::Genome;
use crate::{Fitness, Individual, Objective, Population, Result, StreamRng};

// The ids of the random streams that parallel breeding derives from an algorithm's generator, one
// per algorithm: the streams of a generation are `rng.derive(id).derive(generation)`, and a
// candidate's (a GA's pair of parents, a DE trial, an ES offspring) is `derive(position)` of
// those. They must never change for the same major version: they decide the results of seeded
// runs.
pub(crate) mod breeding_streams {
    // `GaBuilder::parallel_breeding`
    pub(crate) const GA: u64 = 0;
    // `DeBuilder::parallel_breeding`
    pub(crate) const DE: u64 = 1;
    // `EsBuilder::parallel_breeding`
    pub(crate) const ES: u64 = 2;
    // `OpenEsBuilder::parallel_breeding`
    pub(crate) const OPEN_ES: u64 = 3;
}

// `make(position, input, rng)` for each of `inputs`, in order into `made` and `other`, each with
// the stream `streams.derive(position)`: on rayon's threads, so the same results on any number of
// threads
#[cfg(feature = "parallel")]
pub(crate) fn breed_in_parallel<I, T, U>(
    inputs: Vec<I>,
    streams: &StreamRng,
    make: impl Fn(usize, I, &mut StreamRng) -> (T, U) + Sync,
    made: &mut Vec<T>,
    other: &mut Vec<U>,
) where
    I: Send,
    T: Send,
    U: Send,
{
    use rayon::prelude::*;
    // unzipping an indexed parallel iterator keeps the order, whatever the thread count
    inputs
        .into_par_iter()
        .enumerate()
        .map(|(position, input)| make(position, input, &mut streams.derive(position as u64)))
        .unzip_into_vecs(made, other);
}

// without the `parallel` feature (a checkpoint of a run with parallel breeding), the same results,
// one after the other
#[cfg(not(feature = "parallel"))]
pub(crate) fn breed_in_parallel<I, T, U>(
    inputs: Vec<I>,
    streams: &StreamRng,
    make: impl Fn(usize, I, &mut StreamRng) -> (T, U),
    made: &mut Vec<T>,
    other: &mut Vec<U>,
) {
    made.clear();
    other.clear();
    for (position, input) in inputs.into_iter().enumerate() {
        let (a, b) = make(position, input, &mut streams.derive(position as u64));
        made.push(a);
        other.push(b);
    }
}

/// An algorithm as an ask / tell state machine.
///
/// [`ask`](Algorithm::ask) gives the genomes that need a fitness, and [`tell`](Algorithm::tell)
/// takes their fitness values, in the same order, and advances the algorithm. The first ask gives
/// the initial population; that tell completes generation 0. Every later tell completes the next
/// generation.
pub trait Algorithm {
    /// The genome type.
    type Genome: Genome;

    /// Whether higher or lower fitness is better.
    fn objective(&self) -> Objective;

    /// The genomes to evaluate next. Asking again before telling gives the same genomes.
    ///
    /// The list can be empty, e.g. when every child is a copy of a parent and inherits its
    /// fitness. The generation is then completed by telling no fitness values.
    fn ask(&mut self) -> Candidates<'_, Self::Genome>;

    /// The fitness of the genomes of the last [`ask`](Algorithm::ask), in the same order.
    ///
    /// # Errors
    ///
    /// [`Error::TellWithoutAsk`](crate::Error::TellWithoutAsk) if nothing was asked, and
    /// [`Error::FitnessCount`](crate::Error::FitnessCount) if the number of values is not the
    /// number of genomes asked for. The algorithm doesn't change on errors.
    fn tell(&mut self, fitness: &[Fitness]) -> Result<()>;

    /// The current population.
    fn population(&self) -> &Population<Self::Genome>;

    /// The best individual found so far, whether or not it's still in the population. Ties keep
    /// the individual found first.
    ///
    /// `None` before the first [`tell`](Algorithm::tell), and always `Some` after it: the
    /// [`Engine`](crate::Engine) relies on it.
    fn best(&self) -> Option<&Individual<Self::Genome>>;

    /// The individuals evaluated in the last generation that didn't survive into the population,
    /// e.g. the offspring rejected by (μ,λ) selection. Observers such as a hall of fame use them
    /// to see every evaluated individual. Empty by default.
    fn discarded(&self) -> &[Individual<Self::Genome>] {
        &[]
    }

    /// The number of completed generations after the initial one: 0 once the initial population
    /// is evaluated.
    fn generation(&self) -> u64;

    /// The number of fitness values told so far.
    fn evaluations(&self) -> u64;

    /// The generation in which the best individual so far was found.
    fn best_generation(&self) -> u64;

    /// Whether the algorithm has converged and has nothing more to do, e.g. a local method at a
    /// minimum with no restart left, or a [`Cmaes`] with [`cmaes::Restarts::Stop`] whose run has
    /// converged. The engines stop with [`StopReason::Converged`](crate::StopReason::Converged)
    /// after a generation in which it's true, unless a stop condition is met in the same
    /// generation. Asked again, the algorithm goes on, as it would without this method.
    ///
    /// `false` by default: the other population-based algorithms never finish on their own, and
    /// stop by their stop conditions.
    fn is_finished(&self) -> bool {
        false
    }

    /// Called by the [`Engine`](crate::Engine) once at the start of each run, with what the
    /// fitness function [provides](crate::engine::FitnessFunction::provides) besides the fitness:
    /// an algorithm that uses an extra, such as a gradient, resolves where it comes from here
    /// (e.g. with [`Gradients::resolve`](crate::gradient::Gradients::resolve)), and fails for
    /// what's missing. Nothing by default.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`](crate::Error::InvalidSetting) for an extra the algorithm needs
    /// and the fitness function doesn't provide, with the fix in the reason.
    #[inline]
    fn prepare(&mut self, provided: Provided) -> Result<()> {
        let _ = provided;
        Ok(())
    }

    /// What the next [`ask`](Algorithm::ask) wants besides the fitness, such as the gradient at
    /// each genome; between an ask and its tell, what that ask wants. The
    /// [`Engine`](crate::Engine) calls it before every ask, and only what the fitness function
    /// provides can be wanted. Nothing by default: the engine then evaluates as it would without
    /// extras.
    #[inline]
    fn wants(&self) -> Wanted {
        Wanted::NOTHING
    }

    /// The fitness and the [wanted](Algorithm::wants) extras of the genomes of the last
    /// [`ask`](Algorithm::ask), in the same order: what the [`Engine`](crate::Engine) tells an
    /// algorithm that wants extras, instead of [`tell`](Algorithm::tell). By default,
    /// `tell(evaluations.fitness())`.
    ///
    /// # Errors
    ///
    /// As [`tell`](Algorithm::tell).
    #[inline]
    fn tell_evaluations(&mut self, evaluations: &Evaluations<'_>) -> Result<()> {
        self.tell(evaluations.fitness())
    }
}

/// An algorithm that can score again what it keeps, for a fitness function that changes during a
/// run: adaptive penalty weights, a retrained surrogate model, a moving landscape.
///
/// After [`reevaluate`](Reevaluate::reevaluate), the next [`ask`](Algorithm::ask) gives the
/// genomes the algorithm keeps (its population, and whatever else it compares with, such as a
/// particle swarm's personal bests), and its [`tell`](Algorithm::tell) scores them again without
/// starting a new generation. [`best`](Algorithm::best) is then the best of them, and
/// [`best_generation`](Algorithm::best_generation) the current generation: old and new values are
/// never compared. The evaluations count as usual, and no random number is drawn, so a seeded run
/// that re-evaluates at the same generations is reproducible.
///
/// In an [`Engine`](crate::Engine), call it from [`control`](crate::Engine::control).
pub trait Reevaluate: Algorithm {
    /// Marks what the algorithm keeps for evaluation by the next [`ask`](Algorithm::ask).
    ///
    /// # Errors
    ///
    /// [`Error::ReevaluationOutOfTurn`](crate::Error::ReevaluationOutOfTurn) between an ask and
    /// its tell. Nothing changes on errors.
    fn reevaluate(&mut self) -> Result<()>;
}

/// The genomes an [`Algorithm`] asks to evaluate.
///
/// It doesn't allocate, and it's `Copy` and `Sync`, so it can be shared between threads for
/// parallel evaluation.
#[derive(Debug)]
pub struct Candidates<'a, G: Genome, F = Fitness> {
    individuals: &'a [Individual<G, F>],
    indices: &'a [usize],
}

impl<G: Genome, F> Clone for Candidates<'_, G, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G: Genome, F> Copy for Candidates<'_, G, F> {}

impl<'a, G: Genome, F> Candidates<'a, G, F> {
    /// The genomes of `individuals` at `indices`.
    ///
    /// # Panics
    ///
    /// [`get`](Candidates::get) and [`iter`](Candidates::iter) panic if an index is out of bounds.
    pub fn new(individuals: &'a [Individual<G, F>], indices: &'a [usize]) -> Self {
        Self {
            individuals,
            indices,
        }
    }

    /// The number of genomes.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether there are no genomes.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// The genome at `position`, or `None` if out of bounds.
    pub fn get(&self, position: usize) -> Option<&'a G> {
        let index = *self.indices.get(position)?;
        Some(self.individuals[index].genome())
    }

    /// The genomes, in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &'a G> + 'a {
        let individuals = self.individuals;
        self.indices
            .iter()
            .map(move |&index| individuals[index].genome())
    }
}
