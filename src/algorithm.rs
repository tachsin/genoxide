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

pub mod ga;

pub use ga::{Ga, GaBuilder, Scheme, Unset};

use crate::genome::Genome;
use crate::{Fitness, Individual, Objective, Population, Result};

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

    /// The number of completed generations after the initial one: 0 once the initial population
    /// is evaluated.
    fn generation(&self) -> u64;

    /// The number of fitness values told so far.
    fn evaluations(&self) -> u64;

    /// The generation in which the best individual so far was found.
    fn best_generation(&self) -> u64;
}

/// The genomes an [`Algorithm`] asks to evaluate.
///
/// It doesn't allocate, and it's `Copy` and `Sync`, so it can be shared between threads for
/// parallel evaluation.
#[derive(Debug)]
pub struct Candidates<'a, G: Genome> {
    individuals: &'a [Individual<G>],
    indices: &'a [usize],
}

impl<G: Genome> Clone for Candidates<'_, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<G: Genome> Copy for Candidates<'_, G> {}

impl<'a, G: Genome> Candidates<'a, G> {
    /// The genomes of `individuals` at `indices`.
    ///
    /// # Panics
    ///
    /// [`get`](Candidates::get) and [`iter`](Candidates::iter) panic if an index is out of bounds.
    pub fn new(individuals: &'a [Individual<G>], indices: &'a [usize]) -> Self {
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
