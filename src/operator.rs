//! Genetic operators: selection, crossover and mutation.
//!
//! Crossover and mutation are parameterized by the [`Representation`], so an operator that
//! doesn't fit a genome (e.g. point crossover on a permutation) doesn't compile.

pub mod crossover;
pub mod mutate;
pub mod select;

pub use crossover::{PointCrossover, UniformCrossover};
pub use mutate::BitFlip;
pub use select::{
    RandomSelection, Rank, Roulette, StochasticUniversalSampling, Tournament, Truncation,
};

use crate::genome::{Genome, Representation};
use crate::{Error, Objective, Population, Result, StreamRng};
use std::fmt::Debug;

/// Selects parents from an evaluated population.
pub trait Select: Clone + Debug + Send + Sync {
    /// Selects `count` parents, with replacement: the positions of the selected individuals.
    ///
    /// Every individual is expected to be evaluated, unevaluated individuals are treated as
    /// invalid. Returns no parents for an empty population.
    fn select<G: Genome>(
        &self,
        population: &Population<G>,
        objective: Objective,
        count: usize,
        rng: &mut StreamRng,
    ) -> Vec<usize>;
}

/// Recombines two genomes into two children, in place.
pub trait Crossover<R: Representation>: Clone + Debug + Send + Sync {
    /// Recombines `a` and `b`, which become the two children.
    fn crossover(
        &self,
        representation: &R,
        a: &mut R::Genome,
        b: &mut R::Genome,
        rng: &mut StreamRng,
    );
}

/// Changes a genome randomly, in place.
pub trait Mutate<R: Representation>: Clone + Debug + Send + Sync {
    /// Mutates `genome`. The genome always changes (unless its space has a single genome).
    fn mutate(&self, representation: &R, genome: &mut R::Genome, rng: &mut StreamRng);
}

// A rate in (0, 1] for `setting`.
pub(crate) fn check_rate(setting: &'static str, rate: f64) -> Result<f64> {
    if rate > 0.0 && rate <= 1.0 {
        Ok(rate)
    } else {
        Err(Error::InvalidSetting {
            setting,
            reason: format!("must be greater than 0 and at most 1, got {rate}"),
        })
    }
}
