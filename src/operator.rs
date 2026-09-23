//! Genetic operators: selection, crossover and mutation.
//!
//! Crossover and mutation are parameterized by the [`Representation`], so an operator that
//! doesn't fit a genome (e.g. point crossover on a permutation) doesn't compile.

pub mod crossover;
pub mod mutate;
pub mod permutation;
pub mod select;

pub use crossover::{
    ArithmeticCrossover, BlendCrossover, NoCrossover, PointCrossover, SimulatedBinaryCrossover,
    UniformCrossover,
};
pub use mutate::{
    BitFlip, GaussianMutation, PolynomialMutation, SelfAdaptiveMutation, SwapMutation,
    UniformMutation,
};
pub use permutation::{
    CycleCrossover, EdgeRecombinationCrossover, InsertionMutation, InversionMutation,
    OrderCrossover, PartiallyMappedCrossover, ScrambleMutation,
};
pub use select::{
    RandomSelection, Rank, Roulette, StochasticUniversalSampling, Tournament, Truncation,
};

use crate::genome::{Genome, Representation};
use crate::{Error, Objective, Population, Result, StreamRng};
use std::fmt::Debug;

/// Selects parents from an evaluated population.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a selection operator",
    label = "not a selection operator",
    note = "set the selection with `.select(...)`, e.g. `.select(Tournament::new(3)?)`"
)]
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
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a crossover for `{R}`",
    label = "not a crossover for `{R}`",
    note = "set the crossover with `.crossover(...)`; point and uniform crossover need genomes that implement `SwapGenes`, SBX, blend and arithmetic crossover are for `Real`, order, partially mapped, cycle and edge recombination crossover for `Permutation`, and `NoCrossover` fits every genome"
)]
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
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a mutation for `{R}`",
    label = "not a mutation for `{R}`",
    note = "set the mutation with `.mutate(...)` (`.neighbor(...)` for local search): `BitFlip` for `Binary`, `UniformMutation` for `Integer`, `GaussianMutation`, `PolynomialMutation` or `UniformMutation` for `Real`, `SelfAdaptiveMutation` for `AdaptiveReal`, `SwapMutation`, `InversionMutation`, `InsertionMutation` or `ScrambleMutation` for `Permutation`"
)]
pub trait Mutate<R: Representation>: Clone + Debug + Send + Sync {
    /// Mutates `genome`. The genome always changes (unless its space has a single genome).
    fn mutate(&self, representation: &R, genome: &mut R::Genome, rng: &mut StreamRng);
}

// A probability in [0, 1] for `setting`.
pub(crate) fn check_probability(setting: &'static str, probability: f64) -> Result<f64> {
    if (0.0..=1.0).contains(&probability) {
        Ok(probability)
    } else {
        Err(Error::InvalidSetting {
            setting,
            reason: format!("must be between 0 and 1, got {probability}"),
        })
    }
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
