//! Multi-objective algorithms as ask / tell state machines.

use super::Scores;
use crate::algorithm::Candidates;
use crate::genome::Genome;
use crate::{Individual, Objective, Population, Result};

/// A multi-objective algorithm with `M` objectives as an ask / tell state machine, like
/// [`Algorithm`](crate::Algorithm) with [`Scores`] instead of a single fitness.
///
/// [`ask`](MultiObjectiveAlgorithm::ask) gives the genomes that need scores, and
/// [`tell`](MultiObjectiveAlgorithm::tell) takes their scores, in the same order, and advances the
/// algorithm. The first ask gives the initial population; that tell completes generation 0.
/// Every later tell completes the next generation. A [`MultiEngine`](super::MultiEngine) runs
/// the loop.
pub trait MultiObjectiveAlgorithm<const M: usize> {
    /// The genome type.
    type Genome: Genome;

    /// Whether each objective is minimized or maximized.
    fn objectives(&self) -> [Objective; M];

    /// The genomes to evaluate next. Asking again before telling gives the same genomes.
    ///
    /// The list can be empty, e.g. when every child is a copy of a parent and inherits its
    /// scores. The generation is then completed by telling no scores.
    fn ask(&mut self) -> Candidates<'_, Self::Genome, Scores<M>>;

    /// The scores of the genomes of the last [`ask`](MultiObjectiveAlgorithm::ask), in the same
    /// order.
    ///
    /// # Errors
    ///
    /// [`Error::TellWithoutAsk`](crate::Error::TellWithoutAsk) if nothing was asked, and
    /// [`Error::FitnessCount`](crate::Error::FitnessCount) if the number of scores is not the
    /// number of genomes asked for. The algorithm doesn't change on errors.
    fn tell(&mut self, scores: &[Scores<M>]) -> Result<()>;

    /// The current population.
    fn population(&self) -> &Population<Self::Genome, Scores<M>>;

    /// The non-dominated individuals of the current population: its first front, the best
    /// trade-offs found. Empty before the first [`tell`](MultiObjectiveAlgorithm::tell).
    fn front(&self) -> &[Individual<Self::Genome, Scores<M>>];

    /// The individuals evaluated in the last generation that didn't survive into the population.
    /// Empty by default.
    fn discarded(&self) -> &[Individual<Self::Genome, Scores<M>>] {
        &[]
    }

    /// The number of completed generations after the initial one: 0 once the initial population
    /// is evaluated.
    fn generation(&self) -> u64;

    /// The number of scores told so far.
    fn evaluations(&self) -> u64;

    /// The last generation in which the front gained a solution that no member of the previous
    /// front dominated or equaled, for [`Stop::stagnation`](crate::Stop::stagnation).
    fn front_generation(&self) -> u64;
}
