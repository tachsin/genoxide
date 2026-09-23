//! A genome with its fitness and age.

use crate::Fitness;
use crate::genome::Genome;

/// A genome with its fitness and age.
///
/// The fitness is `None` until the individual is evaluated. That is different from
/// [`Fitness::invalid`], which is the result of an evaluation.
///
/// Changing the genome through [`genome_mut`](Individual::genome_mut) makes it a new individual:
/// the fitness and age are cleared, so a stale fitness is impossible.
///
/// ```
/// use genoxide::{Fitness, Individual};
/// use genoxide::genome::Bits;
///
/// let mut individual = Individual::new(Bits::zeros(3));
/// assert_eq!(individual.fitness(), None); // not evaluated
/// individual.set_fitness(Fitness::new(0.0));
/// individual.genome_mut().flip(0);
/// assert_eq!(individual.fitness(), None); // changed, so not evaluated
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Individual<G: Genome> {
    genome: G,
    fitness: Option<Fitness>,
    age: u32,
}

impl<G: Genome> Individual<G> {
    /// A new, not yet evaluated individual.
    pub fn new(genome: G) -> Self {
        Self {
            genome,
            fitness: None,
            age: 0,
        }
    }

    /// The genome.
    pub fn genome(&self) -> &G {
        &self.genome
    }

    /// Mutable access to the genome, which makes this a new individual: the fitness and age are
    /// cleared.
    pub fn genome_mut(&mut self) -> &mut G {
        self.fitness = None;
        self.age = 0;
        &mut self.genome
    }

    /// The genome, consuming the individual.
    pub fn into_genome(self) -> G {
        self.genome
    }

    /// The fitness, or `None` if not evaluated yet.
    pub fn fitness(&self) -> Option<Fitness> {
        self.fitness
    }

    /// Whether the individual has been evaluated.
    pub fn is_evaluated(&self) -> bool {
        self.fitness.is_some()
    }

    /// Sets the fitness, the result of evaluating the genome.
    pub fn set_fitness(&mut self, fitness: Fitness) {
        self.fitness = Some(fitness);
    }

    /// The number of generations this individual has survived.
    pub fn age(&self) -> u32 {
        self.age
    }

    /// Increments the age, at the end of a generation it survived.
    pub fn increment_age(&mut self) {
        self.age = self.age.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Bits;

    #[test]
    fn genome_mut_clears_fitness_and_age() {
        let mut individual = Individual::new(Bits::zeros(2));
        individual.set_fitness(Fitness::invalid());
        individual.increment_age();
        assert!(individual.is_evaluated());
        assert_eq!(individual.age(), 1);

        individual.genome_mut().set(0, true);
        assert!(!individual.is_evaluated());
        assert_eq!(individual.age(), 0);
        assert_eq!(individual.genome().to_string(), "10");
    }

    #[test]
    fn invalid_is_evaluated() {
        let mut individual = Individual::new(Bits::zeros(2));
        individual.set_fitness(Fitness::invalid());
        assert_eq!(individual.fitness(), Some(Fitness::invalid()));
        assert!(individual.is_evaluated());
    }

    #[test]
    fn age_saturates() {
        let mut individual = Individual::new(Bits::zeros(1));
        individual.age = u32::MAX;
        individual.increment_age();
        assert_eq!(individual.age(), u32::MAX);
    }
}
