//! The best distinct solutions of a run.

use super::{Observer, Snapshot};
use crate::genome::Genome;
use crate::{Error, Individual, Objective, Result};

/// The best `capacity` individuals with distinct genomes seen during a run, best first.
///
/// After every generation, it's offered the population and the individuals evaluated but not
/// kept (e.g. the offspring rejected by (μ,λ) selection), so it sees every evaluated individual.
/// On ties, the individual seen first comes first.
///
/// ```
/// use genoxide::prelude::*;
///
/// let ga = Ga::builder(Binary::new(16)?)
///     .population_size(20)
///     .select(Tournament::new(2)?)
///     .crossover(UniformCrossover::new())
///     .mutate(BitFlip::count(1)?)
///     .seed(3)
///     .build()?;
/// let mut hall_of_fame = HallOfFame::new(5)?;
/// Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
///     .stop_when(Stop::generations(10))
///     .observe(&mut hall_of_fame)
///     .run()?;
/// assert_eq!(hall_of_fame.individuals().len(), 5);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HallOfFame<G: Genome> {
    capacity: usize,
    individuals: Vec<Individual<G>>,
}

impl<G: Genome> HallOfFame<G> {
    /// An empty hall of fame for up to `capacity` individuals, at least 1.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a capacity of 0.
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::InvalidSetting {
                setting: "hall_of_fame_capacity",
                reason: "must be at least 1".to_string(),
            });
        }
        // no memory reserved ahead: the capacity can be huge
        Ok(Self {
            capacity,
            individuals: Vec::new(),
        })
    }

    /// The maximum number of individuals.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The individuals, best first.
    pub fn individuals(&self) -> &[Individual<G>] {
        &self.individuals
    }

    /// The best individual.
    pub fn best(&self) -> Option<&Individual<G>> {
        self.individuals.first()
    }

    /// Offers an individual: it enters if it's evaluated, its genome is new and it's better than
    /// the worst individual (or there's room).
    pub fn offer(&mut self, candidate: &Individual<G>, objective: Objective) {
        let Some(fitness) = candidate.fitness() else {
            return;
        };
        // every individual in the hall of fame is evaluated
        let fitness_of = |individual: &Individual<G>| individual.fitness().unwrap_or(fitness);
        if self.individuals.len() == self.capacity
            && self
                .individuals
                .last()
                .is_some_and(|worst| !objective.is_better(fitness, fitness_of(worst)))
        {
            return;
        }
        if self
            .individuals
            .iter()
            .any(|individual| individual.genome() == candidate.genome())
        {
            return;
        }
        let position = self
            .individuals
            .iter()
            .position(|individual| objective.is_better(fitness, fitness_of(individual)))
            .unwrap_or(self.individuals.len());
        self.individuals.insert(position, candidate.clone());
        self.individuals.truncate(self.capacity);
    }
}

impl<G: Genome> Observer<G> for HallOfFame<G> {
    fn observe(&mut self, snapshot: &Snapshot<'_, G>) {
        let objective = snapshot.progress().objective();
        for individual in snapshot.population().iter().chain(snapshot.discarded()) {
            self.offer(individual, objective);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Fitness;
    use crate::genome::Bits;
    use proptest::prelude::*;

    fn individual(genome: u8, score: Option<f64>) -> Individual<Bits> {
        let mut individual = Individual::new((0..8).map(|bit| genome >> bit & 1 == 1).collect());
        individual.set_fitness(score.map_or(Fitness::invalid(), Fitness::new));
        individual
    }

    #[test]
    fn validation() {
        assert!(HallOfFame::<Bits>::new(0).is_err());
    }

    #[test]
    fn keeps_the_best_distinct() {
        let mut hall = HallOfFame::new(2).unwrap();
        hall.offer(&individual(1, Some(1.0)), Objective::Maximize);
        hall.offer(&individual(2, Some(5.0)), Objective::Maximize);
        hall.offer(&individual(2, Some(5.0)), Objective::Maximize); // duplicate
        hall.offer(&individual(3, Some(3.0)), Objective::Maximize);
        hall.offer(&individual(4, Some(3.0)), Objective::Maximize); // tie with the worst: stays out
        hall.offer(&Individual::new(Bits::zeros(8)), Objective::Maximize); // not evaluated
        let genomes: Vec<_> = hall
            .individuals()
            .iter()
            .map(|i| i.genome().to_string())
            .collect();
        assert_eq!(genomes, ["01000000", "11000000"]);
    }

    proptest! {
        #[test]
        fn matches_sorting_the_distinct_individuals(
            offers in prop::collection::vec((0u8..16, prop::option::of(-5i32..5)), 0..40),
            capacity in 1usize..6,
            minimize: bool,
        ) {
            let objective = if minimize { Objective::Minimize } else { Objective::Maximize };
            let mut hall = HallOfFame::new(capacity).unwrap();
            // the fitness is a function of the genome, as with a deterministic fitness function
            let offered: Vec<_> = offers
                .iter()
                .map(|&(genome, _)| individual(genome, offers.iter().find(|o| o.0 == genome).unwrap().1.map(f64::from)))
                .collect();
            for candidate in &offered {
                hall.offer(candidate, objective);
            }
            let mut expected: Vec<Individual<Bits>> = Vec::new();
            for candidate in &offered {
                if !expected.contains(candidate) {
                    expected.push(candidate.clone());
                }
            }
            expected.sort_by(|a, b| objective.compare(b.fitness().unwrap(), a.fitness().unwrap()));
            expected.truncate(capacity);
            prop_assert_eq!(hall.individuals(), expected.as_slice());
        }
    }
}
