//! The best distinct solutions of a run.

use super::{Observer, Snapshot};
use crate::engine::InfoStore;
use crate::genome::Genome;
use crate::{Error, Individual, Objective, Result};
use std::any::Any;

/// The best `capacity` individuals with distinct genomes seen during a run, best first.
///
/// After every generation, it's offered the population and the individuals evaluated but not
/// kept (e.g. the offspring rejected by (μ,λ) selection), so it sees every evaluated individual.
/// On ties, the individual seen first comes first.
///
/// As an observer, it keeps the info of its members that the fitness function returned in an
/// [`Evaluated`](crate::engine::Evaluated), see [`info`](HallOfFame::info). Two halls of fame are
/// equal when their capacities and individuals are: the info isn't compared. With the `serde`
/// feature, the info isn't serialized.
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
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HallOfFame<G: Genome> {
    capacity: usize,
    individuals: Vec<Individual<G>>,
    // the info of its members, from the snapshots that offered them
    #[cfg_attr(feature = "serde", serde(skip, default = "InfoStore::default"))]
    infos: InfoStore<G>,
}

impl<G: Genome> PartialEq for HallOfFame<G> {
    fn eq(&self, other: &Self) -> bool {
        self.capacity == other.capacity && self.individuals == other.individuals
    }
}

impl<G: Genome> Eq for HallOfFame<G> {}

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
            infos: InfoStore::default(),
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

    /// The info the fitness function returned with the fitness of the member with `genome`, in an
    /// [`Evaluated`](crate::engine::Evaluated), as a `T`. `None` for another type, a genome that
    /// isn't a member, or a member without info: offered by hand with
    /// [`offer`](HallOfFame::offer), evaluated without info, or evaluated before resuming from a
    /// checkpoint.
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
    /// // the fitness, and the longest run of ones
    /// let fitness = |bits: &Bits| {
    ///     let (mut longest, mut run) = (0, 0);
    ///     for bit in bits.iter() {
    ///         run = if bit { run + 1 } else { 0 };
    ///         longest = longest.max(run);
    ///     }
    ///     Evaluated::new(bits.count_ones() as f64, longest)
    /// };
    /// let mut hall_of_fame = HallOfFame::new(5)?;
    /// Engine::new(ga, fitness)
    ///     .stop_when(Stop::generations(10))
    ///     .observe(&mut hall_of_fame)
    ///     .run()?;
    /// for individual in hall_of_fame.individuals() {
    ///     let longest = hall_of_fame.info::<i32>(individual.genome()).expect("evaluated with info");
    ///     println!("{} {:?}, longest run {longest}", individual.genome(), individual.fitness());
    /// }
    /// # Ok::<(), genoxide::Error>(())
    /// ```
    pub fn info<T: Any>(&self, genome: &G) -> Option<&T> {
        self.infos.info(genome)
    }

    /// Offers an individual: it enters if it's evaluated, its genome is new and it's better than
    /// the worst individual (or there's room).
    pub fn offer(&mut self, candidate: &Individual<G>, objective: Objective) {
        self.enter(candidate, objective);
    }

    // offers an individual, returning whether it entered
    fn enter(&mut self, candidate: &Individual<G>, objective: Objective) -> bool {
        let Some(fitness) = candidate.fitness() else {
            return false;
        };
        // every individual in the hall of fame is evaluated
        let fitness_of = |individual: &Individual<G>| individual.fitness().unwrap_or(fitness);
        if self.individuals.len() == self.capacity
            && self
                .individuals
                .last()
                .is_some_and(|worst| !objective.is_better(fitness, fitness_of(worst)))
        {
            return false;
        }
        if self
            .individuals
            .iter()
            .any(|individual| individual.genome() == candidate.genome())
        {
            return false;
        }
        let position = self
            .individuals
            .iter()
            .position(|individual| objective.is_better(fitness, fitness_of(individual)))
            .unwrap_or(self.individuals.len());
        let entering = if self.individuals.len() == self.capacity {
            // the worst leaves, with its info, and the candidate is copied into its memory; it's
            // better than the worst, so its position is before the worst's
            let mut worst = self.individuals.pop().expect("a full hall of fame");
            self.infos.remove(worst.genome());
            worst.clone_from(candidate);
            worst
        } else {
            candidate.clone()
        };
        self.individuals.insert(position, entering);
        true
    }
}

impl<G: Genome> Observer<G> for HallOfFame<G> {
    fn observe(&mut self, snapshot: &Snapshot<'_, G>) {
        let objective = snapshot.progress().objective();
        for individual in snapshot.population().iter().chain(snapshot.discarded()) {
            if self.enter(individual, objective)
                && let Some(info) = snapshot.infos.get(individual.genome())
            {
                self.infos.insert(individual.genome().clone(), info.clone());
            }
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

    #[test]
    fn keeps_the_info_of_its_members() {
        use crate::engine::{Info, InfoStore, Progress};
        let population: crate::Population<Bits> = (1..=4)
            .map(|genome| individual(genome, Some(f64::from(genome))))
            .collect();
        let mut infos = InfoStore::default();
        let mut evaluated = population
            .iter()
            .map(|individual| {
                (
                    individual.genome().clone(),
                    Info::new(individual.genome().to_string()),
                )
            })
            .collect();
        infos.update(&mut evaluated, population.iter().map(Individual::genome));
        let progress = Progress::for_test(0, Objective::Maximize);
        let mut hall = HallOfFame::new(2).unwrap();
        // one at a time: each better one pushes the worst out, with its info
        for individual in population.iter() {
            let one = std::slice::from_ref(individual);
            hall.observe(&Snapshot::new(
                &crate::Population::default(),
                one,
                individual,
                &progress,
                &infos,
            ));
        }
        assert_eq!(hall.infos.len(), 2);
        for member in hall.individuals() {
            assert_eq!(
                hall.info::<String>(member.genome()),
                Some(&member.genome().to_string())
            );
        }
        assert!(hall.info::<String>(population[0].genome()).is_none());
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
