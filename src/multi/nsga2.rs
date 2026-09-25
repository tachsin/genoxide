//! NSGA-II: the non-dominated sorting genetic algorithm.

use super::breed::{Variation, scores_of};
use super::pareto::gains;
use super::{MultiObjectiveAlgorithm, Scores, crowding_distance, non_dominated_sort};
use crate::algorithm::{Candidates, Unset};
use crate::genome::Representation;
use crate::operator::{Crossover, Mutate, check_rates, check_size};
use crate::rng::Chance;
use crate::{Error, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;
use std::cmp::Ordering;

/// NSGA-II (Deb et al., 2002): the classic multi-objective genetic algorithm, as an ask / tell
/// [`MultiObjectiveAlgorithm`].
///
/// Every generation:
///
/// 1. Parents are chosen by binary tournament with the crowded comparison: the lower
///    [non-dominated](non_dominated_sort) rank wins, then the larger
///    [crowding distance](crowding_distance), then a coin flip.
/// 2. Pairs of parents are recombined with the [`Crossover`] with probability `crossover_rate`,
///    and each child is mutated with the [`Mutate`] operator with probability `mutation_rate`,
///    as many children as the population size. A child that equals a member of the population
///    or an earlier child is dropped and another bred instead (see
///    [`eliminate_duplicates`](Nsga2Builder::eliminate_duplicates)); with copies allowed, a child
///    that equals a parent inherits its scores and isn't evaluated again.
/// 3. Parents and children compete: the next population takes whole fronts, best first, and
///    fills the rest from the next front by crowding distance (random on ties), which keeps the
///    front spread out.
///
/// Constraints are handled by constrained dominance ([`dominates`](super::dominates)).
///
/// Built with [`Nsga2::builder`], run with a [`MultiEngine`](super::MultiEngine). For real
/// genomes, the usual operators are [`SimulatedBinaryCrossover`](crate::operator::SimulatedBinaryCrossover)
/// (η 15 to 20, rate 0.9) and [`PolynomialMutation`](crate::operator::PolynomialMutation)
/// (η 20, a rate of 1 / the number of genes).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "R: serde::Serialize, C: serde::Serialize, X: serde::Serialize, R::Genome: serde::Serialize",
        deserialize = "R: serde::Deserialize<'de>, C: serde::Deserialize<'de>, X: serde::Deserialize<'de>, R::Genome: serde::Deserialize<'de>"
    ))
)]
pub struct Nsga2<R: Representation, C, X, const M: usize> {
    variation: Variation<R, C, X>,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays::array"))]
    objectives: [Objective; M],
    population_size: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    seed: u64,
    rng: StreamRng,
    population: Population<R::Genome, Scores<M>>,
    // the non-dominated rank and the crowding distance of each member of the population
    ranks: Vec<usize>,
    crowding: Vec<f64>,
    offspring: Vec<Individual<R::Genome, Scores<M>>>,
    pending: Vec<usize>,
    front: Vec<Individual<R::Genome, Scores<M>>>,
    discarded: Vec<Individual<R::Genome, Scores<M>>>,
    started: bool,
    asked: bool,
    generation: u64,
    evaluations: u64,
    front_generation: u64,
}

impl<R: Representation, const M: usize> Nsga2<R, Unset, Unset, M> {
    /// A builder for NSGA-II on `representation`, with the direction of each objective: the
    /// fitness function returns as many values as there are objectives.
    pub fn builder(representation: R, objectives: [Objective; M]) -> Nsga2Builder<R, M> {
        Nsga2Builder {
            representation,
            objectives,
            crossover: Unset,
            mutate: Unset,
            population_size: None,
            crossover_rate: 0.9,
            mutation_rate: 1.0,
            seed: None,
            eliminate_duplicates: true,
            initial_genomes: Vec::new(),
        }
    }
}

impl<R, C, X, const M: usize> Nsga2<R, C, X, M>
where
    R: Representation,
    C: Crossover<R>,
    X: Mutate<R>,
{
    /// The representation.
    pub fn representation(&self) -> &R {
        &self.variation.representation
    }

    /// The crossover operator.
    pub fn crossover(&self) -> &C {
        &self.variation.crossover
    }

    /// The mutation operator.
    pub fn mutate(&self) -> &X {
        &self.variation.mutate
    }

    /// The population size.
    pub fn population_size(&self) -> usize {
        self.population_size
    }

    /// The probability that a pair of parents is recombined.
    pub fn crossover_rate(&self) -> f64 {
        self.crossover_rate
    }

    /// The probability that a child is mutated.
    pub fn mutation_rate(&self) -> f64 {
        self.mutation_rate
    }

    /// The non-dominated rank of each member of the population, in its order: 0 for the first
    /// front. Empty before the first [`tell`](MultiObjectiveAlgorithm::tell).
    pub fn ranks(&self) -> &[usize] {
        &self.ranks
    }

    /// The crowding distance of each member of the population within its front, in its order.
    /// Empty before the first [`tell`](MultiObjectiveAlgorithm::tell).
    pub fn crowding_distances(&self) -> &[f64] {
        &self.crowding
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn breed(&mut self) {
        let (ranks, crowding) = (&self.ranks, &self.crowding);
        self.variation.breed(
            &self.population,
            self.population_size,
            &mut self.rng,
            |rng| crowded_tournament(ranks, crowding, rng),
            &mut self.offspring,
        );
    }

    // the next population from the parents and the offspring, and its ranks and crowding
    fn survive(&mut self) {
        let parents = std::mem::take(&mut self.population).into_vec();
        let parent_count = parents.len();
        let mut pool = parents;
        pool.append(&mut self.offspring);
        let scores = scores_of(&pool);
        let fronts = non_dominated_sort(&scores, &self.objectives);
        let mut chosen: Vec<(usize, usize, f64)> = Vec::with_capacity(self.population_size);
        for (rank, front) in fronts.iter().enumerate() {
            let distances = crowding_distance(&scores, front);
            let room = self.population_size - chosen.len();
            if front.len() <= room {
                chosen.extend(front.iter().zip(&distances).map(|(&i, &d)| (i, rank, d)));
            } else {
                // the least crowded first, random on ties
                let keys: Vec<u64> = front.iter().map(|_| self.rng.next_u64()).collect();
                let mut order: Vec<usize> = (0..front.len()).collect();
                order.sort_by(|&a, &b| {
                    distances[b]
                        .total_cmp(&distances[a])
                        .then(keys[a].cmp(&keys[b]))
                });
                chosen.extend(
                    order[..room]
                        .iter()
                        .map(|&position| (front[position], rank, distances[position])),
                );
            }
            if chosen.len() == self.population_size {
                break;
            }
        }
        let mut selected = vec![false; pool.len()];
        for &(index, _, _) in &chosen {
            selected[index] = true;
        }
        let mut slots: Vec<Option<Individual<R::Genome, Scores<M>>>> =
            pool.into_iter().map(Some).collect();
        self.discarded.clear();
        for (index, slot) in slots.iter_mut().enumerate() {
            if !selected[index] && index >= parent_count {
                self.discarded.push(slot.take().expect("not taken yet"));
            }
        }
        self.ranks.clear();
        self.crowding.clear();
        let mut population = Vec::with_capacity(chosen.len());
        for (index, rank, distance) in chosen {
            let mut individual = slots[index].take().expect("chosen once");
            if index < parent_count {
                individual.increment_age();
            }
            population.push(individual);
            self.ranks.push(rank);
            self.crowding.push(distance);
        }
        self.population = Population::new(population);
    }

    // the ranks and crowding of the initial population
    fn rank_initial(&mut self) {
        let scores = scores_of(self.population.as_slice());
        let fronts = non_dominated_sort(&scores, &self.objectives);
        self.ranks = vec![0; scores.len()];
        self.crowding = vec![0.0; scores.len()];
        for (rank, front) in fronts.iter().enumerate() {
            let distances = crowding_distance(&scores, front);
            for (&index, distance) in front.iter().zip(distances) {
                self.ranks[index] = rank;
                self.crowding[index] = distance;
            }
        }
    }

    // the new front, and whether it improved on the previous one
    fn update_front(&mut self) {
        let front: Vec<Individual<R::Genome, Scores<M>>> = self
            .population
            .iter()
            .zip(&self.ranks)
            .filter(|(_, rank)| **rank == 0)
            .map(|(individual, _)| individual.clone())
            .collect();
        if gains(
            &scores_of(&front),
            &scores_of(&self.front),
            &self.objectives,
        ) {
            self.front_generation = self.generation;
        }
        self.front = front;
    }
}

// the winner of a binary tournament by the crowded comparison: the lower rank, then the larger
// crowding distance, then a coin flip
fn crowded_tournament(ranks: &[usize], crowding: &[f64], rng: &mut StreamRng) -> usize {
    let size = ranks.len();
    let a = rng.below(size);
    let mut b = rng.below(size - 1);
    if b >= a {
        b += 1;
    }
    match ranks[a].cmp(&ranks[b]) {
        Ordering::Less => a,
        Ordering::Greater => b,
        Ordering::Equal => {
            if crowding[a] > crowding[b] {
                a
            } else if crowding[b] > crowding[a] || rng.below(2) != 0 {
                b
            } else {
                a
            }
        }
    }
}

impl<R, C, X, const M: usize> MultiObjectiveAlgorithm<M> for Nsga2<R, C, X, M>
where
    R: Representation,
    C: Crossover<R>,
    X: Mutate<R>,
{
    type Genome = R::Genome;

    fn objectives(&self) -> [Objective; M] {
        self.objectives
    }

    fn ask(&mut self) -> Candidates<'_, R::Genome, Scores<M>> {
        if !self.asked {
            self.pending.clear();
            if self.started {
                self.breed();
                self.pending.extend(
                    (0..self.offspring.len()).filter(|&i| !self.offspring[i].is_evaluated()),
                );
            } else {
                self.pending.extend(0..self.population.len());
            }
            self.asked = true;
        }
        let individuals = if self.started {
            &self.offspring
        } else {
            self.population.as_slice()
        };
        Candidates::new(individuals, &self.pending)
    }

    fn tell(&mut self, scores: &[Scores<M>]) -> Result<()> {
        if !self.asked {
            return Err(Error::TellWithoutAsk);
        }
        if scores.len() != self.pending.len() {
            return Err(Error::FitnessCount {
                expected: self.pending.len(),
                got: scores.len(),
            });
        }
        self.asked = false;
        self.evaluations += scores.len() as u64;
        if self.started {
            for (&index, &score) in self.pending.iter().zip(scores) {
                self.offspring[index].set_fitness(score);
            }
            self.generation += 1;
            self.survive();
        } else {
            for (individual, &score) in self.population.iter_mut().zip(scores) {
                individual.set_fitness(score);
            }
            self.rank_initial();
        }
        self.update_front();
        self.started = true;
        Ok(())
    }

    fn population(&self) -> &Population<R::Genome, Scores<M>> {
        &self.population
    }

    fn front(&self) -> &[Individual<R::Genome, Scores<M>>] {
        &self.front
    }

    fn discarded(&self) -> &[Individual<R::Genome, Scores<M>>] {
        &self.discarded
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn evaluations(&self) -> u64 {
        self.evaluations
    }

    fn front_generation(&self) -> u64 {
        self.front_generation
    }
}

/// A builder for [`Nsga2`], from [`Nsga2::builder`].
///
/// The population size, the crossover and the mutation are required. Defaults: `crossover_rate`
/// 0.9, `mutation_rate` 1.0, a random initial population and a random seed.
#[derive(Clone, Debug)]
pub struct Nsga2Builder<R: Representation, const M: usize, C = Unset, X = Unset> {
    representation: R,
    objectives: [Objective; M],
    crossover: C,
    mutate: X,
    population_size: Option<usize>,
    crossover_rate: f64,
    mutation_rate: f64,
    seed: Option<u64>,
    eliminate_duplicates: bool,
    initial_genomes: Vec<R::Genome>,
}

impl<R: Representation, const M: usize, C, X> Nsga2Builder<R, M, C, X> {
    /// The crossover operator. Required.
    pub fn crossover<T>(self, crossover: T) -> Nsga2Builder<R, M, T, X> {
        Nsga2Builder {
            representation: self.representation,
            objectives: self.objectives,
            crossover,
            mutate: self.mutate,
            population_size: self.population_size,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            seed: self.seed,
            eliminate_duplicates: self.eliminate_duplicates,
            initial_genomes: self.initial_genomes,
        }
    }

    /// The mutation operator. Required.
    pub fn mutate<T>(self, mutate: T) -> Nsga2Builder<R, M, C, T> {
        Nsga2Builder {
            representation: self.representation,
            objectives: self.objectives,
            crossover: self.crossover,
            mutate,
            population_size: self.population_size,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            seed: self.seed,
            eliminate_duplicates: self.eliminate_duplicates,
            initial_genomes: self.initial_genomes,
        }
    }

    /// The population size, at least 2 and at most 2^24; also the number of children per
    /// generation. Required.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// The probability that a pair of parents is recombined, between 0 and 1. 0.9 by default.
    pub fn crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// The probability that a child is mutated, between 0 and 1. 1 by default: how many genes
    /// change is up to the mutation operator.
    pub fn mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// The seed of the random numbers, for a reproducible run. Random by default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Whether a child that equals a member of the population, or an earlier child of the same
    /// generation, is dropped and another bred instead, which keeps the population and its front
    /// free of copies. On by default, as in pymoo. When copies are all a population can breed,
    /// after 100 dropped children per child needed, copies are accepted.
    pub fn eliminate_duplicates(mut self, eliminate: bool) -> Self {
        self.eliminate_duplicates = eliminate;
        self
    }

    /// Genomes for the initial population, at most the population size, each valid for the
    /// representation. The rest is random.
    pub fn initial_genomes<I: IntoIterator<Item = R::Genome>>(mut self, genomes: I) -> Self {
        self.initial_genomes = genomes.into_iter().collect();
        self
    }

    /// Validates the settings and creates the algorithm, with its initial population.
    ///
    /// # Errors
    ///
    /// - [`Error::MissingSetting`] without a population size.
    /// - [`Error::InvalidSetting`] for a population size below 2 or above 2^24, no objectives,
    ///   rates out of range, a mutation rate of 0 with a crossover rate of 0 or
    ///   [`NoCrossover`](crate::operator::NoCrossover), or more initial genomes than the
    ///   population size.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<Nsga2<R, C, X, M>>
    where
        C: Crossover<R>,
        X: Mutate<R>,
    {
        let size = self.population_size.ok_or(Error::MissingSetting {
            setting: "population_size",
        })?;
        let invalid = |setting, reason: String| Err(Error::InvalidSetting { setting, reason });
        if size < 2 {
            return invalid(
                "population_size",
                format!("NSGA-II needs at least 2 individuals, got {size}"),
            );
        }
        check_size("population_size", size)?;
        if M == 0 {
            return invalid("objectives", "at least 1 objective is needed".to_string());
        }
        let (crossover_rate, mutation_rate) = check_rates(
            self.crossover_rate,
            self.mutation_rate,
            self.crossover.recombines(),
        )?;
        if self.initial_genomes.len() > size {
            return invalid(
                "initial_genomes",
                format!(
                    "at most the population size {size}, got {}",
                    self.initial_genomes.len()
                ),
            );
        }
        for genome in &self.initial_genomes {
            self.representation.validate(genome)?;
        }
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let mut rng = StreamRng::seed_from_u64(seed);
        let random = size - self.initial_genomes.len();
        let mut genomes = self.initial_genomes;
        genomes.extend((0..random).map(|_| self.representation.random_genome(&mut rng)));
        Ok(Nsga2 {
            variation: Variation {
                representation: self.representation,
                crossover: self.crossover,
                mutate: self.mutate,
                crossover_chance: Chance::new(crossover_rate),
                mutation_chance: Chance::new(mutation_rate),
                eliminate_duplicates: self.eliminate_duplicates,
            },
            objectives: self.objectives,
            population_size: size,
            crossover_rate,
            mutation_rate,
            seed,
            rng,
            population: genomes.into_iter().map(Individual::unevaluated).collect(),
            ranks: Vec::new(),
            crowding: Vec::new(),
            offspring: Vec::new(),
            pending: Vec::new(),
            front: Vec::new(),
            discarded: Vec::new(),
            started: false,
            asked: false,
            generation: 0,
            evaluations: 0,
            front_generation: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective::{Maximize, Minimize};
    use crate::genome::{Binary, Bits, Real, Reals};
    use crate::multi::dominates;
    use crate::operator::{
        BitFlip, PolynomialMutation, SimulatedBinaryCrossover, UniformCrossover,
    };
    use proptest::prelude::*;

    type Real2 = Nsga2<Real, SimulatedBinaryCrossover, PolynomialMutation, 2>;

    fn builder(
        size: usize,
        seed: u64,
    ) -> Nsga2Builder<Real, 2, SimulatedBinaryCrossover, PolynomialMutation> {
        Nsga2::builder(Real::uniform(3, 0.0..=1.0).unwrap(), [Minimize, Minimize])
            .population_size(size)
            .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
            .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0).unwrap())
            .seed(seed)
    }

    // a trade-off between the first gene and the others
    fn scores(x: &Reals) -> Scores<2> {
        let g = 1.0 + x[1] + x[2];
        Scores::new([x[0], g * (1.0 - (x[0] / g).sqrt())])
    }

    fn step(nsga2: &mut Real2) {
        let told: Vec<Scores<2>> = nsga2.ask().iter().map(scores).collect();
        nsga2.tell(&told).unwrap();
    }

    fn setting<T: std::fmt::Debug>(result: Result<T>) -> &'static str {
        match result {
            Err(Error::InvalidSetting { setting, .. } | Error::MissingSetting { setting }) => {
                setting
            }
            other => panic!("expected a setting error, got {other:?}"),
        }
    }

    #[test]
    fn validation() {
        let base = || {
            Nsga2::builder(Real::uniform(3, 0.0..=1.0).unwrap(), [Minimize, Maximize])
                .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
                .mutate(PolynomialMutation::per_gene(0.3, 20.0).unwrap())
        };
        assert_eq!(setting(base().build()), "population_size");
        assert_eq!(
            setting(base().population_size(1).build()),
            "population_size"
        );
        assert_eq!(
            setting(base().population_size(4).crossover_rate(1.5).build()),
            "crossover_rate"
        );
        assert_eq!(
            setting(
                base()
                    .population_size(4)
                    .crossover_rate(0.0)
                    .mutation_rate(0.0)
                    .build()
            ),
            "mutation_rate"
        );
        let genome = || Reals::from(vec![0.5; 3]);
        assert_eq!(
            setting(
                base()
                    .population_size(2)
                    .initial_genomes(vec![genome(); 3])
                    .build()
            ),
            "initial_genomes"
        );
        assert!(matches!(
            base()
                .population_size(4)
                .initial_genomes([Reals::from(vec![2.0; 3])])
                .build(),
            Err(Error::InvalidGenome { .. })
        ));
        let none: [Objective; 0] = [];
        let no_objectives = Nsga2::builder(Real::uniform(3, 0.0..=1.0).unwrap(), none)
            .population_size(4)
            .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
            .mutate(PolynomialMutation::per_gene(0.3, 20.0).unwrap())
            .build();
        assert_eq!(setting(no_objectives), "objectives");
        let nsga2 = base().population_size(2).build().unwrap();
        assert_eq!(nsga2.objectives(), [Minimize, Maximize]);
    }

    #[test]
    fn ask_tell_protocol() {
        let mut nsga2 = builder(10, 0).build().unwrap();
        assert_eq!(nsga2.tell(&[]), Err(Error::TellWithoutAsk));
        assert!(nsga2.front().is_empty());
        let first: Vec<Reals> = nsga2.ask().iter().cloned().collect();
        assert_eq!(first.len(), 10);
        assert_eq!(nsga2.ask().iter().cloned().collect::<Vec<_>>(), first);
        assert_eq!(
            nsga2.tell(&[Scores::new([0.0, 0.0])]),
            Err(Error::FitnessCount {
                expected: 10,
                got: 1
            })
        );
        step(&mut nsga2);
        assert_eq!((nsga2.generation(), nsga2.evaluations()), (0, 10));
        assert_eq!(
            (nsga2.ranks().len(), nsga2.crowding_distances().len()),
            (10, 10)
        );
        step(&mut nsga2);
        assert_eq!(nsga2.generation(), 1);
        assert_eq!(nsga2.population().len(), 10);
        // 20 competed, 10 survived: the offspring among the other 10 are discarded
        let surviving_parents = nsga2.population().iter().filter(|x| x.age() > 0).count();
        assert_eq!(nsga2.discarded().len(), surviving_parents);
    }

    #[test]
    fn copies_of_parents_inherit_their_scores() {
        // mutating every gene changes every child
        let mut nsga2 = builder(10, 3)
            .crossover_rate(0.0)
            .mutate(PolynomialMutation::per_gene(1.0, 20.0).unwrap())
            .build()
            .unwrap();
        step(&mut nsga2);
        assert_eq!(nsga2.ask().len(), 10);
        // without mutation, the parents that aren't recombined are copied, when copies are kept
        let mut nsga2 = builder(10, 3)
            .mutation_rate(0.0)
            .crossover_rate(0.5)
            .eliminate_duplicates(false)
            .build()
            .unwrap();
        step(&mut nsga2);
        let asked = nsga2.ask().len();
        assert!(asked < 10, "{asked}");
        let told: Vec<Scores<2>> = nsga2.ask().iter().map(scores).collect();
        nsga2.tell(&told).unwrap();
        assert_eq!(nsga2.evaluations(), 10 + asked as u64);
    }

    #[test]
    fn duplicates_are_eliminated() {
        // without mutation, half the pairs would be copies of their parents
        let mut nsga2 = builder(10, 3)
            .mutation_rate(0.0)
            .crossover_rate(0.5)
            .build()
            .unwrap();
        for _ in 0..5 {
            step(&mut nsga2);
            let asked: Vec<Reals> = nsga2.ask().iter().cloned().collect();
            assert_eq!(asked.len(), 10);
            for (index, child) in asked.iter().enumerate() {
                assert!(nsga2.population().iter().all(|x| x.genome() != child));
                assert!(asked[..index].iter().all(|other| other != child));
            }
            let told: Vec<Scores<2>> = asked.iter().map(scores).collect();
            nsga2.tell(&told).unwrap();
        }
    }

    #[test]
    fn a_population_that_can_only_breed_copies_gets_its_children() {
        // two genes of one bit: at most 4 distinct genomes for a population of 6
        let mut nsga2 = Nsga2::builder(Binary::new(2).unwrap(), [Minimize, Minimize])
            .population_size(6)
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::count(1).unwrap())
            .seed(3)
            .build()
            .unwrap();
        let f = |x: &Bits| Scores::new([x.count_ones() as f64, 2.0 - x.count_ones() as f64]);
        for _ in 0..5 {
            let told: Vec<Scores<2>> = nsga2.ask().iter().map(f).collect();
            nsga2.tell(&told).unwrap();
            assert_eq!(nsga2.population().len(), 6);
        }
    }

    #[test]
    fn the_front_is_the_first_rank() {
        let mut nsga2 = builder(20, 1).build().unwrap();
        for _ in 0..10 {
            step(&mut nsga2);
            let front: Vec<Scores<2>> =
                nsga2.front().iter().map(|x| x.fitness().unwrap()).collect();
            let ranked = nsga2.ranks().iter().filter(|&&rank| rank == 0).count();
            assert_eq!(front.len(), ranked);
            for a in &front {
                assert!(!front.iter().any(|b| dominates(b, a, &[Minimize, Minimize])));
            }
        }
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut nsga2 = builder(12, seed).build().unwrap();
            for _ in 0..15 {
                step(&mut nsga2);
            }
            nsga2.population().clone()
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }

    proptest! {
        #[test]
        fn the_new_front_is_never_dominated_by_the_old_population(
            seed: u64,
            size in 2usize..16,
            maximize: bool,
        ) {
            let objectives = if maximize { [Maximize, Minimize] } else { [Minimize, Minimize] };
            let mut nsga2 = Nsga2::builder(Real::uniform(3, 0.0..=1.0).unwrap(), objectives)
                .population_size(size)
                .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
                .mutate(PolynomialMutation::per_gene(0.5, 20.0).unwrap())
                .seed(seed)
                .build()
                .unwrap();
            let mut rng = StreamRng::seed_from_u64(seed);
            // random scores, some infeasible or invalid
            let mut random = |_: &Reals| match rng.below(10) {
                0 => Scores::invalid(),
                1 => Scores::constrained([0.0, 0.0], rng.below(3) as f64 + 1.0),
                _ => Scores::new([rng.below(5) as f64, rng.below(5) as f64]),
            };
            let told: Vec<Scores<2>> = nsga2.ask().iter().map(&mut random).collect();
            nsga2.tell(&told).unwrap();
            for _ in 0..8 {
                let old: Vec<Scores<2>> =
                    nsga2.population().iter().map(|x| x.fitness().unwrap()).collect();
                let told: Vec<Scores<2>> = nsga2.ask().iter().map(&mut random).collect();
                nsga2.tell(&told).unwrap();
                prop_assert_eq!(nsga2.population().len(), size);
                for member in nsga2.front() {
                    let scores = member.fitness().unwrap();
                    prop_assert!(!old.iter().any(|o| dominates(o, &scores, &objectives)));
                }
            }
        }
    }
}
