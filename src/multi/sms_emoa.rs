//! SMS-EMOA: the S-metric (hypervolume) selection evolutionary multi-objective algorithm.

use super::breed::{Variation, scores_of};
use super::indicator::hypervolume_contributions;
use super::pareto::gains;
use super::{MultiObjectiveAlgorithm, Scores, dominates, non_dominated_sort};
use crate::algorithm::{Candidates, Unset};
use crate::genome::Representation;
use crate::operator::{Crossover, Mutate, check_probability};
use crate::rng::Chance;
use crate::{Error, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;

/// SMS-EMOA (Beume, Naujoks and Emmerich, 2007): selection by hypervolume contribution, as an ask
/// / tell [`MultiObjectiveAlgorithm`].
///
/// Every generation:
///
/// 1. Parents are chosen by binary tournament: the smaller constraint violation, then Pareto
///    dominance, then a coin flip. Pairs are recombined and mutated as in
///    [`Nsga2`](super::Nsga2): `offspring` children (the population size by default, as in
///    pymoo; 1 for the original steady-state algorithm).
/// 2. Parents and children compete: the next population takes whole fronts, best first. From the
///    last front that fits partly, the member with the smallest
///    [hypervolume contribution](super::indicator::hypervolume_contributions) is removed, one at
///    a time, until the rest fits. The contributions are computed with the objectives normalized
///    by the parents' best and worst values, and the reference point at 11 in each normalized
///    objective (pymoo's settings), so the extremes of the front are kept.
///
/// Maximizing the hypervolume gives fronts that are well spread and converged, at a higher cost
/// per generation than NSGA-II: O(N log N) per removal for 2 objectives, O(N²) for 3. Infeasible
/// solutions are ranked by constrained dominance, and removed at random from a last front of
/// equal violation.
///
/// ```
/// use genoxide::Objective::Minimize;
/// use genoxide::multi::SmsEmoa;
/// use genoxide::multi::problems::{TestProblem, Zdt1};
/// use genoxide::prelude::*;
///
/// let problem = Zdt1::new(30);
/// let sms_emoa = SmsEmoa::builder(problem.real(), [Minimize; 2])
///     .population_size(100)
///     .crossover(SimulatedBinaryCrossover::new(15.0)?)
///     .mutate(PolynomialMutation::per_gene(1.0 / 30.0, 20.0)?)
///     .seed(1)
///     .build()?;
/// let outcome = MultiEngine::new(sms_emoa, problem).stop_when(Stop::generations(150)).run()?;
/// assert_eq!(outcome.front().len(), 100);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct SmsEmoa<R: Representation, C, X, const M: usize> {
    variation: Variation<R, C, X>,
    objectives: [Objective; M],
    population_size: usize,
    offspring_count: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    seed: u64,
    rng: StreamRng,
    population: Population<R::Genome, Scores<M>>,
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

impl<R: Representation, const M: usize> SmsEmoa<R, Unset, Unset, M> {
    /// A builder for SMS-EMOA on `representation`, with the direction of each objective.
    pub fn builder(representation: R, objectives: [Objective; M]) -> SmsEmoaBuilder<R, M> {
        SmsEmoaBuilder {
            representation,
            objectives,
            crossover: Unset,
            mutate: Unset,
            population_size: None,
            offspring: None,
            crossover_rate: 0.9,
            mutation_rate: 1.0,
            seed: None,
            initial_genomes: Vec::new(),
        }
    }
}

// the violation of a solution, infinite if it's invalid
fn violation<const M: usize>(scores: &Scores<M>) -> f64 {
    if scores.is_valid() {
        scores.violation()
    } else {
        f64::INFINITY
    }
}

// the values with every objective turned into one to minimize
fn minimized<const M: usize>(scores: &Scores<M>, objectives: &[Objective; M]) -> [f64; M] {
    let values = scores.raw();
    std::array::from_fn(|j| match objectives[j] {
        Objective::Minimize => values[j],
        Objective::Maximize => -values[j],
    })
}

impl<R, C, X, const M: usize> SmsEmoa<R, C, X, M>
where
    R: Representation,
    C: Crossover<R>,
    X: Mutate<R>,
{
    /// The representation.
    pub fn representation(&self) -> &R {
        &self.variation.representation
    }

    /// The population size.
    pub fn population_size(&self) -> usize {
        self.population_size
    }

    /// The number of children per generation.
    pub fn offspring(&self) -> usize {
        self.offspring_count
    }

    /// The probability that a pair of parents is recombined.
    pub fn crossover_rate(&self) -> f64 {
        self.crossover_rate
    }

    /// The probability that a child is mutated.
    pub fn mutation_rate(&self) -> f64 {
        self.mutation_rate
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn breed(&mut self) {
        let population = &self.population;
        let objectives = &self.objectives;
        self.variation.breed(
            population,
            self.offspring_count,
            &mut self.rng,
            |rng| {
                let size = population.len();
                let a = rng.below(size);
                let mut b = rng.below(size - 1);
                if b >= a {
                    b += 1;
                }
                let score = |i: usize| population[i].fitness().unwrap_or(Scores::invalid());
                let (sa, sb) = (score(a), score(b));
                let (va, vb) = (violation(&sa), violation(&sb));
                if va > 0.0 || vb > 0.0 {
                    if va < vb {
                        return a;
                    }
                    if vb < va {
                        return b;
                    }
                } else if dominates(&sa, &sb, objectives) {
                    return a;
                } else if dominates(&sb, &sa, objectives) {
                    return b;
                }
                if rng.below(2) == 0 { a } else { b }
            },
            &mut self.offspring,
        );
    }

    // the best and worst values of the feasible parents, minimized, or `None` if no parent is
    // feasible
    fn normalization(&self) -> Option<([f64; M], [f64; M])> {
        let mut ideal = [f64::INFINITY; M];
        let mut nadir = [f64::NEG_INFINITY; M];
        let mut any = false;
        for individual in self.population.iter() {
            let scores = individual.fitness().unwrap_or(Scores::invalid());
            if scores.is_feasible() {
                any = true;
                let values = minimized(&scores, &self.objectives);
                for j in 0..M {
                    ideal[j] = ideal[j].min(values[j]);
                    nadir[j] = nadir[j].max(values[j]);
                }
            }
        }
        any.then_some((ideal, nadir))
    }

    // keeps `room` of the members of the last front, removing the smallest hypervolume
    // contributors one at a time (at random if the front is infeasible)
    fn thin(
        &mut self,
        front: &[usize],
        scores: &[Scores<M>],
        room: usize,
        normalization: Option<([f64; M], [f64; M])>,
    ) -> Vec<usize> {
        let mut kept = front.to_vec();
        let feasible = kept.iter().all(|&i| scores[i].is_feasible());
        match (feasible, normalization) {
            (true, Some((ideal, nadir))) => {
                let scale: [f64; M] = std::array::from_fn(|j| {
                    let range = nadir[j] - ideal[j];
                    if range > 0.0 && range.is_finite() {
                        range
                    } else {
                        1.0
                    }
                });
                let reference = [11.0; M];
                while kept.len() > room {
                    let points: Vec<[f64; M]> = kept
                        .iter()
                        .map(|&i| {
                            let values = minimized(&scores[i], &self.objectives);
                            std::array::from_fn(|j| (values[j] - ideal[j]) / scale[j])
                        })
                        .collect();
                    let contributions =
                        hypervolume_contributions(&points, &reference, &[Objective::Minimize; M]);
                    let mut smallest = 0;
                    for (position, contribution) in contributions.iter().enumerate() {
                        if *contribution < contributions[smallest] {
                            smallest = position;
                        }
                    }
                    kept.remove(smallest);
                }
            }
            _ => {
                // equally infeasible, or nothing to normalize by: a random subset
                for i in (1..kept.len()).rev() {
                    kept.swap(i, self.rng.below(i + 1));
                }
                kept.truncate(room);
                kept.sort_unstable();
            }
        }
        kept
    }

    // the next population from the parents and the offspring
    fn survive(&mut self) {
        let normalization = self.normalization();
        let parents = std::mem::take(&mut self.population).into_vec();
        let parent_count = parents.len();
        let mut pool = parents;
        pool.append(&mut self.offspring);
        let scores = scores_of(&pool);
        let fronts = non_dominated_sort(&scores, &self.objectives);
        let mut chosen: Vec<usize> = Vec::with_capacity(self.population_size);
        for front in &fronts {
            let room = self.population_size - chosen.len();
            if front.len() <= room {
                chosen.extend(front);
            } else {
                let kept = self.thin(front, &scores, room, normalization);
                chosen.extend(kept);
            }
            if chosen.len() == self.population_size {
                break;
            }
        }
        let mut selected = vec![false; pool.len()];
        for &index in &chosen {
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
        let mut population = Vec::with_capacity(chosen.len());
        for index in chosen {
            let mut individual = slots[index].take().expect("chosen once");
            if index < parent_count {
                individual.increment_age();
            }
            population.push(individual);
        }
        self.population = Population::new(population);
    }

    // the new front, and whether it improved on the previous one
    fn update_front(&mut self) {
        let scores = scores_of(self.population.as_slice());
        let fronts = non_dominated_sort(&scores, &self.objectives);
        let front: Vec<Individual<R::Genome, Scores<M>>> = fronts
            .first()
            .map(|first| first.iter().map(|&i| self.population[i].clone()).collect())
            .unwrap_or_default();
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

impl<R, C, X, const M: usize> MultiObjectiveAlgorithm<M> for SmsEmoa<R, C, X, M>
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

/// A builder for [`SmsEmoa`], from [`SmsEmoa::builder`].
///
/// The population size, the crossover and the mutation are required. Defaults: as many children
/// as the population size, `crossover_rate` 0.9, `mutation_rate` 1.0, a random initial
/// population and a random seed.
#[derive(Clone, Debug)]
pub struct SmsEmoaBuilder<R: Representation, const M: usize, C = Unset, X = Unset> {
    representation: R,
    objectives: [Objective; M],
    crossover: C,
    mutate: X,
    population_size: Option<usize>,
    offspring: Option<usize>,
    crossover_rate: f64,
    mutation_rate: f64,
    seed: Option<u64>,
    initial_genomes: Vec<R::Genome>,
}

impl<R: Representation, const M: usize, C, X> SmsEmoaBuilder<R, M, C, X> {
    /// The crossover operator. Required.
    pub fn crossover<T>(self, crossover: T) -> SmsEmoaBuilder<R, M, T, X> {
        SmsEmoaBuilder {
            representation: self.representation,
            objectives: self.objectives,
            crossover,
            mutate: self.mutate,
            population_size: self.population_size,
            offspring: self.offspring,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            seed: self.seed,
            initial_genomes: self.initial_genomes,
        }
    }

    /// The mutation operator. Required.
    pub fn mutate<T>(self, mutate: T) -> SmsEmoaBuilder<R, M, C, T> {
        SmsEmoaBuilder {
            representation: self.representation,
            objectives: self.objectives,
            crossover: self.crossover,
            mutate,
            population_size: self.population_size,
            offspring: self.offspring,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            seed: self.seed,
            initial_genomes: self.initial_genomes,
        }
    }

    /// The population size, at least 2. Required.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// The number of children per generation, at least 1. The population size by default; 1 is
    /// the original steady-state SMS-EMOA, which needs more generations but fewer evaluations
    /// per improvement.
    pub fn offspring(mut self, count: usize) -> Self {
        self.offspring = Some(count);
        self
    }

    /// The probability that a pair of parents is recombined, between 0 and 1. 0.9 by default.
    pub fn crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// The probability that a child is mutated, between 0 and 1. 1 by default.
    pub fn mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// The seed of the random numbers, for a reproducible run. Random by default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
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
    /// - [`Error::InvalidSetting`] for a population size below 2, no offspring, no objectives,
    ///   rates out of range or both 0, or more initial genomes than the population size.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<SmsEmoa<R, C, X, M>>
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
                format!("SMS-EMOA needs at least 2 individuals, got {size}"),
            );
        }
        let offspring = self.offspring.unwrap_or(size);
        if offspring == 0 {
            return invalid("offspring", "must be at least 1".to_string());
        }
        if M == 0 {
            return invalid("objectives", "at least 1 objective is needed".to_string());
        }
        let crossover_rate = check_probability("crossover_rate", self.crossover_rate)?;
        let mutation_rate = check_probability("mutation_rate", self.mutation_rate)?;
        if crossover_rate == 0.0 && mutation_rate == 0.0 {
            return invalid(
                "mutation_rate",
                "crossover_rate and mutation_rate are both 0, so every child would be a copy of a parent".to_string(),
            );
        }
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
        Ok(SmsEmoa {
            variation: Variation {
                representation: self.representation,
                crossover: self.crossover,
                mutate: self.mutate,
                crossover_chance: Chance::new(crossover_rate),
                mutation_chance: Chance::new(mutation_rate),
            },
            objectives: self.objectives,
            population_size: size,
            offspring_count: offspring,
            crossover_rate,
            mutation_rate,
            seed,
            rng,
            population: genomes.into_iter().map(Individual::unevaluated).collect(),
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
    use crate::genome::{Real, Reals};
    use crate::operator::{PolynomialMutation, SimulatedBinaryCrossover};
    use proptest::prelude::*;

    fn builder(
        size: usize,
        seed: u64,
    ) -> SmsEmoaBuilder<Real, 2, SimulatedBinaryCrossover, PolynomialMutation> {
        SmsEmoa::builder(Real::uniform(3, 0.0..=1.0).unwrap(), [Minimize, Minimize])
            .population_size(size)
            .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
            .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0).unwrap())
            .seed(seed)
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
    fn validation_and_steady_state() {
        assert_eq!(setting(builder(1, 0).build()), "population_size");
        assert_eq!(setting(builder(4, 0).offspring(0).build()), "offspring");
        assert_eq!(builder(4, 0).build().unwrap().offspring(), 4);
        let mut sms_emoa = builder(6, 0).offspring(1).build().unwrap();
        assert_eq!(sms_emoa.tell(&[]), Err(Error::TellWithoutAsk));
        assert_eq!(sms_emoa.ask().len(), 6);
        let f = |x: &Reals| Scores::new([x[0], 1.0 - x[0] + x[1]]);
        for generation in 0..10 {
            let told: Vec<Scores<2>> = sms_emoa.ask().iter().map(f).collect();
            assert!(generation == 0 || told.len() <= 1);
            sms_emoa.tell(&told).unwrap();
            assert_eq!(sms_emoa.population().len(), 6);
        }
    }

    #[test]
    fn thinning_removes_the_smallest_contributors() {
        let mut sms_emoa = builder(4, 0).build().unwrap();
        // parents spanning 0..4 in both objectives, for the normalization
        let parents = [[0.0, 4.0], [1.0, 3.0], [3.0, 1.0], [4.0, 0.0]];
        for (individual, values) in sms_emoa.population.iter_mut().zip(parents) {
            individual.set_fitness(Scores::new(values));
        }
        let normalization = sms_emoa.normalization();
        assert_eq!(normalization, Some(([0.0, 0.0], [4.0, 4.0])));
        // (1.1, 2.9) is crowded next to (1, 3); the extremes have large contributions
        let scores: Vec<Scores<2>> = [[0.0, 4.0], [1.0, 3.0], [1.1, 2.9], [2.0, 2.0], [4.0, 0.0]]
            .into_iter()
            .map(Scores::new)
            .collect();
        let kept = sms_emoa.thin(&[0, 1, 2, 3, 4], &scores, 4, normalization);
        assert_eq!(kept, [0, 1, 3, 4]);
        let kept = sms_emoa.thin(&[0, 1, 2, 3, 4], &scores, 2, normalization);
        assert_eq!(kept, [0, 4]);
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut sms_emoa = builder(10, seed).build().unwrap();
            let f = |x: &Reals| Scores::new([x[0], x[1] + x[2]]);
            for _ in 0..10 {
                let told: Vec<Scores<2>> = sms_emoa.ask().iter().map(f).collect();
                sms_emoa.tell(&told).unwrap();
            }
            sms_emoa.population().clone()
        };
        assert_eq!(run(8), run(8));
        assert_ne!(run(8), run(9));
    }

    proptest! {
        #[test]
        fn the_new_front_is_never_dominated_by_the_old_population(
            seed: u64,
            size in 2usize..12,
            offspring in 1usize..12,
            maximize: bool,
        ) {
            let objectives = if maximize { [Maximize, Minimize] } else { [Minimize, Minimize] };
            let mut sms_emoa = SmsEmoa::builder(Real::uniform(3, 0.0..=1.0).unwrap(), objectives)
                .population_size(size)
                .offspring(offspring)
                .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
                .mutate(PolynomialMutation::per_gene(0.5, 20.0).unwrap())
                .seed(seed)
                .build()
                .unwrap();
            let mut rng = StreamRng::seed_from_u64(seed);
            let mut random = |_: &Reals| match rng.below(10) {
                0 => Scores::invalid(),
                1 => Scores::constrained([0.0, 0.0], rng.below(3) as f64 + 1.0),
                _ => Scores::new([rng.below(5) as f64, rng.below(5) as f64]),
            };
            let told: Vec<Scores<2>> = sms_emoa.ask().iter().map(&mut random).collect();
            sms_emoa.tell(&told).unwrap();
            for _ in 0..8 {
                let old: Vec<Scores<2>> =
                    sms_emoa.population().iter().map(|x| x.fitness().unwrap()).collect();
                let told: Vec<Scores<2>> = sms_emoa.ask().iter().map(&mut random).collect();
                sms_emoa.tell(&told).unwrap();
                prop_assert_eq!(sms_emoa.population().len(), size);
                for member in sms_emoa.front() {
                    let scores = member.fitness().unwrap();
                    prop_assert!(!old.iter().any(|o| dominates(o, &scores, &objectives)));
                }
            }
        }
    }
}
