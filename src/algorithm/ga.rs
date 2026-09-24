//! The genetic algorithm.

use super::{Algorithm, Candidates};
use crate::genome::{Genome, Representation};
use crate::operator::{Crossover, Mutate, Select, check_probability, neighbor};
use crate::rng::Chance;
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;
use std::mem;

/// How the next population is formed from the parents (μ, the population size) and their
/// offspring.
///
/// In every scheme, ties are broken by position, and the best individual found so far is kept by
/// the algorithm whether or not it survives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Scheme {
    /// μ − `elitism` offspring per generation. The next population is the `elitism` best parents
    /// followed by the offspring. `elitism` is less than μ.
    Generational {
        /// The number of best parents that survive, 0 for none.
        elitism: usize,
    },
    /// `replacements` offspring per generation, which replace the `replacements` worst parents,
    /// even if they are worse. `replacements` is between 1 and μ.
    SteadyState {
        /// The number of offspring per generation.
        replacements: usize,
    },
    /// (μ+λ): `lambda` offspring per generation, and the best μ of parents and offspring survive.
    /// On ties, offspring are preferred, which lets the population drift across plateaus.
    MuPlusLambda {
        /// The number of offspring per generation, at least 1.
        lambda: usize,
    },
    /// (μ,λ): `lambda` offspring per generation, and the best μ offspring survive. `lambda` is at
    /// least μ.
    MuCommaLambda {
        /// The number of offspring per generation, at least μ.
        lambda: usize,
    },
}

impl Default for Scheme {
    /// Generational with an elitism of 1.
    fn default() -> Self {
        Scheme::Generational { elitism: 1 }
    }
}

impl Scheme {
    // the number of offspring per generation for a population of `size`
    fn offspring_count(self, size: usize) -> usize {
        match self {
            Scheme::Generational { elitism } => size - elitism,
            Scheme::SteadyState { replacements } => replacements,
            Scheme::MuPlusLambda { lambda } | Scheme::MuCommaLambda { lambda } => lambda,
        }
    }

    fn validate(self, size: usize) -> Result<()> {
        let invalid = |reason: String| {
            Err(Error::InvalidSetting {
                setting: "scheme",
                reason,
            })
        };
        match self {
            Scheme::Generational { elitism } if elitism >= size => invalid(format!(
                "elitism must be less than the population size {size}, got {elitism}"
            )),
            Scheme::SteadyState { replacements } if replacements == 0 || replacements > size => {
                invalid(format!(
                    "replacements must be between 1 and the population size {size}, got {replacements}"
                ))
            }
            Scheme::MuPlusLambda { lambda: 0 } => invalid("lambda must be at least 1".to_string()),
            Scheme::MuCommaLambda { lambda } if lambda < size => invalid(format!(
                "lambda must be at least the population size {size}, got {lambda}"
            )),
            _ => Ok(()),
        }
    }
}

/// A genetic algorithm, as an ask / tell [`Algorithm`].
///
/// Every generation:
///
/// 1. Parents are selected in pairs with the [`Select`] operator.
/// 2. Each pair is recombined with the [`Crossover`] operator with probability `crossover_rate`,
///    otherwise the children are copies of the parents.
/// 3. Each child is mutated with the [`Mutate`] operator with probability `mutation_rate`.
/// 4. A child that is identical to one of its parents inherits the parent's fitness instead of
///    being evaluated again (fitness functions are expected to be deterministic).
/// 5. With [`memetic`](GaBuilder::memetic), the best parents each try some neighbors, made by the
///    mutation operator and evaluated with the offspring. A parent is replaced by its best
///    neighbor when that neighbor is not worse (Lamarckian local search).
/// 6. The next population is formed according to the [`Scheme`].
///
/// Built with [`Ga::builder`]. Run it with an [`Engine`](crate::Engine), or drive it by hand
/// through [`Algorithm`].
///
/// ```
/// use genoxide::prelude::*;
///
/// let ga = Ga::builder(Binary::new(32)?)
///     .population_size(50)
///     .select(Tournament::new(3)?)
///     .crossover(UniformCrossover::new())
///     .mutate(BitFlip::per_gene(1.0 / 32.0)?)
///     .seed(42)
///     .build()?;
/// let outcome = Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
///     .stop_when(Stop::target(32.0).or(Stop::generations(500)))
///     .run()?;
/// assert_eq!(outcome.best_fitness().score(), Some(32.0));
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "R: serde::Serialize, S: serde::Serialize, C: serde::Serialize, M: serde::Serialize, R::Genome: serde::Serialize",
        deserialize = "R: serde::Deserialize<'de>, S: serde::Deserialize<'de>, C: serde::Deserialize<'de>, M: serde::Deserialize<'de>, R::Genome: serde::Deserialize<'de>"
    ))
)]
pub struct Ga<R: Representation, S, C, M> {
    representation: R,
    select: S,
    crossover: C,
    mutate: M,
    objective: Objective,
    population_size: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    crossover_chance: Chance,
    mutation_chance: Chance,
    scheme: Scheme,
    seed: u64,
    rng: StreamRng,
    population: Population<R::Genome>,
    offspring: Vec<Individual<R::Genome>>,
    // offspring evaluated in the last generation that didn't survive
    discarded: Vec<Individual<R::Genome>>,
    // memetic local search: (parents, neighbors per parent)
    memetic: Option<(usize, usize)>,
    // the parent of each memetic neighbor, which are the last offspring
    refined: Vec<usize>,
    phase: Phase,
    asked: bool,
    // positions in the population (initial phase) or the offspring of the genomes asked for
    pending: Vec<usize>,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<R::Genome>>,
    best_generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum Phase {
    // the initial population is not evaluated yet
    Initial,
    Offspring,
}

impl<R: Representation> Ga<R, Unset, Unset, Unset> {
    /// A builder for a genetic algorithm on `representation`.
    pub fn builder(representation: R) -> GaBuilder<R> {
        GaBuilder {
            representation,
            select: Unset,
            crossover: Unset,
            mutate: Unset,
            population_size: None,
            objective: Objective::default(),
            crossover_rate: 0.9,
            mutation_rate: 1.0,
            scheme: Scheme::default(),
            seed: None,
            initial_genomes: Vec::new(),
            memetic: None,
        }
    }
}

impl<R: Representation, S, C, M> Ga<R, S, C, M> {
    /// The representation.
    pub fn representation(&self) -> &R {
        &self.representation
    }

    /// The selection operator.
    pub fn select(&self) -> &S {
        &self.select
    }

    /// The crossover operator.
    pub fn crossover(&self) -> &C {
        &self.crossover
    }

    /// The mutation operator.
    pub fn mutate(&self) -> &M {
        &self.mutate
    }

    /// The population size, μ.
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

    /// The scheme.
    pub fn scheme(&self) -> Scheme {
        self.scheme
    }

    /// The seed of the random numbers: the given one, or a random one if none was given. The same
    /// seed and settings give the same run.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The memetic local search: the number of best parents refined per generation, and the
    /// number of neighbors each tries. `None` without.
    pub fn memetic(&self) -> Option<(usize, usize)> {
        self.memetic
    }
}

impl<R, S, C, M> Ga<R, S, C, M>
where
    R: Representation,
    S: Select,
    C: Crossover<R>,
    M: Mutate<R>,
{
    // creates the offspring of the current population
    fn breed(&mut self) {
        let count = self.scheme.offspring_count(self.population_size);
        let parents = self.select.select(
            &self.population,
            self.objective,
            count + count % 2,
            &mut self.rng,
        );
        self.offspring.clear();
        for pair in parents.chunks_exact(2) {
            let parents = [&self.population[pair[0]], &self.population[pair[1]]];
            let mut a = parents[0].genome().clone();
            let mut b = parents[1].genome().clone();
            if self.rng.chance(self.crossover_chance) {
                self.crossover
                    .crossover(&self.representation, &mut a, &mut b, &mut self.rng);
            }
            for mut genome in [a, b] {
                if self.offspring.len() == count {
                    break;
                }
                if self.rng.chance(self.mutation_chance) {
                    self.mutate
                        .mutate(&self.representation, &mut genome, &mut self.rng);
                }
                let inherited = parents
                    .iter()
                    .find(|parent| parent.genome() == &genome)
                    .and_then(|parent| parent.fitness());
                let mut child = Individual::new(genome);
                if let Some(fitness) = inherited {
                    child.set_fitness(fitness);
                }
                self.offspring.push(child);
            }
        }

        // memetic: neighbors of the best parents, evaluated with the offspring
        self.refined.clear();
        if let Some((parents, neighbors)) = self.memetic {
            let objective = self.objective;
            let fitness = |index: usize| {
                self.population[index]
                    .fitness()
                    .unwrap_or(Fitness::invalid())
            };
            // best first, the earlier one on ties
            let mut order: Vec<usize> = (0..self.population.len()).collect();
            order.sort_by(|&a, &b| objective.compare(fitness(b), fitness(a)));
            for &parent in order.iter().take(parents) {
                for _ in 0..neighbors {
                    let genome = neighbor(
                        &self.mutate,
                        &self.representation,
                        self.population[parent].genome(),
                        &mut self.rng,
                    );
                    self.offspring.push(Individual::new(genome));
                    self.refined.push(parent);
                }
            }
        }
    }

    // memetic: replaces each refined parent by its best neighbor when that's not worse; returns the
    // neighbors that weren't taken, and copies of the ones that were
    fn refine(&mut self) -> Refinement<R::Genome> {
        let Some((_, neighbors)) = self.memetic else {
            return (Vec::new(), Vec::new());
        };
        let start = self.offspring.len() - self.refined.len();
        let candidates: Vec<_> = self.offspring.drain(start..).collect();
        let mut taken = Vec::new();
        let rejected = lamarckian(
            &mut self.population,
            candidates,
            &self.refined,
            neighbors,
            self.objective,
            &mut taken,
        );
        (rejected, taken)
    }

    // forms the next population from the evaluated offspring
    fn survive(&mut self) {
        let size = self.population_size;
        self.discarded.clear();
        match self.scheme {
            Scheme::Generational { elitism } => {
                self.keep_best_parents(elitism);
            }
            Scheme::SteadyState { replacements } => {
                self.keep_best_parents(size - replacements);
            }
            Scheme::MuPlusLambda { .. } => {
                let mut all = mem::take(&mut self.offspring);
                all.extend(mem::take(&mut self.population).into_iter().map(aged));
                let (population, rest) = best_of(all, size, self.objective);
                self.population = population;
                // the parents were seen before, only the offspring are new: parents have been aged
                self.discarded
                    .extend(rest.into_iter().filter(|individual| individual.age() == 0));
            }
            Scheme::MuCommaLambda { .. } => {
                let offspring = mem::take(&mut self.offspring);
                (self.population, self.discarded) = best_of(offspring, size, self.objective);
            }
        }
    }

    // the next population is the `count` best parents followed by the offspring
    fn keep_best_parents(&mut self, count: usize) {
        self.population.sort_best_first(self.objective);
        self.population.truncate(count);
        self.population
            .iter_mut()
            .for_each(Individual::increment_age);
        self.population.extend(self.offspring.drain(..));
    }
}

// the neighbors of a memetic refinement that weren't taken, and copies of the ones that were
type Refinement<G> = (Vec<Individual<G>>, Vec<Individual<G>>);

// `candidates` are the neighbors of `parents` (the position of each one's parent), `neighbors` per
// parent in a row: each parent is replaced by its best neighbor (the first on ties) when that's not
// worse. Returns the neighbors that weren't taken, and adds copies of the taken ones to `taken`.
fn lamarckian<G: Genome>(
    population: &mut Population<G>,
    candidates: Vec<Individual<G>>,
    parents: &[usize],
    neighbors: usize,
    objective: Objective,
    taken: &mut Vec<Individual<G>>,
) -> Vec<Individual<G>> {
    let fitness = |individual: &Individual<G>| individual.fitness().unwrap_or(Fitness::invalid());
    let mut rejected = Vec::new();
    let mut candidates = candidates.into_iter();
    for &parent in parents.iter().step_by(neighbors) {
        let group: Vec<Individual<G>> = candidates.by_ref().take(neighbors).collect();
        let best = (1..group.len()).fold(0, |best, index| {
            if objective.is_better(fitness(&group[index]), fitness(&group[best])) {
                index
            } else {
                best
            }
        });
        let take = !objective.is_better(fitness(&population[parent]), fitness(&group[best]));
        for (index, candidate) in group.into_iter().enumerate() {
            if take && index == best {
                taken.push(candidate.clone());
                population[parent] = candidate;
            } else {
                rejected.push(candidate);
            }
        }
    }
    rejected
}

fn aged<G: Genome>(mut individual: Individual<G>) -> Individual<G> {
    individual.increment_age();
    individual
}

// the `count` best individuals (the earlier ones first on ties), and the rest
fn best_of<G: Genome>(
    individuals: Vec<Individual<G>>,
    count: usize,
    objective: Objective,
) -> (Population<G>, Vec<Individual<G>>) {
    let mut population = Population::new(individuals);
    population.sort_best_first(objective);
    let mut best = population.into_vec();
    let rest = best.split_off(count.min(best.len()));
    (Population::new(best), rest)
}

// replaces `best` by the first individual that is strictly better; true if it did
fn update_best<G: Genome>(
    best: &mut Option<Individual<G>>,
    candidates: &[Individual<G>],
    objective: Objective,
) -> bool {
    let mut improved = false;
    for candidate in candidates {
        let Some(fitness) = candidate.fitness() else {
            continue;
        };
        let is_better = best.as_ref().is_none_or(|best| {
            best.fitness()
                .is_none_or(|best| objective.is_better(fitness, best))
        });
        if is_better {
            *best = Some(candidate.clone());
            improved = true;
        }
    }
    improved
}

impl<R, S, C, M> super::Migrate for Ga<R, S, C, M>
where
    R: Representation + PartialEq,
    S: Select,
    C: Crossover<R>,
    M: Mutate<R>,
{
    fn immigrate(&mut self, migrants: Vec<Individual<R::Genome>>) -> Result<()> {
        if self.asked || self.phase == Phase::Initial {
            return Err(Error::MigrationOutOfTurn);
        }
        for migrant in &migrants {
            self.representation.validate(migrant.genome())?;
        }
        let count = migrants.len().min(self.population.len());
        let size = self.population.len();
        self.population.sort_best_first(self.objective);
        self.population.truncate(size - count);
        self.population.extend(migrants.into_iter().take(count));
        if update_best(
            &mut self.best,
            &self.population.as_slice()[size - count..],
            self.objective,
        ) {
            self.best_generation = self.generation;
        }
        Ok(())
    }

    fn same_representation(&self, other: &Self) -> bool {
        self.representation == other.representation
    }
}

impl<R, S, C, M> Algorithm for Ga<R, S, C, M>
where
    R: Representation,
    S: Select,
    C: Crossover<R>,
    M: Mutate<R>,
{
    type Genome = R::Genome;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, R::Genome> {
        if !self.asked {
            if self.phase == Phase::Offspring {
                self.breed();
            }
            let individuals = match self.phase {
                Phase::Initial => self.population.as_slice(),
                Phase::Offspring => &self.offspring,
            };
            self.pending.clear();
            self.pending.extend(
                individuals
                    .iter()
                    .enumerate()
                    .filter(|(_, individual)| !individual.is_evaluated())
                    .map(|(index, _)| index),
            );
            self.asked = true;
        }
        let individuals = match self.phase {
            Phase::Initial => self.population.as_slice(),
            Phase::Offspring => &self.offspring,
        };
        Candidates::new(individuals, &self.pending)
    }

    fn tell(&mut self, fitness: &[Fitness]) -> Result<()> {
        if !self.asked {
            return Err(Error::TellWithoutAsk);
        }
        if fitness.len() != self.pending.len() {
            return Err(Error::FitnessCount {
                expected: self.pending.len(),
                got: fitness.len(),
            });
        }
        for (&index, &fitness) in self.pending.iter().zip(fitness) {
            let individual = match self.phase {
                Phase::Initial => &mut self.population[index],
                Phase::Offspring => &mut self.offspring[index],
            };
            individual.set_fitness(fitness);
        }
        self.evaluations += fitness.len() as u64;
        self.asked = false;
        match self.phase {
            Phase::Initial => {
                update_best(&mut self.best, self.population.as_slice(), self.objective);
                self.phase = Phase::Offspring;
            }
            Phase::Offspring => {
                self.generation += 1;
                if update_best(&mut self.best, &self.offspring, self.objective) {
                    self.best_generation = self.generation;
                }
                let (rejected, taken) = self.refine();
                self.survive();
                self.discarded.extend(rejected);
                // with (μ+λ), a refined parent can lose to the offspring: it was evaluated, so
                // observers see it as discarded
                for individual in taken {
                    let survived = self
                        .population
                        .iter()
                        .any(|survivor| survivor.genome() == individual.genome());
                    if !survived {
                        self.discarded.push(individual);
                    }
                }
            }
        }
        Ok(())
    }

    fn population(&self) -> &Population<R::Genome> {
        &self.population
    }

    fn best(&self) -> Option<&Individual<R::Genome>> {
        self.best.as_ref()
    }

    fn discarded(&self) -> &[Individual<R::Genome>] {
        &self.discarded
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn evaluations(&self) -> u64 {
        self.evaluations
    }

    fn best_generation(&self) -> u64 {
        self.best_generation
    }
}

/// A component of a [`GaBuilder`] that is not set yet.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unset;

/// A builder for a [`Ga`], from [`Ga::builder`].
///
/// The selection, crossover and mutation operators and the population size are required. A
/// missing operator, or one that doesn't fit the representation, is a compile error. Invalid
/// values are errors from [`build`](GaBuilder::build).
///
/// Defaults: maximize, `crossover_rate` 0.9, `mutation_rate` 1.0, the generational scheme with an
/// elitism of 1, and a random seed.
#[derive(Clone, Debug)]
pub struct GaBuilder<R: Representation, S = Unset, C = Unset, M = Unset> {
    representation: R,
    select: S,
    crossover: C,
    mutate: M,
    population_size: Option<usize>,
    objective: Objective,
    crossover_rate: f64,
    mutation_rate: f64,
    scheme: Scheme,
    seed: Option<u64>,
    initial_genomes: Vec<R::Genome>,
    memetic: Option<(usize, usize)>,
}

impl<R: Representation, S, C, M> GaBuilder<R, S, C, M> {
    /// The selection operator for parents.
    pub fn select<T>(self, select: T) -> GaBuilder<R, T, C, M> {
        GaBuilder {
            representation: self.representation,
            select,
            crossover: self.crossover,
            mutate: self.mutate,
            population_size: self.population_size,
            objective: self.objective,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            scheme: self.scheme,
            seed: self.seed,
            initial_genomes: self.initial_genomes,
            memetic: self.memetic,
        }
    }

    /// The crossover operator.
    pub fn crossover<T>(self, crossover: T) -> GaBuilder<R, S, T, M> {
        GaBuilder {
            representation: self.representation,
            select: self.select,
            crossover,
            mutate: self.mutate,
            population_size: self.population_size,
            objective: self.objective,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            scheme: self.scheme,
            seed: self.seed,
            initial_genomes: self.initial_genomes,
            memetic: self.memetic,
        }
    }

    /// The mutation operator.
    pub fn mutate<T>(self, mutate: T) -> GaBuilder<R, S, C, T> {
        GaBuilder {
            representation: self.representation,
            select: self.select,
            crossover: self.crossover,
            mutate,
            population_size: self.population_size,
            objective: self.objective,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            scheme: self.scheme,
            seed: self.seed,
            initial_genomes: self.initial_genomes,
            memetic: self.memetic,
        }
    }

    /// The population size, μ, at least 1. Required.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// Whether higher or lower fitness is better. Maximize by default.
    pub fn objective(mut self, objective: Objective) -> Self {
        self.objective = objective;
        self
    }

    /// Higher fitness is better (the default).
    pub fn maximize(self) -> Self {
        self.objective(Objective::Maximize)
    }

    /// Lower fitness is better.
    pub fn minimize(self) -> Self {
        self.objective(Objective::Minimize)
    }

    /// The probability that a pair of parents is recombined, between 0 and 1. 0.9 by default.
    pub fn crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// The probability that a child is mutated, between 0 and 1. 1 by default: the mutation
    /// operator decides how much changes.
    pub fn mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// How the next population is formed. Generational with an elitism of 1 by default.
    pub fn scheme(mut self, scheme: Scheme) -> Self {
        self.scheme = scheme;
        self
    }

    /// The seed of the random numbers, for a reproducible run. Random by default, see
    /// [`Ga::seed`].
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Memetic (Lamarckian) local search: every generation, the `parents` best parents each try
    /// `neighbors` (at least 1) neighbors, made by the mutation operator and evaluated with the
    /// offspring. A parent is replaced by its best neighbor when that neighbor is not worse, before
    /// survivor selection. It adds `parents * neighbors` evaluations per generation. Off by
    /// default.
    ///
    /// The refined parents must be ones that survive the generation, so `parents` is at most the
    /// elitism of [`Scheme::Generational`], the population size minus the replacements of
    /// [`Scheme::SteadyState`], or the population size with [`Scheme::MuPlusLambda`]. It doesn't
    /// work with [`Scheme::MuCommaLambda`], where no parent survives.
    pub fn memetic(mut self, parents: usize, neighbors: usize) -> Self {
        self.memetic = Some((parents, neighbors));
        self
    }

    /// Genomes for the initial population, e.g. known good solutions. The rest of the initial
    /// population is random. At most the population size, and each must be valid for the
    /// representation.
    pub fn initial_genomes<I: IntoIterator<Item = R::Genome>>(mut self, genomes: I) -> Self {
        self.initial_genomes = genomes.into_iter().collect();
        self
    }

    /// Validates the settings and creates the algorithm, with its random initial population.
    ///
    /// # Errors
    ///
    /// - [`Error::MissingSetting`] without a population size.
    /// - [`Error::InvalidSetting`] for a population size of 0, rates outside [0, 1], both rates 0
    ///   (every child would be a copy), a [`Scheme`] that doesn't fit the population size, more
    ///   initial genomes than the population size, or memetic settings out of range.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<Ga<R, S, C, M>>
    where
        S: Select,
        C: Crossover<R>,
        M: Mutate<R>,
    {
        let population_size = self.population_size.ok_or(Error::MissingSetting {
            setting: "population_size",
        })?;
        if population_size == 0 {
            return Err(Error::InvalidSetting {
                setting: "population_size",
                reason: "must be at least 1".to_string(),
            });
        }
        let crossover_rate = check_probability("crossover_rate", self.crossover_rate)?;
        let mutation_rate = check_probability("mutation_rate", self.mutation_rate)?;
        if crossover_rate == 0.0 && mutation_rate == 0.0 {
            return Err(Error::InvalidSetting {
                setting: "mutation_rate",
                reason: "crossover_rate and mutation_rate are both 0, so every child would be a copy of a parent".to_string(),
            });
        }
        self.scheme.validate(population_size)?;
        if let Some((parents, neighbors)) = self.memetic {
            // the refined parents must survive the generation, or their refinement is lost
            let survivors = match self.scheme {
                Scheme::Generational { elitism } => elitism,
                Scheme::SteadyState { replacements } => population_size - replacements,
                Scheme::MuPlusLambda { .. } => population_size,
                Scheme::MuCommaLambda { .. } => 0,
            };
            if parents == 0 || parents > survivors || neighbors == 0 {
                return Err(Error::InvalidSetting {
                    setting: "memetic",
                    reason: format!(
                        "parents must be between 1 and the number of parents that survive a generation, {survivors} with {:?} (the elitism of a generational scheme, the population size minus the replacements of a steady-state one, the population size with (μ+λ), none with (μ,λ)), and neighbors at least 1; got {parents} and {neighbors}",
                        self.scheme
                    ),
                });
            }
        }
        if self.initial_genomes.len() > population_size {
            return Err(Error::InvalidSetting {
                setting: "initial_genomes",
                reason: format!(
                    "at most the population size {population_size}, got {}",
                    self.initial_genomes.len()
                ),
            });
        }
        for genome in &self.initial_genomes {
            self.representation.validate(genome)?;
        }

        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let mut rng = StreamRng::seed_from_u64(seed);
        let random = population_size - self.initial_genomes.len();
        let mut genomes = self.initial_genomes;
        genomes.extend((0..random).map(|_| self.representation.random_genome(&mut rng)));

        Ok(Ga {
            representation: self.representation,
            select: self.select,
            crossover: self.crossover,
            mutate: self.mutate,
            objective: self.objective,
            population_size,
            crossover_rate,
            mutation_rate,
            crossover_chance: Chance::new(crossover_rate),
            mutation_chance: Chance::new(mutation_rate),
            scheme: self.scheme,
            seed,
            rng,
            population: Population::from_genomes(genomes),
            offspring: Vec::new(),
            discarded: Vec::new(),
            memetic: self.memetic,
            refined: Vec::new(),
            phase: Phase::Initial,
            asked: false,
            pending: Vec::new(),
            generation: 0,
            evaluations: 0,
            best: None,
            best_generation: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{Binary, Bits};
    use crate::operator::{BitFlip, NoCrossover, Tournament, UniformCrossover};
    use proptest::prelude::*;

    type OneMaxGa = Ga<Binary, Tournament, UniformCrossover, BitFlip>;

    fn builder(len: usize) -> GaBuilder<Binary, Tournament, UniformCrossover, BitFlip> {
        Ga::builder(Binary::new(len).unwrap())
            .population_size(10)
            .select(Tournament::new(2).unwrap())
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::count(1).unwrap())
            .seed(0)
    }

    fn one_max(ga: &mut OneMaxGa) -> Vec<Fitness> {
        ga.ask()
            .iter()
            .map(|genome| Fitness::new(genome.count_ones() as f64))
            .collect()
    }

    fn step(ga: &mut OneMaxGa) {
        let fitness = one_max(ga);
        ga.tell(&fitness).unwrap();
    }

    fn setting(result: Result<OneMaxGa>) -> &'static str {
        match result {
            Err(Error::InvalidSetting { setting, .. } | Error::MissingSetting { setting }) => {
                setting
            }
            other => panic!("expected a setting error, got {other:?}"),
        }
    }

    #[test]
    fn validation() {
        let missing = Ga::builder(Binary::new(4).unwrap())
            .select(Tournament::new(2).unwrap())
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::count(1).unwrap());
        assert_eq!(setting(missing.build()), "population_size");
        assert_eq!(
            setting(builder(4).population_size(0).build()),
            "population_size"
        );
        assert_eq!(
            setting(builder(4).crossover_rate(1.5).build()),
            "crossover_rate"
        );
        assert_eq!(
            setting(builder(4).mutation_rate(-0.1).build()),
            "mutation_rate"
        );
        assert_eq!(
            setting(builder(4).crossover_rate(0.0).mutation_rate(0.0).build()),
            "mutation_rate"
        );
        for scheme in [
            Scheme::Generational { elitism: 10 },
            Scheme::SteadyState { replacements: 0 },
            Scheme::SteadyState { replacements: 11 },
            Scheme::MuPlusLambda { lambda: 0 },
            Scheme::MuCommaLambda { lambda: 9 },
        ] {
            assert_eq!(setting(builder(4).scheme(scheme).build()), "scheme");
        }
        assert_eq!(
            setting(builder(4).initial_genomes(vec![Bits::zeros(4); 11]).build()),
            "initial_genomes"
        );
        assert!(matches!(
            builder(4).initial_genomes([Bits::zeros(3)]).build(),
            Err(Error::InvalidGenome { .. })
        ));
        for scheme in [
            Scheme::Generational { elitism: 0 },
            Scheme::Generational { elitism: 9 },
            Scheme::SteadyState { replacements: 10 },
            Scheme::MuPlusLambda { lambda: 1 },
            Scheme::MuCommaLambda { lambda: 10 },
        ] {
            assert!(builder(4).scheme(scheme).build().is_ok(), "{scheme:?}");
        }
    }

    #[test]
    fn ask_tell_protocol() {
        let mut ga = builder(8).build().unwrap();
        assert_eq!(ga.tell(&[]), Err(Error::TellWithoutAsk));
        let first: Vec<Bits> = ga.ask().iter().cloned().collect();
        assert_eq!(first.len(), 10);
        // asking again gives the same genomes
        assert!(ga.ask().iter().eq(first.iter()));
        assert_eq!(
            ga.tell(&[Fitness::new(0.0)]),
            Err(Error::FitnessCount {
                expected: 10,
                got: 1
            })
        );
        // an error changes nothing
        assert_eq!(ga.evaluations(), 0);
        assert!(ga.best().is_none());
        step(&mut ga);
        assert_eq!((ga.generation(), ga.evaluations()), (0, 10));
        assert!(ga.best().is_some());
        assert_eq!(ga.tell(&[]), Err(Error::TellWithoutAsk));
        step(&mut ga);
        assert_eq!(ga.generation(), 1);
    }

    #[test]
    fn initial_genomes_are_in_the_initial_population() {
        let seeds = [Bits::ones(8), Bits::zeros(8)];
        let mut ga = builder(8).initial_genomes(seeds.clone()).build().unwrap();
        let asked: Vec<Bits> = ga.ask().iter().cloned().collect();
        assert_eq!(&asked[..2], &seeds);
        assert_eq!(asked.len(), 10);
    }

    #[test]
    fn copies_of_a_parent_inherit_its_fitness() {
        // identical parents: every child of a crossover without mutation is a copy
        let mut ga = builder(8)
            .initial_genomes(vec![Bits::ones(8); 10])
            .crossover_rate(1.0)
            .mutation_rate(0.0)
            .build()
            .unwrap();
        step(&mut ga);
        assert!(ga.ask().is_empty());
        ga.tell(&[]).unwrap();
        assert_eq!((ga.generation(), ga.evaluations()), (1, 10));
        assert!(ga.population().iter().all(Individual::is_evaluated));
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut ga = builder(32).seed(seed).build().unwrap();
            for _ in 0..20 {
                step(&mut ga);
            }
            ga.population().clone()
        };
        assert_eq!(run(1), run(1));
        assert_ne!(run(1), run(2));
    }

    #[test]
    fn random_seed_is_reported() {
        let ga = Ga::builder(Binary::new(8).unwrap())
            .population_size(4)
            .select(Tournament::new(2).unwrap())
            .crossover(NoCrossover)
            .mutate(BitFlip::count(1).unwrap())
            .build()
            .unwrap();
        let again = builder(8)
            .population_size(4)
            .seed(ga.seed())
            .build()
            .unwrap();
        assert_eq!(ga.population(), again.population());
    }

    #[test]
    fn elites_age() {
        let mut ga = builder(16)
            .scheme(Scheme::Generational { elitism: 1 })
            .build()
            .unwrap();
        step(&mut ga);
        for generation in 1..=5 {
            step(&mut ga);
            let ages: Vec<u32> = ga.population().iter().map(Individual::age).collect();
            assert!(ages[0] >= 1 && ages[0] <= generation);
            assert!(ages[1..].iter().all(|&age| age == 0));
        }
    }

    #[test]
    fn discarded_offspring() {
        let objective = Objective::Maximize;
        for scheme in [
            Scheme::MuCommaLambda { lambda: 25 },
            Scheme::MuPlusLambda { lambda: 25 },
            Scheme::Generational { elitism: 2 },
        ] {
            let mut ga = builder(16).scheme(scheme).build().unwrap();
            step(&mut ga);
            assert!(ga.discarded().is_empty());
            for _ in 0..5 {
                step(&mut ga);
                let worst_survivor = ga
                    .population()
                    .iter()
                    .filter_map(Individual::fitness)
                    .min_by(|a, b| objective.compare(*a, *b))
                    .unwrap();
                for individual in ga.discarded() {
                    assert_eq!(individual.age(), 0);
                    assert!(!objective.is_better(individual.fitness().unwrap(), worst_survivor));
                }
                match scheme {
                    Scheme::MuCommaLambda { .. } => assert_eq!(ga.discarded().len(), 15),
                    Scheme::Generational { .. } => assert!(ga.discarded().is_empty()),
                    _ => assert!(ga.discarded().len() <= 25),
                }
            }
        }
    }

    #[test]
    fn memetic_validation() {
        assert_eq!(setting(builder(8).memetic(0, 2).build()), "memetic");
        assert_eq!(setting(builder(8).memetic(1, 0).build()), "memetic");
        // at most the parents that survive: the elitism (1 by default), ...
        assert_eq!(setting(builder(8).memetic(2, 1).build()), "memetic");
        assert!(builder(8).memetic(1, 1).build().is_ok());
        let scheme = |scheme, parents| builder(8).scheme(scheme).memetic(parents, 1).build();
        assert!(scheme(Scheme::Generational { elitism: 3 }, 3).is_ok());
        assert_eq!(
            setting(scheme(Scheme::Generational { elitism: 0 }, 1)),
            "memetic"
        );
        // ... the population size minus the replacements, all with (μ+λ), none with (μ,λ)
        assert!(scheme(Scheme::SteadyState { replacements: 4 }, 6).is_ok());
        assert_eq!(
            setting(scheme(Scheme::SteadyState { replacements: 4 }, 7)),
            "memetic"
        );
        assert!(scheme(Scheme::MuPlusLambda { lambda: 5 }, 10).is_ok());
        assert_eq!(
            setting(scheme(Scheme::MuPlusLambda { lambda: 5 }, 11)),
            "memetic"
        );
        assert_eq!(
            setting(scheme(Scheme::MuCommaLambda { lambda: 20 }, 1)),
            "memetic"
        );
    }

    #[test]
    fn memetic_neighbors_are_evaluated_with_the_offspring() {
        let mut ga = builder(16)
            .scheme(Scheme::Generational { elitism: 3 })
            .memetic(3, 2)
            .build()
            .unwrap();
        step(&mut ga);
        // 7 offspring (elitism 3) and 3 parents with 2 neighbors each
        assert_eq!(ga.ask().len(), 7 + 6);
        step(&mut ga);
        assert_eq!(ga.evaluations(), 10 + 13);
        assert_eq!(ga.population().len(), 10);
    }

    #[test]
    fn memetic_improvements_survive_or_are_discarded() {
        // every evaluated individual is in the population or discarded, for every scheme
        // (μ+λ) with every parent refined and many offspring: refined parents can lose
        for (scheme, parents) in [
            (Scheme::Generational { elitism: 3 }, 3),
            (Scheme::SteadyState { replacements: 5 }, 3),
            (Scheme::MuPlusLambda { lambda: 20 }, 10),
        ] {
            let mut ga = builder(64)
                .scheme(scheme)
                .memetic(parents, 1)
                .build()
                .unwrap();
            step(&mut ga);
            for _ in 0..20 {
                let asked: Vec<Bits> = ga.ask().iter().cloned().collect();
                step_with(&mut ga, &asked);
                for genome in &asked {
                    let seen = ga
                        .population()
                        .iter()
                        .chain(ga.discarded())
                        .any(|individual| individual.genome() == genome);
                    assert!(seen, "{scheme:?}");
                }
            }
        }
    }

    // tells the one max fitness of `asked`, which is what `ga.ask()` gives
    fn step_with(ga: &mut OneMaxGa, asked: &[Bits]) {
        let fitness: Vec<Fitness> = asked
            .iter()
            .map(|genome| Fitness::new(genome.count_ones() as f64))
            .collect();
        ga.tell(&fitness).unwrap();
    }

    fn evaluated(genome: &str, score: f64) -> Individual<Bits> {
        let mut individual = Individual::new(genome.chars().map(|c| c == '1').collect());
        individual.set_fitness(Fitness::new(score));
        individual
    }

    #[test]
    fn lamarckian_takes_the_best_neighbor_when_not_worse() {
        let mut population: Population<Bits> = [evaluated("00", 5.0), evaluated("01", 3.0)]
            .into_iter()
            .collect();
        // parent 0 (5): neighbors 4 and 5, and 5 is not worse, so it replaces the parent; parent 1
        // (3): neighbors 2 and 1, both worse
        let candidates = vec![
            evaluated("10", 4.0),
            evaluated("11", 5.0),
            evaluated("10", 2.0),
            evaluated("00", 1.0),
        ];
        let mut taken = Vec::new();
        let rejected = lamarckian(
            &mut population,
            candidates,
            &[0, 0, 1, 1],
            2,
            Objective::Maximize,
            &mut taken,
        );
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].genome().to_string(), "11");
        assert_eq!(population[0].genome().to_string(), "11");
        assert_eq!(population[0].age(), 0);
        assert_eq!(population[1].genome().to_string(), "01");
        let rejected: Vec<f64> = rejected
            .iter()
            .map(|individual| individual.fitness().unwrap().score().unwrap())
            .collect();
        assert_eq!(rejected, [4.0, 2.0, 1.0]);
    }

    fn any_scheme() -> impl Strategy<Value = Scheme> {
        prop_oneof![
            (0usize..10).prop_map(|elitism| Scheme::Generational { elitism }),
            (1usize..=10).prop_map(|replacements| Scheme::SteadyState { replacements }),
            (1usize..20).prop_map(|lambda| Scheme::MuPlusLambda { lambda }),
            (10usize..20).prop_map(|lambda| Scheme::MuCommaLambda { lambda }),
        ]
    }

    fn is_elitist(scheme: Scheme) -> bool {
        match scheme {
            Scheme::Generational { elitism } => elitism > 0,
            Scheme::SteadyState { replacements } => replacements < 10,
            Scheme::MuPlusLambda { .. } => true,
            Scheme::MuCommaLambda { .. } => false,
        }
    }

    proptest! {
        #[test]
        fn schemes(scheme in any_scheme(), seed: u64) {
            let mut ga = builder(16).scheme(scheme).seed(seed).build().unwrap();
            step(&mut ga);
            let objective = Objective::Maximize;
            let mut best_so_far = ga.best().unwrap().fitness().unwrap();
            for _ in 0..10 {
                let previous_best = ga.population().best(objective).unwrap().fitness().unwrap();
                let asked = ga.ask().len();
                prop_assert!(asked <= scheme.offspring_count(10));
                step(&mut ga);
                prop_assert_eq!(ga.population().len(), 10);
                prop_assert!(ga.population().iter().all(Individual::is_evaluated));
                let best = ga.population().best(objective).unwrap().fitness().unwrap();
                // elitist schemes never lose their best
                if is_elitist(scheme) {
                    prop_assert!(!objective.is_better(previous_best, best));
                }
                // the best so far never gets worse, and is at least the population's best
                let now = ga.best().unwrap().fitness().unwrap();
                prop_assert!(!objective.is_better(best_so_far, now));
                prop_assert!(!objective.is_better(best, now));
                best_so_far = now;
            }
        }
    }
}
