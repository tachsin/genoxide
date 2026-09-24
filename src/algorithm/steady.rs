//! A steady-state genetic algorithm that takes results one at a time and in any order, for
//! asynchronous evaluation.

use crate::genome::{Genome, Representation};
use crate::operator::{Crossover, Mutate, Select};
use crate::rng::Chance;
use crate::{Fitness, Individual, Objective, Population, Result, StreamRng};

/// An algorithm that proposes genomes one at a time and takes their fitness in any order, with
/// more genomes proposed before earlier results arrive: for asynchronous evaluation, where every
/// worker gets a new genome as soon as it's done, however long other evaluations take.
/// [`AsyncEngine`](crate::engine::AsyncEngine) runs it.
pub trait Incremental {
    /// The genome type.
    type Genome: Genome;

    /// Whether higher or lower fitness is better.
    fn objective(&self) -> Objective;

    /// A genome to evaluate. More can be proposed before the results of earlier ones arrive.
    fn propose(&mut self) -> Self::Genome;

    /// The fitness of a genome, proposed or not (e.g. a known solution evaluated elsewhere), in
    /// any order. Returns the individual that didn't make it into the population, if any: the one
    /// it replaced, or itself.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidGenome`](crate::Error::InvalidGenome) for a genome that doesn't fit the
    /// representation. The algorithm doesn't change on errors.
    fn receive(
        &mut self,
        genome: Self::Genome,
        fitness: Fitness,
    ) -> Result<Option<Individual<Self::Genome>>>;

    /// The current population: evaluated individuals only.
    fn population(&self) -> &Population<Self::Genome>;

    /// The number of individuals the population grows to. The engine counts a generation for
    /// every this many evaluations.
    fn population_size(&self) -> usize;

    /// The best individual received so far. Ties keep the individual received first.
    fn best(&self) -> Option<&Individual<Self::Genome>>;

    /// The number of fitness values received so far.
    fn evaluations(&self) -> u64;

    /// The number of evaluations when the best individual so far was received.
    fn best_evaluation(&self) -> u64;
}

/// A steady-state genetic algorithm for asynchronous evaluation, from
/// [`GaBuilder::build_steady`](super::GaBuilder::build_steady): the same settings as a [`Ga`](super::Ga), but
/// it breeds one child at a time and every result takes the place of the worst individual when
/// it's not worse.
///
/// - It first proposes the initial population: the initial genomes, then random ones. Once some of
///   them are evaluated, every proposal is a child of two parents chosen by the selection operator
///   from the individuals evaluated so far, recombined and mutated at the usual rates. A child
///   identical to a parent is bred again (up to 100 times): it would add nothing.
/// - A result joins the population until it holds `population_size` individuals. After that it
///   replaces the worst individual (the earliest on ties) if it's at least as good. A genome
///   already in the population is not added twice.
///
/// With one worker, a seed gives the same run every time. With more, the order of the results
/// depends on how long each evaluation takes, so runs differ.
///
/// ```
/// use genoxide::prelude::*;
///
/// let ga = Ga::builder(Binary::new(64)?)
///     .population_size(40)
///     .select(Tournament::new(3)?)
///     .crossover(UniformCrossover::new())
///     .mutate(BitFlip::per_gene(1.0 / 64.0)?)
///     .seed(1)
///     .build_steady()?;
/// let outcome = AsyncEngine::new(ga, |genome: &Bits| genome.count_ones() as f64)
///     .workers(4)
///     .stop_when(Stop::target(64.0).or(Stop::evaluations(50_000)))
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Target);
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
pub struct SteadyGa<R: Representation, S, C, M> {
    representation: R,
    select: S,
    crossover: C,
    mutate: M,
    objective: Objective,
    population_size: usize,
    crossover_chance: Chance,
    mutation_chance: Chance,
    seed: u64,
    rng: StreamRng,
    // the initial genomes not proposed yet, the last one next
    initial: Vec<R::Genome>,
    // how many genomes of the initial population were proposed
    initial_proposed: usize,
    // the second child of the last crossover, proposed next
    queued: Option<R::Genome>,
    population: Population<R::Genome>,
    evaluations: u64,
    best: Option<Individual<R::Genome>>,
    best_evaluation: u64,
}

// how many times a child identical to a parent is bred again
const ATTEMPTS: usize = 100;

impl<R: Representation, S, C, M> SteadyGa<R, S, C, M> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        representation: R,
        select: S,
        crossover: C,
        mutate: M,
        objective: Objective,
        population_size: usize,
        rates: (f64, f64),
        seed: u64,
        mut initial: Vec<R::Genome>,
    ) -> Self {
        initial.reverse();
        Self {
            representation,
            select,
            crossover,
            mutate,
            objective,
            population_size,
            crossover_chance: Chance::new(rates.0),
            mutation_chance: Chance::new(rates.1),
            seed,
            rng: StreamRng::seed_from_u64(seed),
            initial,
            initial_proposed: 0,
            queued: None,
            population: Population::new(Vec::new()),
            evaluations: 0,
            best: None,
            best_evaluation: 0,
        }
    }

    /// The representation.
    pub fn representation(&self) -> &R {
        &self.representation
    }

    /// The seed of the random numbers: the one given to the builder, or the random one drawn.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl<R, S, C, M> SteadyGa<R, S, C, M>
where
    R: Representation,
    S: Select,
    C: Crossover<R>,
    M: Mutate<R>,
{
    // two children of parents chosen from the population, recombined and mutated
    fn breed(&mut self) -> [R::Genome; 2] {
        let parents = self
            .select
            .select(&self.population, self.objective, 2, &mut self.rng);
        let mut children = [
            self.population[parents[0]].genome().clone(),
            self.population[parents[1]].genome().clone(),
        ];
        let [a, b] = &mut children;
        if self.rng.chance(self.crossover_chance) {
            self.crossover
                .crossover(&self.representation, a, b, &mut self.rng);
        }
        for child in &mut children {
            if self.rng.chance(self.mutation_chance) {
                self.mutate
                    .mutate(&self.representation, child, &mut self.rng);
            }
        }
        children
    }

    // whether `genome` is in the population
    fn contains(&self, genome: &R::Genome) -> bool {
        self.population
            .iter()
            .any(|individual| individual.genome() == genome)
    }
}

impl<R, S, C, M> Incremental for SteadyGa<R, S, C, M>
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

    fn propose(&mut self) -> R::Genome {
        if self.initial_proposed < self.population_size {
            self.initial_proposed += 1;
            return match self.initial.pop() {
                Some(genome) => genome,
                None => self.representation.random_genome(&mut self.rng),
            };
        }
        // nothing to breed from yet: more workers than the population
        if self.population.is_empty() {
            return self.representation.random_genome(&mut self.rng);
        }
        if let Some(genome) = self.queued.take() {
            return genome;
        }
        let mut children = self.breed();
        for _ in 1..ATTEMPTS {
            if children.iter().any(|child| !self.contains(child)) {
                break;
            }
            children = self.breed();
        }
        let [a, b] = children;
        match (self.contains(&a), self.contains(&b)) {
            (false, false) => {
                self.queued = Some(b);
                a
            }
            (true, false) => b,
            _ => a,
        }
    }

    fn receive(
        &mut self,
        genome: R::Genome,
        fitness: Fitness,
    ) -> Result<Option<Individual<R::Genome>>> {
        self.representation.validate(&genome)?;
        self.evaluations += 1;
        let mut individual = Individual::new(genome);
        individual.set_fitness(fitness);
        let better = self.best.as_ref().is_none_or(|best| {
            self.objective
                .is_better(fitness, best.fitness().unwrap_or(Fitness::invalid()))
        });
        if better {
            self.best = Some(individual.clone());
            self.best_evaluation = self.evaluations;
        }
        if self.contains(individual.genome()) {
            return Ok(Some(individual));
        }
        if self.population.len() < self.population_size {
            self.population.push(individual);
            return Ok(None);
        }
        // the worst, the earliest on ties
        let objective = self.objective;
        let fitness_at = |index: usize| {
            self.population[index]
                .fitness()
                .unwrap_or(Fitness::invalid())
        };
        let worst = (1..self.population.len()).fold(0, |worst, index| {
            if objective.is_better(fitness_at(worst), fitness_at(index)) {
                index
            } else {
                worst
            }
        });
        if objective.is_better(fitness_at(worst), fitness) {
            return Ok(Some(individual));
        }
        Ok(Some(std::mem::replace(
            &mut self.population[worst],
            individual,
        )))
    }

    fn population(&self) -> &Population<R::Genome> {
        &self.population
    }

    fn population_size(&self) -> usize {
        self.population_size
    }

    fn best(&self) -> Option<&Individual<R::Genome>> {
        self.best.as_ref()
    }

    fn evaluations(&self) -> u64 {
        self.evaluations
    }

    fn best_evaluation(&self) -> u64 {
        self.best_evaluation
    }
}
