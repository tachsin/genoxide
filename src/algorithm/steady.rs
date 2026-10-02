//! A steady-state genetic algorithm that takes results one at a time and in any order, for
//! asynchronous evaluation.

use crate::engine::{Evaluations, GenomeHashing, Provided, Wanted};
use crate::genome::{Genome, Representation};
use crate::operator::{Crossover, Mutate, Select};
use crate::rng::Chance;
use crate::{Fitness, Individual, Objective, Population, Result, StreamRng};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault, Hasher};

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

    /// Called by the [`AsyncEngine`](crate::engine::AsyncEngine) once at the start of each run,
    /// with what the fitness function [provides](crate::engine::FitnessFunction::provides)
    /// besides the fitness, as [`Algorithm::prepare`](super::Algorithm::prepare). Nothing by
    /// default.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`](crate::Error::InvalidSetting) for an extra the algorithm needs
    /// and the fitness function doesn't provide, with the fix in the reason.
    #[inline]
    fn prepare(&mut self, provided: Provided) -> Result<()> {
        let _ = provided;
        Ok(())
    }

    /// What the evaluations of the next [proposed](Incremental::propose) genome want besides the
    /// fitness, such as the values of the constraints. Nothing by default: the engine then
    /// evaluates as it would without extras.
    #[inline]
    fn wants(&self) -> Wanted {
        Wanted::NOTHING
    }

    /// The fitness of a genome and the [wanted](Incremental::wants) extras of its evaluation, a
    /// single one: what the [`AsyncEngine`](crate::engine::AsyncEngine) gives an algorithm that
    /// wants extras, instead of [`receive`](Incremental::receive). By default,
    /// `receive(genome, evaluation.fitness()[0])`.
    ///
    /// # Errors
    ///
    /// As [`receive`](Incremental::receive), and
    /// [`Error::FitnessCount`](crate::Error::FitnessCount) for an evaluation without exactly one
    /// fitness.
    #[inline]
    fn receive_evaluation(
        &mut self,
        genome: Self::Genome,
        evaluation: &Evaluations<'_>,
    ) -> Result<Option<Individual<Self::Genome>>> {
        match evaluation.fitness() {
            [fitness] => self.receive(genome, *fitness),
            other => Err(crate::Error::FitnessCount {
                expected: 1,
                got: other.len(),
            }),
        }
    }
}

/// A steady-state genetic algorithm for asynchronous evaluation, from
/// [`GaBuilder::build_steady`](super::GaBuilder::build_steady): the same settings as a
/// [`Ga`](super::Ga), but it proposes one child at a time and every result takes the place of the
/// worst individual when it's not worse.
///
/// - It first proposes the initial population: the initial genomes, then random ones. After that,
///   until a result arrives, it proposes random genomes (with more workers than the population
///   size). Then every proposal is a child of two parents chosen by the selection operator from
///   the individuals evaluated so far, recombined and mutated at the usual rates. A crossover
///   gives two children: the first is proposed, and the second next, unless it's identical to the
///   first, or in the population by then. When both children are already in the population (e.g.
///   copies of their parents), the parents are chosen and bred again, up to 99 times (100
///   breedings in all): such a child would add nothing. After that, a child in the population is
///   proposed anyway.
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
    // finds genomes and the worst individual in the population; rebuilt after a checkpoint
    #[cfg_attr(feature = "serde", serde(skip))]
    lookup: Lookup,
    evaluations: u64,
    best: Option<Individual<R::Genome>>,
    best_evaluation: u64,
}

// a genome's hash, which only narrows down the genomes it's compared with
fn hash<G: Genome>(genome: &G) -> u64 {
    GenomeHashing::default().hash_one(genome)
}

// What finds a genome and the worst individual in a steady-state GA's population without going
// through all of it: the hash of each genome, in the population's order, how many genomes have
// each hash, and the population's positions in a binary heap with the worst individual at the
// top, the earliest on ties. Not serialized, and rebuilt when it doesn't match the population.
#[derive(Clone, Default)]
struct Lookup {
    hashes: Vec<u64>,
    counts: HashMap<u64, u32, BuildHasherDefault<Prehashed>>,
    heap: Vec<usize>,
}

impl std::fmt::Debug for Lookup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Lookup")
    }
}

impl Lookup {
    fn rebuild<G: Genome>(&mut self, population: &Population<G>, objective: Objective) {
        let len = population.len();
        self.hashes.clear();
        self.hashes.extend(
            population
                .iter()
                .map(|individual| hash(individual.genome())),
        );
        self.counts.clear();
        for &hash in &self.hashes {
            *self.counts.entry(hash).or_default() += 1;
        }
        self.heap.clear();
        self.heap.extend(0..len);
        for slot in (0..len / 2).rev() {
            self.sift_down(slot, population, objective);
        }
    }

    // whether `genome`, whose hash is `hash`, is in `population`
    fn contains<G: Genome>(&self, population: &Population<G>, genome: &G, hash: u64) -> bool {
        self.counts.contains_key(&hash)
            && self
                .hashes
                .iter()
                .zip(population.iter())
                .any(|(&other, individual)| other == hash && individual.genome() == genome)
    }

    // after an individual whose genome has `hash` was added at the end of `population`
    fn push<G: Genome>(&mut self, population: &Population<G>, objective: Objective, hash: u64) {
        self.hashes.push(hash);
        *self.counts.entry(hash).or_default() += 1;
        let slot = self.heap.len();
        self.heap.push(population.len() - 1);
        self.sift_up(slot, population, objective);
    }

    // the position of the worst individual, the earliest on ties
    fn worst(&self) -> usize {
        self.heap[0]
    }

    // after the worst individual was replaced by one whose genome has `hash`
    fn replace_worst<G: Genome>(
        &mut self,
        population: &Population<G>,
        objective: Objective,
        hash: u64,
    ) {
        let worst = self.heap[0];
        let old = std::mem::replace(&mut self.hashes[worst], hash);
        if let Some(count) = self.counts.get_mut(&old) {
            *count -= 1;
            if *count == 0 {
                self.counts.remove(&old);
            }
        }
        *self.counts.entry(hash).or_default() += 1;
        self.sift_down(0, population, objective);
    }

    // whether the individual at position `a` goes above the one at `b` in the heap: worse, or as
    // good and earlier
    fn above<G: Genome>(
        population: &Population<G>,
        objective: Objective,
        a: usize,
        b: usize,
    ) -> bool {
        let fitness = |index: usize| population[index].fitness().unwrap_or(Fitness::invalid());
        match objective.compare(fitness(a), fitness(b)) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => a < b,
        }
    }

    fn sift_up<G: Genome>(
        &mut self,
        mut slot: usize,
        population: &Population<G>,
        objective: Objective,
    ) {
        while slot > 0 {
            let parent = (slot - 1) / 2;
            if !Self::above(population, objective, self.heap[slot], self.heap[parent]) {
                break;
            }
            self.heap.swap(slot, parent);
            slot = parent;
        }
    }

    fn sift_down<G: Genome>(
        &mut self,
        mut slot: usize,
        population: &Population<G>,
        objective: Objective,
    ) {
        let len = self.heap.len();
        loop {
            let mut top = slot;
            for child in [2 * slot + 1, 2 * slot + 2] {
                if child < len
                    && Self::above(population, objective, self.heap[child], self.heap[top])
                {
                    top = child;
                }
            }
            if top == slot {
                break;
            }
            self.heap.swap(slot, top);
            slot = top;
        }
    }
}

// a map keyed by hashes, which are hashed already
#[derive(Default)]
struct Prehashed(u64);

impl Hasher for Prehashed {
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("only hashes are hashed")
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

// how many times a child identical to a parent is bred again
const ATTEMPTS: usize = 100;

impl<R: Representation, S, C, M> SteadyGa<R, S, C, M> {
    #[expect(clippy::too_many_arguments)]
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
            lookup: Lookup::default(),
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
        self.lookup.contains(&self.population, genome, hash(genome))
    }

    // the lookup of the population, after a checkpoint
    fn rehash(&mut self) {
        if self.lookup.hashes.len() != self.population.len() {
            self.lookup.rebuild(&self.population, self.objective);
        }
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
        // the second child of the last crossover, unless it joined the population meanwhile
        self.rehash();
        if let Some(genome) = self.queued.take()
            && !self.contains(&genome)
        {
            return genome;
        }
        let mut children = self.breed();
        // whether each child is in the population, found once
        let mut present = children.each_ref().map(|child| self.contains(child));
        for _ in 1..ATTEMPTS {
            if present != [true, true] {
                break;
            }
            children = self.breed();
            present = children.each_ref().map(|child| self.contains(child));
        }
        let [a, b] = children;
        match (present[0], present[1]) {
            (false, false) => {
                // twins would be the same evaluation twice
                if b != a {
                    self.queued = Some(b);
                }
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
        self.rehash();
        let hash = hash(individual.genome());
        if self
            .lookup
            .contains(&self.population, individual.genome(), hash)
        {
            return Ok(Some(individual));
        }
        if self.population.len() < self.population_size {
            self.population.push(individual);
            self.lookup.push(&self.population, self.objective, hash);
            return Ok(None);
        }
        // the worst, the earliest on ties
        let worst = self.lookup.worst();
        let worst_fitness = self.population[worst]
            .fitness()
            .unwrap_or(Fitness::invalid());
        if self.objective.is_better(worst_fitness, fitness) {
            return Ok(Some(individual));
        }
        let replaced = std::mem::replace(&mut self.population[worst], individual);
        self.lookup
            .replace_worst(&self.population, self.objective, hash);
        Ok(Some(replaced))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::Ga;
    use crate::genome::{Binary, Bits};
    use crate::operator::{BitFlip, Tournament, UniformCrossover};

    fn one_max(genome: &Bits) -> Fitness {
        Fitness::new(genome.count_ones() as f64)
    }

    // With one worker, each result arrives before the next proposal: a proposal already in the
    // population, e.g. the twin of the last one, would be an evaluation wasted.
    #[test]
    fn one_worker_never_gets_a_genome_already_in_the_population() {
        let mut steady = Ga::builder(Binary::new(16).unwrap())
            .population_size(10)
            .select(Tournament::new(2).unwrap())
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::count(1).unwrap())
            .seed(3)
            .build_steady()
            .unwrap();
        for step in 0..5_000 {
            let genome = steady.propose();
            // after the initial population
            if step >= 10 {
                assert!(
                    !steady.contains(&genome),
                    "proposed a genome already in the population at step {step}"
                );
            }
            let fitness = one_max(&genome);
            steady.receive(genome, fitness).unwrap();
        }
    }

    // A population of one genome of one bit: both children of a crossover are the other genome,
    // twins. Every result replaces the individual (the fitness is the same), so the second twin
    // would be in the population when it's proposed.
    #[test]
    fn the_twin_of_a_child_is_not_proposed() {
        let mut steady = Ga::builder(Binary::new(1).unwrap())
            .population_size(1)
            .select(Tournament::new(2).unwrap())
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::count(1).unwrap())
            .seed(0)
            .build_steady()
            .unwrap();
        let first = steady.propose();
        steady.receive(first.clone(), Fitness::new(0.0)).unwrap();
        let mut proposals = vec![first];
        for _ in 0..10 {
            let genome = steady.propose();
            assert!(!steady.contains(&genome), "{proposals:?}, then {genome:?}");
            proposals.push(genome.clone());
            steady.receive(genome, Fitness::new(0.0)).unwrap();
        }
    }

    // The lookup finds the worst as a scan of the population does, the earliest on ties, with
    // ties, invalid and infeasible fitness, and after a checkpoint's rebuild (a clone of the
    // population without the lookup).
    #[test]
    fn the_worst_is_the_first_of_the_worst() {
        for objective in [Objective::Maximize, Objective::Minimize] {
            let mut steady = Ga::builder(Binary::new(12).unwrap())
                .population_size(20)
                .select(Tournament::new(2).unwrap())
                .crossover(UniformCrossover::new())
                .mutate(BitFlip::count(1).unwrap())
                .objective(objective)
                .seed(5)
                .build_steady()
                .unwrap();
            let mut rng = StreamRng::seed_from_u64(9);
            for step in 0..3_000 {
                if step % 700 == 0 {
                    steady.lookup = Lookup::default();
                }
                let genome = steady.propose();
                let fitness = match rng.below(8) {
                    0 => Fitness::invalid(),
                    1 => Fitness::constrained(1.0, rng.below(3) as f64),
                    value => Fitness::new(value as f64),
                };
                let population = steady.population().clone();
                let full = population.len() == 20;
                let contained = steady.contains(&genome);
                let returned = steady.receive(genome.clone(), fitness).unwrap();
                if full && !contained {
                    let fitness_at = |index: usize| population[index].fitness().unwrap();
                    let worst = (1..population.len()).fold(0, |worst, index| {
                        if objective.is_better(fitness_at(worst), fitness_at(index)) {
                            index
                        } else {
                            worst
                        }
                    });
                    if objective.is_better(fitness_at(worst), fitness) {
                        assert_eq!(steady.population(), &population);
                    } else {
                        assert_eq!(returned.as_ref(), Some(&population[worst]));
                        assert_eq!(steady.population()[worst].genome(), &genome);
                    }
                }
            }
        }
    }
}
