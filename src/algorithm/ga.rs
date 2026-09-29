//! The genetic algorithm.

use super::steady::SteadyGa;
use super::{Algorithm, Candidates};
use crate::genome::{Genome, Representation};
use crate::operator::{Crossover, MAX_SIZE, Mutate, Select, check_rates, check_size, neighbor};
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
        /// The number of offspring per generation, between 1 and 2^24, and few enough that the
        /// parents and offspring fit in memory: at most `isize::MAX` bytes of individuals.
        lambda: usize,
    },
    /// (μ,λ): `lambda` offspring per generation, and the best μ offspring survive. `lambda` is at
    /// least μ.
    MuCommaLambda {
        /// The number of offspring per generation, between μ and 2^24, and few enough that the
        /// offspring fit in memory: at most `isize::MAX` bytes of individuals.
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

    // `individuals` is the most individuals a vector can hold: with more offspring, or with more
    // parents and offspring for (μ+λ), which compete together, the allocation fails
    fn validate(self, size: usize, individuals: usize) -> Result<()> {
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
            Scheme::MuPlusLambda { lambda }
                if lambda > MAX_SIZE.min(individuals.saturating_sub(size)) =>
            {
                invalid(format!(
                    "lambda must be at most {}, got {lambda}",
                    MAX_SIZE.min(individuals.saturating_sub(size))
                ))
            }
            Scheme::MuCommaLambda { lambda } if lambda > MAX_SIZE.min(individuals) => {
                invalid(format!(
                    "lambda must be at most {}, got {lambda}",
                    MAX_SIZE.min(individuals)
                ))
            }
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
    // each pair of parents bred on a stream of its own, in parallel with the `parallel` feature
    parallel_breeding: bool,
    population: Population<R::Genome>,
    offspring: Vec<Individual<R::Genome>>,
    // offspring evaluated in the last generation that didn't survive
    discarded: Vec<Individual<R::Genome>>,
    // genomes no longer in use, whose memory the next offspring reuse
    #[cfg_attr(feature = "serde", serde(skip))]
    spare: Spare<R::Genome>,
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
            parallel_breeding: false,
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

    /// Whether each pair of parents is bred on a random stream of its own, in parallel: see
    /// [`GaBuilder::parallel_breeding`].
    pub fn parallel_breeding(&self) -> bool {
        self.parallel_breeding
    }

    /// Marks the population as not evaluated, for a fitness function that changed during the
    /// run: adaptive penalty weights, a retrained surrogate model, a moving optimum. The next
    /// [`ask`](Algorithm::ask) gives the whole population instead of offspring, and its
    /// [`tell`](Algorithm::tell) sets their fitness without breeding.
    ///
    /// - It isn't a generation: [`generation`](Algorithm::generation) doesn't change. The
    ///   evaluations are counted.
    /// - Old and new fitness values measure different things, so after that tell,
    ///   [`best`](Algorithm::best) is the best of the re-evaluated population, found in the
    ///   current generation.
    /// - It draws no random numbers: a seeded run that re-evaluates at the same points gives the
    ///   same results.
    /// - Before the first tell nothing is evaluated yet, and it changes nothing.
    ///
    /// ```
    /// use genoxide::prelude::*;
    ///
    /// let mut ga = Ga::builder(Binary::new(16)?)
    ///     .population_size(20)
    ///     .select(Tournament::new(2)?)
    ///     .crossover(UniformCrossover::new())
    ///     .mutate(BitFlip::count(1)?)
    ///     .seed(3)
    ///     .build()?;
    /// // the ones, then the zeros
    /// let score = |bits: &Bits, ones: bool| {
    ///     let count = bits.count_ones() as f64;
    ///     Fitness::new(if ones { count } else { 16.0 - count })
    /// };
    /// for generation in 0..60 {
    ///     let ones = generation < 30;
    ///     if generation == 30 {
    ///         ga.reevaluate()?;
    ///         assert_eq!(ga.ask().len(), 20);
    ///     }
    ///     let fitness: Vec<Fitness> = ga.ask().iter().map(|bits| score(bits, ones)).collect();
    ///     ga.tell(&fitness)?;
    /// }
    /// // the best under the new measure
    /// assert!(ga.best().unwrap().genome().count_ones() < 8);
    /// # Ok::<(), genoxide::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::ReevaluationOutOfTurn`] between an ask and its tell. Nothing changes on errors.
    pub fn reevaluate(&mut self) -> Result<()> {
        if self.asked {
            return Err(Error::ReevaluationOutOfTurn);
        }
        self.population
            .iter_mut()
            .for_each(Individual::clear_fitness);
        // they were discarded by the last generation, which observers have seen
        let limit = self.spare_limit();
        self.spare
            .recycle(self.discarded.drain(..).map(Individual::into_genome), limit);
        self.phase = Phase::Initial;
        Ok(())
    }

    /// Mutable access to the selection operator, to change it during a run (parameter control).
    /// A change applies from the next generation's breeding, the next
    /// [`ask`](Algorithm::ask) after a [`tell`](Algorithm::tell).
    pub fn select_mut(&mut self) -> &mut S {
        &mut self.select
    }

    /// Mutable access to the crossover operator, to change it during a run, e.g. to replace it
    /// with one built by its own validating constructor. As [`select_mut`](Ga::select_mut).
    pub fn crossover_mut(&mut self) -> &mut C {
        &mut self.crossover
    }

    /// Mutable access to the mutation operator, to change it during a run, e.g. a Gaussian step
    /// annealed over the run. As [`select_mut`](Ga::select_mut).
    ///
    /// ```
    /// use genoxide::prelude::*;
    ///
    /// let mut ga = Ga::builder(Real::uniform(4, -5.0..=5.0)?)
    ///     .population_size(20)
    ///     .select(Tournament::new(2)?)
    ///     .crossover(UniformCrossover::new())
    ///     .mutate(GaussianMutation::per_gene(0.5, 0.2)?)
    ///     .minimize()
    ///     .seed(1)
    ///     .build()?;
    /// let sphere = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
    /// for generation in 0..50 {
    ///     let fitness: Vec<Fitness> = ga.ask().iter().map(|x| Fitness::new(sphere(x))).collect();
    ///     ga.tell(&fitness)?;
    ///     // the step shrinks from 20% to 1% of each gene's range
    ///     let sigma = 0.2 * (0.01_f64 / 0.2).powf(f64::from(generation) / 49.0);
    ///     *ga.mutate_mut() = GaussianMutation::per_gene(0.5, sigma)?;
    /// }
    /// # Ok::<(), genoxide::Error>(())
    /// ```
    pub fn mutate_mut(&mut self) -> &mut M {
        &mut self.mutate
    }

    // the most spare genomes worth keeping: one per genome a generation copies, each child of the
    // selected pairs and each memetic neighbor
    fn spare_limit(&self) -> usize {
        let count = self.scheme.offspring_count(self.population_size);
        let neighbors = self
            .memetic
            .map_or(0, |(parents, neighbors)| parents * neighbors);
        count + count % 2 + neighbors
    }
}

impl<R, S, C, M> Ga<R, S, C, M>
where
    R: Representation,
    C: Crossover<R>,
{
    /// Changes the probability that a pair of parents is recombined, during a run. As
    /// [`select_mut`](Ga::select_mut): it applies from the next generation's breeding.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] as for [`GaBuilder::crossover_rate`]: a rate outside [0, 1], or
    /// 0 while the mutation rate is 0 too. The rate doesn't change on errors.
    pub fn set_crossover_rate(&mut self, rate: f64) -> Result<()> {
        let (crossover_rate, _) =
            check_rates(rate, self.mutation_rate, self.crossover.recombines())?;
        self.crossover_rate = crossover_rate;
        self.crossover_chance = Chance::new(crossover_rate);
        Ok(())
    }

    /// Changes the probability that a child is mutated, during a run. As
    /// [`select_mut`](Ga::select_mut): it applies from the next generation's breeding.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] as for [`GaBuilder::mutation_rate`]: a rate outside [0, 1], or
    /// 0 while every child would otherwise be a copy of a parent. The rate doesn't change on
    /// errors.
    pub fn set_mutation_rate(&mut self, rate: f64) -> Result<()> {
        let (_, mutation_rate) =
            check_rates(self.crossover_rate, rate, self.crossover.recombines())?;
        self.mutation_rate = mutation_rate;
        self.mutation_chance = Chance::new(mutation_rate);
        Ok(())
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
        let breeding = Breeding {
            representation: &self.representation,
            crossover: &self.crossover,
            mutate: &self.mutate,
            crossover_chance: self.crossover_chance,
            mutation_chance: self.mutation_chance,
            population: &self.population,
        };
        if self.parallel_breeding {
            // a stream per pair, from the seed, the generation and the pair's position: the same
            // children on any number of threads
            let streams = self.rng.derive(BREEDING_STREAMS).derive(self.generation);
            let genomes: Vec<Option<R::Genome>> =
                parents.iter().map(|_| self.spare.0.pop()).collect();
            breed_pairs(
                &breeding,
                &parents,
                count,
                genomes,
                &streams,
                &mut self.offspring,
            );
        } else {
            breeding.pairs(
                &parents,
                count,
                &mut self.rng,
                &mut self.spare,
                &mut self.offspring,
            );
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
        let limit = self.spare_limit();
        // observers have seen the last generation's discarded offspring
        self.spare
            .recycle(self.discarded.drain(..).map(Individual::into_genome), limit);
        match self.scheme {
            Scheme::Generational { elitism } => {
                self.keep_best_parents(elitism, limit);
            }
            Scheme::SteadyState { replacements } => {
                self.keep_best_parents(size - replacements, limit);
            }
            Scheme::MuPlusLambda { .. } => {
                let mut all = mem::take(&mut self.offspring);
                all.extend(mem::take(&mut self.population).into_iter().map(aged));
                let (population, rest) = best_of(all, size, self.objective);
                self.population = population;
                // the parents were seen before, only the offspring are new: parents have been aged
                for individual in rest {
                    if individual.age() == 0 {
                        self.discarded.push(individual);
                    } else {
                        self.spare.recycle([individual.into_genome()], limit);
                    }
                }
            }
            Scheme::MuCommaLambda { .. } => {
                let offspring = mem::take(&mut self.offspring);
                let parents = mem::take(&mut self.population);
                self.spare
                    .recycle(parents.into_iter().map(Individual::into_genome), limit);
                (self.population, self.discarded) = best_of(offspring, size, self.objective);
            }
        }
    }

    // the next population is the `count` best parents followed by the offspring; the others'
    // genomes are spare, up to `limit`
    fn keep_best_parents(&mut self, count: usize, limit: usize) {
        let others = self.population.keep_best(count, self.objective);
        self.spare
            .recycle(others.map(Individual::into_genome), limit);
        self.population
            .iter_mut()
            .for_each(Individual::increment_age);
        self.population.append(&mut self.offspring);
    }
}

// The id of the streams derived from a GA's generator for parallel breeding: its stream for a
// generation is `rng.derive(BREEDING_STREAMS).derive(generation)`, and a pair's stream is
// `generation_stream.derive(pair)`. It must never change for the same major version: it decides
// the results of seeded runs.
const BREEDING_STREAMS: u64 = super::breeding_streams::GA;

// what crossing over and mutating a pair of parents needs
struct Breeding<'a, R: Representation, C, M> {
    representation: &'a R,
    crossover: &'a C,
    mutate: &'a M,
    crossover_chance: Chance,
    mutation_chance: Chance,
    population: &'a Population<R::Genome>,
}

impl<R, C, M> Breeding<'_, R, C, M>
where
    R: Representation,
    C: Crossover<R>,
    M: Mutate<R>,
{
    // the children of the pairs of parents at the positions `parents`, one pair after the other,
    // pushed to `offspring` until `wanted` are made (at most two per pair), in the memory of
    // genomes popped from `spare` while there are any, drawing from `rng`: recombined with the
    // crossover rate, each mutated with the mutation rate, and a copy of a parent with the
    // parent's fitness. The only code that breeds a GA's children, sequentially
    // and in parallel: out of line, so that each operator has this one caller and is inlined
    // here, as it was before parallel breeding.
    #[inline(never)]
    fn pairs(
        &self,
        parents: &[usize],
        wanted: usize,
        rng: &mut StreamRng,
        spare: &mut Spare<R::Genome>,
        offspring: &mut Vec<Individual<R::Genome>>,
    ) {
        let end = offspring.len() + wanted;
        for pair in parents.chunks_exact(2) {
            let parents = [&self.population[pair[0]], &self.population[pair[1]]];
            let mut a = spare.copy(parents[0].genome());
            let mut b = spare.copy(parents[1].genome());
            if rng.chance(self.crossover_chance) {
                self.crossover
                    .crossover(self.representation, &mut a, &mut b, rng);
            }
            for mut genome in [a, b] {
                if offspring.len() == end {
                    break;
                }
                if rng.chance(self.mutation_chance) {
                    self.mutate.mutate(self.representation, &mut genome, rng);
                }
                let inherited = parents
                    .iter()
                    .find(|parent| parent.genome() == &genome)
                    .and_then(|parent| parent.fitness());
                let mut child = Individual::new(genome);
                if let Some(fitness) = inherited {
                    child.set_fitness(fitness);
                }
                offspring.push(child);
            }
        }
    }
}

// the `count` children of the selected `parents` into `offspring`, each pair bred on its stream,
// derived from `streams` with the pair's position, in the memory of its `genomes` (one per
// parent, if any), in parallel
#[cfg(feature = "parallel")]
fn breed_pairs<R, C, M>(
    breeding: &Breeding<'_, R, C, M>,
    parents: &[usize],
    count: usize,
    mut genomes: Vec<Option<R::Genome>>,
    streams: &StreamRng,
    offspring: &mut Vec<Individual<R::Genome>>,
) where
    R: Representation,
    C: Crossover<R>,
    M: Mutate<R>,
{
    use rayon::prelude::*;
    type Bred<G> = (Spare<G>, Vec<Individual<G>>);
    // the children of consecutive pairs, in order: collecting keeps the order of the folds,
    // whatever the thread count
    let bred: Vec<Bred<R::Genome>> = parents
        .par_chunks_exact(2)
        .zip(genomes.par_chunks_mut(2))
        .enumerate()
        .fold(
            || (Spare::default(), Vec::new()),
            |(mut spare, mut children): Bred<R::Genome>, (index, (pair, genomes))| {
                spare.0.extend(genomes.iter_mut().filter_map(Option::take));
                let mut rng = streams.derive(index as u64);
                breeding.pairs(pair, count - 2 * index, &mut rng, &mut spare, &mut children);
                (spare, children)
            },
        )
        .collect();
    offspring.extend(bred.into_iter().flat_map(|(_, children)| children));
}

// without the `parallel` feature (a checkpoint of a run with parallel breeding), the same
// children, one pair after the other
#[cfg(not(feature = "parallel"))]
fn breed_pairs<R, C, M>(
    breeding: &Breeding<'_, R, C, M>,
    parents: &[usize],
    count: usize,
    mut genomes: Vec<Option<R::Genome>>,
    streams: &StreamRng,
    offspring: &mut Vec<Individual<R::Genome>>,
) where
    R: Representation,
    C: Crossover<R>,
    M: Mutate<R>,
{
    let mut spare = Spare::default();
    for (index, (pair, genomes)) in parents
        .chunks_exact(2)
        .zip(genomes.chunks_mut(2))
        .enumerate()
    {
        spare.0.extend(genomes.iter_mut().filter_map(Option::take));
        let mut rng = streams.derive(index as u64);
        breeding.pairs(pair, count - 2 * index, &mut rng, &mut spare, offspring);
    }
}

// genomes no longer in use, whose memory new ones reuse (a GA's, and islands' copies): none in a clone, a checkpoint or its
// debug output, as they change no result
pub(super) struct Spare<G>(Vec<G>);

impl<G> Spare<G> {
    // keeps `genomes` until `limit` are kept, and drops the rest
    pub(super) fn recycle(&mut self, genomes: impl IntoIterator<Item = G>, limit: usize) {
        let room = limit.saturating_sub(self.0.len());
        self.0.extend(genomes.into_iter().take(room));
    }

    // a copy of `source`, in the memory of a spare genome if there is one: inlined into the
    // breeding loop, with the genome's `clone_from`
    #[inline(always)]
    pub(super) fn copy(&mut self, source: &G) -> G
    where
        G: Clone,
    {
        match self.0.pop() {
            Some(mut genome) => {
                genome.clone_from(source);
                genome
            }
            None => source.clone(),
        }
    }
}

impl<G> Default for Spare<G> {
    fn default() -> Self {
        Spare(Vec::new())
    }
}

impl<G> Clone for Spare<G> {
    fn clone(&self) -> Self {
        Spare::default()
    }
}

impl<G> std::fmt::Debug for Spare<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Spare")
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
        let limit = self.spare_limit();
        let replaced = self.population.keep_best(size - count, self.objective);
        self.spare
            .recycle(replaced.map(Individual::into_genome), limit);
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

impl<R, S, C, M> super::Reevaluate for Ga<R, S, C, M>
where
    R: Representation,
    S: Select,
    C: Crossover<R>,
    M: Mutate<R>,
{
    /// As [`Ga::reevaluate`]: the next ask gives the whole population.
    fn reevaluate(&mut self) -> Result<()> {
        Ga::reevaluate(self)
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
                // the first evaluation, or a re-evaluation, after which the old best is measured
                // by another function: the population is the best found so far
                self.best = None;
                update_best(&mut self.best, self.population.as_slice(), self.objective);
                self.best_generation = self.generation;
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
/// elitism of 1, a random seed, and sequential breeding.
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
    parallel_breeding: bool,
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
            parallel_breeding: self.parallel_breeding,
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
            parallel_breeding: self.parallel_breeding,
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
            parallel_breeding: self.parallel_breeding,
        }
    }

    /// The population size, μ, at least 1, or 2 with the default scheme (an elitism of 1), and at
    /// most 2^24. Required.
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
    /// survivor selection. It adds `parents * neighbors` evaluations per generation, at most 2^24.
    /// Off by default.
    ///
    /// The refined parents must be ones that survive the generation, so `parents` is at most the
    /// elitism of [`Scheme::Generational`], the population size minus the replacements of
    /// [`Scheme::SteadyState`], or the population size with [`Scheme::MuPlusLambda`]. It doesn't
    /// work with [`Scheme::MuCommaLambda`], where no parent survives.
    pub fn memetic(mut self, parents: usize, neighbors: usize) -> Self {
        self.memetic = Some((parents, neighbors));
        self
    }

    /// Breeds the offspring in parallel with rayon: each pair of selected parents is crossed over
    /// and mutated on a thread pool. Off by default.
    ///
    /// - **Selection** stays sequential, on the run's random numbers, and so does the
    ///   [`memetic`](GaBuilder::memetic) search. Copies of a parent still inherit its fitness,
    ///   and every [`Scheme`] works as without it.
    /// - **Random numbers:** each pair draws from a stream of its own, derived from the seed, the
    ///   generation and the pair's position ([`StreamRng::derive`]). A seeded run gives the same
    ///   results on any number of threads, but not the results it gives without parallel
    ///   breeding.
    /// - **When it pays off:** when breeding is a large part of a generation. That happens with
    ///   parallel or batch evaluation of a fast fitness function, operators that do real work per
    ///   gene (a mutation that draws several numbers and repairs each gene, long genomes), and
    ///   thousands of children per generation. With a slow fitness function, breeding takes
    ///   little of the time; with small populations and short genomes, the threads can cost
    ///   more than they save.
    ///
    /// Checkpoints keep the setting. A run with it resumes identically, also without the
    /// `parallel` feature, then breeding the same children on one thread.
    ///
    /// ```
    /// use genoxide::prelude::*;
    ///
    /// let ga = Ga::builder(Real::uniform(50, -5.0..=5.0)?)
    ///     .population_size(1000)
    ///     .select(Tournament::new(3)?)
    ///     .crossover(SimulatedBinaryCrossover::new(15.0)?)
    ///     .mutate(PolynomialMutation::per_gene(0.1, 20.0)?)
    ///     .parallel_breeding(true)
    ///     .minimize()
    ///     .seed(1)
    ///     .build()?;
    /// let outcome = Engine::new(ga, |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>())
    ///     .parallel(true)
    ///     .stop_when(Stop::generations(20))
    ///     .run()?;
    /// assert!(outcome.best_fitness().score().unwrap() < 100.0);
    /// # Ok::<(), genoxide::Error>(())
    /// ```
    #[cfg(feature = "parallel")]
    pub fn parallel_breeding(mut self, parallel: bool) -> Self {
        self.parallel_breeding = parallel;
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
    /// - [`Error::InvalidSetting`] for a population size of 0 or above 2^24, rates outside
    ///   [0, 1], a mutation rate of 0 with a crossover rate of 0 or [`NoCrossover`] (every child
    ///   would be a copy), a [`Scheme`] that doesn't fit the population size or with more than
    ///   2^24 offspring, more initial genomes than the population size, or memetic settings out
    ///   of range.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    ///
    /// [`NoCrossover`]: crate::operator::NoCrossover
    pub fn build(self) -> Result<Ga<R, S, C, M>>
    where
        S: Select,
        C: Crossover<R>,
        M: Mutate<R>,
    {
        let (population_size, crossover_rate, mutation_rate) = self.check()?;
        let individuals = isize::MAX as usize / mem::size_of::<Individual<R::Genome>>();
        self.scheme.validate(population_size, individuals)?;
        if let Some((parents, neighbors)) = self.memetic {
            // the refined parents must survive the generation, or their refinement is lost
            let survivors = match self.scheme {
                Scheme::Generational { elitism } => elitism,
                Scheme::SteadyState { replacements } => population_size - replacements,
                Scheme::MuPlusLambda { .. } => population_size,
                Scheme::MuCommaLambda { .. } => 0,
            };
            if parents == 0
                || parents > survivors
                || neighbors == 0
                || parents.saturating_mul(neighbors) > MAX_SIZE
            {
                return Err(Error::InvalidSetting {
                    setting: "memetic",
                    reason: format!(
                        "parents must be between 1 and the number of parents that survive a generation, {survivors} with {:?} (the elitism of a generational scheme, the population size minus the replacements of a steady-state one, the population size with (μ+λ), none with (μ,λ)), neighbors at least 1, and parents * neighbors at most {MAX_SIZE}; got {parents} and {neighbors}",
                        self.scheme
                    ),
                });
            }
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
            parallel_breeding: self.parallel_breeding,
            population: Population::from_genomes(genomes),
            offspring: Vec::new(),
            discarded: Vec::new(),
            spare: Spare::default(),
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

    /// Validates the settings and creates a [`SteadyGa`] for asynchronous evaluation with an
    /// [`AsyncEngine`](crate::engine::AsyncEngine): it proposes one child at a time, and each
    /// result replaces the worst individual when it's not worse. The scheme, memetic and parallel
    /// breeding settings don't apply to it.
    ///
    /// # Errors
    ///
    /// As [`build`](GaBuilder::build), except for the scheme, and [`Error::InvalidSetting`] for
    /// a scheme other than the default one, memetic search, or parallel breeding.
    pub fn build_steady(self) -> Result<SteadyGa<R, S, C, M>>
    where
        S: Select,
        C: Crossover<R>,
        M: Mutate<R>,
    {
        let (population_size, crossover_rate, mutation_rate) = self.check()?;
        if self.scheme != Scheme::default() {
            return Err(Error::InvalidSetting {
                setting: "scheme",
                reason: "a steady-state GA for asynchronous evaluation replaces the worst individual with each result; it has no scheme".to_string(),
            });
        }
        if self.memetic.is_some() {
            return Err(Error::InvalidSetting {
                setting: "memetic",
                reason: "a steady-state GA for asynchronous evaluation has no memetic search"
                    .to_string(),
            });
        }
        if self.parallel_breeding {
            return Err(Error::InvalidSetting {
                setting: "parallel_breeding",
                reason: "a steady-state GA for asynchronous evaluation breeds one child at a time"
                    .to_string(),
            });
        }
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        Ok(SteadyGa::new(
            self.representation,
            self.select,
            self.crossover,
            self.mutate,
            self.objective,
            population_size,
            (crossover_rate, mutation_rate),
            seed,
            self.initial_genomes,
        ))
    }

    // the population size and rates, checked, and the initial genomes checked
    fn check(&self) -> Result<(usize, f64, f64)>
    where
        C: Crossover<R>,
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
        check_size("population_size", population_size)?;
        let (crossover_rate, mutation_rate) = check_rates(
            self.crossover_rate,
            self.mutation_rate,
            self.crossover.recombines(),
        )?;
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
        Ok((population_size, crossover_rate, mutation_rate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{Binary, Bits, Real, Reals};
    use crate::operator::{
        BitFlip, NoCrossover, PolynomialMutation, SimulatedBinaryCrossover, Tournament,
        UniformCrossover,
    };
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
    fn settings_changed_during_a_run_are_validated() {
        let mut ga = builder(8).build().unwrap();
        step(&mut ga);
        assert!(matches!(
            ga.set_crossover_rate(1.5),
            Err(Error::InvalidSetting {
                setting: "crossover_rate",
                ..
            })
        ));
        assert!(matches!(
            ga.set_mutation_rate(-0.1),
            Err(Error::InvalidSetting {
                setting: "mutation_rate",
                ..
            })
        ));
        ga.set_crossover_rate(0.0).unwrap();
        // both 0: every child would be a copy of a parent
        assert!(matches!(
            ga.set_mutation_rate(0.0),
            Err(Error::InvalidSetting {
                setting: "mutation_rate",
                ..
            })
        ));
        // unchanged on errors
        assert_eq!(ga.crossover_rate(), 0.0);
        assert_eq!(ga.mutation_rate(), 1.0);

        let mut ga = Ga::builder(Binary::new(8).unwrap())
            .population_size(10)
            .select(Tournament::new(2).unwrap())
            .crossover(NoCrossover)
            .mutate(BitFlip::count(1).unwrap())
            .seed(0)
            .build()
            .unwrap();
        // without recombination, mutation is the only change
        assert!(matches!(
            ga.set_mutation_rate(0.0),
            Err(Error::InvalidSetting {
                setting: "mutation_rate",
                ..
            })
        ));
    }

    #[test]
    fn a_setting_changed_before_breeding_gives_the_run_built_with_it() {
        let run = |mut ga: OneMaxGa, change: fn(&mut OneMaxGa)| {
            step(&mut ga);
            change(&mut ga);
            for _ in 0..20 {
                step(&mut ga);
            }
            ga.population().clone()
        };
        let built = builder(32)
            .crossover_rate(0.3)
            .mutation_rate(0.5)
            .select(Tournament::new(4).unwrap())
            .mutate(BitFlip::count(3).unwrap())
            .build()
            .unwrap();
        let changed = |ga: &mut OneMaxGa| {
            ga.set_crossover_rate(0.3).unwrap();
            ga.set_mutation_rate(0.5).unwrap();
            *ga.select_mut() = Tournament::new(4).unwrap();
            *ga.mutate_mut() = BitFlip::count(3).unwrap();
        };
        let expected = run(built, |_| {});
        assert_eq!(run(builder(32).build().unwrap(), changed), expected);
        // and they do change the run
        assert_ne!(run(builder(32).build().unwrap(), |_| {}), expected);
    }

    #[test]
    fn a_reevaluation_scores_the_population_again_without_breeding() {
        let mut ga = builder(16).build().unwrap();
        for _ in 0..5 {
            step(&mut ga);
        }
        let genomes: Vec<Bits> = ga.population().iter().map(|i| i.genome().clone()).collect();
        let (generation, evaluations) = (ga.generation(), ga.evaluations());

        ga.reevaluate().unwrap();
        let asked: Vec<Bits> = ga.ask().iter().cloned().collect();
        assert_eq!(asked, genomes);
        // the zeros now
        let fitness: Vec<Fitness> = asked
            .iter()
            .map(|genome| Fitness::new(genome.count_zeros() as f64))
            .collect();
        ga.tell(&fitness).unwrap();

        assert_eq!(ga.generation(), generation);
        assert_eq!(ga.evaluations(), evaluations + 10);
        let population: Vec<Bits> = ga.population().iter().map(|i| i.genome().clone()).collect();
        assert_eq!(population, genomes);
        let most_zeros = genomes.iter().map(Bits::count_zeros).max().unwrap();
        assert_eq!(
            ga.best().unwrap().fitness(),
            Some(Fitness::new(most_zeros as f64))
        );
        assert_eq!(ga.best_generation(), generation);
        assert!(ga.discarded().is_empty());
    }

    #[test]
    fn a_reevaluation_with_the_same_function_changes_only_the_count() {
        let run = |reevaluate: bool| {
            let mut ga = builder(16).build().unwrap();
            for generation in 0..20 {
                if reevaluate && generation == 10 {
                    ga.reevaluate().unwrap();
                    step(&mut ga);
                }
                step(&mut ga);
            }
            (
                ga.population().clone(),
                ga.best().cloned(),
                ga.evaluations(),
            )
        };
        let (population, best, evaluations) = run(false);
        assert_eq!(run(true), (population, best, evaluations + 10));
    }

    #[test]
    fn a_reevaluation_waits_for_the_tell() {
        let mut ga = builder(8).build().unwrap();
        // before anything is evaluated, it changes nothing
        ga.reevaluate().unwrap();
        step(&mut ga);
        let mut other = builder(8).build().unwrap();
        step(&mut other);
        assert_eq!(ga.population(), other.population());

        let asked = ga.ask().len();
        assert_eq!(ga.reevaluate(), Err(Error::ReevaluationOutOfTurn));
        // still the offspring
        assert_eq!(ga.ask().len(), asked);
        step(&mut ga);
    }

    #[test]
    fn lambda_is_at_most_the_offspring_that_fit_in_memory() {
        // room for 100 individuals: λ offspring, and with (μ+λ) the μ = 10 parents too
        let valid = |scheme: Scheme| scheme.validate(10, 100).is_ok();
        assert!(valid(Scheme::MuPlusLambda { lambda: 90 }));
        assert!(!valid(Scheme::MuPlusLambda { lambda: 91 }));
        assert!(valid(Scheme::MuCommaLambda { lambda: 100 }));
        assert!(!valid(Scheme::MuCommaLambda { lambda: 101 }));
        // and at most 2^24, however much fits
        let valid = |scheme: Scheme| scheme.validate(10, usize::MAX).is_ok();
        assert!(valid(Scheme::MuCommaLambda { lambda: MAX_SIZE }));
        assert!(!valid(Scheme::MuCommaLambda {
            lambda: MAX_SIZE + 1
        }));
        assert!(valid(Scheme::MuPlusLambda { lambda: MAX_SIZE }));
        assert!(!valid(Scheme::MuPlusLambda {
            lambda: MAX_SIZE + 1
        }));
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
        for parallel_breeding in [false, true] {
            // identical parents: every child of a crossover without mutation is a copy
            let mut ga = builder(8)
                .initial_genomes(vec![Bits::ones(8); 10])
                .crossover_rate(1.0)
                .mutation_rate(0.0)
                .build()
                .unwrap();
            ga.parallel_breeding = parallel_breeding;
            step(&mut ga);
            assert!(ga.ask().is_empty());
            ga.tell(&[]).unwrap();
            assert_eq!((ga.generation(), ga.evaluations()), (1, 10));
            assert!(ga.population().iter().all(Individual::is_evaluated));
        }
    }

    type RealGa = Ga<Real, Tournament, SimulatedBinaryCrossover, PolynomialMutation>;

    // parallel breeding is set directly, so that these tests run without the `parallel` feature
    // too, where the pairs are bred one after the other with the same results
    fn real_ga(seed: u64, scheme: Scheme, parallel_breeding: bool) -> RealGa {
        let mut ga = Ga::builder(Real::uniform(8, -5.0..=5.0).unwrap())
            .population_size(21)
            .select(Tournament::new(3).unwrap())
            .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
            .mutate(PolynomialMutation::per_gene(0.2, 20.0).unwrap())
            .scheme(scheme)
            .minimize()
            .seed(seed)
            .build()
            .unwrap();
        ga.parallel_breeding = parallel_breeding;
        ga
    }

    // the population, the best and the evaluations after `generations` generations of a
    // Rosenbrock function with only +, − and ×, which is the same on every platform
    fn run_real(
        mut ga: RealGa,
        generations: u64,
    ) -> (Population<Reals>, Option<Individual<Reals>>, u64) {
        let rosenbrock = |x: &Reals| {
            x.windows(2)
                .map(|w| {
                    let (a, b) = (w[1] - w[0] * w[0], 1.0 - w[0]);
                    100.0 * a * a + b * b
                })
                .sum::<f64>()
        };
        while ga.generation() < generations {
            let fitness: Vec<Fitness> = ga
                .ask()
                .iter()
                .map(|x| Fitness::new(rosenbrock(x)))
                .collect();
            ga.tell(&fitness).unwrap();
        }
        (
            ga.population().clone(),
            ga.best().cloned(),
            ga.evaluations(),
        )
    }

    // an odd number of offspring (21), so the last pair has one child, and every scheme
    const SCHEMES: [Scheme; 5] = [
        Scheme::Generational { elitism: 0 },
        Scheme::Generational { elitism: 2 },
        Scheme::SteadyState { replacements: 5 },
        Scheme::MuPlusLambda { lambda: 21 },
        Scheme::MuCommaLambda { lambda: 42 },
    ];

    #[test]
    fn parallel_breeding_is_reproducible_but_not_sequential_breeding() {
        for scheme in SCHEMES {
            let run =
                |seed, parallel_breeding| run_real(real_ga(seed, scheme, parallel_breeding), 15);
            let parallel = run(1, true);
            assert_eq!(run(1, true), parallel, "{scheme:?}");
            assert_ne!(run(2, true), parallel, "{scheme:?}");
            let sequential = run(1, false);
            assert_ne!(sequential, parallel, "{scheme:?}");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_breeding_is_the_same_on_any_number_of_threads() {
        for scheme in SCHEMES {
            let on_threads = |threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| run_real(real_ga(3, scheme, true), 15))
            };
            let one = on_threads(1);
            assert_eq!(on_threads(2), one, "{scheme:?}");
            assert_eq!(on_threads(8), one, "{scheme:?}");
        }
    }

    /// Fixed values: these must never change for the same major version, on any platform. They
    /// pin the streams of parallel breeding, from the seed, the generation and the pair's
    /// position, with or without the `parallel` feature.
    #[test]
    fn parallel_breeding_values_are_portable() {
        let (_, best, evaluations) =
            run_real(real_ga(1, Scheme::Generational { elitism: 1 }, true), 20);
        assert_eq!(evaluations, 406);
        assert_eq!(
            best.unwrap().genome().to_vec(),
            [
                1.004984224607953,
                0.8873809759635563,
                0.5652053572814714,
                0.27580207032382986,
                0.15310959905804483,
                0.00601326684447917,
                -0.7981794150630991,
                0.4801327414809142
            ]
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_breeding_is_a_setting_of_the_builder_and_not_of_a_steady_ga() {
        let ga = builder(8).parallel_breeding(true).build().unwrap();
        assert!(ga.parallel_breeding());
        assert!(!builder(8).build().unwrap().parallel_breeding());
        assert!(matches!(
            builder(8).parallel_breeding(true).build_steady(),
            Err(Error::InvalidSetting {
                setting: "parallel_breeding",
                ..
            })
        ));
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
            for parallel_breeding in [false, true] {
                let mut ga = builder(64)
                    .scheme(scheme)
                    .memetic(parents, 1)
                    .build()
                    .unwrap();
                ga.parallel_breeding = parallel_breeding;
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
        fn schemes(scheme in any_scheme(), seed: u64, parallel_breeding: bool) {
            let mut ga = builder(16).scheme(scheme).seed(seed).build().unwrap();
            ga.parallel_breeding = parallel_breeding;
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
                // the genomes kept for the next offspring are at most one per copy
                prop_assert!(ga.spare.0.len() <= ga.spare_limit());
            }
            // a clone has no spare genomes, and makes the same run
            let mut clone = ga.clone();
            prop_assert!(clone.spare.0.is_empty());
            for _ in 0..3 {
                step(&mut ga);
                step(&mut clone);
                prop_assert_eq!(clone.population(), ga.population());
                prop_assert_eq!(clone.discarded(), ga.discarded());
            }
        }
    }

    #[test]
    fn offspring_reuse_the_genomes_of_the_replaced_parents() {
        let mut ga = builder(16)
            .scheme(Scheme::Generational { elitism: 0 })
            .build()
            .unwrap();
        step(&mut ga);
        step(&mut ga);
        // the ten parents replaced
        assert_eq!(ga.spare.0.len(), 10);
        let spare: Vec<*const u64> = ga
            .spare
            .0
            .iter()
            .map(|genome| genome.as_words().as_ptr())
            .collect();
        step(&mut ga);
        let offspring = ga
            .population()
            .iter()
            .map(|child| child.genome().as_words().as_ptr());
        assert!(offspring.into_iter().all(|child| spare.contains(&child)));
    }
}
