//! The island model: several populations that evolve apart and exchange their best individuals.

use super::{Algorithm, Candidates};
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;
use std::sync::OnceLock;

/// An [`Algorithm`] whose population can take individuals from elsewhere: the islands of
/// [`Islands`].
pub trait Migrate: Algorithm {
    /// Replaces the worst individuals of the population with `migrants`, which are evaluated; if
    /// there are more migrants than the population size, only the first ones come in.
    ///
    /// # Errors
    ///
    /// [`Error::MigrationOutOfTurn`] before the first [`tell`](Algorithm::tell), or between an
    /// [`ask`](Algorithm::ask) and its tell. The population doesn't change on errors.
    fn immigrate(&mut self, migrants: Vec<Individual<Self::Genome>>) -> Result<()>;
}

/// Where the migrants of [`Islands`] go.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Topology {
    /// Each island sends to the next one, the last to the first (the default): good solutions
    /// spread slowly, which keeps the islands diverse.
    #[default]
    Ring,
    /// Each island sends to every other island: good solutions spread fastest.
    FullyConnected,
    /// Each island sends to another island chosen at random at every migration.
    Random,
}

/// The island model: several algorithms of one type, the islands, that evolve apart and, every
/// few generations, send copies of their best individuals to other islands, where they replace
/// the worst. Isolation keeps the islands diverse, and migration spreads what they find: often
/// better than one large population on multimodal problems.
///
/// `Islands` is an [`Algorithm`] itself, so an [`Engine`](crate::Engine) runs it: every
/// generation asks each island in turn, and the engine evaluates the candidates of all islands
/// together, in parallel if asked. Breeding and migration are sequential and synchronous (the
/// migrants of a migration are chosen before any arrives), so a run is the same with any number
/// of threads.
///
/// ```
/// use genoxide::algorithm::{Islands, islands::Topology};
/// use genoxide::prelude::*;
///
/// // four islands, each with its own seed
/// let islands = (0..4)
///     .map(|seed| {
///         Ga::builder(Binary::new(64)?)
///             .population_size(20)
///             .select(Tournament::new(2)?)
///             .crossover(UniformCrossover::new())
///             .mutate(BitFlip::per_gene(1.0 / 64.0)?)
///             .seed(seed)
///             .build()
///     })
///     .collect::<genoxide::Result<Vec<_>>>()?;
/// let islands = Islands::builder(islands)
///     .topology(Topology::Ring)
///     .interval(5)
///     .migrants(2)
///     .build()?;
/// let outcome = Engine::new(islands, |genome: &Bits| genome.count_ones() as f64)
///     .stop_when(Stop::target(64.0).or(Stop::generations(500)))
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Target);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Islands<A: Migrate> {
    islands: Vec<A>,
    topology: Topology,
    interval: u64,
    migrants: usize,
    seed: u64,
    rng: StreamRng,
    // copies of the genomes asked by the islands, and how many each asked
    candidates: Vec<Individual<A::Genome>>,
    counts: Vec<usize>,
    pending: Vec<usize>,
    // the combined population and discarded individuals, built when first asked for: a run
    // without observers never copies them
    population: OnceLock<Population<A::Genome>>,
    discarded: OnceLock<Vec<Individual<A::Genome>>>,
    asked: bool,
    started: bool,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<A::Genome>>,
    best_generation: u64,
}

impl<A: Migrate> Islands<A> {
    /// A builder for islands of these algorithms, which should have different seeds.
    pub fn builder(islands: Vec<A>) -> IslandsBuilder<A> {
        IslandsBuilder {
            islands,
            topology: Topology::Ring,
            interval: 10,
            migrants: 2,
            seed: None,
        }
    }

    /// The islands.
    pub fn islands(&self) -> &[A] {
        &self.islands
    }

    /// Where the migrants go.
    pub fn topology(&self) -> Topology {
        self.topology
    }

    /// The number of generations between migrations.
    pub fn interval(&self) -> u64 {
        self.interval
    }

    /// The number of individuals each island sends at a migration.
    pub fn migrants(&self) -> usize {
        self.migrants
    }

    /// The seed of the random numbers of the random topology: the given one, or a random one if
    /// none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    // sends copies of the best individuals of every island to its neighbors
    fn migrate(&mut self) -> Result<()> {
        let count = self.islands.len();
        let objective = self.objective();
        let mut arriving: Vec<Vec<Individual<A::Genome>>> = vec![Vec::new(); count];
        for (from, island) in self.islands.iter().enumerate() {
            // copies of the best evaluated individuals, the earlier ones first on ties
            let population = island.population();
            let mut order: Vec<usize> = (0..population.len())
                .filter(|&index| population[index].is_evaluated())
                .collect();
            let fitness = |index: usize| population[index].fitness().unwrap_or(Fitness::invalid());
            order.sort_by(|&a, &b| objective.compare(fitness(b), fitness(a)));
            let emigrants: Vec<Individual<A::Genome>> = order
                .into_iter()
                .take(self.migrants)
                .map(|index| population[index].clone())
                .collect();
            let destinations: Vec<usize> = match self.topology {
                Topology::Ring => vec![(from + 1) % count],
                Topology::FullyConnected => (0..count).filter(|&to| to != from).collect(),
                Topology::Random => {
                    let mut to = self.rng.below(count - 1);
                    if to >= from {
                        to += 1;
                    }
                    vec![to]
                }
            };
            for to in destinations {
                arriving[to].extend(emigrants.iter().cloned());
            }
        }
        for (island, migrants) in self.islands.iter_mut().zip(arriving) {
            if !migrants.is_empty() {
                island.immigrate(migrants)?;
            }
        }
        Ok(())
    }

    // the best so far, and a new combined population and discarded individuals when asked for
    fn collect(&mut self) {
        let objective = self.objective();
        self.population = OnceLock::new();
        self.discarded = OnceLock::new();
        let fitness =
            |individual: &Individual<A::Genome>| individual.fitness().unwrap_or(Fitness::invalid());
        let mut improved = false;
        for island in &self.islands {
            if let Some(candidate) = island.best() {
                let better = self
                    .best
                    .as_ref()
                    .is_none_or(|best| objective.is_better(fitness(candidate), fitness(best)));
                if better {
                    self.best = Some(candidate.clone());
                    improved = true;
                }
            }
        }
        if improved {
            self.best_generation = self.generation;
        }
    }
}

impl<A: Migrate> Algorithm for Islands<A> {
    type Genome = A::Genome;

    fn objective(&self) -> Objective {
        self.islands[0].objective()
    }

    fn ask(&mut self) -> Candidates<'_, A::Genome> {
        if !self.asked {
            self.candidates.clear();
            self.counts.clear();
            for island in &mut self.islands {
                let asked = island.ask();
                self.counts.push(asked.len());
                self.candidates
                    .extend(asked.iter().map(|genome| Individual::new(genome.clone())));
            }
            self.pending.clear();
            self.pending.extend(0..self.candidates.len());
            self.asked = true;
        }
        Candidates::new(&self.candidates, &self.pending)
    }

    fn tell(&mut self, fitness: &[Fitness]) -> Result<()> {
        if !self.asked {
            return Err(Error::TellWithoutAsk);
        }
        if fitness.len() != self.candidates.len() {
            return Err(Error::FitnessCount {
                expected: self.candidates.len(),
                got: fitness.len(),
            });
        }
        self.asked = false;
        self.evaluations += fitness.len() as u64;
        let mut rest = fitness;
        for (island, &count) in self.islands.iter_mut().zip(&self.counts) {
            let (own, others) = rest.split_at(count);
            island.tell(own)?;
            rest = others;
        }
        if self.started {
            self.generation += 1;
            if self.generation % self.interval == 0 {
                self.migrate()?;
            }
        }
        self.started = true;
        self.collect();
        Ok(())
    }

    /// The populations of all islands, in order. Built when first asked for after each
    /// generation.
    fn population(&self) -> &Population<A::Genome> {
        self.population.get_or_init(|| {
            self.islands
                .iter()
                .flat_map(|island| island.population().iter().cloned())
                .collect()
        })
    }

    fn best(&self) -> Option<&Individual<A::Genome>> {
        self.best.as_ref()
    }

    fn discarded(&self) -> &[Individual<A::Genome>] {
        self.discarded.get_or_init(|| {
            self.islands
                .iter()
                .flat_map(|island| island.discarded().iter().cloned())
                .collect()
        })
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

/// A builder for [`Islands`], from [`Islands::builder`].
///
/// Defaults: a ring, a migration every 10 generations, 2 migrants per island, and a random seed
/// for the random topology.
#[derive(Clone, Debug)]
pub struct IslandsBuilder<A: Migrate> {
    islands: Vec<A>,
    topology: Topology,
    interval: u64,
    migrants: usize,
    seed: Option<u64>,
}

impl<A: Migrate> IslandsBuilder<A> {
    /// Where the migrants go. [`Topology::Ring`] by default.
    pub fn topology(mut self, topology: Topology) -> Self {
        self.topology = topology;
        self
    }

    /// The number of generations between migrations, at least 1. 10 by default.
    pub fn interval(mut self, generations: u64) -> Self {
        self.interval = generations;
        self
    }

    /// The number of individuals each island sends at a migration: copies of its best, at
    /// least 1. 2 by default; a few percent of an island's population is common.
    pub fn migrants(mut self, count: usize) -> Self {
        self.migrants = count;
        self
    }

    /// The seed of the random topology's choices, for a reproducible run. Random by default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the settings and creates the island model.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for fewer than 2 islands, islands that already ran or that
    /// don't share an objective, an interval of 0, or no migrants.
    pub fn build(self) -> Result<Islands<A>> {
        let invalid = |setting, reason: String| Err(Error::InvalidSetting { setting, reason });
        if self.islands.len() < 2 {
            return invalid(
                "islands",
                format!("at least 2 islands, got {}", self.islands.len()),
            );
        }
        let objective = self.islands[0].objective();
        if self
            .islands
            .iter()
            .any(|island| island.objective() != objective)
        {
            return invalid("islands", "the islands must share an objective".to_string());
        }
        if self.islands.iter().any(|island| island.evaluations() > 0) {
            return invalid("islands", "the islands must not have run yet".to_string());
        }
        if self.interval == 0 {
            return invalid("interval", "at least 1 generation".to_string());
        }
        if self.migrants == 0 {
            return invalid("migrants", "at least 1".to_string());
        }
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        Ok(Islands {
            islands: self.islands,
            topology: self.topology,
            interval: self.interval,
            migrants: self.migrants,
            seed,
            rng: StreamRng::seed_from_u64(seed),
            candidates: Vec::new(),
            counts: Vec::new(),
            pending: Vec::new(),
            population: OnceLock::new(),
            discarded: OnceLock::new(),
            asked: false,
            started: false,
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
    use crate::algorithm::{De, Ga};
    use crate::genome::{Binary, Bits, Real, Reals};
    use crate::operator::{BitFlip, Tournament, UniformCrossover};

    type BinaryGa = Ga<Binary, Tournament, UniformCrossover, BitFlip>;

    fn ga(seed: u64) -> BinaryGa {
        Ga::builder(Binary::new(32).unwrap())
            .population_size(10)
            .select(Tournament::new(2).unwrap())
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::per_gene(1.0 / 32.0).unwrap())
            .seed(seed)
            .build()
            .unwrap()
    }

    fn ones(genome: &Bits) -> Fitness {
        Fitness::new(genome.count_ones() as f64)
    }

    fn step<A: Algorithm<Genome = Bits>>(algorithm: &mut A) {
        let fitness: Vec<Fitness> = algorithm.ask().iter().map(ones).collect();
        algorithm.tell(&fitness).unwrap();
    }

    fn setting<T: std::fmt::Debug>(result: Result<T>) -> &'static str {
        match result {
            Err(Error::InvalidSetting { setting, .. }) => setting,
            other => panic!("expected a setting error, got {other:?}"),
        }
    }

    #[test]
    fn validation() {
        assert_eq!(setting(Islands::builder(vec![ga(0)]).build()), "islands");
        assert_eq!(
            setting(Islands::builder(vec![ga(0), ga(1)]).interval(0).build()),
            "interval"
        );
        assert_eq!(
            setting(Islands::builder(vec![ga(0), ga(1)]).migrants(0).build()),
            "migrants"
        );
        let mut ran = ga(1);
        step(&mut ran);
        assert_eq!(
            setting(Islands::builder(vec![ga(0), ran]).build()),
            "islands"
        );
        let minimizing = Ga::builder(Binary::new(32).unwrap())
            .population_size(10)
            .select(Tournament::new(2).unwrap())
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::per_gene(1.0 / 32.0).unwrap())
            .minimize()
            .build()
            .unwrap();
        assert_eq!(
            setting(Islands::builder(vec![ga(0), minimizing]).build()),
            "islands"
        );
    }

    #[test]
    fn ask_tell_protocol() {
        let mut islands = Islands::builder(vec![ga(0), ga(1), ga(2)]).build().unwrap();
        assert_eq!(islands.tell(&[]), Err(Error::TellWithoutAsk));
        assert_eq!(islands.ask().len(), 30);
        assert_eq!(
            islands.tell(&[Fitness::new(0.0)]),
            Err(Error::FitnessCount {
                expected: 30,
                got: 1
            })
        );
        step(&mut islands);
        assert_eq!((islands.generation(), islands.evaluations()), (0, 30));
        assert_eq!(islands.population().len(), 30);
        step(&mut islands);
        assert_eq!(islands.generation(), 1);
        assert!(
            islands
                .islands()
                .iter()
                .all(|island| island.generation() == 1)
        );
        // the best of all islands
        let best = islands.best().unwrap().fitness().unwrap();
        for island in islands.islands() {
            assert!(
                !Objective::Maximize.is_better(island.best().unwrap().fitness().unwrap(), best)
            );
        }
    }

    #[test]
    fn a_ring_sends_the_best_to_the_next_island() {
        let mut islands = Islands::builder(vec![ga(0), ga(1), ga(2)])
            .interval(3)
            .migrants(1)
            .build()
            .unwrap();
        for _ in 0..3 {
            step(&mut islands);
        }
        // the next tell completes generation 3, and a migration: record each island's best
        // after its tell, then check that the next island has it
        let fitness: Vec<Fitness> = islands.ask().iter().map(ones).collect();
        let mut rest = &fitness[..];
        let mut bests = Vec::new();
        for (island, &count) in islands.islands.clone().iter_mut().zip(&islands.counts) {
            let (own, others) = rest.split_at(count);
            island.tell(own).unwrap();
            rest = others;
            let mut population = island.population().clone();
            population.sort_best_first(Objective::Maximize);
            bests.push(population[0].genome().clone());
        }
        islands.tell(&fitness).unwrap();
        assert_eq!(islands.generation(), 3);
        for (from, best) in bests.iter().enumerate() {
            let next = &islands.islands()[(from + 1) % 3];
            assert!(next.population().iter().any(|x| x.genome() == best));
        }
    }

    #[test]
    fn migrants_come_in_turn_and_replace_the_worst() {
        let mut island = ga(0);
        let migrant = {
            let mut individual = Individual::new(Bits::ones(32));
            individual.set_fitness(Fitness::new(32.0));
            individual
        };
        assert_eq!(
            island.immigrate(vec![migrant.clone()]),
            Err(Error::MigrationOutOfTurn)
        );
        step(&mut island);
        island.ask();
        assert_eq!(
            island.immigrate(vec![migrant.clone()]),
            Err(Error::MigrationOutOfTurn)
        );
        let fitness: Vec<Fitness> = island.ask().iter().map(ones).collect();
        island.tell(&fitness).unwrap();
        let worst = island
            .population()
            .iter()
            .map(|x| x.fitness().unwrap().score().unwrap())
            .fold(f64::INFINITY, f64::min);
        island.immigrate(vec![migrant.clone()]).unwrap();
        assert_eq!(island.population().len(), 10);
        assert!(
            island
                .population()
                .iter()
                .any(|x| x.genome() == migrant.genome())
        );
        assert_eq!(island.best().unwrap().genome(), migrant.genome());
        // the worst left: every other member is at least as good as it was
        let fewer_worst = island
            .population()
            .iter()
            .filter(|x| x.fitness().unwrap().score().unwrap() == worst)
            .count();
        assert!(fewer_worst < 10);
        // differential evolution
        let mut de = De::builder(Real::uniform(2, 0.0..=1.0).unwrap())
            .population_size(5)
            .minimize()
            .seed(0)
            .build()
            .unwrap();
        let migrant = {
            let mut individual = Individual::new(Reals::from(vec![0.0, 0.0]));
            individual.set_fitness(Fitness::new(-1.0));
            individual
        };
        assert_eq!(
            de.immigrate(vec![migrant.clone()]),
            Err(Error::MigrationOutOfTurn)
        );
        let fitness: Vec<Fitness> = de.ask().iter().map(|x| Fitness::new(x[0] + x[1])).collect();
        de.tell(&fitness).unwrap();
        let worst = (0..5)
            .max_by(|&a, &b| {
                let score = |i: usize| de.population()[i].fitness().unwrap().score().unwrap();
                score(a).total_cmp(&score(b))
            })
            .unwrap();
        de.immigrate(vec![migrant.clone()]).unwrap();
        assert_eq!(de.population()[worst].genome(), migrant.genome());
        assert_eq!(de.best().unwrap().genome(), migrant.genome());
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut islands = Islands::builder(vec![ga(0), ga(1), ga(2), ga(3)])
                .topology(Topology::Random)
                .interval(2)
                .seed(seed)
                .build()
                .unwrap();
            for _ in 0..12 {
                step(&mut islands);
            }
            islands.population().clone()
        };
        assert_eq!(run(5), run(5));
        assert_ne!(run(5), run(6));
    }
}
