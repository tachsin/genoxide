//! Particle swarm optimization: particles that fly toward the best places they and their
//! neighbors have found.

use super::{Algorithm, Candidates};
use crate::genome::{Real, Reals, Representation};
use crate::operator::check_size;
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;

/// Which particles each particle learns from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Topology {
    /// Every particle follows the best position of the whole swarm (gbest). Converges fast, but
    /// the swarm can gather around a local optimum early.
    Global,
    /// Each particle follows the best position among itself and its `neighbors` nearest particles
    /// on each side of a ring, by index (lbest). Good positions spread slowly, which keeps the
    /// swarm exploring longer and suits multimodal problems.
    Ring {
        /// The number of neighbors on each side, at least 1; 1 is common.
        neighbors: usize,
    },
}

/// Particle swarm optimization on [`Real`] genomes, as an ask / tell [`Algorithm`].
///
/// The population is the swarm: each particle has a position (its genome), a velocity, and the
/// best position it has found (its personal best). Every generation, each particle's velocity
/// changes gene by gene to
///
/// `v = w · v + c1 · r1 · (personal best − x) + c2 · r2 · (neighborhood best − x)`
///
/// for new random `r1` and `r2` in `0..1`, is limited to a fraction of the gene's range, and moves
/// the particle: `x = x + v`. A particle that would leave the bounds stops at the bound, and that
/// velocity component becomes 0. The neighborhood best comes from the [`Topology`], and a
/// personal best is replaced by a position that is not worse.
///
/// The defaults are Clerc and Kennedy's constriction coefficients: inertia `w` 0.7298 and
/// accelerations `c1` = `c2` = 1.49618.
///
/// Built with [`Pso::builder`], run with an [`Engine`](crate::Engine).
///
/// ```
/// use genoxide::prelude::*;
///
/// let pso = Pso::builder(Real::uniform(10, -5.0..=5.0)?)
///     .population_size(40)
///     .minimize()
///     .seed(1)
///     .build()?;
/// let sphere = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
/// let outcome = Engine::new(pso, sphere)
///     .stop_when(Stop::target(1e-8).or(Stop::evaluations(200_000)))
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Target);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pso {
    real: Real,
    topology: Topology,
    inertia: f64,
    cognitive: f64,
    social: f64,
    max_velocity: f64,
    objective: Objective,
    seed: u64,
    rng: StreamRng,
    population: Population<Reals>,
    velocities: Vec<Vec<f64>>,
    personal_bests: Vec<Individual<Reals>>,
    started: bool,
    asked: bool,
    pending: Vec<usize>,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<Reals>>,
    best_generation: u64,
}

impl Pso {
    /// A builder for a particle swarm on `real`.
    pub fn builder(real: Real) -> PsoBuilder {
        PsoBuilder {
            real,
            population_size: None,
            topology: Topology::Global,
            inertia: 0.7298,
            acceleration: (1.49618, 1.49618),
            max_velocity: 1.0,
            objective: Objective::default(),
            seed: None,
            initial_genomes: Vec::new(),
        }
    }

    /// The representation.
    pub fn real(&self) -> &Real {
        &self.real
    }

    /// Which particles each particle learns from.
    pub fn topology(&self) -> Topology {
        self.topology
    }

    /// The velocities of the particles, in the order of the population.
    pub fn velocities(&self) -> &[Vec<f64>] {
        &self.velocities
    }

    /// The best position each particle has found, with its fitness, in the order of the
    /// population. Empty before the first [`tell`](Algorithm::tell).
    pub fn personal_bests(&self) -> &[Individual<Reals>] {
        &self.personal_bests
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn personal_best(&self, index: usize) -> Fitness {
        self.personal_bests[index]
            .fitness()
            .unwrap_or(Fitness::invalid())
    }

    // the index of the best personal best among the particles `index - k..=index + k` on the ring,
    // or of the whole swarm; the lowest index on ties
    fn neighborhood_best(&self, index: usize, global: usize) -> usize {
        let Topology::Ring { neighbors } = self.topology else {
            return global;
        };
        let size = self.personal_bests.len();
        // 2 · neighbors + 1 >= size, without overflow
        if neighbors >= size / 2 {
            return global;
        }
        let mut best = index;
        for offset in 1..=neighbors {
            for other in [(index + offset) % size, (index + size - offset) % size] {
                let (candidate, current) = (self.personal_best(other), self.personal_best(best));
                if self.objective.is_better(candidate, current)
                    || (other < best && !self.objective.is_better(current, candidate))
                {
                    best = other;
                }
            }
        }
        best
    }

    // moves every particle
    fn fly(&mut self) {
        let objective = self.objective;
        let size = self.population.len();
        let mut global = 0;
        for index in 1..size {
            if objective.is_better(self.personal_best(index), self.personal_best(global)) {
                global = index;
            }
        }
        let guides: Vec<usize> = (0..size)
            .map(|index| self.neighborhood_best(index, global))
            .collect();
        let bounds = self.real.bounds();
        for (index, &guide) in guides.iter().enumerate() {
            let mut position = self.population[index].genome().clone();
            let personal = self.personal_bests[index].genome();
            let neighborhood = self.personal_bests[guide].genome();
            let velocity = &mut self.velocities[index];
            for j in 0..position.len() {
                let (start, end) = (*bounds[j].start(), *bounds[j].end());
                let limit = self.max_velocity * (end - start);
                let (r1, r2) = (self.rng.unit_f64(), self.rng.unit_f64());
                let mut v = self.inertia * velocity[j]
                    + self.cognitive * r1 * (personal[j] - position[j])
                    + self.social * r2 * (neighborhood[j] - position[j]);
                // NaN from overflowing terms of opposite signs
                if v.is_nan() {
                    v = 0.0;
                }
                v = v.clamp(-limit, limit);
                let x = position[j] + v;
                if x < start {
                    position[j] = start;
                    v = 0.0;
                } else if x > end {
                    position[j] = end;
                    v = 0.0;
                } else {
                    position[j] = x;
                }
                velocity[j] = v;
            }
            self.population[index] = Individual::new(position);
        }
    }
}

impl Algorithm for Pso {
    type Genome = Reals;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, Reals> {
        if !self.asked {
            if self.started {
                self.fly();
            }
            self.pending.clear();
            self.pending.extend(0..self.population.len());
            self.asked = true;
        }
        Candidates::new(self.population.as_slice(), &self.pending)
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
        self.asked = false;
        self.evaluations += fitness.len() as u64;
        if self.started {
            self.generation += 1;
        }
        let objective = self.objective;
        for (individual, &fitness) in self.population.iter_mut().zip(fitness) {
            individual.set_fitness(fitness);
        }
        let mut improved = false;
        for (index, particle) in self.population.iter().enumerate() {
            let fitness = particle.fitness().unwrap_or(Fitness::invalid());
            if !self.started {
                self.personal_bests.push(particle.clone());
            } else if !objective.is_better(self.personal_best(index), fitness) {
                self.personal_bests[index] = particle.clone();
            }
            // the best so far, the first one on ties
            let better = self.best.as_ref().is_none_or(|best| {
                objective.is_better(fitness, best.fitness().unwrap_or(Fitness::invalid()))
            });
            if better {
                self.best = Some(particle.clone());
                improved = true;
            }
        }
        if improved {
            self.best_generation = self.generation;
        }
        self.started = true;
        Ok(())
    }

    fn population(&self) -> &Population<Reals> {
        &self.population
    }

    fn best(&self) -> Option<&Individual<Reals>> {
        self.best.as_ref()
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

/// A builder for a [`Pso`], from [`Pso::builder`].
///
/// The population size is required. Defaults: the global topology, inertia 0.7298, accelerations
/// 1.49618, velocities up to the whole range of each gene, maximize, a random initial swarm and a
/// random seed.
#[derive(Clone, Debug)]
pub struct PsoBuilder {
    real: Real,
    population_size: Option<usize>,
    topology: Topology,
    inertia: f64,
    acceleration: (f64, f64),
    max_velocity: f64,
    objective: Objective,
    seed: Option<u64>,
    initial_genomes: Vec<Reals>,
}

impl PsoBuilder {
    /// The number of particles, at least 2 and at most 2^24; 20 to 50 is common. Required.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// Which particles each particle learns from. [`Topology::Global`] by default.
    pub fn topology(mut self, topology: Topology) -> Self {
        self.topology = topology;
        self
    }

    /// How much of its velocity a particle keeps, `w`: 0 or more and finite, usually below 1.
    /// 0.7298 by default.
    pub fn inertia(mut self, inertia: f64) -> Self {
        self.inertia = inertia;
        self
    }

    /// How strongly a particle is pulled toward its personal best (`c1`, cognitive) and its
    /// neighborhood best (`c2`, social): each 0 or more and finite. 1.49618 each by default.
    pub fn acceleration(mut self, cognitive: f64, social: f64) -> Self {
        self.acceleration = (cognitive, social);
        self
    }

    /// The largest velocity of a gene, as a fraction of its range: greater than 0 and finite, e.g.
    /// 0.1 to 1. 1 by default. It limits the initial velocities too, which are at most half the
    /// range.
    pub fn max_velocity(mut self, fraction: f64) -> Self {
        self.max_velocity = fraction;
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

    /// The seed of the random numbers, for a reproducible run. Random by default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Positions for the initial swarm, at most the population size, each valid for the
    /// representation. The rest is random.
    pub fn initial_genomes<I: IntoIterator<Item = Reals>>(mut self, genomes: I) -> Self {
        self.initial_genomes = genomes.into_iter().collect();
        self
    }

    /// Validates the settings and creates the algorithm, with its initial swarm. Each initial
    /// velocity is half the way from the particle to a random position, limited by
    /// [`max_velocity`](PsoBuilder::max_velocity).
    ///
    /// # Errors
    ///
    /// - [`Error::MissingSetting`] without a population size.
    /// - [`Error::InvalidSetting`] for a population size below 2 or above 2^24, a ring without
    ///   neighbors, an inertia, acceleration or maximum velocity out of range, or more initial
    ///   genomes than the population size.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<Pso> {
        let size = self.population_size.ok_or(Error::MissingSetting {
            setting: "population_size",
        })?;
        let invalid = |setting, reason: String| Err(Error::InvalidSetting { setting, reason });
        if size < 2 {
            return invalid(
                "population_size",
                format!("a swarm needs at least 2 particles, got {size}"),
            );
        }
        check_size("population_size", size)?;
        if let Topology::Ring { neighbors: 0 } = self.topology {
            return invalid("topology", "a ring needs at least 1 neighbor".to_string());
        }
        let (cognitive, social) = self.acceleration;
        if !(self.inertia >= 0.0 && self.inertia.is_finite()) {
            return invalid(
                "inertia",
                format!("must be 0 or more and finite, got {}", self.inertia),
            );
        }
        if !(cognitive >= 0.0 && cognitive.is_finite() && social >= 0.0 && social.is_finite()) {
            return invalid(
                "acceleration",
                format!("must be 0 or more and finite, got {cognitive} and {social}"),
            );
        }
        if !(self.max_velocity > 0.0 && self.max_velocity.is_finite()) {
            return invalid(
                "max_velocity",
                format!(
                    "must be greater than 0 and finite, got {}",
                    self.max_velocity
                ),
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
            self.real.validate(genome)?;
        }
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let mut rng = StreamRng::seed_from_u64(seed);
        let random = size - self.initial_genomes.len();
        let mut genomes = self.initial_genomes;
        genomes.extend((0..random).map(|_| self.real.random_genome(&mut rng)));
        let bounds = self.real.bounds();
        let velocities = genomes
            .iter()
            .map(|genome| {
                let target = self.real.random_genome(&mut rng);
                genome
                    .iter()
                    .zip(target.iter())
                    .zip(bounds)
                    .map(|((x, y), range)| {
                        let limit = self.max_velocity * (range.end() - range.start());
                        (y / 2.0 - x / 2.0).clamp(-limit, limit)
                    })
                    .collect()
            })
            .collect();
        Ok(Pso {
            real: self.real,
            topology: self.topology,
            inertia: self.inertia,
            cognitive,
            social,
            max_velocity: self.max_velocity,
            objective: self.objective,
            seed,
            rng,
            population: Population::from_genomes(genomes),
            velocities,
            personal_bests: Vec::new(),
            started: false,
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
    use crate::{Engine, Stop, StopReason};
    use proptest::prelude::*;

    fn sphere(x: &Reals) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    #[test]
    fn initial_velocities_respect_max_velocity() {
        let real = Real::new([0.0..=1.0, -10.0..=10.0, 3.0..=3.0]).unwrap();
        let swarm = |max_velocity| {
            Pso::builder(real.clone())
                .population_size(10)
                .max_velocity(max_velocity)
                .seed(0)
                .build()
                .unwrap()
        };
        for max_velocity in [0.01, 0.3, 0.5, 1.0] {
            for velocity in swarm(max_velocity).velocities() {
                for (v, bounds) in velocity.iter().zip(real.bounds()) {
                    let limit = max_velocity * (bounds.end() - bounds.start());
                    assert!(v.abs() <= limit, "{v} with the limit {limit}");
                }
            }
        }
        // half the way to a random position is never faster than half the range: from 0.5 up,
        // the limit changes nothing
        assert_eq!(swarm(0.5).velocities(), swarm(1.0).velocities());
        assert_ne!(swarm(0.1).velocities(), swarm(1.0).velocities());
    }

    fn builder(topology: Topology, seed: u64) -> PsoBuilder {
        Pso::builder(Real::uniform(8, -5.0..=5.0).unwrap())
            .population_size(20)
            .topology(topology)
            .minimize()
            .seed(seed)
    }

    fn step(pso: &mut Pso) {
        let fitness: Vec<Fitness> = pso.ask().iter().map(|x| Fitness::new(sphere(x))).collect();
        pso.tell(&fitness).unwrap();
    }

    fn setting(result: Result<Pso>) -> &'static str {
        match result {
            Err(Error::InvalidSetting { setting, .. } | Error::MissingSetting { setting }) => {
                setting
            }
            other => panic!("expected a setting error, got {other:?}"),
        }
    }

    #[test]
    fn validation() {
        let real = || Real::uniform(2, 0.0..=1.0).unwrap();
        let sized = || Pso::builder(real()).population_size(10);
        assert_eq!(setting(Pso::builder(real()).build()), "population_size");
        assert_eq!(
            setting(Pso::builder(real()).population_size(1).build()),
            "population_size"
        );
        let ring = |neighbors| sized().topology(Topology::Ring { neighbors }).build();
        assert_eq!(setting(ring(0)), "topology");
        assert_eq!(setting(sized().inertia(-0.1).build()), "inertia");
        assert_eq!(setting(sized().inertia(f64::INFINITY).build()), "inertia");
        assert_eq!(
            setting(sized().acceleration(-1.0, 1.0).build()),
            "acceleration"
        );
        assert_eq!(
            setting(sized().acceleration(1.0, f64::NAN).build()),
            "acceleration"
        );
        assert_eq!(setting(sized().max_velocity(0.0).build()), "max_velocity");
        assert!(ring(20).is_ok());
        assert!(sized().inertia(0.0).acceleration(0.0, 0.0).build().is_ok());
        assert!(matches!(
            sized().initial_genomes([Reals::from(vec![0.5])]).build(),
            Err(Error::InvalidGenome { .. })
        ));
    }

    #[test]
    fn ask_tell_protocol() {
        let mut pso = builder(Topology::Global, 0).build().unwrap();
        assert_eq!(pso.tell(&[]), Err(Error::TellWithoutAsk));
        assert!(pso.personal_bests().is_empty());
        let first: Vec<Reals> = pso.ask().iter().cloned().collect();
        assert_eq!(first.len(), 20);
        // asking again gives the same particles
        assert_eq!(pso.ask().iter().cloned().collect::<Vec<_>>(), first);
        assert_eq!(
            pso.tell(&[Fitness::new(0.0)]),
            Err(Error::FitnessCount {
                expected: 20,
                got: 1
            })
        );
        step(&mut pso);
        assert_eq!((pso.generation(), pso.evaluations()), (0, 20));
        assert_eq!(pso.personal_bests().len(), 20);
        step(&mut pso);
        assert_eq!((pso.generation(), pso.evaluations()), (1, 40));
        assert!(pso.discarded().is_empty());
    }

    #[test]
    fn ring_neighborhoods() {
        let mut pso = builder(Topology::Ring { neighbors: 1 }, 0).build().unwrap();
        // personal bests with fitness 19, 18, ..., 0: the last particle is the best
        let fitness: Vec<Fitness> = (0..20).map(|i| Fitness::new(19.0 - i as f64)).collect();
        pso.ask();
        pso.tell(&fitness).unwrap();
        assert_eq!(pso.neighborhood_best(0, 19), 19);
        assert_eq!(pso.neighborhood_best(5, 19), 6);
        assert_eq!(pso.neighborhood_best(19, 19), 19);
        assert_eq!(pso.neighborhood_best(18, 19), 19);
        // ties go to the lowest index
        let mut pso = builder(Topology::Ring { neighbors: 2 }, 0).build().unwrap();
        pso.ask();
        pso.tell(&[Fitness::new(1.0); 20]).unwrap();
        assert_eq!(pso.neighborhood_best(0, 0), 0);
        assert_eq!(pso.neighborhood_best(5, 0), 3);
        assert_eq!(pso.neighborhood_best(19, 0), 0);
        // a ring that covers the whole swarm is the global topology, however large
        for neighbors in [10, usize::MAX] {
            let mut pso = builder(Topology::Ring { neighbors }, 0).build().unwrap();
            pso.ask();
            pso.tell(&fitness).unwrap();
            assert_eq!(pso.neighborhood_best(0, 19), 19);
        }
        // 9 neighbors on each side leave out only the particle opposite
        let mut pso = builder(Topology::Ring { neighbors: 9 }, 0).build().unwrap();
        pso.ask();
        pso.tell(&fitness).unwrap();
        assert_eq!(pso.neighborhood_best(10, 19), 19);
        assert_eq!(pso.neighborhood_best(9, 19), 18);
    }

    #[test]
    fn each_topology_solves_the_sphere() {
        for topology in [Topology::Global, Topology::Ring { neighbors: 1 }] {
            let pso = builder(topology, 1).population_size(30).build().unwrap();
            let outcome = Engine::new(pso, sphere)
                .stop_when(Stop::target(1e-8).or(Stop::evaluations(200_000)))
                .run()
                .unwrap();
            assert_eq!(outcome.stop_reason(), StopReason::Target, "{topology:?}");
        }
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut pso = builder(Topology::Ring { neighbors: 1 }, seed)
                .build()
                .unwrap();
            for _ in 0..20 {
                step(&mut pso);
            }
            pso.population().clone()
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3), run(4));
    }

    proptest! {
        #[test]
        fn particles_stay_in_bounds_and_personal_bests_never_get_worse(
            ring: bool,
            inertia in 0.0..=1.5f64,
            cognitive in 0.0..=4.0f64,
            social in 0.0..=4.0f64,
            max_velocity in 0.01..=2.0f64,
            seed: u64,
        ) {
            // one fixed gene, bounds of different widths, and one whose velocity terms overflow
            let real =
                Real::new([3.0..=3.0, -1.0..=1.0, 0.0..=100.0, -1e-3..=1e-3, -8e307..=8e307])
                    .unwrap();
            let topology = if ring { Topology::Ring { neighbors: 1 } } else { Topology::Global };
            let mut pso = Pso::builder(real.clone())
                .population_size(8)
                .topology(topology)
                .inertia(inertia)
                .acceleration(cognitive, social)
                .max_velocity(max_velocity)
                .minimize()
                .seed(seed)
                .build()
                .unwrap();
            step(&mut pso);
            for _ in 0..10 {
                let before: Vec<Fitness> =
                    pso.personal_bests().iter().map(|x| x.fitness().unwrap()).collect();
                let positions: Vec<Reals> = pso.ask().iter().cloned().collect();
                for position in &positions {
                    prop_assert!(real.validate(position).is_ok(), "{position:?}");
                }
                for velocity in pso.velocities() {
                    for (v, bounds) in velocity.iter().zip(real.bounds()) {
                        prop_assert!(v.abs() <= max_velocity * (bounds.end() - bounds.start()));
                    }
                }
                step(&mut pso);
                for (index, best) in pso.personal_bests().iter().enumerate() {
                    prop_assert!(!Objective::Minimize.is_better(before[index], best.fitness().unwrap()));
                }
            }
        }
    }
}
