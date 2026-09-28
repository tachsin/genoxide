//! Particle swarm optimization: particles that fly toward the best places they and their
//! neighbors have found.

use super::{Algorithm, Candidates, Reevaluate};
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
    // a re-evaluation for the next ask, and the genomes it asks for: the positions, then the
    // personal bests at other positions
    reevaluating: bool,
    rescored: Vec<Individual<Reals>>,
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

    /// How much of its velocity a particle keeps, `w`.
    pub fn inertia(&self) -> f64 {
        self.inertia
    }

    /// How strongly a particle is pulled toward its personal best (`c1`, cognitive) and its
    /// neighborhood best (`c2`, social).
    pub fn acceleration(&self) -> (f64, f64) {
        (self.cognitive, self.social)
    }

    /// Changes the inertia `w` during a run (parameter control), e.g. the classic inertia that
    /// decreases linearly from 0.9 to 0.4 over the run (Shi and Eberhart, 1998): a swarm that
    /// explores first and converges later. It applies from the next [`ask`](Algorithm::ask) that
    /// moves the swarm.
    ///
    /// ```
    /// use genoxide::prelude::*;
    ///
    /// let pso = Pso::builder(Real::uniform(10, -5.0..=5.0)?)
    ///     .population_size(30)
    ///     .inertia(0.9)
    ///     .acceleration(2.0, 2.0)
    ///     .max_velocity(0.2)
    ///     .minimize()
    ///     .seed(1)
    ///     .build()?;
    /// let sphere = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
    /// let generations = 500;
    /// let outcome = Engine::new(pso, sphere)
    ///     .stop_when(Stop::generations(generations))
    ///     // from 0.9 to 0.4 over the run
    ///     .control(|pso, progress| {
    ///         let done = progress.generation() as f64 / generations as f64;
    ///         pso.set_inertia(0.9 - 0.5 * done)
    ///     })
    ///     .run()?;
    /// assert!(outcome.best_fitness().score().unwrap() < 1e-6);
    /// # Ok::<(), genoxide::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] as for [`PsoBuilder::inertia`]: an inertia below 0 or not
    /// finite. The inertia doesn't change on errors.
    pub fn set_inertia(&mut self, inertia: f64) -> Result<()> {
        check_inertia(inertia)?;
        self.inertia = inertia;
        Ok(())
    }

    /// Changes the accelerations toward the personal best (`c1`, cognitive) and the neighborhood
    /// best (`c2`, social) during a run, e.g. from a large `c1` and a small `c2` to the reverse
    /// (Ratnaweera, Halgamuge and Watson, 2004). As [`set_inertia`](Pso::set_inertia), it applies
    /// from the next ask that moves the swarm.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] as for [`PsoBuilder::acceleration`]: an acceleration below 0 or
    /// not finite. The accelerations don't change on errors.
    pub fn set_acceleration(&mut self, cognitive: f64, social: f64) -> Result<()> {
        check_acceleration(cognitive, social)?;
        (self.cognitive, self.social) = (cognitive, social);
        Ok(())
    }

    /// Marks the swarm for evaluation again, for a fitness function that changed during the run:
    /// the next [`ask`](Algorithm::ask) gives the particles' positions, then the personal bests
    /// at other positions (a personal best at its particle's position is asked once), and its
    /// [`tell`](Algorithm::tell) sets their fitness without moving the swarm.
    ///
    /// - It isn't a generation: [`generation`](Algorithm::generation) doesn't change. The
    ///   evaluations are counted.
    /// - Each personal best is then the better of the two, the position on ties, as after a move,
    ///   and the neighborhood bests follow from them. [`best`](Algorithm::best) is the best
    ///   personal best, found in the current generation: old and new values are never compared.
    /// - The velocities don't change, and no random number is drawn: a seeded run that
    ///   re-evaluates at the same points gives the same results.
    /// - Before the first tell nothing is evaluated yet, and it changes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::ReevaluationOutOfTurn`] between an ask and its tell. Nothing changes on errors.
    pub fn reevaluate(&mut self) -> Result<()> {
        if self.asked {
            return Err(Error::ReevaluationOutOfTurn);
        }
        self.reevaluating = self.started;
        Ok(())
    }

    // sets the fitness of the positions and the personal bests of a re-evaluation, in the order
    // asked, and the best from them
    fn rescore(&mut self, fitness: &[Fitness]) {
        let objective = self.objective;
        let size = self.population.len();
        let mut elsewhere = fitness[size..].iter();
        for (index, &at_position) in fitness[..size].iter().enumerate() {
            let particle = &mut self.population[index];
            particle.set_fitness(at_position);
            let personal = &mut self.personal_bests[index];
            if personal.genome() == particle.genome() {
                personal.set_fitness(at_position);
            } else {
                let &value = elsewhere
                    .next()
                    .expect("asked for each personal best elsewhere");
                personal.set_fitness(value);
                if !objective.is_better(value, at_position) {
                    *personal = particle.clone();
                }
            }
        }
        // the best of the new values, the first one on ties
        self.best = None;
        for index in 0..size {
            let fitness = self.personal_best(index);
            let better = self.best.as_ref().is_none_or(|best| {
                objective.is_better(fitness, best.fitness().unwrap_or(Fitness::invalid()))
            });
            if better {
                self.best = Some(self.personal_bests[index].clone());
            }
        }
        self.best_generation = self.generation;
        self.rescored.clear();
        self.reevaluating = false;
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

impl Reevaluate for Pso {
    /// As [`Pso::reevaluate`]: the next ask gives the positions and the personal bests.
    fn reevaluate(&mut self) -> Result<()> {
        Pso::reevaluate(self)
    }
}

impl Algorithm for Pso {
    type Genome = Reals;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, Reals> {
        if !self.asked {
            self.pending.clear();
            if self.reevaluating {
                let elsewhere = self
                    .personal_bests
                    .iter()
                    .zip(self.population.iter())
                    .filter(|(personal, particle)| personal.genome() != particle.genome())
                    .map(|(personal, _)| personal);
                self.rescored.clear();
                self.rescored.extend(self.population.iter().cloned());
                self.rescored.extend(elsewhere.cloned());
                self.pending.extend(0..self.rescored.len());
            } else {
                if self.started {
                    self.fly();
                }
                self.pending.extend(0..self.population.len());
            }
            self.asked = true;
        }
        let individuals = if self.reevaluating {
            &self.rescored
        } else {
            self.population.as_slice()
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
        self.asked = false;
        self.evaluations += fitness.len() as u64;
        if self.reevaluating {
            self.rescore(fitness);
            return Ok(());
        }
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
        check_inertia(self.inertia)?;
        check_acceleration(cognitive, social)?;
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
            reevaluating: false,
            rescored: Vec::new(),
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

fn check_inertia(inertia: f64) -> Result<()> {
    if !(inertia >= 0.0 && inertia.is_finite()) {
        return Err(Error::InvalidSetting {
            setting: "inertia",
            reason: format!("must be 0 or more and finite, got {inertia}"),
        });
    }
    Ok(())
}

fn check_acceleration(cognitive: f64, social: f64) -> Result<()> {
    if !(cognitive >= 0.0 && cognitive.is_finite() && social >= 0.0 && social.is_finite()) {
        return Err(Error::InvalidSetting {
            setting: "acceleration",
            reason: format!("must be 0 or more and finite, got {cognitive} and {social}"),
        });
    }
    Ok(())
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

    // the sphere around 1 instead of 0
    fn shifted(x: &Reals) -> f64 {
        x.iter().map(|xi| (xi - 1.0) * (xi - 1.0)).sum()
    }

    #[test]
    fn a_reevaluation_scores_positions_and_personal_bests_again() {
        let mut pso = builder(Topology::Ring { neighbors: 1 }, 0).build().unwrap();
        for _ in 0..5 {
            step(&mut pso);
        }
        let positions: Vec<Reals> = pso
            .population()
            .iter()
            .map(|x| x.genome().clone())
            .collect();
        let personal: Vec<Reals> = pso
            .personal_bests()
            .iter()
            .map(|x| x.genome().clone())
            .collect();
        let velocities = pso.velocities().to_vec();
        let (generation, evaluations) = (pso.generation(), pso.evaluations());

        pso.reevaluate().unwrap();
        let asked: Vec<Reals> = pso.ask().iter().cloned().collect();
        // the positions, then the personal bests elsewhere
        let moved: Vec<Reals> = personal
            .iter()
            .zip(&positions)
            .filter(|(personal, position)| personal != position)
            .map(|(personal, _)| personal.clone())
            .collect();
        assert!(!moved.is_empty() && moved.len() < 20);
        assert_eq!(asked, [positions.clone(), moved].concat());
        let fitness: Vec<Fitness> = asked.iter().map(|x| Fitness::new(shifted(x))).collect();
        pso.tell(&fitness).unwrap();

        assert_eq!(pso.generation(), generation);
        assert_eq!(pso.evaluations(), evaluations + asked.len() as u64);
        assert_eq!(pso.velocities(), velocities);
        for (index, particle) in pso.population().iter().enumerate() {
            assert_eq!(particle.genome(), &positions[index]);
            assert_eq!(
                particle.fitness(),
                Some(Fitness::new(shifted(&positions[index])))
            );
            // the better of the old personal best and the position, the position on ties
            let (old, here) = (shifted(&personal[index]), shifted(&positions[index]));
            let expected = if here <= old {
                &positions[index]
            } else {
                &personal[index]
            };
            let best = &pso.personal_bests()[index];
            assert_eq!(best.genome(), expected);
            assert_eq!(best.fitness(), Some(Fitness::new(old.min(here))));
        }
        let lowest = pso
            .personal_bests()
            .iter()
            .map(|x| shifted(x.genome()))
            .fold(f64::INFINITY, f64::min);
        assert_eq!(pso.best().unwrap().fitness(), Some(Fitness::new(lowest)));
        assert_eq!(pso.best_generation(), generation);
        // and the swarm moves on
        assert_eq!(pso.ask().len(), 20);
        step(&mut pso);
        assert_eq!(pso.generation(), generation + 1);
    }

    #[test]
    fn a_reevaluation_waits_for_the_tell() {
        let mut pso = builder(Topology::Global, 1).build().unwrap();
        // before anything is evaluated, it changes nothing
        pso.reevaluate().unwrap();
        step(&mut pso);
        let mut other = builder(Topology::Global, 1).build().unwrap();
        step(&mut other);
        assert_eq!(pso.population(), other.population());
        step(&mut pso);

        pso.ask();
        assert_eq!(pso.reevaluate(), Err(Error::ReevaluationOutOfTurn));
        // still the moved swarm
        assert_eq!(pso.ask().len(), 20);
        step(&mut pso);
        assert_eq!(pso.generation(), 2);
    }

    #[test]
    fn a_reevaluation_with_the_same_function_changes_only_the_count() {
        let run = |reevaluate: bool| {
            let mut pso = builder(Topology::Global, 2).build().unwrap();
            let mut extra = 0;
            for generation in 0..20 {
                if reevaluate && generation == 10 {
                    pso.reevaluate().unwrap();
                    extra = pso.ask().len() as u64;
                    step(&mut pso);
                }
                step(&mut pso);
            }
            (
                pso.population().clone(),
                pso.personal_bests().to_vec(),
                pso.velocities().to_vec(),
                pso.best().unwrap().fitness(),
                pso.evaluations() - extra,
            )
        };
        assert_eq!(run(true), run(false));
    }

    #[test]
    fn reevaluations_repeat_with_a_seed() {
        let run = || {
            let mut pso = builder(Topology::Ring { neighbors: 1 }, 3).build().unwrap();
            for generation in 0..30 {
                if generation == 10 {
                    pso.reevaluate().unwrap();
                }
                let f = if generation < 10 { sphere } else { shifted };
                let fitness: Vec<Fitness> = pso.ask().iter().map(|x| Fitness::new(f(x))).collect();
                pso.tell(&fitness).unwrap();
            }
            (pso.population().clone(), pso.best().cloned())
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn setters_validate_like_the_builder() {
        let mut pso = builder(Topology::Global, 0).build().unwrap();
        for inertia in [-0.1, f64::INFINITY, f64::NAN] {
            assert!(matches!(
                pso.set_inertia(inertia),
                Err(Error::InvalidSetting {
                    setting: "inertia",
                    ..
                })
            ));
        }
        for (cognitive, social) in [(-1.0, 1.0), (1.0, f64::NAN), (f64::INFINITY, 0.0)] {
            assert!(matches!(
                pso.set_acceleration(cognitive, social),
                Err(Error::InvalidSetting {
                    setting: "acceleration",
                    ..
                })
            ));
        }
        // unchanged on errors
        assert_eq!(pso.inertia(), 0.7298);
        assert_eq!(pso.acceleration(), (1.49618, 1.49618));
        pso.set_inertia(0.0).unwrap();
        pso.set_acceleration(0.0, 0.0).unwrap();
        assert_eq!((pso.inertia(), pso.acceleration()), (0.0, (0.0, 0.0)));
    }

    #[test]
    fn a_setting_changed_before_a_move_gives_the_run_built_with_it() {
        let run = |mut pso: Pso, change: fn(&mut Pso)| {
            step(&mut pso);
            change(&mut pso);
            for _ in 0..20 {
                step(&mut pso);
            }
            pso.population().clone()
        };
        let built = builder(Topology::Global, 4)
            .inertia(0.4)
            .acceleration(2.0, 1.0)
            .build()
            .unwrap();
        let changed = |pso: &mut Pso| {
            pso.set_inertia(0.4).unwrap();
            pso.set_acceleration(2.0, 1.0).unwrap();
        };
        let expected = run(built, |_| {});
        let default = || builder(Topology::Global, 4).build().unwrap();
        assert_eq!(run(default(), changed), expected);
        assert_ne!(run(default(), |_| {}), expected);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn checkpoints_keep_changed_settings_and_a_due_reevaluation() {
        use crate::checkpoint;
        let mut pso = builder(Topology::Global, 5).build().unwrap();
        step(&mut pso);
        pso.set_inertia(0.5).unwrap();
        pso.set_acceleration(1.0, 2.0).unwrap();
        pso.reevaluate().unwrap();
        let mut bytes = Vec::new();
        checkpoint::save(&pso, &mut bytes).unwrap();
        let mut loaded: Pso = checkpoint::load(bytes.as_slice()).unwrap();
        assert_eq!((loaded.inertia(), loaded.acceleration()), (0.5, (1.0, 2.0)));
        for _ in 0..5 {
            step(&mut pso);
            step(&mut loaded);
        }
        assert_eq!(loaded.population(), pso.population());
        assert_eq!(loaded.evaluations(), pso.evaluations());
    }

    #[test]
    fn an_engine_control_reevaluates_once() {
        use std::sync::atomic::{AtomicU64, Ordering};
        // the center of the sphere, moved from 0 to 1 after generation 20
        let center = AtomicU64::new(0.0_f64.to_bits());
        let moving = |x: &Reals| {
            let center = f64::from_bits(center.load(Ordering::Relaxed));
            x.iter()
                .map(|xi| (xi - center) * (xi - center))
                .sum::<f64>()
        };
        let mut controls = 0;
        let mut engine = Engine::new(builder(Topology::Global, 6).build().unwrap(), &moving)
            .stop_when(Stop::generations(60))
            .control(|pso: &mut Pso, progress| {
                controls += 1;
                if progress.generation() == 20 {
                    center.store(1.0_f64.to_bits(), Ordering::Relaxed);
                    pso.reevaluate()?;
                }
                Ok(())
            });
        let outcome = engine.run().unwrap();
        drop(engine);
        // once per generation, not after the re-evaluation
        assert_eq!(controls, 61);
        assert_eq!(outcome.generations(), 60);
        assert!(outcome.evaluations() > 61 * 20);
        // the best by the new function
        let best = outcome.best_fitness().score().unwrap();
        assert_eq!(best, shifted(outcome.best_genome()));
        assert!(best < 0.1, "{best}");
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
