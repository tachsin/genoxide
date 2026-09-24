//! Differential evolution: new solutions from the differences between solutions.

use super::{Algorithm, Candidates};
use crate::genome::{Real, Reals, Representation};
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;

/// How a differential evolution builds the mutant vector `v` for an individual `x`, from random
/// distinct other individuals `r1`, `r2` (and `r3`) and the scale factor `F`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Strategy {
    /// DE/rand/1: `v = r1 + F · (r2 − r3)`. The classic: robust, explores well.
    Rand1,
    /// DE/best/1: `v = best + F · (r1 − r2)`. Converges fast, but with a fixed `F` the population
    /// can collapse onto one point before the optimum: use it with [`Control::Dither`].
    Best1,
    /// DE/current-to-pbest/1 (JADE): `v = x + F · (pbest − x) + F · (r1 − r2)`, where `pbest` is
    /// a random one of the best `p` fraction of the population, and `r2` can also come from an
    /// archive of recently replaced individuals. Greedy but diverse; the basis of JADE and SHADE.
    /// Without an archive and with a fixed `F`, it can collapse like [`Strategy::Best1`].
    CurrentToPBest {
        /// The fraction of the population that `pbest` is chosen from, greater than 0 and at
        /// most 1, e.g. 0.05 to 0.2. At least the best individual is always a candidate.
        p: f64,
        /// The size of the archive as a multiple of the population size, 0 for no archive; e.g.
        /// 1 (JADE) or 2.6 (L-SHADE). When full, a random member makes room.
        archive: f64,
    },
}

/// Where the scale factor `F` and crossover rate `CR` of a differential evolution come from.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Control {
    /// The same `F` and `CR` for every trial.
    Fixed {
        /// The scale factor of the differences, greater than 0 and at most 2; 0.5 is common.
        f: f64,
        /// The probability that a gene comes from the mutant vector, between 0 and 1; 0.9 is
        /// common, and small values suit separable problems.
        cr: f64,
    },
    /// A random `F` for every trial, uniform in `min_f..max_f` ("dither"), and a fixed `CR`. The
    /// random step sizes keep the population from collapsing onto one point, which greedy
    /// strategies like [`Strategy::Best1`] otherwise can with a fixed `F`.
    Dither {
        /// The smallest scale factor, greater than 0.
        min_f: f64,
        /// The largest scale factor, at least `min_f` and at most 2; e.g. 0.5 to 1.
        max_f: f64,
        /// The probability that a gene comes from the mutant vector, between 0 and 1.
        cr: f64,
    },
}

impl Default for Control {
    /// `F` 0.5 and `CR` 0.9.
    fn default() -> Self {
        Control::Fixed { f: 0.5, cr: 0.9 }
    }
}

/// Differential evolution on [`Real`] genomes, as an ask / tell [`Algorithm`].
///
/// Every generation, each individual `x` gets a trial vector: a mutant vector from the
/// [`Strategy`], crossed with `x` by binomial crossover (each gene comes from the mutant with
/// probability `CR`, and at least one does). A trial gene outside the bounds is set halfway
/// between `x`'s gene and the bound ("bounce-back"). The trial replaces `x` if it's not worse.
///
/// Built with [`De::builder`], run with an [`Engine`](crate::Engine).
///
/// ```
/// use genoxide::prelude::*;
///
/// let de = De::builder(Real::uniform(10, -5.0..=5.0)?)
///     .population_size(40)
///     .strategy(de::Strategy::Rand1)
///     .minimize()
///     .seed(1)
///     .build()?;
/// let sphere = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
/// let outcome = Engine::new(de, sphere)
///     .stop_when(Stop::target(1e-8).or(Stop::evaluations(200_000)))
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Target);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct De {
    real: Real,
    strategy: Strategy,
    control: Control,
    objective: Objective,
    seed: u64,
    rng: StreamRng,
    population: Population<Reals>,
    trials: Vec<Individual<Reals>>,
    archive: Vec<Reals>,
    discarded: Vec<Individual<Reals>>,
    started: bool,
    asked: bool,
    pending: Vec<usize>,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<Reals>>,
    best_generation: u64,
}

impl De {
    /// A builder for a differential evolution on `real`.
    pub fn builder(real: Real) -> DeBuilder {
        DeBuilder {
            real,
            population_size: None,
            strategy: Strategy::Rand1,
            control: Control::default(),
            objective: Objective::default(),
            seed: None,
            initial_genomes: Vec::new(),
        }
    }

    /// The representation.
    pub fn real(&self) -> &Real {
        &self.real
    }

    /// The mutation strategy.
    pub fn strategy(&self) -> Strategy {
        self.strategy
    }

    /// Where `F` and `CR` come from.
    pub fn control(&self) -> Control {
        self.control
    }

    /// The archive of replaced individuals, for [`Strategy::CurrentToPBest`].
    pub fn archive(&self) -> &[Reals] {
        &self.archive
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn fitness(&self, index: usize) -> Fitness {
        self.population[index]
            .fitness()
            .unwrap_or(Fitness::invalid())
    }

    // the scale factor and crossover rate of the next trial
    fn next_parameters(&mut self) -> (f64, f64) {
        match self.control {
            Control::Fixed { f, cr } => (f, cr),
            Control::Dither { min_f, max_f, cr } => {
                (min_f + (max_f - min_f) * self.rng.unit_f64(), cr)
            }
        }
    }

    // a random index in `0..n` that isn't in `excluded`; `n` is larger than `excluded`
    fn other_than(&mut self, n: usize, excluded: &[usize]) -> usize {
        loop {
            let index = self.rng.below(n);
            if !excluded.contains(&index) {
                return index;
            }
        }
    }

    // the trial vector for individual `target`
    fn trial(&mut self, target: usize, f: f64, cr: f64, order: &[usize]) -> Reals {
        let size = self.population.len();
        let x = self.population[target].genome().clone();
        let genes = |de: &Self, index: usize| de.population[index].genome().clone();
        let mutant: Vec<f64> = match self.strategy {
            Strategy::Rand1 => {
                let r1 = self.other_than(size, &[target]);
                let r2 = self.other_than(size, &[target, r1]);
                let r3 = self.other_than(size, &[target, r1, r2]);
                let (a, b, c) = (genes(self, r1), genes(self, r2), genes(self, r3));
                (0..x.len()).map(|j| a[j] + f * (b[j] - c[j])).collect()
            }
            Strategy::Best1 => {
                let best = order[0];
                let r1 = self.other_than(size, &[target, best]);
                let r2 = self.other_than(size, &[target, best, r1]);
                let (a, b, c) = (genes(self, best), genes(self, r1), genes(self, r2));
                (0..x.len()).map(|j| a[j] + f * (b[j] - c[j])).collect()
            }
            Strategy::CurrentToPBest { p, .. } => {
                let top = ((p * size as f64).round() as usize).clamp(1, size);
                let pbest = order[self.rng.below(top)];
                let r1 = self.other_than(size, &[target]);
                // r2 from the population and the archive
                let r2 = self.other_than(size + self.archive.len(), &[target, r1]);
                let (best, a) = (genes(self, pbest), genes(self, r1));
                let b = if r2 < size {
                    genes(self, r2)
                } else {
                    self.archive[r2 - size].clone()
                };
                (0..x.len())
                    .map(|j| x[j] + f * (best[j] - x[j]) + f * (a[j] - b[j]))
                    .collect()
            }
        };
        // binomial crossover, with at least one gene from the mutant, and bounce-back at the bounds
        let bounds = self.real.bounds();
        let forced = self.rng.below(x.len());
        (0..x.len())
            .map(|j| {
                let from_mutant = j == forced || self.rng.unit_f64() < cr;
                let (start, end) = (*bounds[j].start(), *bounds[j].end());
                if !from_mutant || start == end {
                    x[j]
                } else if mutant[j] < start {
                    midpoint(start, x[j])
                } else if mutant[j] > end {
                    midpoint(end, x[j])
                } else if mutant[j].is_nan() {
                    x[j]
                } else {
                    mutant[j]
                }
            })
            .collect()
    }

    fn breed(&mut self) {
        let objective = self.objective;
        let size = self.population.len();
        // best first, the earlier one on ties
        let mut order: Vec<usize> = (0..size).collect();
        order.sort_by(|&a, &b| objective.compare(self.fitness(b), self.fitness(a)));
        self.trials.clear();
        for target in 0..size {
            let (f, cr) = self.next_parameters();
            let trial = self.trial(target, f, cr, &order);
            self.trials.push(Individual::new(trial));
        }
    }

    // the trials that replace their targets, then the rejected ones as discarded
    fn select(&mut self) {
        let objective = self.objective;
        let archive_size = match self.strategy {
            Strategy::CurrentToPBest { archive, .. } => {
                (archive * self.population.len() as f64).round() as usize
            }
            _ => 0,
        };
        self.discarded.clear();
        let trials = std::mem::take(&mut self.trials);
        for (target, trial) in trials.into_iter().enumerate() {
            let trial_fitness = trial.fitness().unwrap_or(Fitness::invalid());
            if objective.is_better(self.fitness(target), trial_fitness) {
                self.population[target].increment_age();
                self.discarded.push(trial);
                continue;
            }
            let replaced = std::mem::replace(&mut self.population[target], trial);
            if archive_size > 0 {
                if self.archive.len() >= archive_size {
                    let random = self.rng.below(self.archive.len());
                    self.archive.swap_remove(random);
                }
                self.archive.push(replaced.into_genome());
            }
        }
    }
}

// halfway between `a` and `b`, without overflow near the largest numbers
fn midpoint(a: f64, b: f64) -> f64 {
    let middle = (a + b) / 2.0;
    if middle.is_finite() {
        middle
    } else {
        a / 2.0 + b / 2.0
    }
}

impl Algorithm for De {
    type Genome = Reals;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, Reals> {
        if !self.asked {
            self.pending.clear();
            if self.started {
                self.breed();
                self.pending.extend(0..self.trials.len());
            } else {
                self.pending.extend(0..self.population.len());
            }
            self.asked = true;
        }
        let individuals = if self.started {
            &self.trials
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
        let objective = self.objective;
        let individuals = if self.started {
            self.trials.as_mut_slice()
        } else {
            self.population.iter_mut().into_slice()
        };
        for (individual, &fitness) in individuals.iter_mut().zip(fitness) {
            individual.set_fitness(fitness);
        }
        if self.started {
            self.generation += 1;
        }
        // the best so far, the first one on ties
        let candidates = if self.started {
            &self.trials[..]
        } else {
            self.population.as_slice()
        };
        let mut improved = false;
        for candidate in candidates {
            let fitness = candidate.fitness().unwrap_or(Fitness::invalid());
            let better = self.best.as_ref().is_none_or(|best| {
                objective.is_better(fitness, best.fitness().unwrap_or(Fitness::invalid()))
            });
            if better {
                self.best = Some(candidate.clone());
                improved = true;
            }
        }
        if improved {
            self.best_generation = self.generation;
        }
        if self.started {
            self.select();
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

    fn discarded(&self) -> &[Individual<Reals>] {
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

/// A builder for a [`De`], from [`De::builder`].
///
/// The population size is required. Defaults: DE/rand/1 with `F` 0.5 and `CR` 0.9, maximize, a
/// random initial population and a random seed.
#[derive(Clone, Debug)]
pub struct DeBuilder {
    real: Real,
    population_size: Option<usize>,
    strategy: Strategy,
    control: Control,
    objective: Objective,
    seed: Option<u64>,
    initial_genomes: Vec<Reals>,
}

impl DeBuilder {
    /// The population size, at least 4; 5 to 10 times the number of genes is common. Required.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// How mutant vectors are built. DE/rand/1 by default.
    pub fn strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Where `F` and `CR` come from. `F` 0.5 and `CR` 0.9 by default.
    pub fn control(mut self, control: Control) -> Self {
        self.control = control;
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

    /// Genomes for the initial population, at most the population size, each valid for the
    /// representation. The rest is random.
    pub fn initial_genomes<I: IntoIterator<Item = Reals>>(mut self, genomes: I) -> Self {
        self.initial_genomes = genomes.into_iter().collect();
        self
    }

    /// Validates the settings and creates the algorithm, with its initial population.
    ///
    /// # Errors
    ///
    /// - [`Error::MissingSetting`] without a population size.
    /// - [`Error::InvalidSetting`] for a population size below 4, or strategy or control
    ///   settings out of range.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<De> {
        let size = self.population_size.ok_or(Error::MissingSetting {
            setting: "population_size",
        })?;
        if size < 4 {
            return Err(Error::InvalidSetting {
                setting: "population_size",
                reason: format!("differential evolution needs at least 4 individuals, got {size}"),
            });
        }
        let invalid = |setting, reason: String| Err(Error::InvalidSetting { setting, reason });
        if let Strategy::CurrentToPBest { p, archive } = self.strategy {
            if !(p > 0.0 && p <= 1.0) {
                return invalid(
                    "p",
                    format!("must be greater than 0 and at most 1, got {p}"),
                );
            }
            if !(archive >= 0.0 && archive.is_finite()) {
                return invalid(
                    "archive",
                    format!("must be 0 or more and finite, got {archive}"),
                );
            }
        }
        let cr = match self.control {
            Control::Fixed { f, cr } => {
                if !(f > 0.0 && f <= 2.0) {
                    return invalid(
                        "f",
                        format!("must be greater than 0 and at most 2, got {f}"),
                    );
                }
                cr
            }
            Control::Dither { min_f, max_f, cr } => {
                if !(min_f > 0.0 && min_f <= max_f && max_f <= 2.0) {
                    return invalid(
                        "f",
                        format!("must be 0 < min_f <= max_f <= 2, got {min_f} and {max_f}"),
                    );
                }
                cr
            }
        };
        if !(0.0..=1.0).contains(&cr) {
            return invalid("cr", format!("must be between 0 and 1, got {cr}"));
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
        Ok(De {
            real: self.real,
            strategy: self.strategy,
            control: self.control,
            objective: self.objective,
            seed,
            rng,
            population: Population::from_genomes(genomes),
            trials: Vec::new(),
            archive: Vec::new(),
            discarded: Vec::new(),
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
    // explicit, as the proptest prelude also exports a `Strategy`
    use super::Strategy;

    fn sphere(x: &Reals) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    fn builder(strategy: Strategy, seed: u64) -> DeBuilder {
        De::builder(Real::uniform(8, -5.0..=5.0).unwrap())
            .population_size(20)
            .strategy(strategy)
            .minimize()
            .seed(seed)
    }

    fn step(de: &mut De) {
        let fitness: Vec<Fitness> = de.ask().iter().map(|x| Fitness::new(sphere(x))).collect();
        de.tell(&fitness).unwrap();
    }

    const STRATEGIES: [Strategy; 4] = [
        Strategy::Rand1,
        Strategy::Best1,
        Strategy::CurrentToPBest {
            p: 0.1,
            archive: 0.0,
        },
        Strategy::CurrentToPBest {
            p: 0.2,
            archive: 1.5,
        },
    ];

    fn setting(result: Result<De>) -> &'static str {
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
        assert_eq!(setting(De::builder(real()).build()), "population_size");
        assert_eq!(
            setting(De::builder(real()).population_size(3).build()),
            "population_size"
        );
        let with = |strategy, control| {
            De::builder(real())
                .population_size(10)
                .strategy(strategy)
                .control(control)
                .build()
        };
        let fixed = |f, cr| Control::Fixed { f, cr };
        let dither = |min_f, max_f| Control::Dither {
            min_f,
            max_f,
            cr: 0.5,
        };
        assert_eq!(setting(with(Strategy::Rand1, dither(0.0, 1.0))), "f");
        assert_eq!(setting(with(Strategy::Rand1, dither(0.8, 0.5))), "f");
        assert!(with(Strategy::Rand1, dither(0.5, 0.5)).is_ok());
        assert_eq!(setting(with(Strategy::Rand1, fixed(0.0, 0.5))), "f");
        assert_eq!(setting(with(Strategy::Rand1, fixed(2.5, 0.5))), "f");
        assert_eq!(setting(with(Strategy::Rand1, fixed(0.5, 1.5))), "cr");
        let pbest = |p, archive| Strategy::CurrentToPBest { p, archive };
        assert_eq!(setting(with(pbest(0.0, 1.0), fixed(0.5, 0.9))), "p");
        assert_eq!(setting(with(pbest(0.1, -1.0), fixed(0.5, 0.9))), "archive");
        assert!(with(pbest(1.0, 0.0), fixed(2.0, 0.0)).is_ok());
        assert!(matches!(
            De::builder(real())
                .population_size(10)
                .initial_genomes([Reals::from(vec![0.5])])
                .build(),
            Err(Error::InvalidGenome { .. })
        ));
    }

    #[test]
    fn ask_tell_protocol() {
        let mut de = builder(Strategy::Rand1, 0).build().unwrap();
        assert_eq!(de.tell(&[]), Err(Error::TellWithoutAsk));
        assert_eq!(de.ask().len(), 20);
        assert_eq!(
            de.tell(&[Fitness::new(0.0)]),
            Err(Error::FitnessCount {
                expected: 20,
                got: 1
            })
        );
        step(&mut de);
        assert_eq!((de.generation(), de.evaluations()), (0, 20));
        step(&mut de);
        assert_eq!((de.generation(), de.evaluations()), (1, 40));
        // every trial either replaced its target or was discarded
        let replaced = de.population().iter().filter(|x| x.age() == 0).count();
        assert_eq!(replaced + de.discarded().len(), 20);
    }

    #[test]
    fn each_strategy_solves_the_sphere() {
        let dither = Control::Dither {
            min_f: 0.5,
            max_f: 1.0,
            cr: 0.9,
        };
        // the greedy strategies with dither, as documented
        for (strategy, control) in [
            (Strategy::Rand1, Control::default()),
            (Strategy::Best1, dither),
            (
                Strategy::CurrentToPBest {
                    p: 0.1,
                    archive: 0.0,
                },
                dither,
            ),
            (
                Strategy::CurrentToPBest {
                    p: 0.1,
                    archive: 1.0,
                },
                Control::default(),
            ),
        ] {
            let de = builder(strategy, 1)
                .population_size(40)
                .control(control)
                .build()
                .unwrap();
            let outcome = Engine::new(de, sphere)
                .stop_when(Stop::target(1e-8).or(Stop::evaluations(200_000)))
                .run()
                .unwrap();
            assert_eq!(outcome.stop_reason(), StopReason::Target, "{strategy:?}");
        }
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut de = builder(
                Strategy::CurrentToPBest {
                    p: 0.1,
                    archive: 1.0,
                },
                seed,
            )
            .build()
            .unwrap();
            for _ in 0..20 {
                step(&mut de);
            }
            de.population().clone()
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3), run(4));
    }

    proptest! {
        #[test]
        fn trials_stay_in_bounds_and_targets_never_get_worse(
            strategy in prop::sample::select(STRATEGIES.to_vec()),
            f in 0.01..=2.0f64,
            cr in 0.0..=1.0f64,
            dither: bool,
            seed: u64,
        ) {
            // one fixed gene, bounds of different widths, and bounds near the largest numbers
            let real =
                Real::new([3.0..=3.0, -1.0..=1.0, 0.0..=100.0, -1e-3..=1e-3, 0.0..=1.7e308]).unwrap();
            let mut de = De::builder(real.clone())
                .population_size(8)
                .strategy(strategy)
                .control(if dither { Control::Dither { min_f: f / 2.0, max_f: f, cr } } else { Control::Fixed { f, cr } })
                .minimize()
                .seed(seed)
                .build()
                .unwrap();
            step(&mut de);
            for _ in 0..10 {
                let before: Vec<Fitness> = de.population().iter().map(|x| x.fitness().unwrap()).collect();
                let trials: Vec<Reals> = de.ask().iter().cloned().collect();
                for trial in &trials {
                    prop_assert!(real.validate(trial).is_ok(), "{trial:?}");
                }
                step(&mut de);
                for (index, individual) in de.population().iter().enumerate() {
                    prop_assert!(!Objective::Minimize.is_better(before[index], individual.fitness().unwrap()));
                }
                if let Strategy::CurrentToPBest { archive, .. } = strategy {
                    prop_assert!(de.archive().len() <= (archive * 8.0).round() as usize);
                }
            }
        }
    }
}
