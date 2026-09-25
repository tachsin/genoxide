//! Differential evolution: new solutions from the differences between solutions.

use super::{Algorithm, Candidates};
use crate::genome::{Real, Reals, Representation};
use crate::operator::check_size;
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;

/// How a differential evolution builds the mutant vector `v` for an individual `x`, from random
/// distinct other individuals `r1`, `r2` (and `r3`) and the scale factor `F`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// JADE's adaptation: each trial draws `F` from a Cauchy distribution (scale 0.1, redrawn
    /// until positive, at most 1) and `CR` from a normal distribution (standard deviation 0.1,
    /// clamped to 0..=1), around means that start at 0.5. After every generation, the means move
    /// by the fraction `c` toward the values of the trials that improved on their parents: the
    /// Lehmer mean of their `F`, the arithmetic mean of their `CR`.
    Jade {
        /// How fast the means adapt, greater than 0 and at most 1; 0.1 is common.
        c: f64,
    },
    /// SHADE's adaptation: a memory of `memory` (`F`, `CR`) pairs, starting at 0.5. Each trial
    /// picks a random pair and draws `F` and `CR` around it like JADE. After every generation, one
    /// memory slot (in turn) takes the means of the successful values, weighted by how much each
    /// trial improved: the weighted Lehmer mean of `F` and the weighted arithmetic mean of `CR`
    /// (as in SHADE), with a `CR` that stays 0 once the successes only had `CR` 0 (as in L-SHADE).
    Shade {
        /// The number of (`F`, `CR`) pairs remembered, at least 1 and at most 2^24. Small
        /// memories adapt faster: 5 to 10 is a good choice (L-SHADE uses 6). Each slot is updated
        /// once every `memory` generations.
        memory: usize,
    },
}

impl Default for Control {
    /// `F` 0.5 and `CR` 0.9.
    fn default() -> Self {
        Control::Fixed { f: 0.5, cr: 0.9 }
    }
}

/// When a differential evolution starts over from new random individuals.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Restarts {
    /// Never: a converged population goes on sampling around the same point.
    #[default]
    Never,
    /// When the population has converged or stalled: every individual but the best is replaced
    /// by a random one, and the adaptation of `F` and `CR` and the archive start over. The new
    /// individuals are evaluated in a generation of their own, and the best individual carries
    /// what was found into the new population.
    ///
    /// The restart happens when the next generation is asked for: observers see the population
    /// that converged or stalled, and migrants that arrive in between (see
    /// [`Islands`](crate::algorithm::Islands)) are kept too.
    ///
    /// Small populations need far fewer evaluations to reach a target, but converge early on
    /// hard problems; restarts make them reliable.
    OnStagnation {
        /// Converged: the scores of the population are within `tolerance` of each other,
        /// relative to the best score (`max - min <= tolerance · (1 + |best|)`), e.g. 1e-8, and
        /// so are their constraint violations (`max - min <= tolerance · (1 + min)`).
        tolerance: f64,
        /// Stalled: the best score since the last restart hasn't improved for `patience`
        /// generations, at least 1, e.g. 200.
        patience: u64,
    },
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct De {
    real: Real,
    strategy: Strategy,
    control: Control,
    objective: Objective,
    seed: u64,
    rng: StreamRng,
    population: Population<Reals>,
    trials: Vec<Individual<Reals>>,
    // the scale factor and crossover rate of each trial
    parameters: Vec<(f64, f64)>,
    archive: Vec<Reals>,
    // JADE: the means of F and CR; SHADE: the memory of (F, CR), CR NaN for "stays 0"
    means: (f64, f64),
    memory: Vec<(f64, f64)>,
    memory_slot: usize,
    // L-SHADE: (initial size, minimum size, evaluations)
    reduction: Option<(usize, usize, u64)>,
    // restarts: a checkpoint from before them resumes without them, as it was saved
    #[cfg_attr(feature = "serde", serde(default))]
    restarts: Restarts,
    // the new individuals of a restart, evaluated before the next generation is bred
    #[cfg_attr(feature = "serde", serde(default))]
    fresh: Vec<usize>,
    #[cfg_attr(feature = "serde", serde(default))]
    restart_count: u64,
    // the best fitness since the last restart, and when it last improved
    #[cfg_attr(feature = "serde", serde(default))]
    start_best: Option<Fitness>,
    #[cfg_attr(feature = "serde", serde(default))]
    start_best_generation: u64,
    // a restart for the next ask, and the positions of the migrants since the last tell, which
    // it keeps
    #[cfg_attr(feature = "serde", serde(default))]
    restart_due: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    immigrated: Vec<usize>,
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
    ///
    /// The defaults are the settings that reached targets in the fewest evaluations in
    /// genoxide's measurements (shifted Rastrigin, Rosenbrock and Ackley with 10 and 30 genes):
    /// current-to-pbest/1 with an archive, SHADE's adaptation, a population of the number of
    /// genes + 10, and restarts on stagnation. See [`DeBuilder`].
    pub fn builder(real: Real) -> DeBuilder {
        DeBuilder {
            real,
            population_size: None,
            strategy: Strategy::CurrentToPBest {
                p: 0.1,
                archive: 1.0,
            },
            control: Control::Shade { memory: 6 },
            objective: Objective::default(),
            seed: None,
            initial_genomes: Vec::new(),
            reduction: None,
            restarts: Restarts::OnStagnation {
                tolerance: 1e-8,
                patience: 200,
            },
        }
    }

    /// A builder with the settings of L-SHADE (Tanabe and Fukunaga, 2014), for a run of
    /// `max_evaluations` evaluations: a population of 18 times the number of genes, shrinking
    /// linearly to 4; current-to-pbest/1 with `p` 0.11 and an archive of 2.6 times the population;
    /// SHADE with a memory of 6. Set the objective, and stop the run at `max_evaluations`.
    ///
    /// Unlike the paper, the `CR` memory takes the weighted arithmetic mean of SHADE (see
    /// [`Control::Shade`]), `pbest` can come from the best individual alone when `p` times the
    /// population rounds to 1, and a target replaced by an equally good trial is archived too.
    ///
    /// ```
    /// use genoxide::prelude::*;
    ///
    /// let budget = 50_000;
    /// let de = De::l_shade(Real::uniform(5, -5.0..=5.0)?, budget).minimize().seed(1).build()?;
    /// let sphere = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
    /// let outcome = Engine::new(de, sphere)
    ///     .stop_when(Stop::target(1e-8).or(Stop::evaluations(budget)))
    ///     .run()?;
    /// assert_eq!(outcome.stop_reason(), StopReason::Target);
    /// # Ok::<(), genoxide::Error>(())
    /// ```
    pub fn l_shade(real: Real, max_evaluations: u64) -> DeBuilder {
        let size = (18 * real.genome_len()).max(4);
        De::builder(real)
            .population_size(size)
            .strategy(Strategy::CurrentToPBest {
                p: 0.11,
                archive: 2.6,
            })
            .control(Control::Shade { memory: 6 })
            .linear_reduction(4, max_evaluations)
            .restarts(Restarts::Never)
    }

    /// The means of `F` and `CR` of [`Control::Jade`], or the memory of (`F`, `CR`) pairs of
    /// [`Control::Shade`] (a `CR` of NaN stays 0); empty for the other controls.
    pub fn adapted(&self) -> Vec<(f64, f64)> {
        match self.control {
            Control::Jade { .. } => vec![self.means],
            Control::Shade { .. } => self.memory.clone(),
            _ => Vec::new(),
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

    /// When the population starts over.
    pub fn restarts(&self) -> Restarts {
        self.restarts
    }

    /// The number of restarts so far.
    pub fn restart_count(&self) -> u64 {
        self.restart_count
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
            Control::Jade { .. } => {
                let (mean_f, mean_cr) = self.means;
                (self.adaptive_f(mean_f), self.adaptive_cr(mean_cr))
            }
            Control::Shade { .. } => {
                let (mean_f, mean_cr) = self.memory[self.rng.below(self.memory.len())];
                let f = self.adaptive_f(mean_f);
                // a CR that stays 0
                let cr = if mean_cr.is_nan() {
                    0.0
                } else {
                    self.adaptive_cr(mean_cr)
                };
                (f, cr)
            }
        }
    }

    // F from a Cauchy distribution around `mean`, scale 0.1, redrawn until positive, at most 1
    fn adaptive_f(&mut self, mean: f64) -> f64 {
        loop {
            let f = cauchy(&mut self.rng, mean, 0.1);
            if f > 0.0 {
                return f.min(1.0);
            }
        }
    }

    // CR from a normal distribution around `mean`, standard deviation 0.1, clamped to 0..=1
    fn adaptive_cr(&mut self, mean: f64) -> f64 {
        (mean + 0.1 * self.rng.normal()).clamp(0.0, 1.0)
    }

    // adapts JADE's means or SHADE's memory to the successful (F, CR, improvement) of a generation
    fn adapt(&mut self, successes: &[(f64, f64, f64)]) {
        if successes.is_empty() {
            return;
        }
        let f: Vec<f64> = successes.iter().map(|success| success.0).collect();
        let cr: Vec<f64> = successes.iter().map(|success| success.1).collect();
        match self.control {
            Control::Jade { c } => {
                let equal = vec![1.0; successes.len()];
                let (mean_f, mean_cr) = self.means;
                self.means = (
                    (1.0 - c) * mean_f + c * lehmer_mean(&f, &equal),
                    (1.0 - c) * mean_cr + c * cr.iter().sum::<f64>() / cr.len() as f64,
                );
            }
            Control::Shade { .. } => {
                let improvements: Vec<f64> = successes.iter().map(|success| success.2).collect();
                // weights proportional to the improvements; equal if those aren't usable
                let total: f64 = improvements.iter().sum();
                let weights = if total > 0.0 && total.is_finite() {
                    improvements
                } else {
                    vec![1.0; successes.len()]
                };
                let slot = self.memory_slot;
                let (_, previous_cr) = self.memory[slot];
                let new_cr = if previous_cr.is_nan() || cr.iter().all(|&cr| cr == 0.0) {
                    f64::NAN
                } else {
                    // the weighted arithmetic mean
                    cr.iter().zip(&weights).map(|(cr, w)| cr * w).sum::<f64>()
                        / weights.iter().sum::<f64>()
                };
                self.memory[slot] = (lehmer_mean(&f, &weights), new_cr);
                self.memory_slot = (slot + 1) % self.memory.len();
            }
            _ => {}
        }
    }

    // L-SHADE: the population shrinks linearly with the evaluations, the worst first
    fn reduce(&mut self) {
        let Some((initial, minimum, max_evaluations)) = self.reduction else {
            return;
        };
        let used = self.evaluations.min(max_evaluations) as f64 / max_evaluations as f64;
        let size = (initial as f64 + (minimum as f64 - initial as f64) * used).round() as usize;
        let size = size.clamp(minimum, self.population.len());
        if size < self.population.len() {
            self.population.sort_best_first(self.objective);
            // the trials of this generation (age 0) that are dropped are discarded
            let dropped = self.population.as_slice()[size..]
                .iter()
                .filter(|individual| individual.age() == 0)
                .cloned();
            self.discarded.extend(dropped);
            self.population.truncate(size);
            if let Strategy::CurrentToPBest { archive, .. } = self.strategy {
                let archive_size = (archive * size as f64).round() as usize;
                while self.archive.len() > archive_size {
                    let random = self.rng.below(self.archive.len());
                    self.archive.swap_remove(random);
                }
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
        // one of the genes that can change: a fixed one would leave the trial a copy
        let variable = self.real.variable_genes();
        let forced = match variable.len() {
            0 => usize::MAX,
            len => variable[self.rng.below(len)],
        };
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
        self.parameters.clear();
        for target in 0..size {
            let (f, cr) = self.next_parameters();
            let trial = self.trial(target, f, cr, &order);
            self.trials.push(Individual::new(trial));
            self.parameters.push((f, cr));
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
        let mut successes = Vec::new();
        for (target, trial) in trials.into_iter().enumerate() {
            let trial_fitness = trial.fitness().unwrap_or(Fitness::invalid());
            let target_fitness = self.fitness(target);
            if objective.is_better(target_fitness, trial_fitness) {
                self.population[target].increment_age();
                self.discarded.push(trial);
                continue;
            }
            if objective.is_better(trial_fitness, target_fitness) {
                let (f, cr) = self.parameters[target];
                successes.push((f, cr, improvement(target_fitness, trial_fitness)));
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
        self.adapt(&successes);
        self.reduce();
        self.track_start();
        self.restart_due = self.stagnated();
    }

    // the index of the best individual, the first one on ties
    fn best_index(&self) -> usize {
        let objective = self.objective;
        (0..self.population.len())
            .reduce(|a, b| {
                if objective.is_better(self.fitness(b), self.fitness(a)) {
                    b
                } else {
                    a
                }
            })
            .unwrap_or(0)
    }

    // the best fitness since the last restart, and when it last improved
    fn track_start(&mut self) {
        let best = self.fitness(self.best_index());
        let objective = self.objective;
        if self
            .start_best
            .is_none_or(|start_best| objective.is_better(best, start_best))
        {
            self.start_best = Some(best);
            self.start_best_generation = self.generation;
        }
    }

    // whether the population has converged or stalled, with restarts on
    fn stagnated(&self) -> bool {
        let Restarts::OnStagnation {
            tolerance,
            patience,
        } = self.restarts
        else {
            return false;
        };
        if self.generation - self.start_best_generation >= patience {
            return true;
        }
        let mut scores = Vec::with_capacity(self.population.len());
        let mut violations = Vec::with_capacity(self.population.len());
        for index in 0..self.population.len() {
            let fitness = self.fitness(index);
            // an invalid individual: not converged
            let Some(score) = fitness.score() else {
                return false;
            };
            scores.push(score);
            violations.push(fitness.violation());
        }
        let range = |values: &[f64]| {
            values
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
                    (min.min(x), max.max(x))
                })
        };
        let (min, max) = range(&scores);
        let best = match self.objective {
            Objective::Maximize => max,
            Objective::Minimize => min,
        };
        let (min_violation, max_violation) = range(&violations);
        max - min <= tolerance * (1.0 + best.abs())
            && max_violation - min_violation <= tolerance * (1.0 + min_violation)
    }

    // every individual but the best and the migrants since the last tell is replaced by a random
    // one; the adaptation and the archive start over
    fn restart(&mut self) {
        let best = self.best_index();
        self.fresh.clear();
        for index in 0..self.population.len() {
            if index != best && !self.immigrated.contains(&index) {
                let genome = self.real.random_genome(&mut self.rng);
                self.population[index] = Individual::new(genome);
                self.fresh.push(index);
            }
        }
        self.archive.clear();
        self.means = (0.5, 0.5);
        self.memory.fill((0.5, 0.5));
        self.memory_slot = 0;
        self.restart_count += 1;
        self.restart_due = false;
        self.start_best = None;
        self.start_best_generation = self.generation;
    }
}

// a Cauchy random number: `location + scale · Z1 / Z2` for standard normal Z1 and Z2, which only
// needs portable math (no tangent)
fn cauchy(rng: &mut StreamRng, location: f64, scale: f64) -> f64 {
    loop {
        let (numerator, denominator) = (rng.normal(), rng.normal());
        if denominator != 0.0 {
            return location + scale * numerator / denominator;
        }
    }
}

// the weighted Lehmer mean `Σ w x² / Σ w x`, which leans toward the larger values
fn lehmer_mean(values: &[f64], weights: &[f64]) -> f64 {
    let squares: f64 = values.iter().zip(weights).map(|(x, w)| w * x * x).sum();
    let sum: f64 = values.iter().zip(weights).map(|(x, w)| w * x).sum();
    if sum > 0.0 { squares / sum } else { 0.0 }
}

// how much `better` improves on `worse`: the score difference between feasible fitness values,
// the violation difference between infeasible ones, and 1 otherwise
fn improvement(worse: Fitness, better: Fitness) -> f64 {
    let difference = match (worse.score(), better.score()) {
        (Some(a), Some(b)) if worse.is_feasible() && better.is_feasible() => (a - b).abs(),
        (Some(_), Some(_)) if !worse.is_feasible() && !better.is_feasible() => {
            (worse.violation() - better.violation()).abs()
        }
        _ => 1.0,
    };
    if difference.is_finite() {
        difference
    } else {
        1.0
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

impl super::Migrate for De {
    fn immigrate(&mut self, migrants: Vec<Individual<Reals>>) -> Result<()> {
        if self.asked || !self.started {
            return Err(Error::MigrationOutOfTurn);
        }
        for migrant in &migrants {
            self.real.validate(migrant.genome())?;
        }
        let objective = self.objective;
        // the worst positions, the later one first on ties
        let mut order: Vec<usize> = (0..self.population.len()).collect();
        order.sort_by(|&a, &b| {
            objective
                .compare(self.fitness(a), self.fitness(b))
                .then(b.cmp(&a))
        });
        let mut improved = false;
        for (position, migrant) in order.into_iter().zip(migrants) {
            let fitness = migrant.fitness().unwrap_or(Fitness::invalid());
            let better = self.best.as_ref().is_none_or(|best| {
                objective.is_better(fitness, best.fitness().unwrap_or(Fitness::invalid()))
            });
            if better {
                self.best = Some(migrant.clone());
                improved = true;
            }
            self.population[position] = migrant;
            if !self.immigrated.contains(&position) {
                self.immigrated.push(position);
            }
        }
        if improved {
            self.best_generation = self.generation;
        }
        Ok(())
    }

    fn same_representation(&self, other: &Self) -> bool {
        self.real == other.real
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
            if self.started && self.restart_due {
                self.restart();
            }
            if !self.started {
                self.pending.extend(0..self.population.len());
            } else if !self.fresh.is_empty() {
                // the new individuals of a restart
                self.pending.extend_from_slice(&self.fresh);
            } else {
                self.breed();
                self.pending.extend(0..self.trials.len());
            }
            self.asked = true;
        }
        let individuals = if self.started && self.fresh.is_empty() {
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
        self.immigrated.clear();
        let objective = self.objective;
        // a generation of trials, or the evaluation of the initial population or of a restart
        let trials = self.started && self.fresh.is_empty();
        if trials {
            for (individual, &fitness) in self.trials.iter_mut().zip(fitness) {
                individual.set_fitness(fitness);
            }
        } else {
            for (&index, &fitness) in self.pending.iter().zip(fitness) {
                self.population[index].set_fitness(fitness);
            }
        }
        if self.started {
            self.generation += 1;
        }
        // the best so far, the first one on ties
        let candidates: Vec<&Individual<Reals>> = if trials {
            self.trials.iter().collect()
        } else {
            self.pending
                .iter()
                .map(|&index| &self.population[index])
                .collect()
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
        if trials {
            self.select();
        } else {
            // a restart's generation has no trials to discard
            self.fresh.clear();
            self.discarded.clear();
            self.track_start();
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
/// Defaults: DE/current-to-pbest/1 with `p` 0.1 and an archive of the population's size,
/// SHADE's adaptation with a memory of 6, a population of the number of genes + 10, restarts on
/// stagnation (tolerance 1e-8, patience 200), maximize, a random initial population and a random
/// seed.
///
/// Small populations reach a target in fewer evaluations, and the restarts keep them from
/// getting stuck: with 10 and 30 genes, genes + 10 reached shifted Rastrigin, Rosenbrock and
/// Ackley targets in 2 to 5 times fewer evaluations than a population of 100. Non-separable,
/// highly multimodal problems (e.g. rotated Rastrigin) do better with larger populations, e.g. 100.
#[derive(Clone, Debug)]
pub struct DeBuilder {
    real: Real,
    population_size: Option<usize>,
    strategy: Strategy,
    control: Control,
    objective: Objective,
    seed: Option<u64>,
    initial_genomes: Vec<Reals>,
    reduction: Option<(usize, u64)>,
    restarts: Restarts,
}

impl DeBuilder {
    /// The population size, at least 4 and at most 2^24. The number of genes + 10 by default.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// How mutant vectors are built. DE/current-to-pbest/1 with `p` 0.1 and an archive of the
    /// population's size by default.
    pub fn strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Where `F` and `CR` come from. SHADE's adaptation with a memory of 6 by default.
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

    /// Linear population size reduction (L-SHADE): after every generation, the population
    /// shrinks to the size on the line from the initial size to `min_size` (at least 4) over
    /// `max_evaluations` evaluations, dropping the worst individuals. Stop the run at
    /// `max_evaluations` too. Off by default.
    pub fn linear_reduction(mut self, min_size: usize, max_evaluations: u64) -> Self {
        self.reduction = Some((min_size, max_evaluations));
        self
    }

    /// Genomes for the initial population, at most the population size, each valid for the
    /// representation. The rest is random.
    pub fn initial_genomes<I: IntoIterator<Item = Reals>>(mut self, genomes: I) -> Self {
        self.initial_genomes = genomes.into_iter().collect();
        self
    }

    /// When the population starts over from new random individuals. On stagnation by default,
    /// with a tolerance of 1e-8 and a patience of 200 generations.
    pub fn restarts(mut self, restarts: Restarts) -> Self {
        self.restarts = restarts;
        self
    }

    /// Validates the settings and creates the algorithm, with its initial population.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidSetting`] for a population size below 4 or above 2^24, strategy,
    ///   control, linear reduction or restart settings out of range, or more initial genomes than
    ///   the population size.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<De> {
        let size = self.population_size.unwrap_or(self.real.genome_len() + 10);
        if size < 4 {
            return Err(Error::InvalidSetting {
                setting: "population_size",
                reason: format!("differential evolution needs at least 4 individuals, got {size}"),
            });
        }
        check_size("population_size", size)?;
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
            Control::Jade { c } => {
                if !(c > 0.0 && c <= 1.0) {
                    return invalid(
                        "c",
                        format!("must be greater than 0 and at most 1, got {c}"),
                    );
                }
                0.5
            }
            Control::Shade { memory } => {
                if memory == 0 {
                    return invalid("memory", "must be at least 1".to_string());
                }
                check_size("memory", memory)?;
                0.5
            }
        };
        if let Some((min_size, max_evaluations)) = self.reduction {
            if min_size < 4 || min_size > size || max_evaluations == 0 {
                return invalid(
                    "linear_reduction",
                    format!(
                        "the minimum size must be between 4 and the population size {size}, and the evaluations at least 1; got {min_size} and {max_evaluations}"
                    ),
                );
            }
        }
        if !(0.0..=1.0).contains(&cr) {
            return invalid("cr", format!("must be between 0 and 1, got {cr}"));
        }
        if let Restarts::OnStagnation {
            tolerance,
            patience,
        } = self.restarts
        {
            if !(tolerance >= 0.0 && tolerance.is_finite()) || patience == 0 {
                return invalid(
                    "restarts",
                    format!(
                        "the tolerance must be 0 or more and finite, and the patience at least 1; got {tolerance} and {patience}"
                    ),
                );
            }
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
            parameters: Vec::new(),
            archive: Vec::new(),
            means: (0.5, 0.5),
            memory: match self.control {
                Control::Shade { memory } => vec![(0.5, 0.5); memory],
                _ => Vec::new(),
            },
            memory_slot: 0,
            reduction: self
                .reduction
                .map(|(min_size, max_evaluations)| (size, min_size, max_evaluations)),
            restarts: self.restarts,
            fresh: Vec::new(),
            restart_count: 0,
            start_best: None,
            start_best_generation: 0,
            restart_due: false,
            immigrated: Vec::new(),
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
        // the number of genes + 10 by default
        assert_eq!(De::builder(real()).build().unwrap().population().len(), 12);
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
    fn adaptive_validation() {
        let real = || Real::uniform(2, 0.0..=1.0).unwrap();
        let with = |control| {
            De::builder(real())
                .population_size(10)
                .control(control)
                .build()
        };
        assert_eq!(setting(with(Control::Jade { c: 0.0 })), "c");
        assert_eq!(setting(with(Control::Jade { c: 1.5 })), "c");
        assert_eq!(setting(with(Control::Shade { memory: 0 })), "memory");
        assert!(with(Control::Jade { c: 1.0 }).is_ok());
        let reduction = |min, evaluations| {
            De::builder(real())
                .population_size(10)
                .linear_reduction(min, evaluations)
                .build()
        };
        assert_eq!(setting(reduction(3, 100)), "linear_reduction");
        assert_eq!(setting(reduction(11, 100)), "linear_reduction");
        assert_eq!(setting(reduction(4, 0)), "linear_reduction");
        assert!(reduction(10, 100).is_ok());
    }

    #[test]
    fn cauchy_is_centered_with_its_scale() {
        let mut rng = StreamRng::seed_from_u64(0);
        let mut samples: Vec<f64> = (0..40_000).map(|_| cauchy(&mut rng, 0.5, 0.1)).collect();
        samples.sort_by(f64::total_cmp);
        // the median is the location, and half of the mass is within one scale of it
        let median = samples[samples.len() / 2];
        let within = samples.iter().filter(|x| (*x - 0.5).abs() < 0.1).count() as f64 / 40_000.0;
        assert!((median - 0.5).abs() < 0.005, "median {median}");
        assert!((within - 0.5).abs() < 0.01, "within {within}");
    }

    #[test]
    fn means() {
        assert!((lehmer_mean(&[1.0, 2.0, 3.0], &[1.0; 3]) - 14.0 / 6.0).abs() < 1e-12);
        assert!((lehmer_mean(&[1.0, 3.0], &[3.0, 1.0]) - 12.0 / 6.0).abs() < 1e-12);
        assert_eq!(lehmer_mean(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        let (worse, better) = (Fitness::new(5.0), Fitness::new(3.5));
        assert_eq!(improvement(worse, better), 1.5);
        let (worse, better) = (
            Fitness::constrained(0.0, 2.0),
            Fitness::constrained(9.0, 0.5),
        );
        assert_eq!(improvement(worse, better), 1.5);
        assert_eq!(improvement(Fitness::invalid(), Fitness::new(1.0)), 1.0);
    }

    #[test]
    fn jade_means_move_toward_the_successes() {
        let mut de = builder(Strategy::Rand1, 0)
            .control(Control::Jade { c: 0.5 })
            .build()
            .unwrap();
        assert_eq!(de.adapted(), [(0.5, 0.5)]);
        de.adapt(&[(0.9, 0.1, 1.0), (0.9, 0.3, 5.0)]);
        // F: the Lehmer mean 0.9; CR: the mean 0.2
        let (f, cr) = de.adapted()[0];
        assert!(
            (f - 0.7).abs() < 1e-12 && (cr - 0.35).abs() < 1e-12,
            "{f} {cr}"
        );
        de.adapt(&[]);
        assert_eq!(de.adapted()[0], (f, cr));
    }

    #[test]
    fn shade_memory_takes_the_weighted_means_in_turn() {
        let mut de = builder(Strategy::Rand1, 0)
            .control(Control::Shade { memory: 2 })
            .build()
            .unwrap();
        // weights 1 and 3
        de.adapt(&[(0.2, 0.2, 1.0), (0.6, 0.6, 3.0)]);
        let memory = de.adapted();
        let lehmer = (0.04 + 3.0 * 0.36) / (0.2 + 3.0 * 0.6);
        assert!((memory[0].0 - lehmer).abs() < 1e-12);
        assert!((memory[0].1 - 0.5).abs() < 1e-12);
        assert_eq!(memory[1], (0.5, 0.5));
        // the next slot; successes with CR 0 make the CR stay 0
        de.adapt(&[(0.5, 0.0, 1.0)]);
        assert!(de.adapted()[1].1.is_nan());
        // back to the first slot
        de.adapt(&[(0.5, 0.4, 1.0)]);
        assert!((de.adapted()[0].1 - 0.4).abs() < 1e-12);
    }

    #[test]
    fn linear_reduction_shrinks_the_population_on_schedule() {
        let mut de = builder(
            Strategy::CurrentToPBest {
                p: 0.1,
                archive: 2.0,
            },
            0,
        )
        .population_size(40)
        .control(Control::Shade { memory: 6 })
        .linear_reduction(4, 2_000)
        .build()
        .unwrap();
        step(&mut de);
        let mut previous = de.population().len();
        while de.evaluations() < 2_000 {
            step(&mut de);
            // every trial survived (age 0) or was discarded, also when the population shrank
            let survived = de.population().iter().filter(|x| x.age() == 0).count();
            assert_eq!(survived + de.discarded().len(), previous);
            let size = de.population().len();
            let expected =
                (40.0 + (4.0 - 40.0) * de.evaluations().min(2_000) as f64 / 2_000.0).round();
            assert!(size <= previous);
            assert_eq!(size, (expected as usize).clamp(4, previous));
            assert!(de.archive().len() <= 2 * size);
            previous = size;
        }
        assert_eq!(de.population().len(), 4);
    }

    #[test]
    fn l_shade_preset() {
        let de = De::l_shade(Real::uniform(10, -1.0..=1.0).unwrap(), 100_000)
            .build()
            .unwrap();
        assert_eq!(de.population().len(), 180);
        assert_eq!(
            de.strategy(),
            Strategy::CurrentToPBest {
                p: 0.11,
                archive: 2.6
            }
        );
        assert_eq!(de.control(), Control::Shade { memory: 6 });
        assert_eq!(de.adapted().len(), 6);
    }

    #[test]
    fn restarts_validation() {
        let restarts = |tolerance, patience| {
            setting(
                builder(Strategy::Rand1, 0)
                    .restarts(Restarts::OnStagnation {
                        tolerance,
                        patience,
                    })
                    .build(),
            )
        };
        assert_eq!(restarts(-1.0, 10), "restarts");
        assert_eq!(restarts(f64::NAN, 10), "restarts");
        assert_eq!(restarts(1e-8, 0), "restarts");
        assert!(
            builder(Strategy::Rand1, 0)
                .restarts(Restarts::OnStagnation {
                    tolerance: 1e-8,
                    patience: 10,
                })
                .build()
                .is_ok()
        );
    }

    #[test]
    fn a_converged_population_starts_over_from_its_best() {
        let shade = Strategy::CurrentToPBest {
            p: 0.1,
            archive: 1.0,
        };
        let mut de = builder(shade, 3)
            .control(Control::Shade { memory: 6 })
            // converged as soon as the scores are within 1 of each other
            .restarts(Restarts::OnStagnation {
                tolerance: 1.0,
                patience: 1_000,
            })
            .build()
            .unwrap();
        step(&mut de);
        let mut generations = 0;
        // a converged population is still what the last generation left
        while !de.restart_due {
            step(&mut de);
            generations += 1;
            assert!(generations < 1_000, "no restart");
            assert!(de.population().iter().all(Individual::is_evaluated));
        }
        assert_eq!(de.restart_count(), 0);
        // at the next ask, the best individual stays, the others are new and evaluated in a
        // generation of their own
        let best = de.best().unwrap().clone();
        let evaluations = de.evaluations();
        let fresh: Vec<Reals> = de.ask().iter().cloned().collect();
        assert_eq!(de.restart_count(), 1);
        assert_eq!(fresh.len(), 19);
        assert_eq!(de.archive().len(), 0);
        assert!(de.adapted().iter().all(|&slot| slot == (0.5, 0.5)));
        assert!(de.population().iter().any(|x| x.genome() == best.genome()));
        let told: Vec<Fitness> = fresh.iter().map(|x| Fitness::new(sphere(x))).collect();
        de.tell(&told).unwrap();
        assert_eq!(de.evaluations(), evaluations + 19);
        assert!(de.discarded().is_empty());
        // the best so far is never lost
        let (now, before) = (
            de.best().unwrap().fitness().unwrap(),
            best.fitness().unwrap(),
        );
        assert!(!Objective::Minimize.is_better(before, now));
        // and the run goes on with trials again
        assert_eq!(de.ask().len(), 20);
    }

    #[test]
    fn a_stalled_population_starts_over() {
        let mut de = builder(Strategy::Rand1, 4)
            .restarts(Restarts::OnStagnation {
                tolerance: 0.0,
                patience: 5,
            })
            .build()
            .unwrap();
        // different scores, so not converged, and trials that are all worse: stalled
        let fitness: Vec<Fitness> = (0..20).map(|i| Fitness::new(f64::from(i))).collect();
        de.ask();
        de.tell(&fitness).unwrap();
        for generation in 1..=7 {
            let fitness = vec![Fitness::new(100.0); de.ask().len()];
            de.tell(&fitness).unwrap();
            // 5 generations without improvement; the restart is at the next ask
            assert_eq!(de.restart_due, generation == 5);
            assert_eq!(de.restart_count(), u64::from(generation >= 6));
        }
    }

    #[test]
    fn restarts_repeat_with_a_seed() {
        let run = |seed| {
            let mut de = builder(Strategy::Rand1, seed)
                .restarts(Restarts::OnStagnation {
                    tolerance: 1e-3,
                    patience: 20,
                })
                .build()
                .unwrap();
            for _ in 0..200 {
                step(&mut de);
            }
            (
                de.restart_count(),
                de.evaluations(),
                de.best().unwrap().clone(),
            )
        };
        let (restarts, evaluations, best) = run(9);
        assert!(restarts > 0);
        assert_eq!(run(9), (restarts, evaluations, best));
    }

    #[test]
    fn small_populations_with_restarts_solve_rastrigin() {
        // shifted Rastrigin in 10 dimensions, as in the benchmarks
        let shift = |i: usize| 2.0 * ((37 * i + 11) % 101) as f64 / 101.0 - 1.0;
        let rastrigin = |x: &Reals| {
            10.0 * x.len() as f64
                + x.iter()
                    .enumerate()
                    .map(|(i, xi)| {
                        let y = xi - shift(i);
                        y * y - 10.0 * (2.0 * std::f64::consts::PI * y).cos()
                    })
                    .sum::<f64>()
        };
        let solved = (0..5)
            .filter(|&seed| {
                let de = De::builder(Real::uniform(10, -5.12..=5.12).unwrap())
                    .population_size(20)
                    .strategy(Strategy::CurrentToPBest {
                        p: 0.1,
                        archive: 1.0,
                    })
                    .control(Control::Shade { memory: 6 })
                    .restarts(Restarts::OnStagnation {
                        tolerance: 1e-8,
                        patience: 200,
                    })
                    .minimize()
                    .seed(seed)
                    .build()
                    .unwrap();
                let outcome = Engine::new(de, rastrigin)
                    .stop_when(Stop::target(0.01).or(Stop::evaluations(100_000)))
                    .run()
                    .unwrap();
                outcome.stop_reason() == StopReason::Target
            })
            .count();
        assert_eq!(solved, 5);
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
            kind in 0usize..4,
            seed: u64,
        ) {
            // one fixed gene, bounds of different widths, and bounds near the largest numbers
            let real =
                Real::new([3.0..=3.0, -1.0..=1.0, 0.0..=100.0, -1e-3..=1e-3, 0.0..=1.7e308]).unwrap();
            let mut de = De::builder(real.clone())
                .population_size(8)
                .strategy(strategy)
                .control(match kind {
                    0 => Control::Fixed { f, cr },
                    1 => Control::Dither { min_f: f / 2.0, max_f: f, cr },
                    2 => Control::Jade { c: 0.1 },
                    _ => Control::Shade { memory: 3 },
                })
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
