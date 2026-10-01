//! The Nelder-Mead simplex method, with Gao and Han's adaptive coefficients and random restarts.

use super::local::Restarts;
use super::{Algorithm, Candidates, Reevaluate};
use crate::genome::{Real, Reals, Representation};
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;

/// The coefficients of a [`NelderMead`]: how far its simplex reflects, expands, contracts and
/// shrinks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Coefficients {
    /// Gao and Han's (2012) coefficients for `n` searched genes (the default): reflection 1,
    /// expansion 1 + 2/n, contraction 0.75 − 1/(2n) and shrink 1 − 1/n. With 2 genes they are the
    /// standard ones; with more, smaller expansions and gentler contractions and shrinks keep the
    /// simplex from flattening, which makes the method much faster from about 5 genes up. With a
    /// single gene, where the formulas would shrink the simplex to a point, the standard ones.
    #[default]
    Adaptive,
    /// The standard coefficients of Nelder and Mead (1965): reflection 1, expansion 2,
    /// contraction 1/2 and shrink 1/2, for any number of genes.
    Standard,
    /// Coefficients of your own, satisfying the conditions of Nelder and Mead (Lagarias et al.,
    /// 1998, eq. 2.1): `reflection` > 0, `expansion` > 1 and > `reflection`, and `contraction`
    /// and `shrink` between 0 and 1 (exclusive).
    Custom {
        /// ρ: the reflection is `x̄ + ρ (x̄ − x_worst)`, with `x̄` the centroid of the other
        /// vertices.
        reflection: f64,
        /// χ: the expansion is `x̄ + ρ χ (x̄ − x_worst)`.
        expansion: f64,
        /// γ: the outside contraction is `x̄ + ρ γ (x̄ − x_worst)`, the inside one
        /// `x̄ − γ (x̄ − x_worst)`.
        contraction: f64,
        /// σ: a shrink moves every vertex to `x_best + σ (x − x_best)`.
        shrink: f64,
    },
}

impl Coefficients {
    /// The reflection, expansion, contraction and shrink coefficients (ρ, χ, γ, σ) for `n`
    /// searched genes.
    ///
    /// ```
    /// use genoxide::algorithm::nelder_mead::Coefficients;
    ///
    /// assert_eq!(Coefficients::Adaptive.values(2), Coefficients::Standard.values(2));
    /// assert_eq!(Coefficients::Adaptive.values(10), [1.0, 1.2, 0.7, 0.9]);
    /// ```
    pub fn values(self, n: usize) -> [f64; 4] {
        match self {
            Coefficients::Adaptive if n >= 2 => {
                let n = n as f64;
                [1.0, 1.0 + 2.0 / n, 0.75 - 1.0 / (2.0 * n), 1.0 - 1.0 / n]
            }
            Coefficients::Adaptive | Coefficients::Standard => [1.0, 2.0, 0.5, 0.5],
            Coefficients::Custom {
                reflection,
                expansion,
                contraction,
                shrink,
            } => [reflection, expansion, contraction, shrink],
        }
    }

    fn validate(self) -> Result<()> {
        if let Coefficients::Custom {
            reflection,
            expansion,
            contraction,
            shrink,
        } = self
        {
            // an infinite expansion fails `expansion.is_finite()`, and with it any reflection
            // below it is finite
            let valid = reflection > 0.0
                && expansion > 1.0
                && expansion > reflection
                && expansion.is_finite()
                && contraction > 0.0
                && contraction < 1.0
                && shrink > 0.0
                && shrink < 1.0;
            if !valid {
                return Err(Error::InvalidSetting {
                    setting: "coefficients",
                    reason: format!(
                        "need reflection > 0, finite expansion > 1 and > reflection, and contraction and shrink between 0 and 1 (exclusive); got {reflection}, {expansion}, {contraction} and {shrink}"
                    ),
                });
            }
        }
        Ok(())
    }
}

// what the next ask of a `NelderMead` evaluates
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum Step {
    // the vertices of a new simplex: the first one, or after a restart
    Start,
    // the reflection; speculative, also the expansion and both contractions
    Reflect,
    Expand,
    ContractOutside,
    ContractInside,
    // the n vertices moved towards the best one
    Shrink,
}

// what an iteration does once the values it needs are known
enum Decision {
    // accept the trial point at this index in place of the worst vertex
    Accept(usize),
    Shrink,
    // evaluate the point of this step first
    Ask(Step),
}

/// The Nelder-Mead simplex method on [`Real`] genomes, as an ask / tell [`Algorithm`]: a local
/// method without derivatives, for low dimensions (up to about 10 genes, more with the adaptive
/// coefficients) and for functions that are non-smooth or noisy in their last digits.
///
/// It keeps a simplex of `n + 1` points for `n` searched genes, ordered from best to worst. Each
/// iteration replaces the worst point by one on the line through it and the centroid `x̄` of the
/// others: the reflection `x̄ + ρ (x̄ − x_worst)`, if it's neither the best nor the worst;
/// further out, the expansion, if the reflection is the best; nearer, a contraction, if the
/// reflection is no better than the second worst. If the contraction isn't better either, the
/// whole simplex shrinks towards its best point. The steps, their conditions, strict or not,
/// and the order of tied points are those of Lagarias, Reeds, Wright and Wright (1998,
/// section 2), "the" Nelder-Mead algorithm; the default coefficients are Gao and Han's (2012)
/// ([`Coefficients`]).
///
/// The method only compares fitness values, never subtracts them, so it works for any fitness:
/// invalid points ([`Fitness::invalid`]) are worse than any other, and constrained ones are
/// compared by Deb's rules, as everywhere in genoxide.
///
/// - **Start.** The first simplex is the initial genome (random by default) and, for each
///   searched gene, a copy moved by [`initial_step`](NelderMeadBuilder::initial_step) of the
///   gene's range: up, or down if up leaves the bounds. Genes whose bounds are equal stay fixed
///   and aren't searched.
/// - **Bounds.** Points outside the bounds are moved to the nearest point inside, gene by gene.
///   The simplex can then flatten against a bound; when the minimum is on it, that's where it
///   converges.
/// - **Convergence.** A run has converged when every vertex is within
///   [`tolerance`](NelderMeadBuilder::tolerance) of the best one, in every gene, as a fraction of
///   the gene's range. Without [`Restarts`] left, the method has then
///   [finished](Algorithm::is_finished): the [`Engine`](crate::Engine) stops with
///   [`StopReason::Converged`](crate::StopReason::Converged). With them, the next ask starts a
///   new simplex at a random point.
/// - **Generations.** A generation is one round of evaluations: the `n + 1` vertices of a new
///   simplex, one trial point, or the `n` points of a shrink. An iteration takes one to three
///   rounds ([`iterations`](NelderMead::iterations) counts them). With
///   [`speculative`](NelderMeadBuilder::speculative) asks, the reflection, the expansion and
///   both contractions are evaluated in one round, for parallel evaluation.
/// - **Best.** [`best`](Algorithm::best) is the best point evaluated, which may be a trial point
///   that wasn't accepted: a rejected expansion, or a speculative point.
///
/// Every operation is a sum, product or quotient in a fixed order, so a seed gives the same run
/// on every platform. The simplex holds `n + 1` genomes: its memory grows as `n²`.
///
/// Built with [`NelderMead::builder`], run with an [`Engine`](crate::Engine).
///
/// ```
/// use genoxide::prelude::*;
///
/// // Rosenbrock's valley, from the classic start (-1.2, 1)
/// let rosenbrock = |x: &Reals| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2);
/// let nelder_mead = NelderMead::builder(Real::uniform(2, -5.0..=5.0)?)
///     .initial_genome(Reals::from(vec![-1.2, 1.0]))
///     .minimize()
///     .build()?;
/// let outcome = Engine::new(nelder_mead, rosenbrock)
///     .stop_when(Stop::evaluations(1_000))
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Converged);
/// assert!(outcome.best_fitness().score().unwrap() < 1e-15);
/// # Ok::<(), genoxide::Error>(())
/// ```
///
/// References: Nelder, J. A. and Mead, R. (1965). A simplex method for function minimization.
/// *The Computer Journal* 7(4): 308-313. Lagarias, J. C., Reeds, J. A., Wright, M. H. and
/// Wright, P. E. (1998). Convergence properties of the Nelder-Mead simplex method in low
/// dimensions. *SIAM Journal on Optimization* 9(1): 112-147. Gao, F. and Han, L. (2012).
/// Implementing the Nelder-Mead simplex algorithm with adaptive parameters. *Computational
/// Optimization and Applications* 51(1): 259-277.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NelderMead {
    real: Real,
    // the genes with more than one value, the dimensions of the simplex
    free: Vec<usize>,
    coefficients: Coefficients,
    // ρ, χ, γ and σ for the searched genes
    reflection: f64,
    expansion: f64,
    contraction: f64,
    shrink: f64,
    initial_step: f64,
    tolerance: f64,
    restarts: Restarts,
    speculative: bool,
    objective: Objective,
    seed: u64,
    rng: StreamRng,
    // the vertices, best first after their first tell
    simplex: Population<Reals>,
    step: Step,
    // the trial points of the iteration: the reflection first, then the expansion and the
    // contractions of a speculative ask, or the point of the round after the reflection
    trials: Vec<Individual<Reals>>,
    discarded: Vec<Individual<Reals>>,
    converged: bool,
    restart_count: u64,
    iterations: u64,
    // a re-evaluation of the simplex for the next ask
    reevaluating: bool,
    started: bool,
    asked: bool,
    pending: Vec<usize>,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<Reals>>,
    best_generation: u64,
}

impl NelderMead {
    /// A builder for a Nelder-Mead search on `real`.
    pub fn builder(real: Real) -> NelderMeadBuilder {
        NelderMeadBuilder {
            real,
            coefficients: Coefficients::default(),
            initial_step: 0.1,
            tolerance: 1e-10,
            restarts: Restarts::Never,
            speculative: false,
            initial_genome: None,
            objective: Objective::default(),
            seed: None,
        }
    }

    /// The representation.
    pub fn real(&self) -> &Real {
        &self.real
    }

    /// The coefficients.
    pub fn coefficients(&self) -> Coefficients {
        self.coefficients
    }

    /// The size of the simplex: the largest difference between a vertex and the best vertex in
    /// any searched gene, as a fraction of the gene's range. The run has converged once it's
    /// within the [tolerance](NelderMeadBuilder::tolerance).
    pub fn size(&self) -> f64 {
        let vertices = self.simplex.as_slice();
        let Some((first, others)) = vertices.split_first() else {
            return 0.0;
        };
        let bounds = self.real.bounds();
        let mut size = 0.0f64;
        for vertex in others {
            for &gene in &self.free {
                let width = bounds[gene].end() - bounds[gene].start();
                let difference = (vertex.genome()[gene] - first.genome()[gene]).abs();
                size = size.max(difference / width);
            }
        }
        size
    }

    /// Whether the current run has converged: its simplex is within the
    /// [tolerance](NelderMeadBuilder::tolerance). With [`Restarts`] left, the next
    /// [`ask`](Algorithm::ask) starts a new run.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// The number of completed iterations, of every run: those that replaced the worst vertex,
    /// and the shrinks.
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// The number of restarts so far.
    pub fn restart_count(&self) -> u64 {
        self.restart_count
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Marks the simplex as not evaluated, for a fitness function that changed during the run.
    /// The next [`ask`](Algorithm::ask) gives its vertices, and its [`tell`](Algorithm::tell)
    /// sets their fitness and orders them again, without a step.
    ///
    /// - It isn't a generation: [`generation`](Algorithm::generation) doesn't change. The
    ///   evaluations are counted.
    /// - An iteration under way, waiting for its expansion, contraction or shrink, is dropped:
    ///   the next one starts from the reordered simplex. A restart that was due still happens
    ///   at the next ask after the re-evaluation.
    /// - [`best`](Algorithm::best) is then the best vertex, found in the current generation: old
    ///   and new values are never compared.
    /// - No random number is drawn: a seeded run that re-evaluates at the same points gives the
    ///   same results.
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

    // the fitness of a re-evaluation: the simplex in its new order, and the best of it
    #[cold]
    #[inline(never)]
    fn rescore(&mut self, fitness: &[Fitness]) {
        for (vertex, &fitness) in self.simplex.iter_mut().zip(fitness) {
            vertex.set_fitness(fitness);
        }
        self.simplex.sort_best_first(self.objective);
        self.best = Some(self.simplex[0].clone());
        self.best_generation = self.generation;
        self.trials.clear();
        if self.step != Step::Start {
            self.step = Step::Reflect;
        }
        self.reevaluating = false;
    }

    // a new simplex around a random point
    fn restart(&mut self) {
        self.restart_count += 1;
        let start = self.real.random_genome(&mut self.rng);
        self.simplex = Population::new(simplex_around(
            &self.real,
            &self.free,
            self.initial_step,
            start,
        ));
        self.converged = false;
        self.trials.clear();
    }

    // the centroid of every vertex but the worst
    fn centroid(&self) -> Vec<f64> {
        let n = self.free.len();
        let mut centroid = vec![0.0; self.real.bounds().len()];
        // each vertex divided by n, so that wide bounds don't overflow the sum
        let count = n as f64;
        for vertex in &self.simplex.as_slice()[..n] {
            for (sum, x) in centroid.iter_mut().zip(vertex.genome().iter()) {
                *sum += x / count;
            }
        }
        centroid
    }

    // `x̄ + t (x̄ − x_worst)`, moved into the bounds
    fn along(&self, centroid: &[f64], t: f64) -> Reals {
        let worst = self.simplex[self.free.len()].genome();
        let bounds = self.real.bounds();
        bounds
            .iter()
            .zip(centroid)
            .zip(worst.iter())
            .map(|((range, &c), &w)| (c + t * (c - w)).clamp(*range.start(), *range.end()))
            .collect()
    }

    // `self.trials[index]`'s fitness, evaluated by now
    fn trial_fitness(&self, index: usize) -> Fitness {
        self.trials[index].fitness().unwrap_or(Fitness::invalid())
    }

    // what the iteration does with the trial points evaluated so far
    fn decide(&self) -> Decision {
        let objective = self.objective;
        let n = self.free.len();
        let vertex = |index: usize| -> Fitness {
            self.simplex[index].fitness().unwrap_or(Fitness::invalid())
        };
        let better = |a, b| objective.is_better(a, b);
        let (best, second_worst, worst) = (vertex(0), vertex(n - 1), vertex(n));
        let reflected = self.trial_fitness(0);
        // Lagarias et al., steps 2 to 4: f1 <= fr < fn accepts the reflection; fr < f1
        // expands; fn <= fr < fn+1 contracts outside, and fr >= fn+1 inside
        let next = if better(reflected, best) {
            Step::Expand
        } else if better(reflected, second_worst) {
            return Decision::Accept(0);
        } else if better(reflected, worst) {
            Step::ContractOutside
        } else {
            Step::ContractInside
        };
        // its point: at its place in a speculative ask, or after the reflection in the next round
        let index = match (self.speculative, next) {
            (false, _) => 1,
            (true, Step::Expand) => 1,
            (true, Step::ContractOutside) => 2,
            (true, _) => 3,
        };
        let Some(fitness) = self.trials.get(index).and_then(Individual::fitness) else {
            return Decision::Ask(next);
        };
        match next {
            // fe < fr accepts the expansion, else the reflection
            Step::Expand if better(fitness, reflected) => Decision::Accept(index),
            Step::Expand => Decision::Accept(0),
            // fc <= fr accepts the outside contraction
            Step::ContractOutside if !better(reflected, fitness) => Decision::Accept(index),
            // fcc < fn+1 accepts the inside contraction
            Step::ContractInside if better(fitness, worst) => Decision::Accept(index),
            _ => Decision::Shrink,
        }
    }

    // the trial point at `index` in place of the worst vertex, after every vertex at least as
    // good (Lagarias et al.'s ordering rule: the highest place its fitness allows)
    fn accept(&mut self, index: usize) {
        let accepted = self.trials.remove(index);
        self.discarded.append(&mut self.trials);
        let fitness = accepted.fitness().unwrap_or(Fitness::invalid());
        let objective = self.objective;
        let vertices = self.simplex.individuals_mut();
        vertices.pop();
        let place = vertices.partition_point(|vertex| {
            !objective.is_better(fitness, vertex.fitness().unwrap_or(Fitness::invalid()))
        });
        vertices.insert(place, accepted);
        self.end_iteration();
    }

    // the vertices of a shrink in place of all but the best, ordered again: the best stays first
    // on ties (Lagarias et al.'s shrink rule), and the others keep their order
    fn end_shrink(&mut self) {
        let vertices = self.simplex.individuals_mut();
        for (vertex, shrunk) in vertices[1..].iter_mut().zip(self.trials.drain(..)) {
            *vertex = shrunk;
        }
        self.simplex.sort_best_first(self.objective);
        self.end_iteration();
    }

    fn end_iteration(&mut self) {
        self.iterations += 1;
        self.step = Step::Reflect;
        self.check_convergence();
    }

    // whether the run has converged, and if so whether a restart is due
    fn check_convergence(&mut self) {
        self.converged = self.size() <= self.tolerance;
        if self.converged && self.restart_count < self.restarts.times() {
            self.step = Step::Start;
        }
    }
}

// the best so far, from the evaluated `individuals` in order: the first on ties
fn update_best<'a>(
    best: &mut Option<Individual<Reals>>,
    best_generation: &mut u64,
    generation: u64,
    objective: Objective,
    individuals: impl Iterator<Item = &'a Individual<Reals>>,
) {
    for individual in individuals {
        let fitness = individual.fitness().unwrap_or(Fitness::invalid());
        let better = best.as_ref().is_none_or(|best| {
            objective.is_better(fitness, best.fitness().unwrap_or(Fitness::invalid()))
        });
        if better {
            *best = Some(individual.clone());
            *best_generation = generation;
        }
    }
}

// the first simplex of a run: `start`, and for each searched gene a copy moved by `step` of its
// range: up, or down if up leaves the bounds, or to the farther bound if both do
fn simplex_around(real: &Real, free: &[usize], step: f64, start: Reals) -> Vec<Individual<Reals>> {
    let bounds = real.bounds();
    let mut vertices = Vec::with_capacity(free.len() + 1);
    for &gene in free {
        let (low, high) = (*bounds[gene].start(), *bounds[gene].end());
        let distance = step * (high - low);
        let x = start[gene];
        let moved = if x + distance <= high {
            x + distance
        } else if x - distance >= low {
            x - distance
        } else if high - x >= x - low {
            high
        } else {
            low
        };
        let mut genome = start.clone();
        genome[gene] = moved;
        vertices.push(Individual::new(genome));
    }
    vertices.insert(0, Individual::new(start));
    vertices
}

impl Reevaluate for NelderMead {
    /// As [`NelderMead::reevaluate`]: the next ask gives the simplex.
    fn reevaluate(&mut self) -> Result<()> {
        NelderMead::reevaluate(self)
    }
}

impl Algorithm for NelderMead {
    type Genome = Reals;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, Reals> {
        if !self.asked {
            self.pending.clear();
            if self.reevaluating {
                self.pending.extend(0..self.simplex.len());
            } else {
                let [rho, chi, gamma, sigma] = [
                    self.reflection,
                    self.expansion,
                    self.contraction,
                    self.shrink,
                ];
                match self.step {
                    Step::Start => {
                        if self.started {
                            self.restart();
                        }
                        self.pending.extend(0..self.simplex.len());
                    }
                    Step::Reflect => {
                        let centroid = self.centroid();
                        self.trials.clear();
                        self.trials
                            .push(Individual::new(self.along(&centroid, rho)));
                        if self.speculative {
                            for t in [rho * chi, rho * gamma, -gamma] {
                                self.trials.push(Individual::new(self.along(&centroid, t)));
                            }
                        }
                        self.pending.extend(0..self.trials.len());
                    }
                    Step::Expand | Step::ContractOutside | Step::ContractInside => {
                        let t = match self.step {
                            Step::Expand => rho * chi,
                            Step::ContractOutside => rho * gamma,
                            _ => -gamma,
                        };
                        let centroid = self.centroid();
                        let point = Individual::new(self.along(&centroid, t));
                        self.trials.truncate(1);
                        self.trials.push(point);
                        self.pending.push(1);
                    }
                    Step::Shrink => {
                        self.trials.clear();
                        let bounds = self.real.bounds();
                        let best = self.simplex[0].genome();
                        for vertex in &self.simplex.as_slice()[1..] {
                            let genes = bounds.iter().zip(best.iter()).zip(vertex.genome().iter());
                            let genome: Reals = genes
                                .map(|((range, &b), &x)| {
                                    (b + sigma * (x - b)).clamp(*range.start(), *range.end())
                                })
                                .collect();
                            self.trials.push(Individual::new(genome));
                        }
                        self.pending.extend(0..self.trials.len());
                    }
                }
            }
            self.asked = true;
        }
        let individuals = if self.reevaluating || self.step == Step::Start {
            self.simplex.as_slice()
        } else {
            &self.trials
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
        self.discarded.clear();
        if self.reevaluating {
            self.rescore(fitness);
            return Ok(());
        }
        if self.started {
            self.generation += 1;
        }
        if self.step == Step::Start {
            for (vertex, &fitness) in self.simplex.iter_mut().zip(fitness) {
                vertex.set_fitness(fitness);
            }
            self.started = true;
            update_best(
                &mut self.best,
                &mut self.best_generation,
                self.generation,
                self.objective,
                self.simplex.iter(),
            );
            self.simplex.sort_best_first(self.objective);
            self.step = Step::Reflect;
            self.check_convergence();
            return Ok(());
        }
        for (&index, &fitness) in self.pending.iter().zip(fitness) {
            self.trials[index].set_fitness(fitness);
        }
        let trials = &self.trials;
        update_best(
            &mut self.best,
            &mut self.best_generation,
            self.generation,
            self.objective,
            self.pending.iter().map(|&index| &trials[index]),
        );
        if self.step == Step::Shrink {
            self.end_shrink();
            return Ok(());
        }
        match self.decide() {
            Decision::Accept(index) => self.accept(index),
            Decision::Shrink => {
                self.discarded.append(&mut self.trials);
                self.step = Step::Shrink;
            }
            Decision::Ask(step) => self.step = step,
        }
        Ok(())
    }

    fn population(&self) -> &Population<Reals> {
        &self.simplex
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

    /// Whether the run has converged with no restart left.
    fn is_finished(&self) -> bool {
        self.converged && self.step != Step::Start
    }
}

/// A builder for a [`NelderMead`], from [`NelderMead::builder`].
///
/// Defaults: maximize, [`Coefficients::Adaptive`], an initial step of 0.1 and a tolerance of
/// 1e-10 of each gene's range, no restarts, one point per round, a random initial genome and a
/// random seed.
#[derive(Clone, Debug)]
pub struct NelderMeadBuilder {
    real: Real,
    coefficients: Coefficients,
    initial_step: f64,
    tolerance: f64,
    restarts: Restarts,
    speculative: bool,
    initial_genome: Option<Reals>,
    objective: Objective,
    seed: Option<u64>,
}

impl NelderMeadBuilder {
    /// The coefficients of reflection, expansion, contraction and shrink.
    /// [`Coefficients::Adaptive`] by default.
    pub fn coefficients(mut self, coefficients: Coefficients) -> Self {
        self.coefficients = coefficients;
        self
    }

    /// The size of the first simplex of every run, as a fraction of each gene's range, greater
    /// than 0 and at most 1: 0.1 by default. A larger simplex looks further at the start, a
    /// smaller one stays near the initial genome.
    pub fn initial_step(mut self, fraction: f64) -> Self {
        self.initial_step = fraction;
        self
    }

    /// The simplex size at which a run has converged ([`NelderMead::size`]), as a fraction of
    /// each gene's range, greater than 0 and smaller than the initial step: 1e-10 by default.
    pub fn tolerance(mut self, fraction: f64) -> Self {
        self.tolerance = fraction;
        self
    }

    /// Whether to start again from random points when a run has converged.
    /// [`Restarts::Never`] by default.
    pub fn restarts(mut self, restarts: Restarts) -> Self {
        self.restarts = restarts;
        self
    }

    /// Evaluates the reflection, the expansion and both contractions of an iteration in one
    /// round, instead of one or two at a time: the same simplexes in fewer rounds, for 4
    /// evaluations per iteration and its shrink. It pays off when the fitness function is slow
    /// and evaluated in parallel ([`Engine::parallel`](crate::Engine::parallel)). Off by default.
    pub fn speculative(mut self, speculative: bool) -> Self {
        self.speculative = speculative;
        self
    }

    /// The genome to start from, e.g. a known good design or the best of a global method.
    /// Random by default. Restarts start from random points.
    pub fn initial_genome(mut self, genome: Reals) -> Self {
        self.initial_genome = Some(genome);
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

    /// The seed of the random numbers, for a reproducible run: the initial genome, unless it's
    /// given, and the starts of restarts. Random by default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the settings and creates the search, with its first simplex.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidSetting`] for a representation without a gene that has more than one
    ///   value, invalid custom coefficients, an initial step that isn't greater than 0 and at
    ///   most 1, a tolerance that isn't greater than 0 and smaller than the initial step, or
    ///   random restarts 0 times.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<NelderMead> {
        let invalid =
            |setting: &'static str, reason: String| Err(Error::InvalidSetting { setting, reason });
        let free = self.real.variable_genes().to_vec();
        if free.is_empty() {
            return invalid(
                "real",
                "Nelder-Mead needs a gene with more than one value".to_string(),
            );
        }
        self.coefficients.validate()?;
        self.restarts.validate()?;
        if !(self.initial_step > 0.0 && self.initial_step <= 1.0) {
            return invalid(
                "initial_step",
                format!(
                    "must be greater than 0 and at most 1, got {}",
                    self.initial_step
                ),
            );
        }
        if !(self.tolerance > 0.0 && self.tolerance < self.initial_step) {
            return invalid(
                "tolerance",
                format!(
                    "must be greater than 0 and smaller than the initial step {}, got {}",
                    self.initial_step, self.tolerance
                ),
            );
        }
        if let Some(genome) = &self.initial_genome {
            self.real.validate(genome)?;
        }
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let mut rng = StreamRng::seed_from_u64(seed);
        let start = match self.initial_genome {
            Some(genome) => genome,
            None => self.real.random_genome(&mut rng),
        };
        let simplex = simplex_around(&self.real, &free, self.initial_step, start);
        let [reflection, expansion, contraction, shrink] = self.coefficients.values(free.len());
        Ok(NelderMead {
            real: self.real,
            free,
            coefficients: self.coefficients,
            reflection,
            expansion,
            contraction,
            shrink,
            initial_step: self.initial_step,
            tolerance: self.tolerance,
            restarts: self.restarts,
            speculative: self.speculative,
            objective: self.objective,
            seed,
            rng,
            simplex: Population::new(simplex),
            step: Step::Start,
            trials: Vec::new(),
            discarded: Vec::new(),
            converged: false,
            restart_count: 0,
            iterations: 0,
            reevaluating: false,
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
