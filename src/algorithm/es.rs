//! Evolution strategies: (μ/ρ +, λ)-ES with self-adapted step sizes.

use super::{Algorithm, Candidates};
use crate::genome::{Real, Reals, Representation};
use crate::math::{exp, log};
use crate::operator::mutate::{MAX_STEP, reflect};
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;

/// How an [`Es`] combines `rho` random parents (ρ) into an offspring, before mutation. With
/// `rho` 1, an offspring is a mutated copy of one random parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Recombination {
    /// Intermediate recombination (μ/ρ_I): the centroid of the parents' genes, and the geometric
    /// mean of their step sizes. Averages out the parents' errors: (μ/μ_I, λ) with all parents is
    /// the most common choice. (The geometric mean suits step sizes that mutate log-normally: with
    /// the arithmetic mean, per-gene step sizes collapsed prematurely in half of the runs on a
    /// 10-gene ellipsoid.)
    Intermediate {
        /// The number of parents per offspring, from 1 to μ.
        rho: usize,
    },
    /// Dominant, or discrete, recombination (μ/ρ_D): each gene, with its step size, comes from
    /// a random one of the parents.
    Dominant {
        /// The number of parents per offspring, from 1 to μ.
        rho: usize,
    },
}

/// Which individuals survive in an [`Es`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Selection {
    /// (μ,λ): the best μ offspring become the parents, which never survive. Forgets misadapted
    /// step sizes, so it suits self-adaptation best (the default). λ is at least μ.
    #[default]
    Comma,
    /// (μ+λ): the best μ of the parents and the offspring survive, offspring first on ties.
    /// Elitist: never loses the best solution, but can keep poor step sizes.
    Plus,
}

/// The step sizes (mutation strengths) of each individual of an [`Es`], which evolve with it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepSizes {
    /// One step size for all genes, with the learning rate `τ = 1/√n` for `n` genes.
    One,
    /// A step size per gene (the default), with a global and a per-gene learning rate,
    /// `τ' = 1/√(2n)` and `τ = 1/√(2√n)` (Schwefel), which learns the scaling of each gene.
    #[default]
    PerGene,
}

// the smallest step size, above 0 so that it can grow again
const MIN_STEP: f64 = f64::MIN_POSITIVE;

/// A (μ/ρ +, λ) evolution strategy on [`Real`] genomes with self-adaptation, as an ask / tell
/// [`Algorithm`].
///
/// Each generation creates λ offspring from μ parents: [`Recombination`] of ρ random parents,
/// then a mutation of the step sizes ([`StepSizes`]), log-normal as in Schwefel's ES, then a
/// Gaussian mutation of each gene with its step size, reflected into the bounds. The step sizes
/// are fractions of each gene's range, and good step sizes survive with the good solutions that
/// they produced. [`Selection`] chooses the next parents.
///
/// Built with [`Es::builder`], run with an [`Engine`](crate::Engine).
///
/// ```
/// use genoxide::prelude::*;
///
/// // (5/5_I, 35)-ES with a step size per gene
/// let es = Es::builder(Real::uniform(10, -5.0..=5.0)?)
///     .parents(5)
///     .offspring(35)
///     .minimize()
///     .seed(1)
///     .build()?;
/// let ellipsoid = |x: &Reals| {
///     x.iter().enumerate().map(|(i, xi)| (i + 1) as f64 * xi * xi).sum::<f64>()
/// };
/// let outcome = Engine::new(es, ellipsoid)
///     .stop_when(Stop::target(1e-10).or(Stop::evaluations(100_000)))
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Target);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Es {
    real: Real,
    mu: usize,
    lambda: usize,
    recombination: Recombination,
    selection: Selection,
    step_sizes: StepSizes,
    objective: Objective,
    seed: u64,
    rng: StreamRng,
    // the parents, and the step sizes of each: one, or one per variable gene
    population: Population<Reals>,
    steps: Vec<Vec<f64>>,
    offspring: Vec<Individual<Reals>>,
    offspring_steps: Vec<Vec<f64>>,
    discarded: Vec<Individual<Reals>>,
    started: bool,
    asked: bool,
    pending: Vec<usize>,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<Reals>>,
    best_generation: u64,
}

impl Es {
    /// A builder for an evolution strategy on `real`.
    pub fn builder(real: Real) -> EsBuilder {
        EsBuilder {
            real,
            parents: None,
            offspring: None,
            recombination: None,
            selection: Selection::Comma,
            step_sizes: StepSizes::PerGene,
            initial_step: 0.3,
            objective: Objective::default(),
            seed: None,
            initial_genomes: Vec::new(),
        }
    }

    /// The representation.
    pub fn real(&self) -> &Real {
        &self.real
    }

    /// How parents are combined.
    pub fn recombination(&self) -> Recombination {
        self.recombination
    }

    /// The step sizes of the parents, in the order of the population, as fractions of the
    /// ranges: one per parent with [`StepSizes::One`], or one per gene that has more than one
    /// value with [`StepSizes::PerGene`].
    pub fn step_sizes(&self) -> &[Vec<f64>] {
        &self.steps
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn fitness(individual: &Individual<Reals>) -> Fitness {
        individual.fitness().unwrap_or(Fitness::invalid())
    }

    // a recombined and mutated offspring, and its step sizes
    fn offspring(&mut self) -> (Reals, Vec<f64>) {
        let variable = self.real.variable_genes();
        let n = variable.len();
        let (rho, dominant) = match self.recombination {
            Recombination::Intermediate { rho } => (rho, false),
            Recombination::Dominant { rho } => (rho, true),
        };
        let parents = self.rng.sample_distinct(rho, self.mu);
        let mut genes = self.population[parents[0]].genome().clone();
        let mut steps = self.steps[parents[0]].clone();
        if rho > 1 && dominant {
            for (index, &gene) in variable.iter().enumerate() {
                let parent = parents[self.rng.below(rho)];
                genes[gene] = self.population[parent].genome()[gene];
                if self.step_sizes == StepSizes::PerGene {
                    steps[index] = self.steps[parent][index];
                }
            }
            if self.step_sizes == StepSizes::One {
                steps = self.steps[parents[self.rng.below(rho)]].clone();
            }
        } else if rho > 1 {
            let weight = 1.0 / rho as f64;
            for &gene in variable {
                genes[gene] = parents
                    .iter()
                    .map(|&parent| weight * self.population[parent].genome()[gene])
                    .sum();
            }
            // the geometric mean of the step sizes, which mutate log-normally
            for (index, step) in steps.iter_mut().enumerate() {
                let log_mean: f64 = parents
                    .iter()
                    .map(|&parent| weight * log(self.steps[parent][index]))
                    .sum();
                *step = exp(log_mean);
            }
        }
        // the log-normal mutation of the step sizes
        let dimensions = n as f64;
        match self.step_sizes {
            StepSizes::One => {
                let tau = 1.0 / dimensions.sqrt();
                steps[0] = (steps[0] * exp(tau * self.rng.normal())).clamp(MIN_STEP, MAX_STEP);
            }
            StepSizes::PerGene => {
                let global = self.rng.normal() / (2.0 * dimensions).sqrt();
                let tau = 1.0 / (2.0 * dimensions.sqrt()).sqrt();
                for step in &mut steps {
                    *step =
                        (*step * exp(global + tau * self.rng.normal())).clamp(MIN_STEP, MAX_STEP);
                }
            }
        }
        // the Gaussian mutation of the genes, reflected into the bounds
        let bounds = self.real.bounds();
        for (index, &gene) in variable.iter().enumerate() {
            let range = &bounds[gene];
            let step = steps[if self.step_sizes == StepSizes::One {
                0
            } else {
                index
            }];
            let current = genes[gene];
            let value = reflect(
                current + step * (range.end() - range.start()) * self.rng.normal(),
                range,
            );
            genes[gene] = if range.contains(&value) {
                value
            } else {
                // not computable with such huge bounds (NaN): a uniform value instead
                crate::genome::real::random_other_in(range, current, &mut self.rng)
            };
        }
        (genes, steps)
    }

    fn breed(&mut self) {
        self.offspring.clear();
        self.offspring_steps.clear();
        for _ in 0..self.lambda {
            let (genes, steps) = self.offspring();
            self.offspring.push(Individual::new(genes));
            self.offspring_steps.push(steps);
        }
    }

    // the next parents, and the offspring that didn't make it as discarded
    fn select(&mut self) {
        let objective = self.objective;
        let offspring = std::mem::take(&mut self.offspring);
        let offspring_steps = std::mem::take(&mut self.offspring_steps);
        // offspring first, so that they win ties
        let mut pool: Vec<(Individual<Reals>, Vec<f64>, bool)> = offspring
            .into_iter()
            .zip(offspring_steps)
            .map(|(individual, steps)| (individual, steps, true))
            .collect();
        if self.selection == Selection::Plus {
            let parents = std::mem::replace(&mut self.population, Population::new(Vec::new()));
            let steps = std::mem::take(&mut self.steps);
            pool.extend(
                parents
                    .into_vec()
                    .into_iter()
                    .zip(steps)
                    .map(|(individual, steps)| (individual, steps, false)),
            );
        }
        // best first, stable
        pool.sort_by(|a, b| objective.compare(Self::fitness(&b.0), Self::fitness(&a.0)));
        self.discarded.clear();
        let mut parents = Vec::with_capacity(self.mu);
        self.steps.clear();
        for (rank, (mut individual, steps, new)) in pool.into_iter().enumerate() {
            if rank < self.mu {
                if !new {
                    individual.increment_age();
                }
                parents.push(individual);
                self.steps.push(steps);
            } else if new {
                self.discarded.push(individual);
            }
        }
        self.population = Population::new(parents);
    }
}

impl Algorithm for Es {
    type Genome = Reals;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, Reals> {
        if !self.asked {
            self.pending.clear();
            if self.started {
                self.breed();
                self.pending.extend(0..self.offspring.len());
            } else {
                self.pending.extend(0..self.population.len());
            }
            self.asked = true;
        }
        let individuals = if self.started {
            &self.offspring
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
        if self.started {
            self.generation += 1;
        }
        let objective = self.objective;
        let individuals = if self.started {
            self.offspring.as_mut_slice()
        } else {
            self.population.iter_mut().into_slice()
        };
        for (individual, &fitness) in individuals.iter_mut().zip(fitness) {
            individual.set_fitness(fitness);
        }
        // the best so far, the first one on ties
        let candidates = if self.started {
            &self.offspring[..]
        } else {
            self.population.as_slice()
        };
        let mut improved = false;
        for candidate in candidates {
            let better = self.best.as_ref().is_none_or(|best| {
                objective.is_better(Self::fitness(candidate), Self::fitness(best))
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

/// A builder for an [`Es`], from [`Es::builder`].
///
/// The numbers of parents and offspring are required. Defaults: intermediate recombination of
/// all parents, comma selection, a step size per gene starting at 0.3 of each range, maximize,
/// random initial parents and a random seed.
#[derive(Clone, Debug)]
pub struct EsBuilder {
    real: Real,
    parents: Option<usize>,
    offspring: Option<usize>,
    recombination: Option<Recombination>,
    selection: Selection,
    step_sizes: StepSizes,
    initial_step: f64,
    objective: Objective,
    seed: Option<u64>,
    initial_genomes: Vec<Reals>,
}

impl EsBuilder {
    /// The number of parents μ, at least 1. Required.
    pub fn parents(mut self, mu: usize) -> Self {
        self.parents = Some(mu);
        self
    }

    /// The number of offspring per generation λ, at least 1, and at least μ with comma
    /// selection; e.g. 5 to 7 times μ. Required.
    pub fn offspring(mut self, lambda: usize) -> Self {
        self.offspring = Some(lambda);
        self
    }

    /// How parents are combined. Intermediate recombination of all μ parents by default.
    pub fn recombination(mut self, recombination: Recombination) -> Self {
        self.recombination = Some(recombination);
        self
    }

    /// Which individuals survive. [`Selection::Comma`] by default.
    pub fn selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    /// One step size per individual, or one per gene. [`StepSizes::PerGene`] by default.
    pub fn step_sizes(mut self, step_sizes: StepSizes) -> Self {
        self.step_sizes = step_sizes;
        self
    }

    /// The initial step size as a fraction of each gene's range, greater than 0 and at most 10.
    /// 0.3 by default.
    pub fn initial_step(mut self, fraction: f64) -> Self {
        self.initial_step = fraction;
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

    /// Genomes for the initial parents, at most μ, each valid for the representation. The rest
    /// is random.
    pub fn initial_genomes<I: IntoIterator<Item = Reals>>(mut self, genomes: I) -> Self {
        self.initial_genomes = genomes.into_iter().collect();
        self
    }

    /// Validates the settings and creates the algorithm, with its initial parents.
    ///
    /// # Errors
    ///
    /// - [`Error::MissingSetting`] without the number of parents or offspring.
    /// - [`Error::InvalidSetting`] for no parents or offspring, fewer offspring than parents with
    ///   comma selection, `rho` out of range, an initial step size out of range, or a
    ///   representation without a gene that has more than one value.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<Es> {
        let mu = self
            .parents
            .ok_or(Error::MissingSetting { setting: "parents" })?;
        let lambda = self.offspring.ok_or(Error::MissingSetting {
            setting: "offspring",
        })?;
        let invalid = |setting, reason: String| Err(Error::InvalidSetting { setting, reason });
        if mu == 0 {
            return invalid("parents", "must be at least 1".to_string());
        }
        if lambda == 0 || (self.selection == Selection::Comma && lambda < mu) {
            return invalid(
                "offspring",
                format!(
                    "must be at least 1, and at least the {mu} parents with comma selection, got {lambda}"
                ),
            );
        }
        let recombination = self
            .recombination
            .unwrap_or(Recombination::Intermediate { rho: mu });
        let (Recombination::Intermediate { rho } | Recombination::Dominant { rho }) = recombination;
        if rho == 0 || rho > mu {
            return invalid(
                "rho",
                format!("must be between 1 and the {mu} parents, got {rho}"),
            );
        }
        if !(self.initial_step > 0.0 && self.initial_step <= MAX_STEP) {
            return invalid(
                "initial_step",
                format!(
                    "must be greater than 0 and at most {MAX_STEP}, got {}",
                    self.initial_step
                ),
            );
        }
        let n = self.real.variable_genes().len();
        if n == 0 {
            return invalid(
                "real",
                "an evolution strategy needs a gene with more than one value".to_string(),
            );
        }
        if self.initial_genomes.len() > mu {
            return invalid(
                "initial_genomes",
                format!(
                    "at most the {mu} parents, got {}",
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
        let random = mu - self.initial_genomes.len();
        let mut genomes = self.initial_genomes;
        genomes.extend((0..random).map(|_| self.real.random_genome(&mut rng)));
        let steps_per_parent = match self.step_sizes {
            StepSizes::One => 1,
            StepSizes::PerGene => n,
        };
        Ok(Es {
            real: self.real,
            mu,
            lambda,
            recombination,
            selection: self.selection,
            step_sizes: self.step_sizes,
            objective: self.objective,
            seed,
            rng,
            population: Population::from_genomes(genomes),
            steps: vec![vec![self.initial_step; steps_per_parent]; mu],
            offspring: Vec::new(),
            offspring_steps: Vec::new(),
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

    fn sphere(x: &Reals) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    // an ellipsoid with weights from 1 to 10⁴
    fn ellipsoid(x: &Reals) -> f64 {
        let n = x.len() as f64;
        x.iter()
            .enumerate()
            .map(|(i, xi)| crate::math::pow(10.0, 4.0 * i as f64 / (n - 1.0)) * xi * xi)
            .sum()
    }

    fn builder(seed: u64) -> EsBuilder {
        Es::builder(Real::uniform(10, -5.0..=5.0).unwrap())
            .parents(5)
            .offspring(35)
            .minimize()
            .seed(seed)
    }

    fn step(es: &mut Es, f: impl Fn(&Reals) -> f64) {
        let fitness: Vec<Fitness> = es.ask().iter().map(|x| Fitness::new(f(x))).collect();
        es.tell(&fitness).unwrap();
    }

    fn setting(result: Result<Es>) -> &'static str {
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
        let sized = || Es::builder(real()).parents(3).offspring(10);
        assert_eq!(
            setting(Es::builder(real()).offspring(10).build()),
            "parents"
        );
        assert_eq!(setting(Es::builder(real()).parents(3).build()), "offspring");
        assert_eq!(setting(sized().parents(0).build()), "parents");
        assert_eq!(setting(sized().offspring(0).build()), "offspring");
        assert_eq!(setting(sized().offspring(2).build()), "offspring");
        assert!(
            sized()
                .offspring(2)
                .selection(Selection::Plus)
                .build()
                .is_ok()
        );
        let intermediate = |rho| sized().recombination(Recombination::Intermediate { rho });
        assert_eq!(setting(intermediate(0).build()), "rho");
        assert_eq!(setting(intermediate(4).build()), "rho");
        let dominant = |rho| sized().recombination(Recombination::Dominant { rho });
        assert_eq!(setting(dominant(4).build()), "rho");
        assert!(dominant(3).build().is_ok() && intermediate(1).build().is_ok());
        assert_eq!(setting(sized().initial_step(0.0).build()), "initial_step");
        assert_eq!(setting(sized().initial_step(11.0).build()), "initial_step");
        let fixed = Real::new([1.0..=1.0]).unwrap();
        assert_eq!(
            setting(Es::builder(fixed).parents(1).offspring(1).build()),
            "real"
        );
        let genome = || Reals::from(vec![0.5, 0.5]);
        assert_eq!(
            setting(sized().initial_genomes(vec![genome(); 4]).build()),
            "initial_genomes"
        );
        assert!(matches!(
            sized().initial_genomes([Reals::from(vec![0.5])]).build(),
            Err(Error::InvalidGenome { .. })
        ));
        // (μ/μ_I) by default
        assert_eq!(
            sized().build().unwrap().recombination(),
            Recombination::Intermediate { rho: 3 }
        );
    }

    #[test]
    fn ask_tell_protocol() {
        let mut es = builder(0).build().unwrap();
        assert_eq!(es.tell(&[]), Err(Error::TellWithoutAsk));
        assert_eq!(es.ask().len(), 5);
        assert_eq!(
            es.tell(&[Fitness::new(0.0)]),
            Err(Error::FitnessCount {
                expected: 5,
                got: 1
            })
        );
        step(&mut es, sphere);
        assert_eq!((es.generation(), es.evaluations()), (0, 5));
        assert_eq!(es.ask().len(), 35);
        step(&mut es, sphere);
        assert_eq!((es.generation(), es.evaluations()), (1, 40));
        // comma: the parents are the best 5 offspring, the other 30 are discarded
        assert_eq!((es.population().len(), es.discarded().len()), (5, 30));
        assert!(es.population().iter().all(|x| x.age() == 0));
        assert_eq!(es.step_sizes().len(), 5);
        assert!(es.step_sizes().iter().all(|steps| steps.len() == 10));
    }

    #[test]
    fn recombination() {
        // parents at 1 and 2, with step sizes too small to change a gene
        let parents = [Reals::from(vec![1.0; 4]), Reals::from(vec![2.0; 4])];
        let run = |recombination| {
            let mut es = Es::builder(Real::uniform(4, -10.0..=10.0).unwrap())
                .parents(2)
                .offspring(20)
                .recombination(recombination)
                .initial_step(1e-300)
                .initial_genomes(parents.clone())
                .seed(0)
                .build()
                .unwrap();
            step(&mut es, sphere);
            es.ask().iter().cloned().collect::<Vec<Reals>>()
        };
        for offspring in run(Recombination::Intermediate { rho: 2 }) {
            assert_eq!(offspring.to_vec(), [1.5; 4]);
        }
        let dominant: Vec<f64> = run(Recombination::Dominant { rho: 2 })
            .iter()
            .flat_map(|x| x.to_vec())
            .collect();
        assert!(dominant.iter().all(|&x| x == 1.0 || x == 2.0));
        assert!(dominant.contains(&1.0) && dominant.contains(&2.0));
        // copies of one parent
        for offspring in run(Recombination::Intermediate { rho: 1 }) {
            assert!(offspring.to_vec() == [1.0; 4] || offspring.to_vec() == [2.0; 4]);
        }
    }

    #[test]
    fn per_gene_step_sizes_learn_the_scaling() {
        let run = |step_sizes| {
            let es = builder(1).step_sizes(step_sizes).build().unwrap();
            Engine::new(es, ellipsoid)
                .stop_when(Stop::target(1e-10).or(Stop::evaluations(12_000)))
                .run()
                .unwrap()
        };
        let outcome = run(StepSizes::PerGene);
        assert_eq!(outcome.stop_reason(), StopReason::Target);
        assert_eq!(run(StepSizes::One).stop_reason(), StopReason::Evaluations);
        // the steps of the steepest gene are the smallest, by about √10⁴
        let mut es = builder(1).build().unwrap();
        for _ in 0..100 {
            step(&mut es, ellipsoid);
        }
        for steps in es.step_sizes() {
            assert!(steps[0] / steps[9] > 10.0, "{steps:?}");
        }
    }

    #[test]
    fn one_step_size_is_faster_on_the_sphere() {
        let es = builder(2).step_sizes(StepSizes::One).build().unwrap();
        let outcome = Engine::new(es, sphere)
            .stop_when(Stop::target(1e-10).or(Stop::evaluations(3_000)))
            .run()
            .unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Target);
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut es = builder(seed)
                .recombination(Recombination::Dominant { rho: 2 })
                .build()
                .unwrap();
            for _ in 0..20 {
                step(&mut es, sphere);
            }
            es.population().clone()
        };
        assert_eq!(run(3), run(3));
        assert_ne!(run(3), run(4));
    }

    proptest! {
        #[test]
        fn offspring_stay_in_bounds_and_plus_selection_keeps_the_best(
            seed: u64,
            mu in 1usize..5,
            extra in 0usize..10,
            dominant: bool,
            rho_fraction in 0.0..=1.0f64,
            plus: bool,
            one: bool,
            initial_step in 1e-6..=10.0f64,
        ) {
            // one fixed gene, bounds of different widths, and huge bounds
            let real = Real::new([3.0..=3.0, -1.0..=1.0, 0.0..=100.0, -1e-3..=1e-3, -8e307..=8e307])
                .unwrap();
            let rho = 1 + (rho_fraction * (mu - 1) as f64) as usize;
            let mut es = Es::builder(real.clone())
                .parents(mu)
                .offspring(mu + extra)
                .recombination(if dominant {
                    Recombination::Dominant { rho }
                } else {
                    Recombination::Intermediate { rho }
                })
                .selection(if plus { Selection::Plus } else { Selection::Comma })
                .step_sizes(if one { StepSizes::One } else { StepSizes::PerGene })
                .initial_step(initial_step)
                .minimize()
                .seed(seed)
                .build()
                .unwrap();
            // finite despite the huge bounds
            let f = |x: &Reals| x[..4].iter().map(|xi| xi * xi).sum::<f64>() + (x[4] / 1e307).abs();
            step(&mut es, f);
            for _ in 0..10 {
                let best_parent = es.population().best(Objective::Minimize).unwrap().fitness();
                let offspring: Vec<Reals> = es.ask().iter().cloned().collect();
                for x in &offspring {
                    prop_assert!(real.validate(x).is_ok(), "{x:?}");
                }
                step(&mut es, f);
                let steps = es.step_sizes();
                prop_assert_eq!(steps.len(), mu);
                for step in steps.iter().flatten() {
                    prop_assert!((MIN_STEP..=MAX_STEP).contains(step), "{step}");
                }
                if plus {
                    let best = es.population().best(Objective::Minimize).unwrap().fitness();
                    prop_assert!(!Objective::Minimize.is_better(best_parent.unwrap(), best.unwrap()));
                }
            }
        }
    }
}
