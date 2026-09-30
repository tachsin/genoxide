//! OpenAI's evolution strategy (Salimans et al. 2017): a search distribution whose mean follows a
//! gradient estimated from mirrored samples, for problems of thousands of genes and more, such as
//! a neural network's weights.

use super::{Algorithm, Candidates, Reevaluate, breed_in_parallel, breeding_streams};
use crate::genome::{Real, Reals, Representation};
use crate::operator::check_size;
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;

/// How the mean follows the gradient estimate, in units of each gene's range.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Optimizer {
    /// Gradient ascent with momentum: `v = momentum · v + (1 − momentum) · g`, then the mean
    /// moves by `learning_rate · v`.
    Sgd {
        /// The step for a gradient of 1, a fraction of each gene's range: greater than 0 and
        /// finite.
        learning_rate: f64,
        /// How much of the last direction is kept, in `0..1`; 0.9 is common, 0 for none.
        momentum: f64,
    },
    /// Adam (Kingma and Ba, 2015): steps of about `learning_rate` per gene whatever the scale of
    /// the gradient, from averages of the gradient and of its square, corrected for their start
    /// at 0.
    Adam {
        /// The size of a step, a fraction of each gene's range: greater than 0 and finite.
        learning_rate: f64,
        /// The decay of the average of the gradient, in `0..1`; 0.9 is Kingma and Ba's.
        beta1: f64,
        /// The decay of the average of its square, in `0..1`; 0.999 is Kingma and Ba's.
        beta2: f64,
    },
}

impl Optimizer {
    /// Adam with Kingma and Ba's decays, 0.9 and 0.999.
    pub fn adam(learning_rate: f64) -> Self {
        Optimizer::Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
        }
    }

    /// Gradient ascent with momentum.
    pub fn sgd(learning_rate: f64, momentum: f64) -> Self {
        Optimizer::Sgd {
            learning_rate,
            momentum,
        }
    }

    /// The learning rate.
    pub fn learning_rate(self) -> f64 {
        match self {
            Optimizer::Sgd { learning_rate, .. } | Optimizer::Adam { learning_rate, .. } => {
                learning_rate
            }
        }
    }

    fn validate(self) -> Result<()> {
        let invalid = |reason: String| {
            Err(Error::InvalidSetting {
                setting: "optimizer",
                reason,
            })
        };
        let learning_rate = self.learning_rate();
        if !(learning_rate > 0.0 && learning_rate.is_finite()) {
            return invalid(format!(
                "the learning rate must be greater than 0 and finite, got {learning_rate}"
            ));
        }
        let decays = match self {
            Optimizer::Sgd { momentum, .. } => vec![("momentum", momentum)],
            Optimizer::Adam { beta1, beta2, .. } => vec![("beta1", beta1), ("beta2", beta2)],
        };
        for (name, decay) in decays {
            if !(0.0..1.0).contains(&decay) {
                return invalid(format!("{name} must be in 0..1, got {decay}"));
            }
        }
        Ok(())
    }
}

// Adam's denominator term, against a division by 0
const ADAM_EPSILON: f64 = 1e-8;

// the optimizer's memory: the average gradient (SGD's velocity, Adam's first moment), Adam's
// second moment, and Adam's β₁ᵗ and β₂ᵗ, as products so as to be exact
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Moments {
    first: Vec<f64>,
    second: Vec<f64>,
    beta1_power: f64,
    beta2_power: f64,
}

/// OpenAI's evolution strategy (Salimans, Ho, Chen, Sidor and Sutskever, 2017) on [`Real`]
/// genomes, as an ask / tell [`Algorithm`]: the baseline of neuroevolution at scale.
///
/// A mean `m` moves along an estimate of the gradient of the expected fitness of the samples
/// `m + σ ε`, `ε` standard normal, gene by gene in units of each gene's range:
///
/// - **Mirrored sampling:** each generation draws `population_size / 2` perturbations `ε` and
///   asks for `m + σ ε` and `m − σ ε` (Brockhoff et al., 2010), clamped to the bounds.
/// - **Centered ranks:** the samples' fitness is replaced by their rank, spread evenly over
///   `−0.5..=0.5`, the best at 0.5, so the step depends on the order of the samples only. Ties go
///   to the earlier sample; invalid fitness ranks last.
/// - **Gradient:** `g = Σ wᵢ εᵢ / (population_size · σ)`, from the rank weights `wᵢ`, minus
///   [`weight_decay`](OpenEsBuilder::weight_decay) times the mean, which pulls it toward 0.
/// - **Update:** the [`Optimizer`] moves the mean along `g`, Adam by default, and the mean stays
///   within the bounds.
///
/// Its cost per sample is linear in the number of genes, without the covariance matrix of
/// [`Cmaes`](super::Cmaes), so it scales to tens of thousands of genes. The population is the last
/// samples, followed by the mean if it is [evaluated](OpenEsBuilder::evaluate_mean).
///
/// Built with [`OpenEs::builder`], run with an [`Engine`](crate::Engine).
///
/// ```
/// use genoxide::algorithm::open_es::Optimizer;
/// use genoxide::prelude::*;
///
/// let open_es = OpenEs::builder(Real::uniform(100, -5.0..=5.0)?)
///     .population_size(50)
///     .sigma(0.01)
///     .optimizer(Optimizer::adam(0.003))
///     .evaluate_mean(true)
///     .minimize()
///     .seed(1)
///     .build()?;
/// let sphere = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
/// let outcome = Engine::new(open_es, sphere)
///     .stop_when(Stop::target(0.1).or(Stop::generations(2_000)))
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Target);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OpenEs {
    real: Real,
    population_size: usize,
    sigma: f64,
    optimizer: Optimizer,
    weight_decay: f64,
    evaluate_mean: bool,
    parallel_breeding: bool,
    objective: Objective,
    seed: u64,
    rng: StreamRng,
    mean: Vec<f64>,
    moments: Moments,
    // the perturbations of the samples, one per pair
    noise: Vec<Vec<f64>>,
    population: Population<Reals>,
    reevaluating: bool,
    started: bool,
    asked: bool,
    pending: Vec<usize>,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<Reals>>,
    best_generation: u64,
}

impl OpenEs {
    /// A builder for the strategy on `real`.
    pub fn builder(real: Real) -> OpenEsBuilder {
        OpenEsBuilder {
            real,
            population_size: None,
            sigma: 0.02,
            optimizer: Optimizer::adam(0.01),
            weight_decay: 0.0,
            evaluate_mean: false,
            parallel_breeding: false,
            objective: Objective::default(),
            seed: None,
            initial_mean: None,
        }
    }

    /// The representation.
    pub fn real(&self) -> &Real {
        &self.real
    }

    /// The mean of the search distribution.
    pub fn mean(&self) -> Reals {
        Reals::from(self.mean.clone())
    }

    /// The standard deviation of the perturbations, a fraction of each gene's range.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// The optimizer.
    pub fn optimizer(&self) -> Optimizer {
        self.optimizer
    }

    /// The weight decay.
    pub fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    /// Whether the mean is evaluated each generation.
    pub fn evaluate_mean(&self) -> bool {
        self.evaluate_mean
    }

    /// Whether the samples are drawn in parallel, see [`OpenEsBuilder::parallel_breeding`].
    pub fn parallel_breeding(&self) -> bool {
        self.parallel_breeding
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Changes σ during a run (parameter control), e.g. a σ that decays over the run. It applies
    /// from the next [`ask`](Algorithm::ask) that draws samples.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] as for [`OpenEsBuilder::sigma`]. σ doesn't change on errors.
    pub fn set_sigma(&mut self, sigma: f64) -> Result<()> {
        check_sigma(sigma)?;
        self.sigma = sigma;
        Ok(())
    }

    /// Changes the learning rate during a run, e.g. a learning rate that decays over the run. It
    /// applies from the next [`tell`](Algorithm::tell) that moves the mean; the optimizer's
    /// memory is kept.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] as for [`OpenEsBuilder::optimizer`]. Nothing changes on errors.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        let mut optimizer = self.optimizer;
        match &mut optimizer {
            Optimizer::Sgd {
                learning_rate: rate,
                ..
            }
            | Optimizer::Adam {
                learning_rate: rate,
                ..
            } => *rate = learning_rate,
        }
        optimizer.validate()?;
        self.optimizer = optimizer;
        Ok(())
    }

    /// Marks the population, the last samples (and the mean, if evaluated), as not evaluated,
    /// for a fitness function that changed during the run. The next [`ask`](Algorithm::ask)
    /// gives them instead of new samples, and its [`tell`](Algorithm::tell) sets their fitness
    /// without moving the mean, which has learned from them already.
    ///
    /// - It isn't a generation: [`generation`](Algorithm::generation) doesn't change. The
    ///   evaluations are counted.
    /// - [`best`](Algorithm::best) is then the best of the re-evaluated population, found in the
    ///   current generation: old and new values are never compared.
    /// - The mean and the optimizer's memory don't change, and no random number is drawn: a
    ///   seeded run that re-evaluates at the same points gives the same results.
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

    // new perturbations and the samples from them, and the mean if evaluated
    fn sample(&mut self) {
        let genes = self.mean.len();
        let pairs = self.population_size / 2;
        if self.parallel_breeding {
            // a stream per pair, from the seed, the generation and the pair's position: the same
            // samples on any number of threads
            let streams = self
                .rng
                .derive(breeding_streams::OPEN_ES)
                .derive(self.generation);
            let mut units = Vec::new();
            breed_in_parallel(
                vec![(); pairs],
                &streams,
                |_, (), rng| ((0..genes).map(|_| rng.normal()).collect(), ()),
                &mut self.noise,
                &mut units,
            );
        } else {
            self.noise.resize_with(pairs, Vec::new);
            for epsilon in &mut self.noise {
                epsilon.clear();
                epsilon.extend((0..genes).map(|_| self.rng.normal()));
            }
        }
        let bounds = self.real.bounds();
        let sigma = self.sigma;
        let point = |sign: f64, epsilon: &[f64]| -> Reals {
            self.mean
                .iter()
                .zip(epsilon)
                .zip(bounds)
                .map(|((&m, &e), range)| {
                    let (start, end) = (*range.start(), *range.end());
                    (m + sign * sigma * (end - start) * e).clamp(start, end)
                })
                .collect()
        };
        let mut genomes = Vec::with_capacity(self.population_size + 1);
        for epsilon in &self.noise {
            genomes.push(point(1.0, epsilon));
            genomes.push(point(-1.0, epsilon));
        }
        if self.evaluate_mean {
            genomes.push(Reals::from(self.mean.clone()));
        }
        self.population = Population::from_genomes(genomes);
    }

    // the rank weights of the samples, in their order: from 0.5 for the best to −0.5 for the
    // worst, the earlier sample first on ties
    fn rank_weights(&self) -> Vec<f64> {
        let n = self.population_size;
        let objective = self.objective;
        let fitness = |index: usize| {
            self.population[index]
                .fitness()
                .unwrap_or(Fitness::invalid())
        };
        let mut order: Vec<usize> = (0..n).collect();
        // best first, stable: the earlier sample first on ties
        order.sort_by(|&a, &b| objective.compare(fitness(b), fitness(a)));
        let mut weights = vec![0.0; n];
        let last = (n - 1) as f64;
        for (rank, &index) in order.iter().enumerate() {
            weights[index] = 0.5 - rank as f64 / last;
        }
        weights
    }

    // moves the mean along the gradient estimate
    fn update(&mut self) {
        let gradient = self.gradient();
        self.step(&gradient);
    }

    // the estimate of the gradient of the ranks, in units of each gene's range, with the weight
    // decay
    fn gradient(&self) -> Vec<f64> {
        let weights = self.rank_weights();
        let genes = self.mean.len();
        let scale = 1.0 / (self.population_size as f64 * self.sigma);
        let mut gradient = vec![0.0; genes];
        for (pair, epsilon) in self.noise.iter().enumerate() {
            let weight = weights[2 * pair] - weights[2 * pair + 1];
            if weight != 0.0 {
                for (g, &e) in gradient.iter_mut().zip(epsilon) {
                    *g += weight * e;
                }
            }
        }
        let bounds = self.real.bounds();
        for ((g, &m), range) in gradient.iter_mut().zip(&self.mean).zip(bounds) {
            *g *= scale;
            let width = range.end() - range.start();
            if self.weight_decay > 0.0 && width > 0.0 {
                *g -= self.weight_decay * m / width;
            }
        }
        gradient
    }

    // the optimizer's step along `gradient`, and the mean moved by it within the bounds
    fn step(&mut self, gradient: &[f64]) {
        let bounds = self.real.bounds();
        let Moments {
            first,
            second,
            beta1_power,
            beta2_power,
        } = &mut self.moments;
        let steps: Vec<f64> = match self.optimizer {
            Optimizer::Sgd {
                learning_rate,
                momentum,
            } => first
                .iter_mut()
                .zip(gradient)
                .map(|(v, &g)| {
                    *v = momentum * *v + (1.0 - momentum) * g;
                    learning_rate * *v
                })
                .collect(),
            Optimizer::Adam {
                learning_rate,
                beta1,
                beta2,
            } => {
                *beta1_power *= beta1;
                *beta2_power *= beta2;
                let rate = learning_rate * (1.0 - *beta2_power).sqrt() / (1.0 - *beta1_power);
                first
                    .iter_mut()
                    .zip(second.iter_mut())
                    .zip(gradient)
                    .map(|((m, v), &g)| {
                        *m = beta1 * *m + (1.0 - beta1) * g;
                        *v = beta2 * *v + (1.0 - beta2) * g * g;
                        rate * *m / (v.sqrt() + ADAM_EPSILON)
                    })
                    .collect()
            }
        };
        for ((m, step), range) in self.mean.iter_mut().zip(steps).zip(bounds) {
            let (start, end) = (*range.start(), *range.end());
            let moved = *m + step * (end - start);
            // a step that overflows or is NaN leaves the gene where it was
            if moved.is_finite() {
                *m = moved.clamp(start, end);
            }
        }
    }

    // the best of the population, the first one on ties
    fn best_of_population(&self) -> Option<&Individual<Reals>> {
        let objective = self.objective;
        let fitness =
            |individual: &Individual<Reals>| individual.fitness().unwrap_or(Fitness::invalid());
        self.population.iter().reduce(|best, individual| {
            if objective.is_better(fitness(individual), fitness(best)) {
                individual
            } else {
                best
            }
        })
    }
}

impl Reevaluate for OpenEs {
    /// As [`OpenEs::reevaluate`]: the next ask gives the last samples.
    fn reevaluate(&mut self) -> Result<()> {
        OpenEs::reevaluate(self)
    }
}

impl Algorithm for OpenEs {
    type Genome = Reals;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, Reals> {
        if !self.asked {
            if self.started && !self.reevaluating {
                self.sample();
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
        for (individual, &fitness) in self.population.iter_mut().zip(fitness) {
            individual.set_fitness(fitness);
        }
        if self.reevaluating {
            self.best = self.best_of_population().cloned();
            self.best_generation = self.generation;
            self.reevaluating = false;
            return Ok(());
        }
        if self.started {
            self.generation += 1;
        }
        let objective = self.objective;
        if let Some(candidate) = self.best_of_population() {
            let fitness = candidate.fitness().unwrap_or(Fitness::invalid());
            let better = self.best.as_ref().is_none_or(|best| {
                objective.is_better(fitness, best.fitness().unwrap_or(Fitness::invalid()))
            });
            if better {
                self.best = Some(candidate.clone());
                self.best_generation = self.generation;
            }
        }
        self.update();
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

/// A builder for an [`OpenEs`], from [`OpenEs::builder`].
///
/// The population size is required. Defaults: σ 0.02 of each gene's range, Adam with a learning
/// rate of 0.01 of each gene's range, no weight decay, the mean not evaluated, maximize, a random
/// initial mean and a random seed.
#[derive(Clone, Debug)]
pub struct OpenEsBuilder {
    real: Real,
    population_size: Option<usize>,
    sigma: f64,
    optimizer: Optimizer,
    weight_decay: f64,
    evaluate_mean: bool,
    parallel_breeding: bool,
    objective: Objective,
    seed: Option<u64>,
    initial_mean: Option<Reals>,
}

impl OpenEsBuilder {
    /// The number of samples per generation, even (mirrored pairs), at least 2 and at most 2^24.
    /// Tens to thousands. Required.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// The standard deviation of the perturbations, a fraction of each gene's range: greater
    /// than 0 and finite. 0.02 by default.
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// How the mean follows the gradient estimate. [`Optimizer::adam`] with a learning rate of
    /// 0.01 by default.
    pub fn optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// The pull of the mean toward 0 (L2 regularization, as Salimans et al. used on networks'
    /// weights): the gradient loses `decay` times the mean, in units of each gene's range. 0 or
    /// more and finite; 0 by default.
    pub fn weight_decay(mut self, decay: f64) -> Self {
        self.weight_decay = decay;
        self
    }

    /// Whether the mean is evaluated each generation too, after the samples: one more
    /// evaluation per generation, counted, and a candidate for [`best`](Algorithm::best). The
    /// mean isn't among the samples, and on many problems it is better than all of them. Off by
    /// default.
    pub fn evaluate_mean(mut self, evaluate: bool) -> Self {
        self.evaluate_mean = evaluate;
        self
    }

    /// Whether the samples are drawn on rayon's threads (the `parallel` feature; without it, one
    /// after the other with the same results). Off by default.
    ///
    /// - **Random numbers:** each pair draws from a stream of its own, derived from the seed, the
    ///   generation and the pair's position: seeded results differ from those without parallel
    ///   breeding, but not between thread counts.
    /// - **When:** for thousands of genes and a fast fitness function, when drawing the normal
    ///   numbers is much of a generation.
    pub fn parallel_breeding(mut self, parallel: bool) -> Self {
        self.parallel_breeding = parallel;
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

    /// The initial mean, valid for the representation. A random point within the bounds by
    /// default.
    pub fn initial_mean(mut self, genome: Reals) -> Self {
        self.initial_mean = Some(genome);
        self
    }

    /// Validates the settings and creates the algorithm, with its first samples.
    ///
    /// # Errors
    ///
    /// - [`Error::MissingSetting`] without a population size.
    /// - [`Error::InvalidSetting`] for a population size that is odd, below 2 or above 2^24, a σ,
    ///   learning rate, decay or weight decay out of range.
    /// - [`Error::InvalidGenome`] for an initial mean that doesn't fit the representation.
    pub fn build(self) -> Result<OpenEs> {
        let size = self.population_size.ok_or(Error::MissingSetting {
            setting: "population_size",
        })?;
        let invalid = |setting, reason: String| Err(Error::InvalidSetting { setting, reason });
        if size < 2 || size % 2 != 0 {
            return invalid(
                "population_size",
                format!("must be even and at least 2 (mirrored pairs), got {size}"),
            );
        }
        check_size("population_size", size)?;
        check_sigma(self.sigma)?;
        self.optimizer.validate()?;
        if !(self.weight_decay >= 0.0 && self.weight_decay.is_finite()) {
            return invalid(
                "weight_decay",
                format!("must be 0 or more and finite, got {}", self.weight_decay),
            );
        }
        if let Some(genome) = &self.initial_mean {
            self.real.validate(genome)?;
        }
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let mut rng = StreamRng::seed_from_u64(seed);
        let mean = match self.initial_mean {
            Some(genome) => genome.to_vec(),
            None => self.real.random_genome(&mut rng).to_vec(),
        };
        let genes = mean.len();
        let mut open_es = OpenEs {
            real: self.real,
            population_size: size,
            sigma: self.sigma,
            optimizer: self.optimizer,
            weight_decay: self.weight_decay,
            evaluate_mean: self.evaluate_mean,
            parallel_breeding: self.parallel_breeding,
            objective: self.objective,
            seed,
            rng,
            mean,
            moments: Moments {
                first: vec![0.0; genes],
                second: vec![0.0; genes],
                beta1_power: 1.0,
                beta2_power: 1.0,
            },
            noise: Vec::new(),
            population: Population::new(Vec::new()),
            reevaluating: false,
            started: false,
            asked: false,
            pending: Vec::new(),
            generation: 0,
            evaluations: 0,
            best: None,
            best_generation: 0,
        };
        open_es.sample();
        Ok(open_es)
    }
}

fn check_sigma(sigma: f64) -> Result<()> {
    if !(sigma > 0.0 && sigma.is_finite()) {
        return Err(Error::InvalidSetting {
            setting: "sigma",
            reason: format!("must be greater than 0 and finite, got {sigma}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_es(size: usize, optimizer: Optimizer) -> OpenEs {
        OpenEs::builder(Real::uniform(3, -1.0..=1.0).unwrap())
            .population_size(size)
            .optimizer(optimizer)
            .initial_mean(Reals::from(vec![0.0; 3]))
            .minimize()
            .seed(1)
            .build()
            .unwrap()
    }

    #[test]
    fn centered_ranks() {
        let mut es = open_es(6, Optimizer::adam(0.01));
        es.ask();
        // minimized: 1.0 is the best; the two 2.0 in order; NaN (invalid) the worst
        let values = [3.0, 2.0, f64::NAN, 1.0, 2.0, 5.0];
        let fitness: Vec<Fitness> = values.iter().map(|&v| Fitness::new(v)).collect();
        for (individual, &fitness) in es.population.iter_mut().zip(&fitness) {
            individual.set_fitness(fitness);
        }
        // ranks from the best: 3, 1, 4 (after 1 on the tie), 0, 5, 2
        let weight = |rank: f64| 0.5 - rank / 5.0;
        let expected = [3.0, 1.0, 5.0, 0.0, 2.0, 4.0].map(weight);
        assert_eq!(es.rank_weights(), expected);
        assert_eq!(
            expected.map(|w| (w * 10.0).round()),
            [-1.0, 3.0, -5.0, 5.0, 1.0, -3.0]
        );
    }

    #[test]
    fn adam_and_sgd_steps() {
        // on [-1, 1], a range of 2: the mean moves by twice the step
        let mut adam = open_es(2, Optimizer::adam(0.01));
        let gradient = [0.5, -2.0, 0.0];
        adam.step(&gradient);
        // Adam's first step: lr √(1 − β₂) / (1 − β₁) · (1 − β₁) g / (√(1 − β₂) |g| + ε), about
        // lr · sign(g)
        let rate = 0.01 * (1.0 - 0.999_f64).sqrt() / (1.0 - 0.9);
        let expected: Vec<f64> = gradient
            .iter()
            .map(|&g| {
                let (m, v) = ((1.0 - 0.9) * g, (1.0 - 0.999) * g * g);
                rate * m / (v.sqrt() + ADAM_EPSILON) * 2.0
            })
            .collect();
        assert_eq!(adam.mean, expected);
        assert!((adam.mean[0] - 0.02).abs() < 1e-7 && (adam.mean[1] + 0.02).abs() < 1e-7);
        // the second step, from the moments
        adam.step(&gradient);
        let (b1, b2) = (0.9_f64, 0.999_f64);
        let rate = 0.01 * (1.0 - b2 * b2).sqrt() / (1.0 - b1 * b1);
        for (k, &g) in gradient.iter().enumerate() {
            let m = b1 * ((1.0 - b1) * g) + (1.0 - b1) * g;
            let v = b2 * ((1.0 - b2) * g * g) + (1.0 - b2) * g * g;
            assert_eq!(
                adam.mean[k],
                expected[k] + rate * m / (v.sqrt() + ADAM_EPSILON) * 2.0
            );
        }

        let mut sgd = open_es(2, Optimizer::sgd(0.1, 0.5));
        sgd.step(&gradient);
        // v = 0.5 g, the mean moved by 2 · 0.1 · v
        assert_eq!(sgd.mean, [0.05, -0.2, 0.0]);
        sgd.step(&gradient);
        // v = 0.5 · 0.5 g + 0.5 g = 0.75 g
        assert_eq!(sgd.mean, [0.05 + 0.075, -0.2 - 0.3, 0.0]);
        // the bounds hold
        for _ in 0..10 {
            sgd.step(&gradient);
        }
        assert_eq!(sgd.mean[1], -1.0);
    }

    #[test]
    fn the_gradient_points_downhill() {
        // minimizing the sphere from (0.5, -0.25, 0.1): the gradient of the ranks, averaged over
        // many samples, points against x (toward better fitness)
        let mut es = OpenEs::builder(Real::uniform(3, -1.0..=1.0).unwrap())
            .population_size(20)
            .sigma(0.01)
            .initial_mean(Reals::from(vec![0.5, -0.25, 0.1]))
            .minimize()
            .seed(3)
            .build()
            .unwrap();
        let mut sum = [0.0; 3];
        for _ in 0..500 {
            es.sample();
            for individual in es.population.iter_mut() {
                let value = individual.genome().iter().map(|x| x * x).sum::<f64>();
                individual.set_fitness(Fitness::new(value));
            }
            for (s, g) in sum.iter_mut().zip(es.gradient()) {
                *s += g;
            }
        }
        let x = [0.5, -0.25, 0.1];
        let dot: f64 = sum.iter().zip(&x).map(|(s, x)| s * x).sum();
        let norms = sum.iter().map(|s| s * s).sum::<f64>().sqrt()
            * x.iter().map(|x| x * x).sum::<f64>().sqrt();
        let cosine = -dot / norms;
        assert!(cosine > 0.99, "cosine {cosine}");
    }
}
