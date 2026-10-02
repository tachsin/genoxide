//! Bayesian optimization: a surrogate model of an expensive function, and acquisition functions
//! that pick where to evaluate it next.
//!
//! See [`Bo`]. [`acquisition`] has the acquisition functions, from a model's predictive mean and
//! standard deviation at a point: they work with any surrogate model that gives those.

pub mod acquisition;

use super::{Algorithm, Candidates, Lbfgsb, Reevaluate};
use crate::genome::{Real, Reals, Representation};
use crate::model::gp::{self, GaussianProcess, Kernel, Noise, Scaling, map_in_order};
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;

/// The acquisition function of a [`Bo`]: how much a point is worth evaluating, from the model's
/// posterior mean `μ` and standard deviation `σ` there. See [`acquisition`] for the formulas.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Acquisition {
    /// The expected improvement over the best value (Močkus, 1975; Jones, Schonlau and Welch,
    /// 1998). It underflows to 0 with its gradient far from the best, so its maximization from
    /// most starts goes nowhere once the model is sure of itself;
    /// [`LogExpectedImprovement`](Acquisition::LogExpectedImprovement) doesn't.
    ExpectedImprovement,
    /// The logarithm of the expected improvement, computed so that it and its gradient stay
    /// finite where the expected improvement underflows (Ament, Daulton, Eriksson, Balandat and
    /// Bakshy, 2023): the same maximizer, found far more reliably. The default.
    #[default]
    LogExpectedImprovement,
    /// The probability of improving on the best by more than `xi` (Kushner, 1964), maximized
    /// through its logarithm, which has the same maximizer and keeps a gradient where the
    /// probability underflows.
    ProbabilityOfImprovement {
        /// ξ ≥ 0, in the units the model fits ([`Output`]): larger asks for larger improvements,
        /// exploring more. 0 is the pure probability of improvement, which exploits greedily.
        xi: f64,
    },
    /// The confidence bound `μ − √β σ`, minimized (`μ + √β σ` maximized when maximizing):
    /// Srinivas, Krause, Kakade and Seeger (2010). Changeable during a run for a schedule
    /// ([`Bo::set_acquisition`]).
    UpperConfidenceBound {
        /// β ≥ 0: 0 exploits the model's mean alone, larger explores more.
        beta: f64,
    },
}

impl Acquisition {
    fn validate(self) -> Result<()> {
        let parameter = match self {
            Acquisition::ProbabilityOfImprovement { xi } => Some(("xi", xi)),
            Acquisition::UpperConfidenceBound { beta } => Some(("beta", beta)),
            _ => None,
        };
        match parameter {
            Some((name, value)) if !(value >= 0.0 && value.is_finite()) => {
                Err(Error::InvalidSetting {
                    setting: "acquisition",
                    reason: format!("{name} must be finite and at least 0, got {value}"),
                })
            }
            _ => Ok(()),
        }
    }
}

/// What the model of a [`Bo`] fits: a transform of the values to minimize (the scores, negated
/// when maximizing), which the model then standardizes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Output {
    /// The values themselves (the default), standardized as Gaussian processes usually are.
    #[default]
    Standardize,
    /// The logarithm of each value's distance above the best, `ln(v − v_best + δ)`, with `δ` the
    /// first quartile of the distances (the ⌊N/4⌋-th smallest of N, the best's own 0 counted):
    /// for objectives that span orders of magnitude, such as Goldstein-Price's from 3 to 10⁶,
    /// whose large values would otherwise flatten the model where the best ones are. A monotone
    /// transform: the best value stays the best, and the model resolves small differences near it
    /// and compresses large ones far from it. `δ` grows with the spread of the values near the
    /// best, so the best point is never an outlier far below the others (as it would be with a
    /// tiny `δ`), and shrinks as the search closes in.
    ///
    /// Measured on 20 seeds per problem (default settings otherwise, to f* + 1e-3): Goldstein-Price
    /// in [−2, 2]² reached in 20 runs of 20 within 80 evaluations (0 with
    /// [`Standardize`](Output::Standardize)), the six-hump camel in [−5, 5]² in 20 (6), Branin in
    /// a median of 25 evaluations (30).
    Log,
}

// the ids of the random streams derived from the seed: they decide the results of seeded runs and
// must never change for the same major version
mod streams {
    // the Latin hypercube of the initial design
    pub(super) const DESIGN: u64 = 0;
    // the starts of the hyperparameters' maximization, per generation
    pub(super) const HYPERPARAMETERS: u64 = 1;
    // the raw samples of the acquisition's maximization, per generation
    pub(super) const RAW_SAMPLES: u64 = 2;
    // a random point when there's no model to ask, per generation
    pub(super) const RANDOM: u64 = 3;
}

// the evaluations of the acquisition function per start of its maximization
const ACQUISITION_EVALUATIONS: u64 = 200;
// the least variance of the model's prediction, in its standardized units, for the acquisition
const VARIANCE_FLOOR: f64 = 1e-12;

/// Bayesian optimization on [`Real`] genomes, as an ask / tell [`Algorithm`]: for expensive
/// black-box functions, such as a simulation that runs for minutes or a physical experiment, where
/// tens to a few hundred evaluations must do. A Gaussian process ([`model::gp`](crate::model::gp))
/// models the function from every evaluation so far, and an [`Acquisition`] function of its
/// posterior picks the next point to evaluate, trading the model's best guesses against its
/// uncertainty.
///
/// - **The initial design** (generation 0): [`initial_points`](BoBuilder::initial_points) points,
///   2(n + 1) for n searched genes by default, the
///   [`initial_genomes`](BoBuilder::initial_genomes) first and a Latin hypercube sample (McKay,
///   Beckman and Conover, 1979; [`Real::latin_hypercube`]) for the rest. A small design leaves
///   most of the budget to the model's choices. For an accurate model of the whole box rather
///   than its minimum, Loeppky, Sacks and Welch (2009) recommend 10n points.
/// - **Each later generation** asks one point: the model is fitted to every evaluation (its
///   hyperparameters by maximum likelihood, from the last fit's and random starts), then the
///   acquisition function is maximized in the box: evaluated at
///   [`raw_samples`](BoBuilder::raw_samples) random points, then improved by [`Lbfgsb`] with its
///   analytic gradient from the best [`acquisition_starts`](BoBuilder::acquisition_starts) of them
///   and from the best point evaluated so far. The best result that isn't an evaluated point is
///   asked: a point is never asked twice.
/// - **The model** fits the values to minimize, the scores negated when maximizing, through the
///   [`Output`] transform. A point with an invalid fitness, or a score that isn't finite, enters
///   the model at the worst value of the others, so the search learns to avoid where the function
///   fails; until a point has a valid score, the next point is random. The search uses the score
///   only: a constraint violation is ignored by it (use a penalty), though
///   [`best`](Algorithm::best) compares by Deb's rules, as everywhere in genoxide.
/// - **Cost.** The model's fit is O(N³) for N evaluations, its predictions O(N²): Bayesian
///   optimization suits up to a few hundred evaluations of a function that costs far more than
///   that, in up to about 10 to 20 genes.
///
/// [`model`](Bo::model) gives the model that chose the last point. Every random number comes from
/// streams derived from the [seed](BoBuilder::seed), every operation is a sum, product, quotient
/// or square root in a fixed order with [`math`](crate::math)'s functions, and the multi-starts
/// run on rayon (with the `parallel` feature) with the winner chosen by value, then start: a seed
/// gives the same run on every platform and thread count.
///
/// Built with [`Bo::builder`], run with an [`Engine`](crate::Engine).
///
/// ```
/// use genoxide::prelude::*;
/// use genoxide::problems::{Branin, Problem};
///
/// // Branin's function, whose three global minima are 0.397887, in 40 evaluations
/// let bo = Bo::builder(Branin.representation()).minimize().seed(1).build()?;
/// let outcome = Engine::new(bo, Branin)
///     .stop_when(Stop::evaluations(40))
///     .run()?;
/// assert!(outcome.best_fitness().score().unwrap() < 0.397887 + 1e-3);
/// # Ok::<(), genoxide::Error>(())
/// ```
///
/// References: Močkus, J. (1975). On Bayesian methods for seeking the extremum. *Optimization
/// Techniques IFIP Technical Conference 1974*, LNCS 27: 400-404. Jones, D. R., Schonlau, M. and
/// Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal
/// of Global Optimization* 13(4): 455-492. Rasmussen, C. E. and Williams, C. K. I. (2006).
/// *Gaussian Processes for Machine Learning.* MIT Press. Ament, S., Daulton, S., Eriksson, D.,
/// Balandat, M. and Bakshy, E. (2023). Unexpected improvements to expected improvement for
/// Bayesian optimization. *NeurIPS 2023*, arXiv:2310.20708. McKay, M. D., Beckman, R. J. and
/// Conover, W. J. (1979). A comparison of three methods for selecting values of input variables
/// in the analysis of output from a computer code. *Technometrics* 21(2): 239-245. Loeppky, J. L.,
/// Sacks, J. and Welch, W. J. (2009). Choosing the sample size of a computer experiment: a
/// practical guide. *Technometrics* 51(4): 366-376.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Bo {
    real: Real,
    initial_points: usize,
    acquisition: Acquisition,
    kernel: Kernel,
    noise: Noise,
    output: Output,
    raw_samples: usize,
    acquisition_starts: usize,
    hyperparameter_starts: usize,
    objective: Objective,
    seed: u64,
    // the initial design, asked in generation 0
    design: Vec<Reals>,
    // every evaluated point, in the order evaluated
    observations: Population<Reals>,
    // the logarithms of the last fit's hyperparameters, the next fit's first start
    warm: Option<Vec<f64>>,
    // the points of the current ask
    pending: Vec<Individual<Reals>>,
    indices: Vec<usize>,
    asked: bool,
    reevaluating: bool,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<Reals>>,
    best_generation: u64,
    // the model that chose the last point, and the best value it was told, in its standardized
    // units
    #[cfg_attr(feature = "serde", serde(skip))]
    model: Option<GaussianProcess>,
    #[cfg_attr(feature = "serde", serde(skip))]
    model_best: f64,
}

impl Bo {
    /// A builder for Bayesian optimization on `real`.
    pub fn builder(real: Real) -> BoBuilder {
        BoBuilder {
            real,
            initial_points: None,
            initial_genomes: Vec::new(),
            acquisition: Acquisition::default(),
            kernel: Kernel::default(),
            noise: Noise::default(),
            output: Output::default(),
            raw_samples: 1000,
            acquisition_starts: 10,
            hyperparameter_starts: 5,
            objective: Objective::default(),
            seed: None,
        }
    }

    /// The representation.
    pub fn real(&self) -> &Real {
        &self.real
    }

    /// The number of points of the initial design.
    pub fn initial_points(&self) -> usize {
        self.initial_points
    }

    /// The acquisition function.
    pub fn acquisition(&self) -> Acquisition {
        self.acquisition
    }

    /// Changes the acquisition function from the next point on, e.g. UCB's β on a schedule from
    /// [`Engine::control`](crate::Engine::control).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a negative or non-finite ξ or β. Nothing changes on errors.
    pub fn set_acquisition(&mut self, acquisition: Acquisition) -> Result<()> {
        acquisition.validate()?;
        self.acquisition = acquisition;
        Ok(())
    }

    /// The model's kernel.
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// The model's noise.
    pub fn noise(&self) -> Noise {
        self.noise
    }

    /// What the model fits.
    pub fn output(&self) -> Output {
        self.output
    }

    /// The random points at which the acquisition function is evaluated before its maximization.
    pub fn raw_samples(&self) -> usize {
        self.raw_samples
    }

    /// The best raw samples from which the acquisition function is maximized.
    pub fn acquisition_starts(&self) -> usize {
        self.acquisition_starts
    }

    /// The starts of the maximization of the model's likelihood at each fit.
    pub fn hyperparameter_starts(&self) -> usize {
        self.hyperparameter_starts
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The Gaussian process that chose the last point asked, fitted to the evaluations before it:
    /// a model of the values the search minimizes (the scores, negated when maximizing, through
    /// the [`Output`] transform). `None` before the first point after the initial design, after a
    /// random point (when no evaluation had a valid score), and after loading a checkpoint until
    /// the next point is asked.
    pub fn model(&self) -> Option<&GaussianProcess> {
        self.model.as_ref()
    }

    /// The acquisition function at `genome` under [`model`](Bo::model), as the search maximizes
    /// it: the log expected improvement, the expected improvement, the logarithm of the
    /// probability of improvement, or the negated lower confidence bound, in the model's
    /// standardized units. `None` without a model.
    ///
    /// # Panics
    ///
    /// If `genome` doesn't have a value per gene.
    pub fn acquisition_at(&self, genome: &[f64]) -> Option<f64> {
        let model = self.model.as_ref()?;
        assert_eq!(
            genome.len(),
            self.real.genome_len(),
            "a genome of {} genes",
            self.real.genome_len()
        );
        let scaling = Scaling::new(&self.real);
        let mut unit = vec![0.0; scaling.dims()];
        scaling.to_unit(genome, &mut unit);
        Some(self.search(model).value(&unit, None))
    }

    fn search<'a>(&self, model: &'a GaussianProcess) -> Search<'a> {
        Search {
            model,
            best: self.model_best,
            acquisition: self.acquisition,
            scale: model.standardization().1,
        }
    }

    // the value the model fits for a fitness, before the output transform: the score to minimize,
    // or None for an invalid fitness or a score that isn't finite
    fn model_value(&self, fitness: Option<Fitness>) -> Option<f64> {
        let score = fitness?.score()?;
        let value = match self.objective {
            Objective::Minimize => score,
            Objective::Maximize => -score,
        };
        value.is_finite().then_some(value)
    }

    // whether `genome` was evaluated already
    fn observed(&self, genome: &[f64]) -> bool {
        self.observations
            .iter()
            .any(|individual| individual.genome()[..] == genome[..])
    }

    // a random point of the box that wasn't evaluated, from the stream of this generation
    fn random_point(&self, generation: u64) -> Reals {
        let mut rng = StreamRng::seed_from_u64(self.seed)
            .derive(streams::RANDOM)
            .derive(generation);
        loop {
            let genome = self.real.random_genome(&mut rng);
            if !self.observed(&genome) {
                return genome;
            }
        }
    }

    // the point to evaluate next: the model fitted, the acquisition maximized
    fn next_point(&mut self) -> Reals {
        let generation = self.generation + 1;
        let root = StreamRng::seed_from_u64(self.seed);
        let scaling = Scaling::new(&self.real);
        let dims = scaling.dims();
        let count = self.observations.len();
        let mut x = vec![0.0; count * dims];
        let mut values = Vec::with_capacity(count);
        for (individual, unit) in self.observations.iter().zip(x.chunks_exact_mut(dims)) {
            scaling.to_unit(individual.genome(), unit);
            values.push(self.model_value(individual.fitness()));
        }
        self.model = None;
        let Some(targets) = transform(&values, self.output) else {
            return self.random_point(generation);
        };
        let settings = gp::Settings {
            kernel: self.kernel,
            noise: self.noise,
            starts: self.hyperparameter_starts,
            seed: root
                .derive(streams::HYPERPARAMETERS)
                .derive(generation)
                .next_u64(),
        };
        let model =
            GaussianProcess::fit_unit(settings, scaling.clone(), x, &targets, self.warm.as_deref());
        self.warm = Some(model.log_parameters().to_vec());
        // the best point evaluated, by the model's values: the incumbent, and a start
        let mut incumbent: Option<usize> = None;
        for (index, value) in values.iter().enumerate() {
            if value.is_some() && incumbent.is_none_or(|best| targets[index] < targets[best]) {
                incumbent = Some(index);
            }
        }
        let incumbent = incumbent.unwrap_or(0);
        let (y_mean, y_scale) = model.standardization();
        self.model_best = (targets[incumbent] - y_mean) / y_scale;
        let search = self.search(&model);
        // the raw samples, and the best of them as starts after the incumbent
        let mut rng = root.derive(streams::RAW_SAMPLES).derive(generation);
        let raw: Vec<f64> = (0..self.raw_samples * dims)
            .map(|_| rng.unit_f64())
            .collect();
        let raw_values = map_in_order(self.raw_samples, |i| {
            search.value(&raw[i * dims..(i + 1) * dims], None)
        });
        let mut order: Vec<usize> = (0..self.raw_samples).collect();
        // highest first, NaN last; the sort is stable, so ties keep the earlier sample
        let key = |v: f64| if v.is_nan() { f64::NEG_INFINITY } else { v };
        order.sort_by(|&a, &b| key(raw_values[b]).total_cmp(&key(raw_values[a])));
        let mut starts = Vec::with_capacity(self.acquisition_starts + 1);
        starts.push(model.unit_point(incumbent).to_vec());
        for &i in order.iter().take(self.acquisition_starts) {
            starts.push(raw[i * dims..(i + 1) * dims].to_vec());
        }
        let unit_box = Real::uniform(dims, 0.0..=1.0).expect("a searched gene");
        let results = map_in_order(starts.len(), |start| {
            Lbfgsb::minimize_with(
                unit_box.clone(),
                Reals::from(starts[start].clone()),
                ACQUISITION_EVALUATIONS,
                |u, gradient| {
                    let value = search.value(u, Some(gradient));
                    for g in gradient.iter_mut() {
                        *g = -*g;
                    }
                    -value
                },
            )
        });
        let mut candidates: Vec<(f64, Reals)> = results
            .into_iter()
            .flatten()
            .map(|(point, value)| (-value, point))
            .collect();
        // highest first; stable, so ties keep the earlier start
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        self.model = Some(model);
        for (_, unit) in &candidates {
            let genome = scaling.to_genome(unit);
            if !self.observed(&genome) {
                return genome;
            }
        }
        self.random_point(generation)
    }

    // the points of the next ask
    fn build_ask(&mut self) {
        self.pending.clear();
        if self.reevaluating {
            let genomes = self.observations.iter().map(|o| o.genome().clone());
            self.pending.extend(genomes.map(Individual::new));
        } else if self.observations.is_empty() {
            let design = self.design.iter().cloned();
            self.pending.extend(design.map(Individual::new));
        } else {
            let point = self.next_point();
            self.pending.push(Individual::new(point));
        }
        self.indices.clear();
        self.indices.extend(0..self.pending.len());
    }

    /// Marks every evaluated point as not evaluated, for a fitness function that changed during
    /// the run: the next [`ask`](Algorithm::ask) gives all of them again, and its
    /// [`tell`](Algorithm::tell) replaces their values. It isn't a generation; the evaluations
    /// are counted. [`best`](Algorithm::best) is then the best of the new values, found in the
    /// current generation. No random number is drawn. Before the first tell it changes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::ReevaluationOutOfTurn`] between an ask and its tell. Nothing changes on errors.
    pub fn reevaluate(&mut self) -> Result<()> {
        if self.asked {
            return Err(Error::ReevaluationOutOfTurn);
        }
        if !self.observations.is_empty() {
            self.reevaluating = true;
        }
        Ok(())
    }
}

// the model's targets from the values to minimize (None for invalid ones): transformed, and the
// invalid ones at the worst of the others; None without a valid value
fn transform(values: &[Option<f64>], output: Output) -> Option<Vec<f64>> {
    let mut best = f64::INFINITY;
    for value in values.iter().flatten() {
        best = best.min(*value);
    }
    if best == f64::INFINITY {
        return None;
    }
    let offset = match output {
        Output::Standardize => 0.0,
        Output::Log => log_offset(values, best),
    };
    let apply = |value: f64| match output {
        Output::Standardize => value,
        Output::Log => crate::math::ln(value - best + offset),
    };
    let mut worst = f64::NEG_INFINITY;
    let mut targets: Vec<f64> = values
        .iter()
        .map(|value| match value {
            Some(value) => {
                let target = apply(*value);
                worst = worst.max(target);
                target
            }
            None => f64::NAN,
        })
        .collect();
    for target in &mut targets {
        if target.is_nan() {
            *target = worst;
        }
    }
    Some(targets)
}

// δ of the log transform: the first quartile of the distances above the best, the ⌊N/4⌋-th
// smallest of the N valid values' (the best's own 0 counted); the smallest positive one if that's
// 0, and 1 if every value is the best
fn log_offset(values: &[Option<f64>], best: f64) -> f64 {
    let mut distances: Vec<f64> = values.iter().flatten().map(|value| value - best).collect();
    distances.sort_by(f64::total_cmp);
    let quartile = distances[distances.len() / 4];
    if quartile > 0.0 {
        return quartile;
    }
    distances
        .into_iter()
        .find(|&distance| distance > 0.0)
        .unwrap_or(1.0)
}

// the acquisition function of a model, in its standardized units, as maximized
struct Search<'a> {
    model: &'a GaussianProcess,
    // the best value, standardized
    best: f64,
    acquisition: Acquisition,
    // the model's scale of the values, for ξ
    scale: f64,
}

impl Search<'_> {
    // the acquisition's value at the unit-cube point `u`, and its gradient
    fn value(&self, u: &[f64], gradient: Option<&mut [f64]>) -> f64 {
        let dims = u.len();
        let (mut dmean, mut dvariance) = (Vec::new(), Vec::new());
        let gradients = if gradient.is_some() {
            dmean.resize(dims, 0.0);
            dvariance.resize(dims, 0.0);
            Some((&mut dmean[..], &mut dvariance[..]))
        } else {
            None
        };
        let (mean, variance) = self.model.predict_unit(u, gradients);
        let floored = variance < VARIANCE_FLOOR;
        let sd = variance.max(VARIANCE_FLOOR).sqrt();
        let [value, by_mean, by_sd] = match self.acquisition {
            Acquisition::ExpectedImprovement => {
                acquisition::expected_improvement_derivatives(mean, sd, self.best)
            }
            Acquisition::LogExpectedImprovement => {
                acquisition::log_expected_improvement_derivatives(mean, sd, self.best)
            }
            Acquisition::ProbabilityOfImprovement { xi } => {
                acquisition::log_probability_of_improvement_derivatives(
                    mean,
                    sd,
                    self.best,
                    xi / self.scale,
                )
            }
            Acquisition::UpperConfidenceBound { beta } => {
                acquisition::upper_confidence_bound_derivatives(mean, sd, beta)
            }
        };
        if let Some(gradient) = gradient {
            for i in 0..dims {
                let dsd = if floored {
                    0.0
                } else {
                    dvariance[i] / (2.0 * sd)
                };
                gradient[i] = by_mean * dmean[i] + by_sd * dsd;
            }
        }
        value
    }
}

impl Reevaluate for Bo {
    /// As [`Bo::reevaluate`]: the next ask gives every evaluated point again.
    fn reevaluate(&mut self) -> Result<()> {
        Bo::reevaluate(self)
    }
}

impl Algorithm for Bo {
    type Genome = Reals;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, Reals> {
        if !self.asked {
            self.build_ask();
            self.asked = true;
        }
        Candidates::new(&self.pending, &self.indices)
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
        self.indices.clear();
        if self.reevaluating {
            self.reevaluating = false;
            self.pending.clear();
            for (observation, &fitness) in self.observations.iter_mut().zip(fitness) {
                observation.set_fitness(fitness);
            }
            self.best = None;
            update_best(
                &mut self.best,
                &mut self.best_generation,
                self.generation,
                self.objective,
                self.observations.as_slice(),
            );
            return Ok(());
        }
        if !self.observations.is_empty() {
            self.generation += 1;
        }
        let first = self.observations.len();
        for (mut individual, &fitness) in self.pending.drain(..).zip(fitness) {
            individual.set_fitness(fitness);
            self.observations.push(individual);
        }
        update_best(
            &mut self.best,
            &mut self.best_generation,
            self.generation,
            self.objective,
            &self.observations.as_slice()[first..],
        );
        Ok(())
    }

    fn population(&self) -> &Population<Reals> {
        &self.observations
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

// the best so far, from the evaluated `individuals` in order: the first on ties
fn update_best(
    best: &mut Option<Individual<Reals>>,
    best_generation: &mut u64,
    generation: u64,
    objective: Objective,
    individuals: &[Individual<Reals>],
) {
    for individual in individuals {
        let fitness = individual.fitness().unwrap_or(Fitness::invalid());
        let better = match best {
            Some(best) => {
                objective.is_better(fitness, best.fitness().unwrap_or(Fitness::invalid()))
            }
            None => true,
        };
        if better {
            *best = Some(individual.clone());
            *best_generation = generation;
        }
    }
}

/// A builder for a [`Bo`], from [`Bo::builder`].
///
/// Defaults: maximize; an initial design of 2(n + 1) points for n searched genes;
/// [`Acquisition::LogExpectedImprovement`]; the [Matérn 5/2 kernel](Kernel::Matern52) with
/// [learned noise](Noise::Learned) of at least 1e-6 of the values' variance;
/// [`Output::Standardize`]; 1000 raw samples and 10 starts for the acquisition's maximization;
/// 5 starts for the hyperparameters'; a random seed.
#[derive(Clone, Debug)]
pub struct BoBuilder {
    real: Real,
    initial_points: Option<usize>,
    initial_genomes: Vec<Reals>,
    acquisition: Acquisition,
    kernel: Kernel,
    noise: Noise,
    output: Output,
    raw_samples: usize,
    acquisition_starts: usize,
    hyperparameter_starts: usize,
    objective: Objective,
    seed: Option<u64>,
}

impl BoBuilder {
    /// The number of points of the initial design, at least 1: 2(n + 1) for n searched genes
    /// (genes whose bounds differ) by default, a small design that leaves most evaluations to the
    /// model. 10n is the usual size for an accurate model of the whole box (Loeppky, Sacks and
    /// Welch, 2009), more than finding the minimum needs.
    pub fn initial_points(mut self, points: usize) -> Self {
        self.initial_points = Some(points);
        self
    }

    /// Genomes to evaluate first, in the initial design: at most
    /// [`initial_points`](BoBuilder::initial_points), and distinct. A Latin hypercube sample
    /// fills the rest of the design.
    pub fn initial_genomes<I: IntoIterator<Item = Reals>>(mut self, genomes: I) -> Self {
        self.initial_genomes = genomes.into_iter().collect();
        self
    }

    /// The acquisition function: [`Acquisition::LogExpectedImprovement`] by default.
    pub fn acquisition(mut self, acquisition: Acquisition) -> Self {
        self.acquisition = acquisition;
        self
    }

    /// The model's kernel: [`Kernel::Matern52`] by default.
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// The model's observation noise: [`Noise::Learned`] with a least variance of 1e-6 of the
    /// values' variance by default.
    pub fn noise(mut self, noise: Noise) -> Self {
        self.noise = noise;
        self
    }

    /// What the model fits: [`Output::Standardize`] by default; [`Output::Log`] for objectives
    /// that span orders of magnitude.
    pub fn output(mut self, output: Output) -> Self {
        self.output = output;
        self
    }

    /// The random points at which the acquisition function is evaluated before its
    /// maximization, at least 1: 1000 by default. They cost a prediction each, O(N²) for N
    /// evaluations, far less than the fit.
    pub fn raw_samples(mut self, samples: usize) -> Self {
        self.raw_samples = samples;
        self
    }

    /// The best raw samples from which L-BFGS-B maximizes the acquisition function, besides the
    /// best point evaluated: 10 by default, at least 1 and at most the raw samples.
    pub fn acquisition_starts(mut self, starts: usize) -> Self {
        self.acquisition_starts = starts;
        self
    }

    /// The starts of the maximization of the model's likelihood at each fit, at least 1: 5 by
    /// default, the last fit's hyperparameters and random points.
    pub fn hyperparameter_starts(mut self, starts: usize) -> Self {
        self.hyperparameter_starts = starts;
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

    /// The seed of the random numbers, for a reproducible run: the design and the starts of both
    /// maximizations. Random by default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the settings and creates the search.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidSetting`] for a representation without a gene that has more than one
    ///   value, an initial design of 0 points or above 2^24, more initial genomes than initial
    ///   points or the same genome twice, an invalid [`Acquisition`] parameter or [`Noise`], 0
    ///   raw samples, acquisition starts of 0 or more than the raw samples, or hyperparameter
    ///   starts of 0.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<Bo> {
        let invalid =
            |setting: &'static str, reason: String| Err(Error::InvalidSetting { setting, reason });
        let dims = self.real.variable_genes().len();
        if dims == 0 {
            return invalid(
                "real",
                "Bayesian optimization needs a gene with more than one value".to_string(),
            );
        }
        let initial_points = self.initial_points.unwrap_or(2 * (dims + 1));
        if initial_points == 0 {
            return invalid("initial_points", "must be at least 1, got 0".to_string());
        }
        crate::operator::check_size("initial_points", initial_points)?;
        if self.initial_genomes.len() > initial_points {
            return invalid(
                "initial_genomes",
                format!(
                    "at most the initial points, {initial_points}, got {}",
                    self.initial_genomes.len()
                ),
            );
        }
        for (index, genome) in self.initial_genomes.iter().enumerate() {
            self.real.validate(genome)?;
            if self.initial_genomes[..index]
                .iter()
                .any(|other| other[..] == genome[..])
            {
                return invalid(
                    "initial_genomes",
                    format!("genome {index} is given twice: {genome:?}"),
                );
            }
        }
        self.acquisition.validate()?;
        self.noise.validate()?;
        if self.raw_samples == 0 {
            return invalid("raw_samples", "must be at least 1, got 0".to_string());
        }
        crate::operator::check_size("raw_samples", self.raw_samples)?;
        if self.acquisition_starts == 0 || self.acquisition_starts > self.raw_samples {
            return invalid(
                "acquisition_starts",
                format!(
                    "must be at least 1 and at most the raw samples, {}, got {}",
                    self.raw_samples, self.acquisition_starts
                ),
            );
        }
        if self.hyperparameter_starts == 0 {
            return invalid(
                "hyperparameter_starts",
                "must be at least 1, got 0".to_string(),
            );
        }
        crate::operator::check_size("hyperparameter_starts", self.hyperparameter_starts)?;
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let mut design = self.initial_genomes;
        let missing = initial_points - design.len();
        if missing > 0 {
            let mut rng = StreamRng::seed_from_u64(seed).derive(streams::DESIGN);
            design.extend(self.real.latin_hypercube(missing, &mut rng)?);
        }
        Ok(Bo {
            real: self.real,
            initial_points,
            acquisition: self.acquisition,
            kernel: self.kernel,
            noise: self.noise,
            output: self.output,
            raw_samples: self.raw_samples,
            acquisition_starts: self.acquisition_starts,
            hyperparameter_starts: self.hyperparameter_starts,
            objective: self.objective,
            seed,
            design,
            observations: Population::default(),
            warm: None,
            pending: Vec::new(),
            indices: Vec::new(),
            asked: false,
            reevaluating: false,
            generation: 0,
            evaluations: 0,
            best: None,
            best_generation: 0,
            model: None,
            model_best: f64::NAN,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a model of a smooth function of 2 genes, from 10 points
    fn model() -> (GaussianProcess, f64) {
        let real = Real::new([-1.0..=2.0, 0.0..=3.0]).unwrap();
        let points = real
            .latin_hypercube(10, &mut StreamRng::seed_from_u64(2))
            .unwrap();
        let values: Vec<f64> = points
            .iter()
            .map(|x| (2.0 * x[0]).sin() + 0.5 * x[1] * x[1] - 0.3 * x[0] * x[1])
            .collect();
        let scaling = Scaling::new(&real);
        let mut x = vec![0.0; 20];
        for (point, unit) in points.iter().zip(x.as_chunks_mut::<2>().0) {
            scaling.to_unit(point, unit);
        }
        let settings = gp::Settings {
            kernel: Kernel::Matern52,
            noise: Noise::default(),
            starts: 3,
            seed: 1,
        };
        let model = GaussianProcess::fit_unit(settings, scaling, x, &values, None);
        let (mean, scale) = model.standardization();
        let best = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        (model, (best - mean) / scale)
    }

    #[test]
    fn the_acquisitions_gradients_match_central_differences() {
        let (model, best) = model();
        let acquisitions = [
            Acquisition::ExpectedImprovement,
            Acquisition::LogExpectedImprovement,
            Acquisition::ProbabilityOfImprovement { xi: 0.01 },
            Acquisition::UpperConfidenceBound { beta: 2.0 },
        ];
        for acquisition in acquisitions {
            let search = Search {
                model: &model,
                best,
                acquisition,
                scale: model.standardization().1,
            };
            for u in [[0.3, 0.6], [0.9, 0.1], [0.55, 0.45], [0.05, 0.95]] {
                let mut gradient = [0.0; 2];
                let value = search.value(&u, Some(&mut gradient));
                assert_eq!(value.to_bits(), search.value(&u, None).to_bits());
                for i in 0..2 {
                    let h = 1e-6;
                    let (mut plus, mut minus) = (u, u);
                    plus[i] += h;
                    minus[i] -= h;
                    let numeric =
                        (search.value(&plus, None) - search.value(&minus, None)) / (2.0 * h);
                    assert!(
                        (gradient[i] - numeric).abs() <= 1e-5 * numeric.abs().max(1.0),
                        "{acquisition:?} at {u:?}, gene {i}: {} against {numeric}",
                        gradient[i]
                    );
                }
            }
        }
    }

    #[test]
    fn the_log_transform_keeps_the_order_and_imputes_the_worst() {
        let values = [Some(3.0), None, Some(10.0), Some(3.5), Some(1e6), Some(4.0)];
        let targets = transform(&values, Output::Log).unwrap();
        // δ is the first quartile of the distances above the best, 0.5 here: the ⌊5/4⌋-th of
        // 0, 0.5, 1, 7 and 999997
        assert_eq!(targets[0], 0.5_f64.ln());
        assert_eq!(targets[3], 1.0_f64.ln());
        assert!(targets[0] < targets[3] && targets[3] < targets[5] && targets[5] < targets[2]);
        assert_eq!(targets[1], targets[4]);
        let targets = transform(&values, Output::Standardize).unwrap();
        assert_eq!(targets, [3.0, 1e6, 10.0, 3.5, 1e6, 4.0]);
        assert_eq!(transform(&[None, None], Output::Log), None);
        // every value the best: δ is 1
        assert_eq!(
            transform(&[Some(2.0), Some(2.0)], Output::Log),
            Some(vec![0.0, 0.0])
        );
    }
}
