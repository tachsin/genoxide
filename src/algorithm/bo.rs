//! Bayesian optimization: a surrogate model of an expensive function, and acquisition functions
//! that pick where to evaluate it next.
//!
//! See [`Bo`]. [`acquisition`] has the acquisition functions, from a model's predictive mean and
//! standard deviation at a point: they work with any surrogate model that gives those.

pub mod acquisition;
mod space;

pub use space::Space;

use super::{Algorithm, Candidates, Incremental, Lbfgsb, Reevaluate};
use crate::engine::{Evaluations, Provided, Wanted};
use crate::genome::{Real, Reals};
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
    /// ([`Bo::set_acquisition`]). Not for a problem with constraints, whose acquisition is
    /// weighed by the probability of feasibility: a bound can be negative.
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

    // the error for an acquisition that can't be weighed by the probability of feasibility
    fn check_constrained(self, constraints: usize) -> Result<()> {
        if constraints > 0 && matches!(self, Acquisition::UpperConfidenceBound { .. }) {
            return Err(Error::InvalidSetting {
                setting: "acquisition",
                reason: format!(
                    "the fitness function gives {constraints} constraints, and Bayesian \
                     optimization with constraints weighs the expected improvement (or its \
                     logarithm, or the probability of improvement) by the probability of \
                     feasibility: the upper confidence bound can't be weighed so"
                ),
            });
        }
        Ok(())
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
    /// Measured on 20 seeds per problem, with the other settings' defaults, to f* + 1e-3 within 80
    /// evaluations: Goldstein-Price in [−2, 2]² reached in 19 runs (0 with
    /// [`Standardize`](Output::Standardize)), the six-hump camel in [−5, 5]² in 20 (4), Branin in
    /// a median of 20 evaluations (30). Not for every function: on Hartmann 3, whose values span
    /// less than an order of magnitude, 13 runs (20).
    Log,
}

/// What a [`Bo`] takes a point to be worth while it isn't evaluated yet: the points of a batch
/// already chosen, as it chooses the next one ([`BoBuilder::batch`]), and under an
/// [`AsyncEngine`](crate::engine::AsyncEngine) the points still being evaluated. The heuristics of
/// Ginsbourger, Le Riche and Carraro (2010, section 4.2): each such point is added to the model
/// with a fantasized value, as if it had been observed, without fitting the hyperparameters again
/// (the constant mean is estimated again, as at every fit). The model is then sure of itself at
/// the point, so the acquisition function looks elsewhere, and how far elsewhere depends on the
/// value.
///
/// With constraints, each constraint's model takes its posterior mean at the point, whatever the
/// fantasy of the objective.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Fantasy {
    /// The Kriging believer, the default: the model's posterior mean at the point, which leaves
    /// the mean where it was and only removes the uncertainty there (Ginsbourger et al.'s
    /// Algorithm 1). A point predicted below the best value lowers the best, and the next points
    /// then tend to cluster around it.
    ///
    /// Measured with batches of 4 against the constant liars over 20 seeds, to f* + 1e-3: Branin
    /// within 80 evaluations in 20 runs, a median of 34 evaluations (the lowest lie 34, the mean
    /// 42, the highest 54); Hartmann 3 in 20, a median of 30 (30, 40, 40); Hartmann 6 within 200 in
    /// 13 (13, 12, 11), the others in the local minimum −3.2032. Under an
    /// [`AsyncEngine`](crate::engine::AsyncEngine) with 4 workers, whose every proposal has 3
    /// points fantasized, Hartmann 3 to f* + 1e-4 within 120 evaluations in all 5 runs tried, from
    /// 39 to 76 evaluations, against 2 of 5 with the lowest lie, which keeps the region of the
    /// points being evaluated as good as the best and so pushes the search away from it.
    #[default]
    KrigingBeliever,
    /// The constant liar: the same value, a [`Lie`], at every such point (Ginsbourger et al.'s
    /// Algorithm 2). The higher the lie, the farther the next points go from the earlier ones.
    ConstantLiar(Lie),
}

/// The value of a [`Fantasy::ConstantLiar`]: a statistic of the values the model fits (after the
/// [`Output`] transform), of every point evaluated so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Lie {
    /// The lowest, the best: a soft repulsion, the next points near but not at the earlier ones;
    /// on Branin, Ginsbourger et al.'s best strategy of the four, which visited the three minima's
    /// regions in 6 points.
    #[default]
    Min,
    /// The mean: the next points spread over the box.
    Mean,
    /// The highest, the worst: the strongest repulsion, the most exploration.
    Max,
}

// the ids of the random streams derived from the seed: they decide the results of seeded runs and
// must never change for the same major version
mod streams {
    // the Latin hypercube of the initial design
    pub(super) const DESIGN: u64 = 0;
    // the starts of the hyperparameters' maximization, per generation
    pub(super) const HYPERPARAMETERS: u64 = 1;
    // the raw samples of the acquisition's maximization, per generation (then per point of a
    // batch after the first)
    pub(super) const RAW_SAMPLES: u64 = 2;
    // a random point when there's no model to ask, per generation (then per point of a batch
    // after the first)
    pub(super) const RANDOM: u64 = 3;
    // the starts of the constraints' models' hyperparameters, per generation, then per constraint
    pub(super) const CONSTRAINTS: u64 = 4;
    // the streams of a proposal to an asynchronous engine, per proposal, then the kinds above
    pub(super) const PROPOSALS: u64 = 5;
}

// when the random numbers of a choice are drawn: a generation of an ask, or a proposal
#[derive(Clone, Copy, Debug)]
enum Step {
    Generation(u64),
    Proposal(u64),
}

// the evaluations of the acquisition function per start of its maximization
const ACQUISITION_EVALUATIONS: u64 = 200;
// the steps of a hill climb on an integer lattice, at most
const LATTICE_STEPS: usize = 10_000;
// the least variance of the model's prediction, in its standardized units, for the acquisition
const VARIANCE_FLOOR: f64 = 1e-12;

/// Bayesian optimization on [`Real`] genomes (and [`Integer`](crate::genome::Integer) ones, see
/// [`Space`]), as an ask / tell [`Algorithm`] and an [`Incremental`] one: for expensive black-box
/// functions, such as a simulation that runs for minutes or a physical experiment, where tens to
/// a few hundred evaluations must do. A Gaussian process ([`model::gp`](crate::model::gp)) models
/// the function from every evaluation so far, and an [`Acquisition`] function of its posterior
/// picks the next points to evaluate, trading the model's best guesses against its uncertainty.
///
/// - **The initial design** (generation 0): [`initial_points`](BoBuilder::initial_points) points,
///   2(n + 1) for n searched genes by default, the
///   [`initial_genomes`](BoBuilder::initial_genomes) first and a Latin hypercube sample (McKay,
///   Beckman and Conover, 1979; [`Real::latin_hypercube`]) for the rest. A small design leaves
///   most of the budget to the model's choices. For an accurate model of the whole box rather
///   than its minimum, Loeppky, Sacks and Welch (2009) recommend 10n points.
/// - **Each later generation** asks a [batch](BoBuilder::batch) of q points, one by default: the
///   model is fitted to every evaluation (its hyperparameters by maximum likelihood, from the
///   last fit's and random starts), then the acquisition function is maximized in the box:
///   evaluated at [`raw_samples`](BoBuilder::raw_samples) random points, then improved by
///   [`Lbfgsb`] with its analytic gradient from the best
///   [`acquisition_starts`](BoBuilder::acquisition_starts) of them and from the best point
///   evaluated so far. The best result that isn't an evaluated point is asked: a point is never
///   asked twice. Each further point of a batch is chosen the same way after the points before it
///   are added to the model with a [`Fantasy`] value, the hyperparameters kept: the q
///   evaluations of a generation can then run at once
///   ([`Engine::parallel`](crate::Engine::parallel)).
/// - **Asynchronous evaluation**: under an [`AsyncEngine`](crate::engine::AsyncEngine), the
///   design is proposed first, then each proposal is a point chosen by the model of the results
///   so far, with the points still being evaluated added with a [`Fantasy`] value as in a batch;
///   until a result arrives, random points. A generation of the engine is
///   [`initial_points`](Bo::initial_points) evaluations. With one worker, a seed gives the same
///   run every time.
/// - **The model** fits the values to minimize, the scores negated when maximizing, through the
///   [`Output`] transform. A point with an invalid fitness, or a score that isn't finite, enters
///   the model at the worst value of the others, so the search learns to avoid where the function
///   fails; until a point has a valid score, the next point is random.
/// - **Constraints** `gᵢ(x) ≤ 0` whose values the fitness function gives one by one, such as a
///   [`Constrained`](crate::constraint::Constrained) one or a test problem with
///   [`constraints`](crate::problems::Problem::constraints): a Gaussian process models each, and
///   the acquisition function is weighed by the probability that a point is feasible, `Π P(gᵢ ≤ 0)`
///   under the constraints' models, taken as independent: the expected constrained improvement of
///   Gardner, Kusner, Xu, Weinberger and Cunningham (2014), the improvement being over the best
///   feasible point (with the log expected improvement, its logarithm plus the probability's).
///   Until a feasible point is evaluated, the search maximizes the probability of feasibility
///   alone. Without constraint values, the search uses the score only, a violation being ignored
///   by it (use a penalty), though [`best`](Algorithm::best) compares by Deb's rules, as everywhere
///   in genoxide.
/// - **Cost.** The model's fit is O(N³) for N evaluations, its predictions O(N²), for the
///   objective and each constraint: Bayesian optimization suits up to a few hundred evaluations of
///   a function that costs far more than that, in up to about 10 to 20 genes.
///
/// [`model`](Bo::model) gives the model that chose the last point. Every random number comes from
/// streams derived from the [seed](BoBuilder::seed), every operation is a sum, product, quotient
/// or square root in a fixed order with [`math`](crate::math)'s functions, and the multi-starts
/// run on rayon (with the `parallel` feature) with the winner chosen by value, then start: a seed
/// gives the same run on every platform and thread count.
///
/// Built with [`Bo::builder`], run with an [`Engine`](crate::Engine) or an
/// [`AsyncEngine`](crate::engine::AsyncEngine).
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
///
/// // 4 points at a time (evaluated in parallel with `Engine::parallel`): 10 rounds
/// let bo = Bo::builder(Branin.representation()).batch(4).minimize().seed(1).build()?;
/// let outcome = Engine::new(bo, Branin)
///     .stop_when(Stop::evaluations(46))
///     .run()?;
/// assert_eq!(outcome.generations(), 10);
/// assert!(outcome.best_fitness().score().unwrap() < 0.397887 + 1e-2);
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
/// practical guide. *Technometrics* 51(4): 366-376. Ginsbourger, D., Le Riche, R. and Carraro, L.
/// (2010). Kriging is well-suited to parallelize optimization. In *Computational Intelligence in
/// Expensive Optimization Problems*, Springer: 131-162. Gardner, J. R., Kusner, M. J., Xu, Z.,
/// Weinberger, K. Q. and Cunningham, J. P. (2014). Bayesian optimization with inequality
/// constraints. *ICML 2014*, PMLR 32(2): 937-945.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "R: serde::Serialize, R::Genome: serde::Serialize",
        deserialize = "R: serde::Deserialize<'de>, R::Genome: serde::Deserialize<'de>"
    ))
)]
pub struct Bo<R: Space = Real> {
    representation: R,
    initial_points: usize,
    acquisition: Acquisition,
    kernel: Kernel,
    noise: Noise,
    output: Output,
    raw_samples: usize,
    acquisition_starts: usize,
    hyperparameter_starts: usize,
    batch: usize,
    fantasy: Fantasy,
    objective: Objective,
    seed: u64,
    // the initial design, asked in generation 0
    design: Vec<R::Genome>,
    // every evaluated point, in the order evaluated
    observations: Population<R::Genome>,
    // the number of inequality constraints whose values the fitness function gives, once known,
    // and their values at each evaluated point, a row per point (NaN for an invalid fitness)
    constraints: Option<usize>,
    constraint_values: Vec<f64>,
    // the logarithms of the last fit's hyperparameters, the next fit's first start, for the
    // objective's model and each constraint's
    warm: Option<Vec<f64>>,
    constraint_warm: Vec<Option<Vec<f64>>>,
    // the points of the current ask
    pending: Vec<Individual<R::Genome>>,
    indices: Vec<usize>,
    asked: bool,
    reevaluating: bool,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<R::Genome>>,
    best_generation: u64,
    best_evaluation: u64,
    // under an asynchronous engine: the points proposed and not received, in order; those to
    // propose again first, after a checkpoint; the design points proposed; and the proposals
    proposed: Vec<R::Genome>,
    requeued: Vec<R::Genome>,
    design_proposed: usize,
    proposals: u64,
    // the models that chose the last point, fitted to the evaluations before it
    #[cfg_attr(feature = "serde", serde(skip))]
    surrogate: Option<Surrogate>,
}

// the models of a step, in their standardized units
#[derive(Clone, Debug)]
struct Surrogate {
    objective: GaussianProcess,
    constraints: Vec<GaussianProcess>,
    // each constraint's bound 0, standardized
    thresholds: Vec<f64>,
    // the best value of the feasible points, standardized; None without one
    best: Option<f64>,
}

impl Bo {
    /// A builder for Bayesian optimization on `representation`, a [`Real`] or an
    /// [`Integer`](crate::genome::Integer) (see [`Space`]).
    pub fn builder<R: Space>(representation: R) -> BoBuilder<R> {
        BoBuilder {
            representation,
            initial_points: None,
            initial_genomes: Vec::new(),
            acquisition: Acquisition::default(),
            kernel: Kernel::default(),
            noise: Noise::default(),
            output: Output::default(),
            raw_samples: 1000,
            acquisition_starts: 10,
            hyperparameter_starts: 5,
            batch: 1,
            fantasy: Fantasy::default(),
            objective: Objective::default(),
            seed: None,
        }
    }
}

impl<R: Space> Bo<R> {
    /// The representation.
    pub fn representation(&self) -> &R {
        &self.representation
    }

    /// Whether higher or lower fitness is better (as [`Algorithm::objective`] and
    /// [`Incremental::objective`]).
    pub fn objective(&self) -> Objective {
        self.objective
    }

    /// Every evaluated point, in the order evaluated (as [`Algorithm::population`] and
    /// [`Incremental::population`]).
    pub fn population(&self) -> &Population<R::Genome> {
        &self.observations
    }

    /// The best point evaluated so far, by Deb's rules (as [`Algorithm::best`] and
    /// [`Incremental::best`]).
    pub fn best(&self) -> Option<&Individual<R::Genome>> {
        self.best.as_ref()
    }

    /// The number of fitness values told or received so far (as [`Algorithm::evaluations`] and
    /// [`Incremental::evaluations`]).
    pub fn evaluations(&self) -> u64 {
        self.evaluations
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
    /// [`Error::InvalidSetting`] for a negative or non-finite ξ or β, or the upper confidence
    /// bound for a problem with constraints. Nothing changes on errors.
    pub fn set_acquisition(&mut self, acquisition: Acquisition) -> Result<()> {
        acquisition.validate()?;
        acquisition.check_constrained(self.constraints.unwrap_or(0))?;
        self.acquisition = acquisition;
        Ok(())
    }

    /// The number of points of each generation after the initial design.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Changes the number of points of each generation from the next ask on.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for 0 or more than 2^24. Nothing changes on errors.
    pub fn set_batch(&mut self, batch: usize) -> Result<()> {
        check_batch(batch)?;
        self.batch = batch;
        Ok(())
    }

    /// What the points of a batch, and those still being evaluated, are taken to be worth.
    pub fn fantasy(&self) -> Fantasy {
        self.fantasy
    }

    /// Changes the [`Fantasy`] from the next choice on.
    pub fn set_fantasy(&mut self, fantasy: Fantasy) {
        self.fantasy = fantasy;
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

    /// The number of inequality constraints whose values the fitness function gives: `None`
    /// before a run has started, and 0 without constraints.
    pub fn constraints(&self) -> Option<usize> {
        self.constraints
    }

    /// The values of the inequality constraints `gᵢ(x) ≤ 0` at the evaluated point `index` of the
    /// [population](Algorithm::population), in order: empty without constraints, NaN for a point
    /// with an invalid fitness.
    ///
    /// # Panics
    ///
    /// If `index` is out of bounds.
    pub fn constraint_values(&self, index: usize) -> &[f64] {
        assert!(
            index < self.observations.len(),
            "point {index} isn't evaluated"
        );
        let m = self.constraints.unwrap_or(0);
        &self.constraint_values[index * m..(index + 1) * m]
    }

    /// The points proposed to an [`AsyncEngine`](crate::engine::AsyncEngine) whose results
    /// haven't arrived, in the order proposed.
    pub fn proposed(&self) -> &[R::Genome] {
        &self.proposed
    }

    /// The Gaussian process that chose the last point asked, fitted to the evaluations before it
    /// (without the [fantasies](Fantasy) of a batch or of pending points): a model of the values
    /// the search minimizes (the scores, negated when maximizing, through the [`Output`]
    /// transform). `None` before the first point after the initial design, after a random point
    /// (when no evaluation had a valid score), and after loading a checkpoint until the next point
    /// is asked.
    pub fn model(&self) -> Option<&GaussianProcess> {
        self.surrogate
            .as_ref()
            .map(|surrogate| &surrogate.objective)
    }

    /// The Gaussian processes of the constraints that chose the last point, one per constraint,
    /// fitted as [`model`](Bo::model): empty without constraints or without a model.
    pub fn constraint_models(&self) -> &[GaussianProcess] {
        self.surrogate
            .as_ref()
            .map_or(&[], |surrogate| &surrogate.constraints[..])
    }

    // the unit-cube point of gene values, on the lattice for integers
    fn unit_of(&self, genome: &[f64]) -> Vec<f64> {
        assert_eq!(
            genome.len(),
            self.representation.genome_len(),
            "a genome of {} genes",
            self.representation.genome_len()
        );
        let scaling = self.representation.scaling();
        let mut unit = vec![0.0; scaling.dims()];
        scaling.to_unit(genome, &mut unit);
        if self.representation.lattice().is_some() {
            let nearest = self.representation.from_unit(&scaling, &unit);
            R::to_unit(&scaling, &nearest, &mut unit);
        }
        unit
    }

    /// The acquisition function at the gene values `genome` (rounded to the nearest integers for
    /// an [`Integer`](crate::genome::Integer) genome) under [`model`](Bo::model), as the search
    /// maximizes it: the log expected improvement, the expected improvement, the logarithm of the
    /// probability of improvement, or the negated lower confidence bound, in the model's
    /// standardized units; with constraints, weighed by the probability of feasibility (its
    /// logarithm added to a logarithm), or the logarithm of that probability alone before a
    /// feasible point is evaluated. `None` without a model.
    ///
    /// # Panics
    ///
    /// If `genome` doesn't have a value per gene.
    pub fn acquisition_at(&self, genome: &[f64]) -> Option<f64> {
        let surrogate = self.surrogate.as_ref()?;
        let unit = self.unit_of(genome);
        Some(self.search(surrogate).value(&unit, None))
    }

    /// The probability that the gene values `genome` (rounded for an
    /// [`Integer`](crate::genome::Integer) genome) are feasible, under the
    /// [constraints' models](Bo::constraint_models): `Π P(gᵢ ≤ 0)`. 1 without constraints, `None`
    /// without a model.
    ///
    /// # Panics
    ///
    /// If `genome` doesn't have a value per gene.
    pub fn probability_of_feasibility_at(&self, genome: &[f64]) -> Option<f64> {
        let surrogate = self.surrogate.as_ref()?;
        let unit = self.unit_of(genome);
        let mut log = 0.0;
        for (model, &threshold) in surrogate.constraints.iter().zip(&surrogate.thresholds) {
            let (mean, variance) = model.predict_unit(&unit, None);
            let sd = variance.max(VARIANCE_FLOOR).sqrt();
            log +=
                acquisition::log_probability_of_improvement_derivatives(mean, sd, threshold, 0.0)
                    [0];
        }
        Some(crate::math::exp(log))
    }

    fn search<'a>(&self, surrogate: &'a Surrogate) -> Search<'a> {
        Search {
            surrogate,
            acquisition: self.acquisition,
            scale: surrogate.objective.standardization().1,
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
    fn observed(&self, genome: &R::Genome) -> bool {
        self.observations
            .iter()
            .any(|individual| individual.genome() == genome)
    }

    // whether `genome` was evaluated, or is among `others`
    fn taken(&self, genome: &R::Genome, others: &[&[R::Genome]]) -> bool {
        self.observed(genome) || others.iter().any(|list| list.contains(genome))
    }

    // the random stream of `kind` for `step`, and its point `pick` of a batch
    fn stream(&self, kind: u64, step: Step, pick: usize) -> StreamRng {
        let root = StreamRng::seed_from_u64(self.seed);
        let rng = match step {
            Step::Generation(generation) => root.derive(kind).derive(generation),
            Step::Proposal(proposal) => root
                .derive(streams::PROPOSALS)
                .derive(proposal)
                .derive(kind),
        };
        if pick == 0 {
            rng
        } else {
            rng.derive(pick as u64)
        }
    }

    // a random point of the box that wasn't evaluated and isn't among `others`, from the stream of
    // this step and pick; None once every point of a lattice is taken
    fn random_point(&self, step: Step, pick: usize, others: &[&[R::Genome]]) -> Option<R::Genome> {
        let mut rng = self.stream(streams::RANDOM, step, pick);
        // a few draws, then, on a lattice, its first point not taken
        for _ in 0..1000 {
            let genome = self.representation.random_genome(&mut rng);
            if !self.taken(&genome, others) {
                return Some(genome);
            }
        }
        let taken = self.observations.len() + others.iter().map(|list| list.len()).sum::<usize>();
        match self.representation.lattice() {
            None => loop {
                let genome = self.representation.random_genome(&mut rng);
                if !self.taken(&genome, others) {
                    return Some(genome);
                }
            },
            Some(points) if points > taken as u128 && points <= 1 << 24 => self
                .representation
                .enumerate(1 << 24)?
                .into_iter()
                .find(|genome| !self.taken(genome, others)),
            Some(_) => None,
        }
    }

    // the models of the evaluations, and the unit-cube point to start the acquisition's
    // maximization from (the best point evaluated) and the values of the lies, standardized: min,
    // mean, max; None if there's no valid value, or a model doesn't factor
    fn fit(&mut self, step: Step) -> Option<(Surrogate, Vec<f64>, [f64; 3])> {
        let scaling = self.representation.scaling();
        let dims = scaling.dims();
        let count = self.observations.len();
        let mut x = vec![0.0; count * dims];
        let mut values = Vec::with_capacity(count);
        for (individual, unit) in self.observations.iter().zip(x.chunks_exact_mut(dims)) {
            R::to_unit(&scaling, individual.genome(), unit);
            values.push(self.model_value(individual.fitness()));
        }
        let targets = transform(&values, self.output)?;
        let m = self.constraints.unwrap_or(0);
        let settings = |seed| gp::Settings {
            kernel: self.kernel,
            noise: self.noise,
            starts: self.hyperparameter_starts,
            seed,
        };
        let seed = self.stream(streams::HYPERPARAMETERS, step, 0).next_u64();
        let points = if m > 0 { x.clone() } else { Vec::new() };
        let fitted = GaussianProcess::fit_unit(
            settings(seed),
            scaling.clone(),
            x,
            &targets,
            self.warm.as_deref(),
        );
        // a kernel matrix that doesn't factor, even with jitter: a random point instead
        let model = fitted.ok()?;
        self.warm = Some(model.log_parameters().to_vec());
        // the constraints' models, of their values standardized, the invalid ones at the worst
        let mut constraints = Vec::with_capacity(m);
        let mut thresholds = Vec::with_capacity(m);
        let constraint_seeds = self.stream(streams::CONSTRAINTS, step, 0);
        for i in 0..m {
            let column: Vec<Option<f64>> = (0..count)
                .map(|o| {
                    let g = self.constraint_values[o * m + i];
                    g.is_finite().then_some(g)
                })
                .collect();
            let targets = transform(&column, Output::Standardize)?;
            let seed = constraint_seeds.derive(i as u64).next_u64();
            let warm = self.constraint_warm[i].as_deref();
            let fitted = GaussianProcess::fit_unit(
                settings(seed),
                scaling.clone(),
                points.clone(),
                &targets,
                warm,
            );
            let constraint = fitted.ok()?;
            self.constraint_warm[i] = Some(constraint.log_parameters().to_vec());
            let (mean, scale) = constraint.standardization();
            thresholds.push((0.0 - mean) / scale);
            constraints.push(constraint);
        }
        // the best point evaluated, by the model's values (of the feasible points, with
        // constraints): the incumbent, and a start
        let mut incumbent: Option<usize> = None;
        for (index, value) in values.iter().enumerate() {
            let feasible = m == 0
                || self.observations.as_slice()[index]
                    .fitness()
                    .is_some_and(Fitness::is_feasible);
            if value.is_some()
                && feasible
                && incumbent.is_none_or(|best| targets[index] < targets[best])
            {
                incumbent = Some(index);
            }
        }
        let (y_mean, y_scale) = model.standardization();
        let best = incumbent.map(|index| (targets[index] - y_mean) / y_scale);
        // without a feasible point, the start is the least infeasible one
        let start = incumbent.unwrap_or_else(|| {
            let mut least = 0;
            let mut violation = f64::INFINITY;
            for (index, individual) in self.observations.iter().enumerate() {
                let v = individual
                    .fitness()
                    .filter(|fitness| fitness.is_valid())
                    .map_or(f64::INFINITY, Fitness::violation);
                if v < violation {
                    (least, violation) = (index, v);
                }
            }
            least
        });
        let start = model.unit_point(start).to_vec();
        let standardized = model.targets();
        let (mut low, mut sum, mut high) = (f64::INFINITY, 0.0, f64::NEG_INFINITY);
        for &y in standardized {
            low = low.min(y);
            sum += y;
            high = high.max(y);
        }
        let lies = [low, sum / standardized.len() as f64, high];
        let surrogate = Surrogate {
            objective: model,
            constraints,
            thresholds,
            best,
        };
        Some((surrogate, start, lies))
    }

    // the fantasized values of the point `unit` under `current`: the objective's, and each
    // constraint's
    fn fantasize(&self, current: &Surrogate, unit: &[f64], lies: [f64; 3]) -> (f64, Vec<f64>) {
        let objective = match self.fantasy {
            Fantasy::KrigingBeliever => current.objective.predict_unit(unit, None).0,
            Fantasy::ConstantLiar(Lie::Min) => lies[0],
            Fantasy::ConstantLiar(Lie::Mean) => lies[1],
            Fantasy::ConstantLiar(Lie::Max) => lies[2],
        };
        let constraints = current
            .constraints
            .iter()
            .map(|model| model.predict_unit(unit, None).0)
            .collect();
        (objective, constraints)
    }

    // up to `count` new points: the models fitted to the evaluations, the `pending` points added
    // with fantasized values, then the acquisition maximized, each point after the first with
    // the points before it added alike
    fn choose(&mut self, count: usize, pending: &[R::Genome], step: Step) -> Vec<R::Genome> {
        let mut chosen: Vec<R::Genome> = Vec::with_capacity(count);
        self.surrogate = None;
        let Some((base, start, lies)) = self.fit(step) else {
            for pick in 0..count {
                match self.random_point(step, pick, &[pending, &chosen]) {
                    Some(genome) => chosen.push(genome),
                    None => break,
                }
            }
            return chosen;
        };
        let scaling = self.representation.scaling();
        let dims = scaling.dims();
        let mut fantasies = Fantasies::default();
        let mut current: Option<Surrogate> = None;
        let mut unit = vec![0.0; dims];
        for genome in pending {
            R::to_unit(&scaling, genome, &mut unit);
            let (objective, constraints) =
                self.fantasize(current.as_ref().unwrap_or(&base), &unit, lies);
            fantasies.push(&unit, objective, &constraints);
            current = condition(&base, &fantasies).or(current);
        }
        for pick in 0..count {
            let surrogate = current.as_ref().unwrap_or(&base);
            let point = self
                .maximize(surrogate, &scaling, &start, step, pick, &[pending, &chosen])
                .or_else(|| self.random_point(step, pick, &[pending, &chosen]));
            let Some(point) = point else { break };
            if pick + 1 < count {
                R::to_unit(&scaling, &point, &mut unit);
                let (objective, constraints) = self.fantasize(surrogate, &unit, lies);
                fantasies.push(&unit, objective, &constraints);
                current = condition(&base, &fantasies).or(current);
            }
            chosen.push(point);
        }
        self.surrogate = Some(base);
        chosen
    }

    // the acquisition's best point under `surrogate` that isn't taken: L-BFGS-B from the raw
    // samples' best and from `start` for reals, a hill climb on the lattice for integers
    fn maximize(
        &self,
        surrogate: &Surrogate,
        scaling: &Scaling,
        start: &[f64],
        step: Step,
        pick: usize,
        others: &[&[R::Genome]],
    ) -> Option<R::Genome> {
        if self.representation.lattice().is_some() {
            return self.maximize_lattice(surrogate, scaling, start, step, pick, others);
        }
        let search = self.search(surrogate);
        let dims = scaling.dims();
        // the raw samples, and the best of them as starts after the incumbent
        let mut rng = self.stream(streams::RAW_SAMPLES, step, pick);
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
        starts.push(start.to_vec());
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
        candidates
            .iter()
            .map(|(_, unit)| self.representation.from_unit(scaling, unit))
            .find(|genome| !self.taken(genome, others))
    }

    // the acquisition's best lattice point that isn't taken: the raw samples (every point of a
    // small lattice), then a hill climb from the best of them and from `start`
    fn maximize_lattice(
        &self,
        surrogate: &Surrogate,
        scaling: &Scaling,
        start: &[f64],
        step: Step,
        pick: usize,
        others: &[&[R::Genome]],
    ) -> Option<R::Genome> {
        let search = self.search(surrogate);
        let dims = scaling.dims();
        // the acquisition at a lattice point, −∞ at a point taken
        let value = |genome: &R::Genome| {
            if self.taken(genome, others) {
                return f64::NEG_INFINITY;
            }
            let mut unit = vec![0.0; dims];
            R::to_unit(scaling, genome, &mut unit);
            let value = search.value(&unit, None);
            if value.is_nan() {
                f64::NEG_INFINITY
            } else {
                value
            }
        };
        let raw = match self.representation.enumerate(self.raw_samples) {
            Some(all) => all,
            None => {
                let mut rng = self.stream(streams::RAW_SAMPLES, step, pick);
                (0..self.raw_samples)
                    .map(|_| self.representation.random_genome(&mut rng))
                    .collect()
            }
        };
        let raw_values = map_in_order(raw.len(), |i| value(&raw[i]));
        let mut order: Vec<usize> = (0..raw.len()).collect();
        // highest first; stable, so ties keep the earlier sample
        order.sort_by(|&a, &b| raw_values[b].total_cmp(&raw_values[a]));
        let mut starts = Vec::with_capacity(self.acquisition_starts + 1);
        starts.push(self.representation.from_unit(scaling, start));
        for &i in order.iter().take(self.acquisition_starts) {
            starts.push(raw[i].clone());
        }
        let climbs = map_in_order(starts.len(), |s| {
            let mut point = starts[s].clone();
            let mut best = value(&point);
            let mut neighbors = Vec::new();
            for _ in 0..LATTICE_STEPS {
                self.representation.neighbors(&point, &mut neighbors);
                let mut next: Option<(f64, usize)> = None;
                for (n, neighbor) in neighbors.iter().enumerate() {
                    let v = value(neighbor);
                    if v > best && next.is_none_or(|(top, _)| v > top) {
                        next = Some((v, n));
                    }
                }
                let Some((v, n)) = next else { break };
                best = v;
                point = neighbors.swap_remove(n);
            }
            (best, point)
        });
        let mut candidates: Vec<(f64, R::Genome)> = climbs;
        candidates.extend(order.iter().map(|&i| (raw_values[i], raw[i].clone())));
        // highest first; stable, so ties keep the earlier start
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        candidates
            .into_iter()
            .filter(|(value, _)| *value > f64::NEG_INFINITY)
            .map(|(_, genome)| genome)
            .find(|genome| !self.taken(genome, others))
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
            let step = Step::Generation(self.generation + 1);
            let pending = self.proposed.clone();
            let points = self.choose(self.batch, &pending, step);
            self.pending.extend(points.into_iter().map(Individual::new));
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

    // the number of constraints from what the fitness function provides
    fn prepare_constraints(&mut self, provided: Provided) -> Result<()> {
        let m = provided.inequalities;
        self.acquisition.check_constrained(m)?;
        match self.constraints {
            Some(known) if known != m && !self.observations.is_empty() => {
                Err(Error::InvalidSetting {
                    setting: "fitness",
                    reason: format!(
                        "the fitness function gives {m} constraints, and the evaluations so far \
                         have {known}"
                    ),
                })
            }
            _ => {
                self.constraints = Some(m);
                self.constraint_warm.resize(m, None);
                self.constraint_warm.truncate(m);
                Ok(())
            }
        }
    }

    // the constraints' values of an evaluation, checked: an error if they're missing
    fn checked_values<'e>(
        &self,
        evaluations: &Evaluations<'e>,
        position: usize,
    ) -> Result<Option<&'e [f64]>> {
        let m = self.constraints.unwrap_or(0);
        if m == 0 {
            return Ok(None);
        }
        match evaluations.inequalities(position) {
            Some(values) if values.len() == m => Ok(Some(values)),
            _ => Err(Error::InvalidSetting {
                setting: "fitness",
                reason: format!(
                    "Bayesian optimization with {m} constraints needs their values with each \
                     fitness: tell them with `tell_evaluations` (as the engines do), from a \
                     fitness function that provides them"
                ),
            }),
        }
    }

    // the constraints' values of a fitness, NaN for an invalid one
    fn push_values(&mut self, fitness: Fitness, values: Option<&[f64]>) {
        let m = self.constraints.unwrap_or(0);
        match values {
            Some(values) if fitness.is_valid() => self.constraint_values.extend_from_slice(values),
            _ => self
                .constraint_values
                .extend(std::iter::repeat_n(f64::NAN, m)),
        }
    }

    // the best so far, from the newly evaluated observations from `first` on: the first on ties
    fn update_best(&mut self, first: usize) {
        let evaluations = self.evaluations - (self.observations.len() - first) as u64;
        for (k, individual) in self.observations.as_slice()[first..].iter().enumerate() {
            let fitness = individual.fitness().unwrap_or(Fitness::invalid());
            let better = match &self.best {
                Some(best) => self
                    .objective
                    .is_better(fitness, best.fitness().unwrap_or(Fitness::invalid())),
                None => true,
            };
            if better {
                self.best = Some(individual.clone());
                self.best_generation = self.generation;
                self.best_evaluation = evaluations + k as u64 + 1;
            }
        }
    }

    // takes a result under an asynchronous engine
    fn receive_one(
        &mut self,
        genome: R::Genome,
        fitness: Fitness,
        values: Option<&[f64]>,
    ) -> Result<Option<Individual<R::Genome>>> {
        self.representation.validate(&genome)?;
        if let Some(position) = self.proposed.iter().position(|g| *g == genome) {
            self.proposed.remove(position);
        } else if let Some(position) = self.requeued.iter().position(|g| *g == genome) {
            self.requeued.remove(position);
        }
        let mut individual = Individual::new(genome);
        individual.set_fitness(fitness);
        if self.observed(individual.genome()) {
            return Ok(Some(individual));
        }
        self.evaluations += 1;
        self.observations.push(individual);
        self.push_values(fitness, values);
        self.update_best(self.observations.len() - 1);
        Ok(None)
    }
}

// the points added to the models with fantasized values, the unit-cube points a row each
#[derive(Default)]
struct Fantasies {
    units: Vec<f64>,
    objective: Vec<f64>,
    // a row of the constraints' values per point
    constraints: Vec<f64>,
}

impl Fantasies {
    fn push(&mut self, unit: &[f64], objective: f64, constraints: &[f64]) {
        self.units.extend_from_slice(unit);
        self.objective.push(objective);
        self.constraints.extend_from_slice(constraints);
    }
}

// the models of `base` with the fantasized points added, as if observed, and the best value with
// those of them that are feasible by their fantasized constraints; None if a model doesn't factor
fn condition(base: &Surrogate, fantasies: &Fantasies) -> Option<Surrogate> {
    let objective = base
        .objective
        .with_points(&fantasies.units, &fantasies.objective)?;
    let m = base.constraints.len();
    let mut constraints = Vec::with_capacity(m);
    for (i, model) in base.constraints.iter().enumerate() {
        let values: Vec<f64> = fantasies
            .objective
            .iter()
            .enumerate()
            .map(|(point, _)| fantasies.constraints[point * m + i])
            .collect();
        constraints.push(model.with_points(&fantasies.units, &values)?);
    }
    let mut best = base.best;
    for (point, &value) in fantasies.objective.iter().enumerate() {
        let feasible = (0..m).all(|i| fantasies.constraints[point * m + i] <= base.thresholds[i]);
        if feasible && best.is_none_or(|best| value < best) {
            best = Some(value);
        }
    }
    Some(Surrogate {
        objective,
        constraints,
        thresholds: base.thresholds.clone(),
        best,
    })
}

// the error for a batch of 0, or too large
fn check_batch(batch: usize) -> Result<()> {
    if batch == 0 {
        return Err(Error::InvalidSetting {
            setting: "batch",
            reason: "must be at least 1, got 0".to_string(),
        });
    }
    crate::operator::check_size("batch", batch).map(|_| ())
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

// the acquisition function of a surrogate, in its standardized units, as maximized
struct Search<'a> {
    surrogate: &'a Surrogate,
    acquisition: Acquisition,
    // the model's scale of the values, for ξ
    scale: f64,
}

impl Search<'_> {
    // the acquisition's value at the unit-cube point `u`, and its gradient
    fn value(&self, u: &[f64], mut gradient: Option<&mut [f64]>) -> f64 {
        let surrogate = self.surrogate;
        let dims = u.len();
        let mut value = 0.0;
        if let Some(best) = surrogate.best {
            value = self.objective(u, best, gradient.as_deref_mut());
        } else if let Some(gradient) = gradient.as_deref_mut() {
            gradient.fill(0.0);
        }
        if surrogate.constraints.is_empty() {
            return value;
        }
        // the logarithm of the probability of feasibility, Σ ln Φ((tᵢ − μᵢ)/σᵢ), and its gradient
        let mut log_feasible = 0.0;
        let mut d_log = vec![0.0; if gradient.is_some() { dims } else { 0 }];
        let (mut dmean, mut dvariance) = (vec![0.0; dims], vec![0.0; dims]);
        for (model, &threshold) in surrogate.constraints.iter().zip(&surrogate.thresholds) {
            let gradients = gradient
                .is_some()
                .then_some((&mut dmean[..], &mut dvariance[..]));
            let (mean, variance) = model.predict_unit(u, gradients);
            let floored = variance < VARIANCE_FLOOR;
            let sd = variance.max(VARIANCE_FLOOR).sqrt();
            let [log, by_mean, by_sd] =
                acquisition::log_probability_of_improvement_derivatives(mean, sd, threshold, 0.0);
            log_feasible += log;
            if gradient.is_some() {
                for i in 0..dims {
                    let dsd = if floored {
                        0.0
                    } else {
                        dvariance[i] / (2.0 * sd)
                    };
                    d_log[i] += by_mean * dmean[i] + by_sd * dsd;
                }
            }
        }
        match (surrogate.best, self.acquisition) {
            // before a feasible point: the probability of feasibility alone, by its logarithm
            (None, _) => {
                if let Some(gradient) = gradient {
                    gradient.copy_from_slice(&d_log);
                }
                log_feasible
            }
            // EI × P: (EI P)′ = P EI′ + EI P (ln P)′
            (Some(_), Acquisition::ExpectedImprovement) => {
                let feasible = crate::math::exp(log_feasible);
                if let Some(gradient) = gradient {
                    for i in 0..dims {
                        gradient[i] = feasible * gradient[i] + value * feasible * d_log[i];
                    }
                }
                value * feasible
            }
            // a logarithm: ln a + ln P
            (Some(_), _) => {
                if let Some(gradient) = gradient {
                    for i in 0..dims {
                        gradient[i] += d_log[i];
                    }
                }
                value + log_feasible
            }
        }
    }

    // the objective's acquisition at `u`, and its gradient
    fn objective(&self, u: &[f64], best: f64, gradient: Option<&mut [f64]>) -> f64 {
        let dims = u.len();
        let (mut dmean, mut dvariance) = (Vec::new(), Vec::new());
        let gradients = if gradient.is_some() {
            dmean.resize(dims, 0.0);
            dvariance.resize(dims, 0.0);
            Some((&mut dmean[..], &mut dvariance[..]))
        } else {
            None
        };
        let (mean, variance) = self.surrogate.objective.predict_unit(u, gradients);
        let floored = variance < VARIANCE_FLOOR;
        let sd = variance.max(VARIANCE_FLOOR).sqrt();
        let [value, by_mean, by_sd] = match self.acquisition {
            Acquisition::ExpectedImprovement => {
                acquisition::expected_improvement_derivatives(mean, sd, best)
            }
            Acquisition::LogExpectedImprovement => {
                acquisition::log_expected_improvement_derivatives(mean, sd, best)
            }
            Acquisition::ProbabilityOfImprovement { xi } => {
                acquisition::log_probability_of_improvement_derivatives(
                    mean,
                    sd,
                    best,
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

impl<R: Space> Reevaluate for Bo<R> {
    /// As [`Bo::reevaluate`]: the next ask gives every evaluated point again.
    fn reevaluate(&mut self) -> Result<()> {
        Bo::reevaluate(self)
    }
}

impl<R: Space> Algorithm for Bo<R> {
    type Genome = R::Genome;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, R::Genome> {
        if !self.asked {
            self.build_ask();
            self.asked = true;
        }
        Candidates::new(&self.pending, &self.indices)
    }

    /// # Errors
    ///
    /// As [`Algorithm::tell`], and [`Error::InvalidSetting`] after a run with constraints, whose
    /// values a tell doesn't have: [`tell_evaluations`](Algorithm::tell_evaluations) gives them.
    fn tell(&mut self, fitness: &[Fitness]) -> Result<()> {
        self.tell_evaluations(&Evaluations::new(fitness))
    }

    fn tell_evaluations(&mut self, evaluations: &Evaluations<'_>) -> Result<()> {
        let fitness = evaluations.fitness();
        if !self.asked {
            return Err(Error::TellWithoutAsk);
        }
        if fitness.len() != self.pending.len() {
            return Err(Error::FitnessCount {
                expected: self.pending.len(),
                got: fitness.len(),
            });
        }
        let mut values = Vec::with_capacity(fitness.len());
        for position in 0..fitness.len() {
            values.push(self.checked_values(evaluations, position)?);
        }
        self.asked = false;
        self.evaluations += fitness.len() as u64;
        self.indices.clear();
        if self.reevaluating {
            self.reevaluating = false;
            self.pending.clear();
            let m = self.constraints.unwrap_or(0);
            for (index, (observation, &fitness)) in
                self.observations.iter_mut().zip(fitness).enumerate()
            {
                observation.set_fitness(fitness);
                let row = &mut self.constraint_values[index * m..(index + 1) * m];
                match values[index] {
                    Some(values) if fitness.is_valid() => row.copy_from_slice(values),
                    _ => row.fill(f64::NAN),
                }
            }
            self.best = None;
            self.update_best(0);
            return Ok(());
        }
        if !self.observations.is_empty() {
            self.generation += 1;
        }
        let first = self.observations.len();
        let pending = std::mem::take(&mut self.pending);
        for ((mut individual, &fitness), values) in pending.into_iter().zip(fitness).zip(values) {
            individual.set_fitness(fitness);
            self.observations.push(individual);
            self.push_values(fitness, values);
        }
        self.update_best(first);
        Ok(())
    }

    fn population(&self) -> &Population<R::Genome> {
        &self.observations
    }

    fn best(&self) -> Option<&Individual<R::Genome>> {
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

    /// Whether every point of an [`Integer`](crate::genome::Integer) genome's lattice is
    /// evaluated: never for [`Real`] genomes.
    fn is_finished(&self) -> bool {
        !self.reevaluating
            && self
                .representation
                .lattice()
                .is_some_and(|points| self.observations.len() as u128 >= points)
    }

    /// Takes the number of inequality constraints whose values the fitness function gives, such
    /// as a [`Constrained`](crate::constraint::Constrained) one: with constraints, each is
    /// modeled.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for the upper confidence bound with constraints, or another
    /// number of constraints than the evaluations so far have.
    fn prepare(&mut self, provided: Provided) -> Result<()> {
        self.prepare_constraints(provided)
    }

    fn wants(&self) -> Wanted {
        if self.constraints.unwrap_or(0) > 0 {
            Wanted::NOTHING.with_inequalities()
        } else {
            Wanted::NOTHING
        }
    }
}

impl<R: Space> Incremental for Bo<R> {
    type Genome = R::Genome;

    fn objective(&self) -> Objective {
        self.objective
    }

    /// The next point: a point proposed before a checkpoint whose result never came, then the
    /// initial design, then the points the model chooses with the points still being evaluated
    /// added with a [`Fantasy`] value (random points until a result has a valid score). Once
    /// every point of an integer lattice is evaluated or being evaluated, the best point again.
    fn propose(&mut self) -> R::Genome {
        // a point proposed again isn't a new proposal: a resumed run's later proposals draw the
        // random numbers that an uninterrupted run's do
        let genome = if self.requeued.is_empty() {
            let genome = self.next_proposal();
            self.proposals += 1;
            genome
        } else {
            self.requeued.remove(0)
        };
        self.proposed.push(genome.clone());
        genome
    }

    /// Takes a result: the genome joins the evaluations, unless it's evaluated already (then it's
    /// returned).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidGenome`] for a genome that doesn't fit the representation, and
    /// [`Error::InvalidSetting`] in a run with constraints, whose values
    /// [`receive_evaluation`](Incremental::receive_evaluation) gives. Nothing changes on errors.
    fn receive(
        &mut self,
        genome: R::Genome,
        fitness: Fitness,
    ) -> Result<Option<Individual<R::Genome>>> {
        let fitness = [fitness];
        self.receive_evaluation(genome, &Evaluations::new(&fitness))
    }

    fn receive_evaluation(
        &mut self,
        genome: R::Genome,
        evaluation: &Evaluations<'_>,
    ) -> Result<Option<Individual<R::Genome>>> {
        let &[fitness] = evaluation.fitness() else {
            return Err(Error::FitnessCount {
                expected: 1,
                got: evaluation.len(),
            });
        };
        let values = self.checked_values(evaluation, 0)?;
        self.receive_one(genome, fitness, values)
    }

    fn population(&self) -> &Population<R::Genome> {
        &self.observations
    }

    /// The initial design's size: the engine counts a generation every this many evaluations.
    fn population_size(&self) -> usize {
        self.initial_points
    }

    fn best(&self) -> Option<&Individual<R::Genome>> {
        self.best.as_ref()
    }

    fn evaluations(&self) -> u64 {
        self.evaluations
    }

    fn best_evaluation(&self) -> u64 {
        self.best_evaluation
    }

    /// As [`Algorithm::prepare`]; besides, the points proposed and not received, whose
    /// evaluations a new run no longer has (a run that continues a checkpoint, or another run of
    /// the same algorithm), are proposed again first.
    fn prepare(&mut self, provided: Provided) -> Result<()> {
        self.prepare_constraints(provided)?;
        let lost = std::mem::take(&mut self.proposed);
        self.requeued.splice(0..0, lost);
        Ok(())
    }

    fn wants(&self) -> Wanted {
        Algorithm::wants(self)
    }
}

impl<R: Space> Bo<R> {
    // a proposal that isn't a requeued point
    fn next_proposal(&mut self) -> R::Genome {
        while self.design_proposed < self.design.len() {
            let genome = self.design[self.design_proposed].clone();
            self.design_proposed += 1;
            if !self.taken(&genome, &[&self.proposed]) {
                return genome;
            }
        }
        let step = Step::Proposal(self.proposals);
        let pending = self.proposed.clone();
        if let Some(genome) = self.choose(1, &pending, step).pop() {
            return genome;
        }
        // every point of the lattice is taken
        match &self.best {
            Some(best) => best.genome().clone(),
            None => self.design[0].clone(),
        }
    }
}

/// A builder for a [`Bo`], from [`Bo::builder`].
///
/// Defaults: maximize; an initial design of 2(n + 1) points for n searched genes;
/// [`Acquisition::LogExpectedImprovement`]; the [Matérn 5/2 kernel](Kernel::Matern52) without
/// noise, the model interpolating the values; [`Output::Standardize`]; 1000 raw samples and 10
/// starts for the acquisition's maximization; 5 starts for the hyperparameters'; batches of 1
/// point, the [Kriging believer](Fantasy::KrigingBeliever); a random seed.
#[derive(Clone, Debug)]
pub struct BoBuilder<R: Space = Real> {
    representation: R,
    initial_points: Option<usize>,
    initial_genomes: Vec<R::Genome>,
    acquisition: Acquisition,
    kernel: Kernel,
    noise: Noise,
    output: Output,
    raw_samples: usize,
    acquisition_starts: usize,
    hyperparameter_starts: usize,
    batch: usize,
    fantasy: Fantasy,
    objective: Objective,
    seed: Option<u64>,
}

impl<R: Space> BoBuilder<R> {
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
    pub fn initial_genomes<I: IntoIterator<Item = R::Genome>>(mut self, genomes: I) -> Self {
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

    /// The model's observation noise: none by default ([`Noise::Fixed`] of 0), the model
    /// interpolating the values of a deterministic function; [`Noise::Learned`] for a noisy one.
    /// See [`Noise`] for the evidence.
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

    /// The number of points q of each generation after the initial design, at least 1: 1 by
    /// default. A batch of q points is chosen one after the other, each after the points before
    /// it are added to the model with a [`Fantasy`] value ([`fantasy`](BoBuilder::fantasy)), and
    /// then evaluated together: in parallel with
    /// [`Engine::parallel`](crate::Engine::parallel), so that a generation takes about as long
    /// as one evaluation. A batch needs more evaluations than one point at a time to come as
    /// close, as each point is chosen knowing less, and fewer generations.
    pub fn batch(mut self, points: usize) -> Self {
        self.batch = points;
        self
    }

    /// What the points of a batch, and those still being evaluated under an
    /// [`AsyncEngine`](crate::engine::AsyncEngine), are taken to be worth: the
    /// [Kriging believer](Fantasy::KrigingBeliever) by default.
    pub fn fantasy(mut self, fantasy: Fantasy) -> Self {
        self.fantasy = fantasy;
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
    ///   value, an initial design of 0 points, above 2^24 or above the points of an integer
    ///   lattice, more initial genomes than initial points or the same genome twice, an invalid
    ///   [`Acquisition`] parameter or [`Noise`], 0 raw samples, acquisition starts of 0 or more
    ///   than the raw samples, hyperparameter starts of 0, or a batch of 0 or above 2^24.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<Bo<R>> {
        let invalid =
            |setting: &'static str, reason: String| Err(Error::InvalidSetting { setting, reason });
        let dims = self.representation.scaling().dims();
        if dims == 0 {
            return invalid(
                "representation",
                "Bayesian optimization needs a gene with more than one value".to_string(),
            );
        }
        let initial_points = self.initial_points.unwrap_or(2 * (dims + 1));
        if initial_points == 0 {
            return invalid("initial_points", "must be at least 1, got 0".to_string());
        }
        crate::operator::check_size("initial_points", initial_points)?;
        if let Some(points) = self.representation.lattice()
            && initial_points as u128 > points
        {
            if self.initial_points.is_some() {
                return invalid(
                    "initial_points",
                    format!("at most the {points} points of the lattice, got {initial_points}"),
                );
            }
        }
        // the default design, on a lattice of fewer points: the whole lattice
        let initial_points = self
            .representation
            .lattice()
            .map_or(initial_points, |points| {
                initial_points.min(points.min(usize::MAX as u128) as usize)
            });
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
            self.representation.validate(genome)?;
            if self.initial_genomes[..index].contains(genome) {
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
        check_batch(self.batch)?;
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let mut design = self.initial_genomes;
        let missing = initial_points - design.len();
        if missing > 0 {
            let mut rng = StreamRng::seed_from_u64(seed).derive(streams::DESIGN);
            let sample = self.representation.design(missing, &mut rng)?;
            // a point the design has already (as on a small lattice): a random one instead
            for genome in sample {
                let mut genome = genome;
                while design.contains(&genome) {
                    genome = self.representation.random_genome(&mut rng);
                }
                design.push(genome);
            }
        }
        Ok(Bo {
            representation: self.representation,
            initial_points,
            acquisition: self.acquisition,
            kernel: self.kernel,
            noise: self.noise,
            output: self.output,
            raw_samples: self.raw_samples,
            acquisition_starts: self.acquisition_starts,
            hyperparameter_starts: self.hyperparameter_starts,
            batch: self.batch,
            fantasy: self.fantasy,
            objective: self.objective,
            seed,
            design,
            observations: Population::default(),
            constraints: None,
            constraint_values: Vec::new(),
            warm: None,
            constraint_warm: Vec::new(),
            pending: Vec::new(),
            indices: Vec::new(),
            asked: false,
            reevaluating: false,
            generation: 0,
            evaluations: 0,
            best: None,
            best_generation: 0,
            best_evaluation: 0,
            proposed: Vec::new(),
            requeued: Vec::new(),
            design_proposed: 0,
            proposals: 0,
            surrogate: None,
        })
    }
}

#[cfg(test)]
mod tests;
