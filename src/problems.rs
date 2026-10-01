//! Single-objective test problems: fitness functions with their search space, their known optimum
//! and the paper that defines them.
//!
//! Each problem implements [`Problem`], and is a [`FitnessFunction`] for an [`Engine`] as it is:
//!
//! ```
//! use genoxide::prelude::*;
//! use genoxide::problems::{Problem, Sphere};
//!
//! let problem = Sphere::new(10);
//! let target = problem.optimum().expect("known").value() + 1e-8;
//! let cmaes = Cmaes::builder(problem.representation()).minimize().seed(1).build()?;
//! let outcome = Engine::new(cmaes, problem)
//!     .stop_when(Stop::target(target).or(Stop::evaluations(100_000)))
//!     .run()?;
//! assert_eq!(outcome.stop_reason(), StopReason::Target);
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! [`all`] lists every problem on [`Real`] genomes at its default size, behind the object-safe
//! [`DynProblem`].
//!
//! # The problems
//!
//! All are minimized. The classic functions are unconstrained, on [`Real`] genomes. `n` is the
//! number of dimensions.
//!
//! | Problem | n (default) | Bounds | Minimum |
//! |---|---|---|---|
//! | [`Sphere`] | any (30) | [−100, 100] | 0 at the origin |
//! | [`AxisParallelEllipsoid`] | any (30) | [−5.12, 5.12] | 0 at the origin |
//! | [`Schwefel1_2`] | any (30) | [−100, 100] | 0 at the origin |
//! | [`Rastrigin`] | any (30) | [−5.12, 5.12] | 0 at the origin |
//! | [`Rosenbrock`] | 2 or more (30) | [−30, 30] | 0 at (1, …, 1) |
//! | [`Ackley`] | any (30) | [−32, 32] | 0 at the origin |
//! | [`Griewank`] | any (30) | [−600, 600] | 0 at the origin |
//! | [`Schwefel2_26`] | any (30) | [−500, 500] | −418.98289 n at xᵢ = 420.96875 |
//! | [`Levy`] | any (30) | [−10, 10] | 0 at (1, …, 1) |
//! | [`Zakharov`] | any (30) | [−5, 10] | 0 at the origin |
//! | [`StyblinskiTang`] | any (30) | [−5, 5] | −39.16617 n at xᵢ = −2.90353 |
//! | [`Michalewicz`] | any (10) | [0, π] | −9.66015 for n = 10, computed for any n |
//! | [`Himmelblau`] | 2 | [−5, 5] | 0 at four points |
//! | [`Branin`] | 2 | [−5, 10] × [0, 15] | 5/(4π) at three points |
//! | [`GoldsteinPrice`] | 2 | [−2, 2] | 3 at (0, −1) |
//! | [`SixHumpCamel`] | 2 | [−5, 5] | −1.03163 at two points |
//! | [`Hartmann3`] | 3 | [0, 1] | −3.86278 at (0.11461, 0.55565, 0.85255), best known |
//! | [`Hartmann6`] | 6 | [0, 1] | −3.32237 at (0.2017, 0.1500, 0.4769, 0.2753, 0.3117, 0.6573), best known |
//! | [`Shekel5`] | 4 | [0, 10] | −10.15320 near (4, 4, 4, 4), best known |
//! | [`Shekel7`] | 4 | [0, 10] | −10.40294 near (4, 4, 4, 4), best known |
//! | [`Shekel10`] | 4 | [0, 10] | −10.53641 near (4, 4, 4, 4), best known |
//! | [`Easom`] | 2 | [−100, 100] | −1 at (π, π) |
//! | [`Eggholder`] | 2 | [−512, 512] | −959.64066 at (512, 404.23181), best known |
//! | [`SchafferF6`] | 2 | [−100, 100] | 0 at the origin |
//! | [`Schwefel2_21`] | any (30) | [−100, 100] | 0 at the origin |
//! | [`Schwefel2_22`] | any (30) | [−10, 10] | 0 at the origin |
//! | [`DixonPrice`] | 2 or more (30) | [−10, 10] | 0 at xᵢ = 2^(−(2ⁱ − 2) / 2ⁱ), and with xₙ negated |
//! | [`Trid`] | 2 or more (10) | [−n², n²] | −n (n + 4) (n − 1) / 6 at xᵢ = i (n + 1 − i) |
//! | [`Powell`] | a multiple of 4 (24) | [−4, 5] | 0 at the origin |
//! | [`Beale`] | 2 | [−4.5, 4.5] | 0 at (3, 0.5) |
//! | [`Booth`] | 2 | [−10, 10] | 0 at (1, 3) |
//! | [`Matyas`] | 2 | [−10, 10] | 0 at the origin |
//! | [`Bohachevsky1`], [`Bohachevsky2`], [`Bohachevsky3`] | 2 | [−100, 100] | 0 at the origin |
//! | [`ThreeHumpCamel`] | 2 | [−5, 5] | 0 at the origin |
//! | [`Langermann`] | 2 | [0, 10] | −4.15581 at (2.79340, 1.59723), best known |
//! | [`ShekelFoxholes`] | 2 | [−65.536, 65.536] | 0.99800 near (−32, −32), best known |
//! | [`Kowalik`] | 4 | [−5, 5] | 3.07486e-4 at (0.19283, 0.19084, 0.12312, 0.13577), best known |
//! | [`SumOfDifferentPowers`] | any (30) | [−1, 1] | 0 at the origin |
//! | [`Step`] | any (30) | [−100, 100] | 0 on [−0.5, 0.5)ⁿ |
//! | [`Quartic`] | any (30) | [−1.28, 1.28] | 0 at the origin, without noise |
//! | [`Penalized1`] | any (30) | [−50, 50] | 0 at (−1, …, −1) |
//! | [`Penalized2`] | any (30) | [−50, 50] | 0 at (1, …, 1) |
//! | [`HighConditionedElliptic`] | 2 or more (30) | [−100, 100] | 0 at the origin |
//! | [`BentCigar`] | 2 or more (30) | [−100, 100] | 0 at the origin |
//! | [`Discus`] | 2 or more (30) | [−100, 100] | 0 at the origin |
//! | [`DifferentPowers`] | 2 or more (30) | [−5, 5] | 0 at the origin |
//! | [`BucheRastrigin`] | 2 or more (30) | [−5, 5] | 0 at the origin |
//! | [`NonContinuousRastrigin`] | any (30) | [−5.12, 5.12] | 0 at the origin |
//! | [`Weierstrass`] | any (30) | [−0.5, 0.5] | 0 at the origin |
//! | [`Katsuura`] | any (30) | [−5, 5] | 0 at the origin, and wherever every gene is a multiple of 1/2 |
//! | [`HappyCat`] | any (30) | [−5, 5] | 0 at (−1, …, −1) |
//! | [`HgBat`] | any (30) | [−5, 5] | 0 at (−1, …, −1) |
//! | [`SchafferF7`] | 2 or more (30) | [−100, 100] | 0 at the origin |
//! | [`RotatedHyperEllipsoid`] | any (30) | [−65.536, 65.536] | 0 at the origin |
//!
//! Two wrappers make instances of any of them, as the CEC and BBOB suites do: [`Shifted`] moves
//! the optimum to a point drawn from a seed, and [`Rotated`] turns the function about its
//! optimum by an orthogonal matrix drawn from a seed, so that the genes interact. The shifted and
//! the shifted rotated Rastrigin of CEC 2005 (F9 and F10) are
//! `Shifted::new(Rastrigin::new(n), seed)` and `Rotated::new(Shifted::new(Rastrigin::new(n), seed), seed)`,
//! with genoxide's own shift and matrix rather than the report's data files.
//!
//! The classic functions but [`Eggholder`], [`Schwefel2_21`] and [`Schwefel2_22`], which aren't
//! differentiable everywhere, supply their analytic gradient to the algorithms that want one (see
//! [`gradient`](crate::gradient)): [`FitnessFunction::provides`] says so, and
//! [`FitnessFunction::evaluate_with`] computes it, with [`math`](crate::math)'s functions. Ackley's
//! has a cone at the origin, where its gradient is taken as 0, and Schwefel 2.26's second
//! derivative is unbounded at 0.
//!
//! Two submodules hold constrained problems, whose fitness is `(score, violation)`:
//!
//! - [`cec2006`]: the CEC 2006 constrained problems g01 to g24
//!   ([`G01`](cec2006::G01) … [`G24`](cec2006::G24)), with 2 to 24 dimensions, inequality and
//!   equality constraints, and the optimum or best known solution of their report;
//! - [`engineering`]: engineering design problems, the welded beam in two forms, the pressure
//!   vessel, the tension/compression spring, the speed reducer, the gear train (on
//!   [`Integer`](crate::genome::Integer) genomes), the three-bar truss, the cantilever beam and
//!   the car side impact.
//!
//! [`Problem::constraints`] gives a constrained problem's constraint values, as `g(x) <= 0` and
//! `h(x) = 0`. The problems with inequalities only (g01, g02, g04, g06 to g10, g12, g16, g18,
//! g19, g24 and the engineering problems on [`Real`] genomes) also give their values to the
//! algorithms that use them, such as [`Mma`](crate::algorithm::Mma): their
//! [`provides`](FitnessFunction::provides) declares them, and
//! [`evaluate_with`](FitnessFunction::evaluate_with) writes them. Their gradients aren't given:
//! [`Constrained::differentiable`](crate::constraint::Constrained::differentiable) adds them.
//!
//! [`control`] holds control tasks instead of functions: the cart-pole and the double pole, with
//! and without velocities, driven by a [`Policy`](control::Policy) such as a neural network of
//! [`nn`](crate::nn), for neuroevolution.
//!
//! Each problem's docs cite its original authors, and say where its definition and bounds come
//! from. Most originals are books or reports that aren't online, and some functions have no known
//! origin: their definitions are taken from later papers that restate them, and are still to be
//! checked against the originals in [#168](https://github.com/tachsin/genoxide/issues/168):
//!
//! - Yao, X., Liu, Y. and Lin, G. (1999). Evolutionary programming made faster. *IEEE
//!   Transactions on Evolutionary Computation* 3(2): 82-102. doi:10.1109/4235.771163 (table I
//!   and the appendix)
//! - Laguna, M. and Martí, R. (2005). Experimental testing of advanced scatter search designs for
//!   global optimization of multimodal functions. *Journal of Global Optimization* 33(2):
//!   235-255. doi:10.1007/s10898-004-1936-z (the appendix)
//! - Molga, M. and Smutnicki, C. (2005). *Test functions for optimization needs.*
//! - Adorio, E. P. (2005). *MVF - Multivariate Test Functions Library in C for Unconstrained
//!   Global Optimization.* University of the Philippines Diliman
//! - Jamil, M. and Yang, X.-S. (2013). A literature survey of benchmark functions for global
//!   optimisation problems. *International Journal of Mathematical Modelling and Numerical
//!   Optimisation* 4(2): 150-194. arXiv:1308.4008
//!
//! The functions of the CEC competitions and of BBOB are taken from their reports, which define
//! them:
//!
//! - Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y.-P., Auger, A. and Tiwari, S.
//!   (2005). *Problem Definitions and Evaluation Criteria for the CEC 2005 Special Session on
//!   Real-Parameter Optimization.* Nanyang Technological University and KanGAL report 2005005
//! - Hansen, N., Finck, S., Ros, R. and Auger, A. (2009). *Real-Parameter Black-Box Optimization
//!   Benchmarking 2009: Noiseless Functions Definitions.* INRIA research report RR-6829
//! - Liang, J. J., Qu, B. Y. and Suganthan, P. N. (2013). *Problem Definitions and Evaluation
//!   Criteria for the CEC 2014 Special Session and Competition on Single Objective
//!   Real-Parameter Numerical Optimization.* Zhengzhou University and Nanyang Technological
//!   University, technical report 201311
//! - Awad, N. H., Ali, M. Z., Suganthan, P. N., Liang, J. J. and Qu, B. Y. (2016). *Problem
//!   Definitions and Evaluation Criteria for the CEC 2017 Special Session and Competition on
//!   Single Objective Real-Parameter Numerical Optimization.* Nanyang Technological University
//!
//! Optima not given to full precision by the sources are derived from the formulas, and the docs
//! say how.
//!
//! The problems compute their trigonometric and exponential functions with [`math`](crate::math),
//! so their values are the same to the bit on every platform, like the rest of genoxide.
//!
//! [`Engine`]: crate::Engine

pub mod cec2006;
mod classic;
pub mod control;
pub mod engineering;
mod gradients;
mod transform;

pub use classic::{
    Ackley, AxisParallelEllipsoid, Branin, GoldsteinPrice, Griewank, Himmelblau, Levy, Michalewicz,
    Rastrigin, Rosenbrock, Schwefel1_2, Schwefel2_26, SixHumpCamel, Sphere, StyblinskiTang,
    Zakharov,
};
pub use classic::{
    Beale, Bohachevsky1, Bohachevsky2, Bohachevsky3, Booth, DixonPrice, Kowalik, Langermann,
    Matyas, Powell, Schwefel2_21, Schwefel2_22, ShekelFoxholes, ThreeHumpCamel, Trid,
};
pub use classic::{
    BentCigar, BucheRastrigin, DifferentPowers, Discus, HappyCat, HgBat, HighConditionedElliptic,
    Katsuura, NonContinuousRastrigin, Penalized1, Penalized2, Quartic, RotatedHyperEllipsoid,
    SchafferF7, Step, SumOfDifferentPowers, Weierstrass,
};
pub use classic::{Easom, Eggholder, Hartmann3, Hartmann6, SchafferF6, Shekel5, Shekel7, Shekel10};
pub use transform::{Rotated, Shifted};

use crate::constraint::{at_most, equal};
use crate::engine::{Extras, FitnessFunction, IntoFitness, Provided};
use crate::genome::{Real, Reals, Representation};
use crate::{Fitness, Objective};

/// A single-objective test problem: a fitness function with its search space, its known optimum
/// and the paper that defines it.
///
/// A problem is minimized unless [`objective`](Problem::objective) says otherwise. Its
/// [`FitnessFunction::Output`] is `f64` for an unconstrained problem.
pub trait Problem:
    FitnessFunction<<<Self as Problem>::Representation as Representation>::Genome>
{
    /// The search space, e.g. [`Real`] bounds.
    type Representation: Representation;

    /// The name, e.g. `"Rastrigin"`.
    fn name(&self) -> &'static str;

    /// The search space: the number of variables and their bounds.
    fn representation(&self) -> Self::Representation;

    /// Whether the score is minimized (the default) or maximized.
    fn objective(&self) -> Objective {
        Objective::Minimize
    }

    /// The global optimum, if it's known for this size: its value and the solutions that reach
    /// it.
    fn optimum(&self) -> Option<Optimum<<Self::Representation as Representation>::Genome>>;

    /// The paper, book or report that defines the problem.
    fn reference(&self) -> &'static str;

    /// Its DOI or URL, if any.
    fn reference_url(&self) -> Option<&'static str> {
        None
    }

    /// The values of the constraints at `genome`, in the paper's order; none for an
    /// unconstrained problem.
    fn constraints(
        &self,
        genome: &<Self::Representation as Representation>::Genome,
    ) -> Constraints {
        let _ = genome;
        Constraints::none()
    }
}

/// The optimum of a problem: its value and the solutions that reach it.
///
/// ```
/// use genoxide::problems::{Branin, Problem};
///
/// let optimum = Branin.optimum().expect("known");
/// assert_eq!(optimum.solutions().len(), 3);
/// assert!(optimum.is_proven());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Optimum<G> {
    value: f64,
    solutions: Vec<G>,
    proven: bool,
}

impl<G> Optimum<G> {
    /// The global optimum: `value`, reached by each of `solutions`.
    pub fn proven(value: f64, solutions: Vec<G>) -> Self {
        Self {
            value,
            solutions,
            proven: true,
        }
    }

    /// The best value known, not proven to be the global optimum, reached by each of
    /// `solutions`.
    pub fn best_known(value: f64, solutions: Vec<G>) -> Self {
        Self {
            value,
            solutions,
            proven: false,
        }
    }

    /// The optimal score.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Solutions with this score: all of them when there are few (Himmelblau's four, Branin's
    /// three), one otherwise, and none when only the value is known.
    pub fn solutions(&self) -> &[G] {
        &self.solutions
    }

    /// Whether the value is the global optimum (analytic or proven), rather than the best known.
    pub fn is_proven(&self) -> bool {
        self.proven
    }
}

/// The constraint values of a solution, in the paper's order and in a common form: inequalities
/// `g(x) <= 0` and equalities `h(x) = 0`.
///
/// ```
/// use genoxide::problems::Constraints;
///
/// // g₁ = 0.5 > 0 is violated, g₂ = −1 is met, and |h₁| = 0.01 is within a tolerance of 0.1
/// let constraints = Constraints::new(vec![0.5, -1.0], vec![0.01]);
/// assert_eq!(constraints.violation(0.1), 0.5);
/// assert!(Constraints::none().is_empty());
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Constraints {
    inequalities: Vec<f64>,
    equalities: Vec<f64>,
}

impl Constraints {
    /// No constraints.
    pub fn none() -> Self {
        Self::default()
    }

    /// The values `g(x)` of inequalities `g(x) <= 0` and `h(x)` of equalities `h(x) = 0`.
    pub fn new(inequalities: Vec<f64>, equalities: Vec<f64>) -> Self {
        Self {
            inequalities,
            equalities,
        }
    }

    /// The values of the inequalities, feasible at 0 or below.
    pub fn inequalities(&self) -> &[f64] {
        &self.inequalities
    }

    /// The values of the equalities, feasible at 0.
    pub fn equalities(&self) -> &[f64] {
        &self.equalities
    }

    /// The number of constraints.
    pub fn len(&self) -> usize {
        self.inequalities.len() + self.equalities.len()
    }

    /// Whether there are no constraints.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The total violation, `Σ max(0, g) + Σ max(0, |h| − tolerance)`: 0 for a feasible
    /// solution, and the violation a constrained problem's fitness has.
    pub fn violation(&self, tolerance: f64) -> f64 {
        let inequalities: f64 = self.inequalities.iter().map(|&g| at_most(g, 0.0)).sum();
        let equalities: f64 = self
            .equalities
            .iter()
            .map(|&h| equal(h, 0.0, tolerance))
            .sum();
        inequalities + equalities
    }
}

/// A problem on [`Real`] genomes, as a trait object: for lists of problems of different types,
/// such as [`all`].
///
/// ```
/// use genoxide::problems;
///
/// for problem in problems::all() {
///     let Some(optimum) = problem.optimum() else { continue };
///     let fitness = problem.evaluate(&optimum.solutions()[0]);
///     let error = (fitness.score().unwrap() - optimum.value()).abs();
///     assert!(error <= 1e-3 * optimum.value().abs().max(1.0), "{}", problem.name());
/// }
/// ```
pub trait DynProblem: Send + Sync {
    /// The name, as [`Problem::name`].
    fn name(&self) -> &'static str;

    /// The search space, as [`Problem::representation`].
    fn real(&self) -> Real;

    /// Whether the score is minimized or maximized, as [`Problem::objective`].
    fn objective(&self) -> Objective;

    /// The fitness of `genome`: its score, and its constraint violation for a constrained
    /// problem.
    fn evaluate(&self, genome: &Reals) -> Fitness;

    /// The global optimum, as [`Problem::optimum`].
    fn optimum(&self) -> Option<Optimum<Reals>>;

    /// The paper that defines the problem, as [`Problem::reference`].
    fn reference(&self) -> &'static str;

    /// Its DOI or URL, as [`Problem::reference_url`].
    fn reference_url(&self) -> Option<&'static str>;

    /// The constraint values at `genome`, as [`Problem::constraints`].
    fn constraints(&self, genome: &Reals) -> Constraints;

    /// What the problem gives besides the fitness, as [`FitnessFunction::provides`]: the
    /// gradient, for the smooth classic functions, and the constraints' values, for the
    /// problems with inequalities only. Nothing by default.
    fn provides(&self) -> Provided {
        Provided::NOTHING
    }

    /// The fitness of `genome`, as [`evaluate`](DynProblem::evaluate), with the extras of
    /// `extras`, as [`FitnessFunction::evaluate_with`]. By default, `evaluate(genome)`.
    ///
    /// ```
    /// use genoxide::engine::Extras;
    /// use genoxide::genome::Reals;
    /// use genoxide::problems;
    ///
    /// for problem in problems::all() {
    ///     if problem.provides().gradient {
    ///         let mut gradient = vec![0.0; problem.real().bounds().len()];
    ///         let x = Reals::from(vec![0.5; gradient.len()]);
    ///         let fitness = problem.evaluate_with(&x, &mut Extras::with_gradient(&mut gradient));
    ///         assert_eq!(fitness, problem.evaluate(&x));
    ///     }
    /// }
    /// ```
    fn evaluate_with(&self, genome: &Reals, extras: &mut Extras<'_>) -> Fitness {
        let _ = extras;
        self.evaluate(genome)
    }
}

// a problem behind `DynProblem`; a wrapper, so that problems don't have the methods of both traits
struct Boxed<P>(P);

impl<P> DynProblem for Boxed<P>
where
    P: Problem<Representation = Real> + Send + Sync,
{
    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn real(&self) -> Real {
        self.0.representation()
    }

    fn objective(&self) -> Objective {
        self.0.objective()
    }

    fn evaluate(&self, genome: &Reals) -> Fitness {
        self.0
            .evaluate(genome)
            .into_fitness()
            .unwrap_or_else(|_| Fitness::invalid())
    }

    fn optimum(&self) -> Option<Optimum<Reals>> {
        self.0.optimum()
    }

    fn reference(&self) -> &'static str {
        self.0.reference()
    }

    fn reference_url(&self) -> Option<&'static str> {
        self.0.reference_url()
    }

    fn constraints(&self, genome: &Reals) -> Constraints {
        self.0.constraints(genome)
    }

    fn provides(&self) -> Provided {
        self.0.provides()
    }

    fn evaluate_with(&self, genome: &Reals, extras: &mut Extras<'_>) -> Fitness {
        self.0
            .evaluate_with(genome, extras)
            .into_fitness()
            .unwrap_or_else(|_| Fitness::invalid())
    }
}

/// `problem` as a [`DynProblem`].
pub fn boxed<P>(problem: P) -> Box<dyn DynProblem>
where
    P: Problem<Representation = Real> + Send + Sync + 'static,
{
    Box::new(Boxed(problem))
}

/// Every problem of this module and its submodules on [`Real`] genomes, at its default size: the
/// classic functions in the order of the table above, then [`cec2006`]'s and [`engineering`]'s.
/// The gear train, on integer genomes, isn't among them.
pub fn all() -> Vec<Box<dyn DynProblem>> {
    vec![
        boxed(Sphere::default()),
        boxed(AxisParallelEllipsoid::default()),
        boxed(Schwefel1_2::default()),
        boxed(Rastrigin::default()),
        boxed(Rosenbrock::default()),
        boxed(Ackley::default()),
        boxed(Griewank::default()),
        boxed(Schwefel2_26::default()),
        boxed(Levy::default()),
        boxed(Zakharov::default()),
        boxed(StyblinskiTang::default()),
        boxed(Michalewicz::default()),
        boxed(Himmelblau),
        boxed(Branin),
        boxed(GoldsteinPrice),
        boxed(SixHumpCamel),
        boxed(Hartmann3),
        boxed(Hartmann6),
        boxed(Shekel5),
        boxed(Shekel7),
        boxed(Shekel10),
        boxed(Easom),
        boxed(Eggholder),
        boxed(SchafferF6),
        boxed(Schwefel2_21::default()),
        boxed(Schwefel2_22::default()),
        boxed(DixonPrice::default()),
        boxed(Trid::default()),
        boxed(Powell::default()),
        boxed(Beale),
        boxed(Booth),
        boxed(Matyas),
        boxed(Bohachevsky1),
        boxed(Bohachevsky2),
        boxed(Bohachevsky3),
        boxed(ThreeHumpCamel),
        boxed(Langermann),
        boxed(ShekelFoxholes),
        boxed(Kowalik),
        boxed(SumOfDifferentPowers::default()),
        boxed(Step::default()),
        boxed(Quartic::default()),
        boxed(Penalized1::default()),
        boxed(Penalized2::default()),
        boxed(HighConditionedElliptic::default()),
        boxed(BentCigar::default()),
        boxed(Discus::default()),
        boxed(DifferentPowers::default()),
        boxed(BucheRastrigin::default()),
        boxed(NonContinuousRastrigin::default()),
        boxed(Weierstrass::default()),
        boxed(Katsuura::default()),
        boxed(HappyCat::default()),
        boxed(HgBat::default()),
        boxed(SchafferF7::default()),
        boxed(RotatedHyperEllipsoid::default()),
        boxed(cec2006::G01),
        boxed(cec2006::G02),
        boxed(cec2006::G03::default()),
        boxed(cec2006::G04),
        boxed(cec2006::G05::default()),
        boxed(cec2006::G06),
        boxed(cec2006::G07),
        boxed(cec2006::G08),
        boxed(cec2006::G09),
        boxed(cec2006::G10),
        boxed(cec2006::G11::default()),
        boxed(cec2006::G12),
        boxed(cec2006::G13::default()),
        boxed(cec2006::G14::default()),
        boxed(cec2006::G15::default()),
        boxed(cec2006::G16),
        boxed(cec2006::G17::default()),
        boxed(cec2006::G18),
        boxed(cec2006::G19),
        boxed(cec2006::G20::default()),
        boxed(cec2006::G21::default()),
        boxed(cec2006::G22::default()),
        boxed(cec2006::G23::default()),
        boxed(cec2006::G24),
        boxed(engineering::WeldedBeam),
        boxed(engineering::WeldedBeamRagsdell),
        boxed(engineering::PressureVessel),
        boxed(engineering::TensionCompressionSpring),
        boxed(engineering::SpeedReducer),
        boxed(engineering::ThreeBarTruss),
        boxed(engineering::CantileverBeam),
        boxed(engineering::CarSideImpact),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamRng;
    use std::collections::HashSet;

    #[test]
    fn the_registry_describes_every_problem() {
        let problems = all();
        assert_eq!(problems.len(), 88);
        let names: HashSet<_> = problems.iter().map(|problem| problem.name()).collect();
        assert_eq!(names.len(), problems.len(), "names are unique");
        let mut rng = StreamRng::seed_from_u64(0);
        for problem in &problems {
            assert!(!problem.reference().is_empty(), "{}", problem.name());
            if let Some(url) = problem.reference_url() {
                assert!(url.starts_with("https://"), "{url}");
            }
            assert_eq!(problem.objective(), Objective::Minimize);
            let real = problem.real();
            let constrained = !problem
                .constraints(&real.random_genome(&mut rng))
                .is_empty();
            let Some(optimum) = problem.optimum() else {
                assert_eq!(problem.name(), "CarSideImpact");
                continue;
            };
            assert!(!optimum.solutions().is_empty(), "{}", problem.name());
            // the unconstrained optima are proven, but for those found numerically
            let numerical = [
                "Hartmann3",
                "Hartmann6",
                "Shekel5",
                "Shekel7",
                "Shekel10",
                "Eggholder",
                "Langermann",
                "ShekelFoxholes",
                "Kowalik",
            ];
            let numerically = numerical.contains(&problem.name());
            assert!(
                constrained || optimum.is_proven() || numerically,
                "{}",
                problem.name()
            );
            for solution in optimum.solutions() {
                assert!(real.validate(solution).is_ok(), "{}", problem.name());
                let fitness = problem.evaluate(solution);
                let score = fitness.score().expect("valid");
                // best known values and their solutions are printed to a few digits
                let tolerance = if optimum.is_proven() { 1e-12 } else { 2e-4 };
                let error = (score - optimum.value()).abs();
                let scale = optimum.value().abs().max(1.0);
                assert!(error <= tolerance * scale, "{}", problem.name());
                // a proven optimum is feasible: under Deb's rules, any feasible point beats an
                // infeasible one, however far worse; the report's x* of g07, g09 and g24 are
                // rounded, and exceed a constraint by 6e-14, 4e-16 and 2e-13
                if optimum.is_proven() {
                    let rounded = ["G07", "G09", "G24"].contains(&problem.name());
                    let slack = if rounded { 1e-12 } else { 0.0 };
                    assert!(
                        fitness.violation() <= slack,
                        "{}: {fitness:?}",
                        problem.name()
                    );
                }
            }
        }
    }

    // the value is finite and deterministic everywhere in the bounds, and no feasible solution is
    // better than the optimum
    #[test]
    fn random_solutions_are_finite_and_no_better_than_the_optimum() {
        let mut rng = StreamRng::seed_from_u64(1);
        for problem in all() {
            let real = problem.real();
            let optimum = problem.optimum().map(|optimum| optimum.value());
            for _ in 0..2_000 {
                let genome = real.random_genome(&mut rng);
                let fitness = problem.evaluate(&genome);
                let score = fitness.score().expect("valid");
                assert!(score.is_finite(), "{}: {genome:?}", problem.name());
                assert_eq!(problem.evaluate(&genome), fitness);
                if let Some(optimum) = optimum.filter(|_| fitness.is_feasible()) {
                    let slack = 1e-9 * optimum.abs().max(1.0);
                    assert!(score >= optimum - slack, "{}: {genome:?}", problem.name());
                }
            }
            // the corners of the box; the three-bar truss's stresses and g08's value are undefined
            // at x₁ = 0, and g20's equalities at the origin
            let low: Reals = real.bounds().iter().map(|range| *range.start()).collect();
            let high: Reals = real.bounds().iter().map(|range| *range.end()).collect();
            for corner in [low, high] {
                if ["ThreeBarTruss", "G08", "G20"].contains(&problem.name()) && corner[0] == 0.0 {
                    continue;
                }
                let fitness = problem.evaluate(&corner);
                let score = fitness.score().expect("valid");
                assert!(score.is_finite(), "{}", problem.name());
                if let Some(optimum) = optimum.filter(|_| fitness.is_feasible()) {
                    assert!(score >= optimum, "{}", problem.name());
                }
            }
        }
    }

    // the problems with inequalities only give their values, those of `constraints`, with the
    // same fitness; the others give none
    #[test]
    fn inequality_values_are_the_constraints() {
        let mut rng = StreamRng::seed_from_u64(2);
        let mut giving = Vec::new();
        for problem in all() {
            let real = problem.real();
            let count = problem.provides().inequalities;
            for _ in 0..50 {
                let x = real.random_genome(&mut rng);
                let constraints = problem.constraints(&x);
                let only_inequalities =
                    constraints.equalities().is_empty() && !constraints.inequalities().is_empty();
                assert_eq!(
                    count > 0,
                    only_inequalities,
                    "{}: gives {count} values",
                    problem.name()
                );
                if count == 0 {
                    break;
                }
                assert_eq!(
                    constraints.inequalities().len(),
                    count,
                    "{}",
                    problem.name()
                );
                let mut g = vec![f64::NAN; count];
                let fitness = problem.evaluate_with(&x, &mut Extras::new(None, Some(&mut g), None));
                assert_eq!(fitness, problem.evaluate(&x), "{}", problem.name());
                assert_eq!(g, constraints.inequalities(), "{}", problem.name());
            }
            if count > 0 {
                giving.push(problem.name());
            }
        }
        assert_eq!(giving.len(), 13 + 8, "{giving:?}");
    }

    #[test]
    fn constraints_measure_their_violation() {
        let constraints = Constraints::new(vec![0.5, -1.0, 0.0], vec![0.25, -0.01]);
        assert_eq!(constraints.len(), 5);
        assert_eq!(constraints.inequalities(), [0.5, -1.0, 0.0]);
        assert_eq!(constraints.equalities(), [0.25, -0.01]);
        // 0.5 from g₁, and 0.25 − 0.05 from h₁; |h₂| is within the tolerance
        assert!((constraints.violation(0.05) - 0.7).abs() < 1e-15);
        assert_eq!(Constraints::none().violation(0.0), 0.0);
        assert_eq!(Constraints::none(), Constraints::new(vec![], vec![]));
    }

    #[test]
    fn optima_are_proven_or_best_known() {
        let proven = Optimum::proven(1.0, vec![Reals::from(vec![0.0])]);
        assert!(proven.is_proven());
        assert_eq!(proven.value(), 1.0);
        let known = Optimum::<Reals>::best_known(2.0, Vec::new());
        assert!(!known.is_proven());
        assert!(known.solutions().is_empty());
    }
}
