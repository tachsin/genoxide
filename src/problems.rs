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
//! [`all`] lists every problem at its default size, behind the object-safe [`DynProblem`].
//!
//! # The problems
//!
//! All are minimized, on [`Real`] genomes. `n` is the number of dimensions.
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
//! - Jamil, M. and Yang, X.-S. (2013). A literature survey of benchmark functions for global
//!   optimisation problems. *International Journal of Mathematical Modelling and Numerical
//!   Optimisation* 4(2): 150-194. arXiv:1308.4008
//!
//! Optima not given to full precision by the sources are derived from the formulas, and the docs
//! say how.
//!
//! The problems use the platform's trigonometric and exponential functions, so their values can
//! differ in the last bit between platforms, unlike the rest of genoxide.
//!
//! [`Engine`]: crate::Engine

mod classic;

pub use classic::{
    Ackley, AxisParallelEllipsoid, Branin, GoldsteinPrice, Griewank, Himmelblau, Levy, Michalewicz,
    Rastrigin, Rosenbrock, Schwefel1_2, Schwefel2_26, SixHumpCamel, Sphere, StyblinskiTang,
    Zakharov,
};

use crate::constraint::{at_most, equal};
use crate::engine::{FitnessFunction, IntoFitness};
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
///     let optimum = problem.optimum().expect("known");
///     let fitness = problem.evaluate(&optimum.solutions()[0]);
///     assert!((fitness.score().unwrap() - optimum.value()).abs() < 1e-9, "{}", problem.name());
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
}

/// `problem` as a [`DynProblem`].
pub fn boxed<P>(problem: P) -> Box<dyn DynProblem>
where
    P: Problem<Representation = Real> + Send + Sync + 'static,
{
    Box::new(Boxed(problem))
}

/// Every problem of this module, at its default size, in the order of the table above.
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
        assert_eq!(problems.len(), 16);
        let names: HashSet<_> = problems.iter().map(|problem| problem.name()).collect();
        assert_eq!(names.len(), problems.len(), "names are unique");
        for problem in &problems {
            assert!(!problem.reference().is_empty(), "{}", problem.name());
            if let Some(url) = problem.reference_url() {
                assert!(url.starts_with("https://"), "{url}");
            }
            assert_eq!(problem.objective(), Objective::Minimize);
            let optimum = problem.optimum().expect("every optimum is known");
            assert!(optimum.is_proven(), "{}", problem.name());
            assert!(!optimum.solutions().is_empty(), "{}", problem.name());
            let real = problem.real();
            for solution in optimum.solutions() {
                assert!(real.validate(solution).is_ok(), "{}", problem.name());
                assert!(problem.constraints(solution).is_empty());
            }
        }
    }

    // the value is finite and deterministic everywhere in the bounds, and never below the optimum
    #[test]
    fn random_solutions_are_finite_and_no_better_than_the_optimum() {
        let mut rng = StreamRng::seed_from_u64(1);
        for problem in all() {
            let real = problem.real();
            let optimum = problem.optimum().expect("known").value();
            for _ in 0..2_000 {
                let genome = real.random_genome(&mut rng);
                let score = problem.evaluate(&genome).score().expect("valid");
                assert!(score.is_finite(), "{}: {genome:?}", problem.name());
                assert_eq!(problem.evaluate(&genome).score(), Some(score));
                let slack = 1e-9 * optimum.abs().max(1.0);
                assert!(score >= optimum - slack, "{}: {genome:?}", problem.name());
            }
            // the corners of the box
            let low: Reals = real.bounds().iter().map(|range| *range.start()).collect();
            let high: Reals = real.bounds().iter().map(|range| *range.end()).collect();
            for corner in [low, high] {
                let score = problem.evaluate(&corner).score().expect("valid");
                assert!(score.is_finite() && score >= optimum, "{}", problem.name());
            }
        }
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
