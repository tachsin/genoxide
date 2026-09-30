//! Test problems of symbolic regression, each with its paper's target, sampling of training and
//! test points, and function set.
//!
//! Each problem is a fitness function, [`Regression`] on its dataset with its primitives (the RMSE
//! after linear scaling), and gives its [`dataset`](RegressionProblem::dataset),
//! [`primitives`](RegressionProblem::primitives) and [`target`](RegressionProblem::target) through
//! [`RegressionProblem`]; [`all`] lists them. Random points come from a fixed seed of genoxide's
//! portable stream, so the data are the same on every platform; another library's random points
//! differ.
//!
//! The optimum is exact recovery: an error at the level of rounding (at most 1e-10 of the
//! targets' [`deviation`](super::Sample::deviation)) on the training and the test points.
//!
//! McDermott et al. (2012) and White et al. (2013) found that genetic programming papers tested
//! mostly on toy problems, Koza's quartic above all, with inconsistent function sets, sampling and
//! budgets; each problem here says which of them it follows.
//!
//! - McDermott, J., White, D. R., Luke, S., Manzoni, L., Castelli, M., Vanneschi, L., Jaśkowski,
//!   W., Krawiec, K., Harper, R., De Jong, K. and O'Reilly, U.-M. (2012). Genetic programming
//!   needs better benchmarks. GECCO 2012: 791-798. doi:10.1145/2330163.2330273
//! - White, D. R., McDermott, J., Castelli, M., Manzoni, L., Goldman, B. W., Kronberger, G.,
//!   Jaśkowski, W., O'Reilly, U.-M. and Luke, S. (2013). Better GP benchmarks: community survey
//!   results and proposals. *Genetic Programming and Evolvable Machines* 14(1): 3-29.
//!   doi:10.1007/s10710-012-9177-2

use super::{Dataset, Math, Regression, Sample, primitives};
use crate::engine::FitnessFunction;
use crate::gp::{PrimitiveSet, Tree};
use crate::math;
use crate::rng::StreamRng;
use rand::RngExt;

/// A test problem of symbolic regression.
pub trait RegressionProblem: FitnessFunction<Tree, Output = Option<f64>> {
    /// The problem's name, e.g. `"Koza-1"`.
    fn name(&self) -> &'static str;

    /// The fitness function: [`Regression`] of the dataset by trees of the primitives, the
    /// RMSE after linear scaling. Change it with [`Regression::metric`] and
    /// [`Regression::linear_scaling`].
    fn regression(&self) -> &Regression;

    /// The primitive set of the paper: build a [`Gp`](crate::gp::Gp) from it.
    fn primitives(&self) -> &PrimitiveSet<Math> {
        self.regression().primitives()
    }

    /// The training and test samples.
    fn dataset(&self) -> &Dataset {
        self.regression().dataset()
    }

    /// The target function at a point, a value per variable.
    fn target(&self, point: &[f64]) -> f64;

    /// The target as a formula, e.g. `"x^4 + x^3 + x^2 + x"`.
    fn formula(&self) -> &'static str;

    /// The paper that defines the problem, as a citation.
    fn reference(&self) -> &'static str;

    /// Its DOI or URL, if any.
    fn reference_url(&self) -> Option<&'static str> {
        None
    }
}

/// Every problem, in the order of this module.
pub fn all() -> Vec<Box<dyn RegressionProblem>> {
    vec![
        Box::new(Koza1::new()),
        Box::new(Koza2::new()),
        Box::new(Koza3::new()),
        Box::new(Nguyen1::new()),
        Box::new(Nguyen2::new()),
        Box::new(Nguyen3::new()),
        Box::new(Nguyen4::new()),
        Box::new(Nguyen5::new()),
        Box::new(Nguyen6::new()),
        Box::new(Nguyen7::new()),
        Box::new(Nguyen8::new()),
        Box::new(Nguyen9::new()),
        Box::new(Nguyen10::new()),
        Box::new(Nguyen11::new()),
        Box::new(Nguyen12::new()),
    ]
}

// `count` points uniform in [low, high] from `seed`, `variables` values each
fn uniform(seed: u64, count: usize, variables: usize, low: f64, high: f64) -> Vec<Vec<f64>> {
    let mut rng = StreamRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            (0..variables)
                .map(|_| rng.random_range(low..=high))
                .collect()
        })
        .collect()
}

// `count` points evenly spaced from `low` to `high`, both included, one variable
fn grid(count: usize, low: f64, high: f64) -> Vec<Vec<f64>> {
    let last = (count - 1) as f64;
    (0..count)
        .map(|i| vec![low + (high - low) * (i as f64 / last)])
        .collect()
}

// the dataset of `target` at the training and test points
fn dataset(training: &[Vec<f64>], test: &[Vec<f64>], target: fn(&[f64]) -> f64) -> Dataset {
    let sample = |points: &[Vec<f64>]| {
        Sample::from_points(points, target).expect("finite targets at the problem's points")
    };
    Dataset::new(sample(training))
        .with_test(sample(test))
        .expect("the same variables")
}

macro_rules! regression_problem {
    ($name:ident, $label:literal, $formula:literal, $reference:expr, $url:expr) => {
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl FitnessFunction<Tree> for $name {
            type Output = Option<f64>;

            /// The RMSE after linear scaling on the training points, as
            /// [`Regression`] gives it.
            fn evaluate(&self, tree: &Tree) -> Option<f64> {
                self.regression.evaluate(tree)
            }
        }

        impl RegressionProblem for $name {
            fn name(&self) -> &'static str {
                $label
            }

            fn regression(&self) -> &Regression {
                &self.regression
            }

            fn target(&self, point: &[f64]) -> f64 {
                Self::target_of(point)
            }

            fn formula(&self) -> &'static str {
                $formula
            }

            fn reference(&self) -> &'static str {
                $reference
            }

            fn reference_url(&self) -> Option<&'static str> {
                $url
            }
        }
    };
}

// Koza's function set, with the protected division and logarithm
fn koza_primitives() -> PrimitiveSet<Math> {
    primitives(
        [
            Math::Add,
            Math::Sub,
            Math::Mul,
            Math::ProtectedDiv,
            Math::Sin,
            Math::Cos,
            Math::Exp,
            Math::ProtectedLog,
        ],
        ["x"],
        None,
    )
    .expect("Koza's primitive set")
}

// Koza's sampling: 20 training points uniform in [−1, 1] (from seed 20), and 101 evenly spaced
// test points
fn koza_regression(target: fn(&[f64]) -> f64) -> Regression {
    let dataset = dataset(
        &uniform(20, 20, 1, -1.0, 1.0),
        &grid(101, -1.0, 1.0),
        target,
    );
    Regression::new(koza_primitives(), dataset).expect("the dataset's variable")
}

const KOZA_1992: &str = "Koza, J. R. (1992). Genetic Programming: On the Programming of Computers \
                         by Means of Natural Selection. MIT Press";
const KOZA_1994: &str = "Koza, J. R. (1994). Genetic Programming II: Automatic Discovery of \
                         Reusable Programs. MIT Press";

/// Koza-1, Koza's quartic: x⁴ + x³ + x² + x (Koza 1992), the classic first problem of genetic
/// programming, and the toy problem that McDermott et al. (2012) found the most overused.
///
/// Koza's function set: `add`, `sub`, `mul`, the protected division `pdiv`, `sin`, `cos`, `exp`
/// and the protected logarithm `plog`, and the variable `x`. 20 training points uniform in
/// [−1, 1], as in the book, drawn from seed 20; 101 evenly spaced test points across [−1, 1],
/// genoxide's choice. The name, target, sampling and set as McDermott et al. (2012)
/// restate them; not yet checked against the book
/// ([#168](https://github.com/tachsin/genoxide/issues/168)).
///
/// ```
/// use genoxide::engine::FitnessFunction;
/// use genoxide::gp::regression::problems::{Koza1, RegressionProblem};
///
/// let problem = Koza1::new();
/// let tree = problem.primitives().parse("add(mul(x, add(x, mul(x, add(x, mul(x, x))))), x)")?;
/// // x(x(x(x + 1) + 1)) + x: the quartic, with an error at the level of rounding
/// assert!(problem.evaluate(&tree).unwrap() < 1e-15);
/// assert_eq!(problem.dataset().training().points(), 20);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Koza1 {
    regression: Regression,
}

impl Koza1 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: koza_regression(Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        let x = point[0];
        x * x * x * x + x * x * x + x * x + x
    }
}

regression_problem!(Koza1, "Koza-1", "x^4 + x^3 + x^2 + x", KOZA_1992, None);

/// Koza-2: x⁵ − 2x³ + x (Koza 1994), with Koza-1's function set and sampling.
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet
/// checked against the book ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Koza2 {
    regression: Regression,
}

impl Koza2 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: koza_regression(Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        let x = point[0];
        x * x * x * x * x - 2.0 * x * x * x + x
    }
}

regression_problem!(Koza2, "Koza-2", "x^5 - 2x^3 + x", KOZA_1994, None);

/// Koza-3: x⁶ − 2x⁴ + x² (Koza 1994), with Koza-1's function set and sampling.
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet
/// checked against the book ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Koza3 {
    regression: Regression,
}

impl Koza3 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: koza_regression(Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        let x = point[0];
        x * x * x * x * x * x - 2.0 * x * x * x * x + x * x
    }
}

regression_problem!(Koza3, "Koza-3", "x^6 - 2x^4 + x^2", KOZA_1994, None);

// Nguyen's function set: Koza's, with the protected division and logarithm, and x (and y)
fn nguyen_primitives(variables: usize) -> PrimitiveSet<Math> {
    let names: &[&str] = if variables == 1 { &["x"] } else { &["x", "y"] };
    primitives(
        [
            Math::Add,
            Math::Sub,
            Math::Mul,
            Math::ProtectedDiv,
            Math::Sin,
            Math::Cos,
            Math::Exp,
            Math::ProtectedLog,
        ],
        names.iter().copied(),
        None,
    )
    .expect("Nguyen's primitive set")
}

// Nguyen's sampling: 20 training points uniform in [low, high] (100 in [low, high]² for two
// variables), from seed 1000 + 2 × the problem's number; five times as many test points, uniform
// in the same range, from the next seed
fn nguyen_regression(
    number: u64,
    variables: usize,
    low: f64,
    high: f64,
    target: fn(&[f64]) -> f64,
) -> Regression {
    let count = if variables == 1 { 20 } else { 100 };
    let seed = 1000 + 2 * number;
    let dataset = dataset(
        &uniform(seed, count, variables, low, high),
        &uniform(seed + 1, 5 * count, variables, low, high),
        target,
    );
    Regression::new(nguyen_primitives(variables), dataset).expect("the dataset's variables")
}

const NGUYEN: &str = "Uy, N. Q., Hoai, N. X., O'Neill, M., McKay, R. I. and Galván-López, E. \
                      (2011). Semantically-based crossover in genetic programming: application \
                      to real-valued symbolic regression. Genetic Programming and Evolvable \
                      Machines 12(2): 91-119";
const NGUYEN_URL: Option<&str> = Some("https://doi.org/10.1007/s10710-010-9121-2");

/// Nguyen-1: x³ + x² + x, from 20 points uniform in [−1, 1] (Uy et al. 2011).
///
/// Nguyen's function set is Koza's: `add`, `sub`, `mul`, the protected division `pdiv`, `sin`,
/// `cos`, `exp` and the protected logarithm `plog`, and the variable `x` (and `y` for the
/// problems of two variables), without constants. The training points are drawn from a fixed
/// seed; the test points, five times as many from another seed in the same range, are genoxide's
/// choice.
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet checked
/// against the paper ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Nguyen1 {
    regression: Regression,
}

impl Nguyen1 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: nguyen_regression(1, 1, -1.0, 1.0, Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        let x = point[0];
        x * x * x + x * x + x
    }
}

regression_problem!(Nguyen1, "Nguyen-1", "x^3 + x^2 + x", NGUYEN, NGUYEN_URL);

/// Nguyen-2: x⁴ + x³ + x² + x, from 20 points uniform in [−1, 1] (Uy et al. 2011).
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet checked
/// against the paper ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Nguyen2 {
    regression: Regression,
}

impl Nguyen2 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: nguyen_regression(2, 1, -1.0, 1.0, Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        let x = point[0];
        x * x * x * x + x * x * x + x * x + x
    }
}

regression_problem!(
    Nguyen2,
    "Nguyen-2",
    "x^4 + x^3 + x^2 + x",
    NGUYEN,
    NGUYEN_URL
);

/// Nguyen-3: x⁵ + x⁴ + x³ + x² + x, from 20 points uniform in [−1, 1] (Uy et al. 2011).
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet checked
/// against the paper ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Nguyen3 {
    regression: Regression,
}

impl Nguyen3 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: nguyen_regression(3, 1, -1.0, 1.0, Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        let x = point[0];
        x * x * x * x * x + x * x * x * x + x * x * x + x * x + x
    }
}

regression_problem!(
    Nguyen3,
    "Nguyen-3",
    "x^5 + x^4 + x^3 + x^2 + x",
    NGUYEN,
    NGUYEN_URL
);

/// Nguyen-4: x⁶ + x⁵ + x⁴ + x³ + x² + x, from 20 points uniform in [−1, 1] (Uy et al. 2011).
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet checked
/// against the paper ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Nguyen4 {
    regression: Regression,
}

impl Nguyen4 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: nguyen_regression(4, 1, -1.0, 1.0, Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        let x = point[0];
        x * x * x * x * x * x + x * x * x * x * x + x * x * x * x + x * x * x + x * x + x
    }
}

regression_problem!(
    Nguyen4,
    "Nguyen-4",
    "x^6 + x^5 + x^4 + x^3 + x^2 + x",
    NGUYEN,
    NGUYEN_URL
);

/// Nguyen-5: sin(x²) cos(x) − 1, from 20 points uniform in [−1, 1] (Uy et al. 2011).
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet checked
/// against the paper ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Nguyen5 {
    regression: Regression,
}

impl Nguyen5 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: nguyen_regression(5, 1, -1.0, 1.0, Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        let x = point[0];
        math::sin(x * x) * math::cos(x) - 1.0
    }
}

regression_problem!(
    Nguyen5,
    "Nguyen-5",
    "sin(x^2) cos(x) - 1",
    NGUYEN,
    NGUYEN_URL
);

/// Nguyen-6: sin(x) + sin(x + x²), from 20 points uniform in [−1, 1] (Uy et al. 2011).
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet checked
/// against the paper ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Nguyen6 {
    regression: Regression,
}

impl Nguyen6 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: nguyen_regression(6, 1, -1.0, 1.0, Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        let x = point[0];
        math::sin(x) + math::sin(x + x * x)
    }
}

regression_problem!(
    Nguyen6,
    "Nguyen-6",
    "sin(x) + sin(x + x^2)",
    NGUYEN,
    NGUYEN_URL
);

/// Nguyen-7: ln(x + 1) + ln(x² + 1), from 20 points uniform in [0, 2] (Uy et al. 2011).
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet checked
/// against the paper ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Nguyen7 {
    regression: Regression,
}

impl Nguyen7 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: nguyen_regression(7, 1, 0.0, 2.0, Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        let x = point[0];
        math::ln(x + 1.0) + math::ln(x * x + 1.0)
    }
}

regression_problem!(
    Nguyen7,
    "Nguyen-7",
    "ln(x + 1) + ln(x^2 + 1)",
    NGUYEN,
    NGUYEN_URL
);

/// Nguyen-8: √x, from 20 points uniform in [0, 4] (Uy et al. 2011).
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet checked
/// against the paper ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Nguyen8 {
    regression: Regression,
}

impl Nguyen8 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: nguyen_regression(8, 1, 0.0, 4.0, Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        point[0].sqrt()
    }
}

regression_problem!(Nguyen8, "Nguyen-8", "sqrt(x)", NGUYEN, NGUYEN_URL);

/// Nguyen-9: sin(x) + sin(y²), from 100 points uniform in [−1, 1]² (Uy et al. 2011).
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet checked
/// against the paper ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Nguyen9 {
    regression: Regression,
}

impl Nguyen9 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: nguyen_regression(9, 2, -1.0, 1.0, Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        let (x, y) = (point[0], point[1]);
        math::sin(x) + math::sin(y * y)
    }
}

regression_problem!(Nguyen9, "Nguyen-9", "sin(x) + sin(y^2)", NGUYEN, NGUYEN_URL);

/// Nguyen-10: 2 sin(x) cos(y), from 100 points uniform in [−1, 1]² (Uy et al. 2011).
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet checked
/// against the paper ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Nguyen10 {
    regression: Regression,
}

impl Nguyen10 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: nguyen_regression(10, 2, -1.0, 1.0, Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        let (x, y) = (point[0], point[1]);
        2.0 * math::sin(x) * math::cos(y)
    }
}

regression_problem!(Nguyen10, "Nguyen-10", "2 sin(x) cos(y)", NGUYEN, NGUYEN_URL);

/// Nguyen-11: xʸ, from 100 points uniform in [0, 1]² (Uy et al. 2011).
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet checked
/// against the paper ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Nguyen11 {
    regression: Regression,
}

impl Nguyen11 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: nguyen_regression(11, 2, 0.0, 1.0, Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        math::powf(point[0], point[1])
    }
}

regression_problem!(Nguyen11, "Nguyen-11", "x^y", NGUYEN, NGUYEN_URL);

/// Nguyen-12: x⁴ − x³ + y²/2 − y, from 100 points uniform in [−1, 1]² (Uy et al. 2011).
///
/// The name, target, sampling and set as McDermott et al. (2012) restate them; not yet checked
/// against the paper ([#168](https://github.com/tachsin/genoxide/issues/168)).
#[derive(Clone, Debug)]
pub struct Nguyen12 {
    regression: Regression,
}

impl Nguyen12 {
    /// The problem, with its data.
    pub fn new() -> Self {
        Self {
            regression: nguyen_regression(12, 2, -1.0, 1.0, Self::target_of),
        }
    }

    fn target_of(point: &[f64]) -> f64 {
        let (x, y) = (point[0], point[1]);
        x * x * x * x - x * x * x + 0.5 * y * y - y
    }
}

regression_problem!(
    Nguyen12,
    "Nguyen-12",
    "x^4 - x^3 + y^2/2 - y",
    NGUYEN,
    NGUYEN_URL
);
