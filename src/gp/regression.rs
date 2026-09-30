//! Symbolic regression: trees of mathematical functions fitted to data.
//!
//! | Part | What it is |
//! |---|---|
//! | [`Math`] | The primitives: arithmetic, the analytic quotient, Koza's protected functions, `sin`, `exp` and the like, and the variables |
//! | [`primitives`] | A primitive set of [`Math`] functions, named variables and optional constants |
//! | [`Sample`], [`Dataset`] | Points and their targets: the training sample the search sees, and a test sample it never sees |
//! | [`Regression`] | The fitness function: the error of a tree on the training sample (RMSE by default), after linear scaling (on by default) |
//! | [`problems`] | Test problems from the literature, each with its paper's target, sampling and function set |
//!
//! [`Regression`] evaluates a tree on all the points at once with
//! [`Tree::evaluate_columns`], in a workspace kept per thread: evaluating in parallel allocates
//! nothing once the workspaces have grown. Every function goes through [`math`] or
//! IEEE operations, and every sum is taken in the order of the points, so the errors are the same
//! to the bit on every platform.
//!
//! **Linear scaling** (Keijzer 2003, 2004): the error is the one of `a + b·f(x)`, with `a` and
//! `b` fitted by least squares on the training sample, so that the search looks for the shape of
//! the target and not for its scale and offset, which random constants find slowly. It's on by
//! default; [`Regression::scaling`] gives `a` and `b` of a tree, and
//! [`Regression::display`] the scaled expression.
//!
//! **Invalid trees.** A tree whose value isn't finite at a training point (a division by zero, the
//! logarithm of a negative number) is invalid: its fitness is `None`, worse than any error, as
//! genoxide treats NaN. The analytic quotient ([`Math::Aq`]) is defined everywhere, and is the
//! recommended division (Ni et al. 2013); Koza's protected functions are there to replicate papers
//! that use them.
//!
//! ```
//! use genoxide::gp::regression::{self, Dataset, Math, Regression, Sample};
//! use genoxide::gp::{Gp, SubtreeCrossover, SubtreeMutation};
//! use genoxide::prelude::*;
//!
//! // 3x² + 2 from 21 points: linear scaling finds the 3 and the 2, the search only x²
//! let xs: Vec<f64> = (0..=20).map(|i| f64::from(i) / 10.0 - 1.0).collect();
//! let ys = xs.iter().map(|x| 3.0 * x * x + 2.0).collect();
//! let dataset = Dataset::new(Sample::new(vec![xs], ys)?);
//! let set = regression::primitives([Math::Add, Math::Sub, Math::Mul], ["x"], None)?;
//! let fitness = Regression::new(set.clone(), dataset)?;
//! let ga = Ga::builder(Gp::builder(set).build()?)
//!     .population_size(100)
//!     .select(Tournament::new(3)?)
//!     .crossover(SubtreeCrossover::new())
//!     .mutate(SubtreeMutation::new())
//!     .mutation_rate(0.1)
//!     .minimize()
//!     .seed(1)
//!     .build()?;
//! let outcome = Engine::new(ga, fitness.clone())
//!     .stop_when(Stop::target(1e-12).or(Stop::generations(50)))
//!     .run()?;
//! assert_eq!(outcome.stop_reason(), StopReason::Target);
//! // e.g. 2 + 3 * (mul(x, x)): the intercept and slope that linear scaling fitted
//! println!("{}", fitness.display(outcome.best_genome()));
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! References:
//!
//! - Keijzer, M. (2003). Improving symbolic regression with interval arithmetic and linear
//!   scaling. EuroGP 2003, LNCS 2610: 70-82. doi:10.1007/3-540-36599-0_7. Linear scaling.
//! - Keijzer, M. (2004). Scaled symbolic regression. *Genetic Programming and Evolvable Machines*
//!   5(3): 259-269. doi:10.1023/B:GENP.0000030195.77571.f9. Linear scaling's effect on the search.
//! - Ni, J., Drieberg, R. H. and Rockett, P. I. (2013). The use of an analytic quotient operator in
//!   genetic programming. *IEEE Transactions on Evolutionary Computation* 17(1): 146-152.
//!   doi:10.1109/TEVC.2012.2195319. The analytic quotient, `a / √(1 + b²)`.
//! - Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of
//!   Natural Selection.* MIT Press. The protected division and logarithm.

pub mod problems;

use super::evaluate::Columns;
use super::primitives::{Constants, PrimitiveSet};
use super::tree::Tree;
use crate::engine::FitnessFunction;
use crate::{Error, Result, math};
use std::cell::RefCell;
use std::fmt;

/// The primitives of symbolic regression. Each computes its value with [`math`] or
/// IEEE operations, the same bits on every platform; a value that isn't finite at a training
/// point makes the tree invalid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Math {
    /// a + b.
    Add,
    /// a − b.
    Sub,
    /// a · b.
    Mul,
    /// a / b, IEEE division: ±∞ or NaN for b = 0, making the tree invalid.
    Div,
    /// The analytic quotient a / √(1 + b²) (Ni et al. 2013): close to a / b for large |b|, and
    /// defined and smooth everywhere.
    Aq,
    /// −a.
    Neg,
    /// 1 / a: ±∞ at ±0, making the tree invalid.
    Inv,
    /// a².
    Square,
    /// a³.
    Cube,
    /// sin a.
    Sin,
    /// cos a.
    Cos,
    /// e^a.
    Exp,
    /// ln a: NaN for a < 0 and −∞ at 0, making the tree invalid.
    Log,
    /// √a: NaN for a < 0, making the tree invalid.
    Sqrt,
    /// tanh a.
    Tanh,
    /// |a|.
    Abs,
    /// Koza's protected division: a / b, and 1 for b = 0.
    ProtectedDiv,
    /// Koza's protected logarithm: ln |a|, and 0 at 0.
    ProtectedLog,
    /// The protected square root √|a|.
    ProtectedSqrt,
    /// The variable of this index: a column of the data.
    Variable(u16),
}

impl Math {
    /// The functions, in the order of the enum: every variant but [`Variable`](Math::Variable).
    pub const FUNCTIONS: [Math; 19] = [
        Math::Add,
        Math::Sub,
        Math::Mul,
        Math::Div,
        Math::Aq,
        Math::Neg,
        Math::Inv,
        Math::Square,
        Math::Cube,
        Math::Sin,
        Math::Cos,
        Math::Exp,
        Math::Log,
        Math::Sqrt,
        Math::Tanh,
        Math::Abs,
        Math::ProtectedDiv,
        Math::ProtectedLog,
        Math::ProtectedSqrt,
    ];

    /// The name of a function in a primitive set made by [`primitives`], as trees are displayed
    /// and parsed: `add`, `aq`, `pdiv`, `plog`, ...; a variable's name is the set's.
    pub fn name(self) -> &'static str {
        match self {
            Math::Add => "add",
            Math::Sub => "sub",
            Math::Mul => "mul",
            Math::Div => "div",
            Math::Aq => "aq",
            Math::Neg => "neg",
            Math::Inv => "inv",
            Math::Square => "square",
            Math::Cube => "cube",
            Math::Sin => "sin",
            Math::Cos => "cos",
            Math::Exp => "exp",
            Math::Log => "log",
            Math::Sqrt => "sqrt",
            Math::Tanh => "tanh",
            Math::Abs => "abs",
            Math::ProtectedDiv => "pdiv",
            Math::ProtectedLog => "plog",
            Math::ProtectedSqrt => "psqrt",
            Math::Variable(_) => "variable",
        }
    }

    /// The number of arguments: 2 for the binary functions, 1 for the unary ones, 0 for a
    /// variable.
    pub fn arity(self) -> usize {
        match self {
            Math::Add | Math::Sub | Math::Mul | Math::Div | Math::Aq | Math::ProtectedDiv => 2,
            Math::Variable(_) => 0,
            _ => 1,
        }
    }

    /// The value at one point: `args` are the arguments' values, `variables` the point.
    ///
    /// # Panics
    ///
    /// If `args` has fewer than [`arity`](Math::arity) values, or a variable's index is beyond
    /// `variables`.
    #[inline]
    pub fn apply(self, args: &[f64], variables: &[f64]) -> f64 {
        match self {
            Math::Variable(k) => variables[usize::from(k)],
            _ if self.arity() == 2 => self.binary(args[0], args[1]),
            _ => self.unary(args[0]),
        }
    }

    /// The values at every point at once, into `output`: `args` are the arguments' columns,
    /// `variables` the data's columns. The same values, to the bit, as
    /// [`apply`](Math::apply) at each point.
    ///
    /// # Panics
    ///
    /// If `args` has fewer than [`arity`](Math::arity) columns, a column is shorter than
    /// `output`, or a variable's index is beyond `variables`.
    pub fn apply_columns(self, args: &[&[f64]], variables: &[Vec<f64>], output: &mut [f64]) {
        match self {
            Math::Variable(k) => output.copy_from_slice(&variables[usize::from(k)][..output.len()]),
            // each arm a loop of its own, which the compiler can vectorize
            Math::Add => binary_columns(args, output, |a, b| a + b),
            Math::Sub => binary_columns(args, output, |a, b| a - b),
            Math::Mul => binary_columns(args, output, |a, b| a * b),
            Math::Div => binary_columns(args, output, |a, b| a / b),
            Math::Aq => binary_columns(args, output, aq),
            Math::ProtectedDiv => binary_columns(args, output, protected_div),
            Math::Neg => unary_columns(args, output, |a| -a),
            Math::Inv => unary_columns(args, output, |a| 1.0 / a),
            Math::Square => unary_columns(args, output, |a| a * a),
            Math::Cube => unary_columns(args, output, |a| a * a * a),
            Math::Sin => unary_columns(args, output, math::sin),
            Math::Cos => unary_columns(args, output, math::cos),
            Math::Exp => unary_columns(args, output, math::exp),
            Math::Log => unary_columns(args, output, math::ln),
            Math::Sqrt => unary_columns(args, output, f64::sqrt),
            Math::Tanh => unary_columns(args, output, math::tanh),
            Math::Abs => unary_columns(args, output, f64::abs),
            Math::ProtectedLog => unary_columns(args, output, protected_log),
            Math::ProtectedSqrt => unary_columns(args, output, protected_sqrt),
        }
    }

    // a binary function's value
    #[inline]
    fn binary(self, a: f64, b: f64) -> f64 {
        match self {
            Math::Add => a + b,
            Math::Sub => a - b,
            Math::Mul => a * b,
            Math::Div => a / b,
            Math::Aq => aq(a, b),
            Math::ProtectedDiv => protected_div(a, b),
            _ => unreachable!("a binary function"),
        }
    }

    // a unary function's value
    #[inline]
    fn unary(self, a: f64) -> f64 {
        match self {
            Math::Neg => -a,
            Math::Inv => 1.0 / a,
            Math::Square => a * a,
            Math::Cube => a * a * a,
            Math::Sin => math::sin(a),
            Math::Cos => math::cos(a),
            Math::Exp => math::exp(a),
            Math::Log => math::ln(a),
            Math::Sqrt => a.sqrt(),
            Math::Tanh => math::tanh(a),
            Math::Abs => a.abs(),
            Math::ProtectedLog => protected_log(a),
            Math::ProtectedSqrt => protected_sqrt(a),
            _ => unreachable!("a unary function"),
        }
    }
}

#[inline]
fn aq(a: f64, b: f64) -> f64 {
    a / (1.0 + b * b).sqrt()
}

#[inline]
fn protected_div(a: f64, b: f64) -> f64 {
    if b == 0.0 { 1.0 } else { a / b }
}

#[inline]
fn protected_log(a: f64) -> f64 {
    if a == 0.0 { 0.0 } else { math::ln(a.abs()) }
}

#[inline]
fn protected_sqrt(a: f64) -> f64 {
    a.abs().sqrt()
}

#[inline]
fn unary_columns(args: &[&[f64]], output: &mut [f64], f: impl Fn(f64) -> f64) {
    for (out, &a) in output.iter_mut().zip(args[0]) {
        *out = f(a);
    }
}

#[inline]
fn binary_columns(args: &[&[f64]], output: &mut [f64], f: impl Fn(f64, f64) -> f64) {
    for ((out, &a), &b) in output.iter_mut().zip(args[0]).zip(args[1]) {
        *out = f(a, b);
    }
}

// the most variables: their index is a u16
const MAX_VARIABLES: usize = 1 << 16;

/// A primitive set of symbolic regression, of one type (`real`): the `functions`, named as
/// [`Math::name`] gives, and the variables, named `variables` in the order of the data's columns
/// ([`Math::Variable`] `0` is the first), with `constants` if given (ephemeral random constants,
/// e.g. [`Constants::uniform`]).
///
/// ```
/// use genoxide::gp::regression::{self, Math};
///
/// let set = regression::primitives([Math::Add, Math::Mul, Math::Sin], ["x", "y"], None)?;
/// let tree = set.parse("add(mul(x, y), sin(x))")?;
/// assert_eq!(tree.display(&set).to_string(), "add(mul(x, y), sin(x))");
/// # Ok::<(), genoxide::Error>(())
/// ```
///
/// # Errors
///
/// [`Error::InvalidSetting`] for no variables, more than 2^16, a name used twice, or a
/// [`Math::Variable`] among the functions; the errors of
/// [`PrimitiveSetBuilder::build`](super::PrimitiveSetBuilder::build).
pub fn primitives<S: Into<String>>(
    functions: impl IntoIterator<Item = Math>,
    variables: impl IntoIterator<Item = S>,
    constants: Option<Constants>,
) -> Result<PrimitiveSet<Math>> {
    let invalid = |reason: String| Error::InvalidSetting {
        setting: "primitives",
        reason,
    };
    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real");
    let mut names: Vec<String> = Vec::new();
    for function in functions {
        if let Math::Variable(_) = function {
            return Err(invalid(
                "a variable isn't a function: name it among the variables".to_string(),
            ));
        }
        names.push(function.name().to_string());
        set.function(
            function.name(),
            function,
            vec![real; function.arity()],
            real,
        );
    }
    let mut count = 0;
    for (index, name) in variables.into_iter().enumerate() {
        let name = name.into();
        let index = u16::try_from(index)
            .map_err(|_| invalid(format!("at most {MAX_VARIABLES} variables")))?;
        names.push(name.clone());
        set.terminal(name, Math::Variable(index), real);
        count += 1;
    }
    if count == 0 {
        return Err(invalid("at least one variable".to_string()));
    }
    names.sort_unstable();
    if let Some(pair) = names.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(invalid(format!("the name {:?} is used twice", pair[0])));
    }
    if let Some(constants) = constants {
        set.constants(real, constants);
    }
    set.build(real)
}

/// Points and their targets: a column of values per variable, and a target per point. All
/// finite.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Sample {
    columns: Vec<Vec<f64>>,
    targets: Vec<f64>,
}

impl Sample {
    /// A sample of `columns`, one per variable with a value per point, and the `targets`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for no columns or no points, columns of other lengths than the
    /// targets, more than 2^16 columns, or a value that isn't finite.
    pub fn new(columns: Vec<Vec<f64>>, targets: Vec<f64>) -> Result<Self> {
        let invalid = |reason: String| Error::InvalidSetting {
            setting: "sample",
            reason,
        };
        if columns.is_empty() || columns.len() > MAX_VARIABLES {
            return Err(invalid(format!(
                "1 to {MAX_VARIABLES} columns of variables, got {}",
                columns.len()
            )));
        }
        if targets.is_empty() {
            return Err(invalid("at least one point".to_string()));
        }
        if let Some((k, column)) = columns
            .iter()
            .enumerate()
            .find(|(_, column)| column.len() != targets.len())
        {
            return Err(invalid(format!(
                "column {k} has {} values for {} targets",
                column.len(),
                targets.len()
            )));
        }
        if columns
            .iter()
            .flatten()
            .chain(&targets)
            .any(|v| !v.is_finite())
        {
            return Err(invalid("every value finite".to_string()));
        }
        Ok(Self { columns, targets })
    }

    /// A sample of `points`, each a value per variable, with `target` of each as its target.
    ///
    /// # Errors
    ///
    /// As [`new`](Sample::new), and for points of different lengths.
    pub fn from_points(
        points: impl IntoIterator<Item = impl AsRef<[f64]>>,
        target: impl Fn(&[f64]) -> f64,
    ) -> Result<Self> {
        let mut columns: Vec<Vec<f64>> = Vec::new();
        let mut targets = Vec::new();
        for (index, point) in points.into_iter().enumerate() {
            let point = point.as_ref();
            if index == 0 {
                columns = vec![Vec::new(); point.len()];
            } else if point.len() != columns.len() {
                return Err(Error::InvalidSetting {
                    setting: "sample",
                    reason: format!(
                        "point {index} has {} values, the first {}",
                        point.len(),
                        columns.len()
                    ),
                });
            }
            for (column, &value) in columns.iter_mut().zip(point) {
                column.push(value);
            }
            targets.push(target(point));
        }
        Self::new(columns, targets)
    }

    /// The number of variables.
    pub fn variables(&self) -> usize {
        self.columns.len()
    }

    /// The number of points.
    pub fn points(&self) -> usize {
        self.targets.len()
    }

    /// The values of each variable, a column per variable.
    pub fn columns(&self) -> &[Vec<f64>] {
        &self.columns
    }

    /// The targets.
    pub fn targets(&self) -> &[f64] {
        &self.targets
    }

    /// The point of this index, a value per variable.
    ///
    /// # Panics
    ///
    /// If `index` isn't below [`points`](Sample::points).
    pub fn point(&self, index: usize) -> Vec<f64> {
        self.columns.iter().map(|column| column[index]).collect()
    }

    /// The standard deviation of the targets (of the population: divided by the number of
    /// points), the scale of an error: exact recovery is usually an error of at most 1e-10 of it.
    pub fn deviation(&self) -> f64 {
        let n = self.targets.len() as f64;
        let mean = self.targets.iter().sum::<f64>() / n;
        let squares: f64 = self.targets.iter().map(|y| (y - mean) * (y - mean)).sum();
        (squares / n).sqrt()
    }
}

/// The data of a regression: the training sample that the search fits, and a test sample it
/// never sees, to measure how the result generalizes.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dataset {
    training: Sample,
    test: Option<Sample>,
}

impl Dataset {
    /// A dataset of a training sample, without a test sample.
    pub fn new(training: Sample) -> Self {
        Self {
            training,
            test: None,
        }
    }

    /// With a test sample.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] unless it has as many variables as the training sample.
    pub fn with_test(mut self, test: Sample) -> Result<Self> {
        if test.variables() != self.training.variables() {
            return Err(Error::InvalidSetting {
                setting: "test",
                reason: format!(
                    "the test sample has {} variables, the training sample {}",
                    test.variables(),
                    self.training.variables()
                ),
            });
        }
        self.test = Some(test);
        Ok(self)
    }

    /// The training sample.
    pub fn training(&self) -> &Sample {
        &self.training
    }

    /// The test sample, if any.
    pub fn test(&self) -> Option<&Sample> {
        self.test.as_ref()
    }

    /// The number of variables.
    pub fn variables(&self) -> usize {
        self.training.variables()
    }
}

/// How [`Regression`] measures the error of a tree's predictions `p` against the targets `y` at
/// the `n` points.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Metric {
    /// The root mean squared error √(Σ (p − y)² / n).
    #[default]
    Rmse,
    /// The mean squared error Σ (p − y)² / n.
    Mse,
    /// The mean absolute error Σ |p − y| / n.
    Mae,
}

/// The linear scaling of a tree's values `f`: the prediction is `intercept + slope · f`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scaling {
    /// `a`, the mean target less the slope times the tree's mean value.
    pub intercept: f64,
    /// `b`, the covariance of the targets and the tree's values over the variance of the tree's
    /// values; 0 for a tree of the same value everywhere.
    pub slope: f64,
}

impl Scaling {
    /// No scaling: an intercept of 0 and a slope of 1.
    pub const IDENTITY: Scaling = Scaling {
        intercept: 0.0,
        slope: 1.0,
    };

    // the least-squares scaling of `values` to `targets`, None unless every value and the result
    // are finite; sums in the order of the points
    fn fit(values: &[f64], targets: &[f64]) -> Option<Scaling> {
        if values.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let n = values.len() as f64;
        let mean_value = values.iter().sum::<f64>() / n;
        let mean_target = targets.iter().sum::<f64>() / n;
        let (mut covariance, mut variance) = (0.0, 0.0);
        for (&v, &y) in values.iter().zip(targets) {
            let d = v - mean_value;
            covariance += (y - mean_target) * d;
            variance += d * d;
        }
        let slope = if variance > 0.0 {
            covariance / variance
        } else {
            0.0
        };
        let scaling = Scaling {
            intercept: mean_target - slope * mean_value,
            slope,
        };
        (scaling.intercept.is_finite() && scaling.slope.is_finite()).then_some(scaling)
    }
}

/// The fitness function of symbolic regression: the error of a tree's predictions on the
/// training sample of a [`Dataset`], minimized; `None` (invalid) if a value isn't finite.
///
/// The error is [`Metric::Rmse`] by default, after linear scaling
/// ([`linear_scaling`](Regression::linear_scaling), on by default). A tree is evaluated on all
/// points at once with [`Tree::evaluate_columns`], in a workspace kept per thread.
#[derive(Clone, Debug)]
pub struct Regression {
    primitives: PrimitiveSet<Math>,
    dataset: Dataset,
    metric: Metric,
    linear_scaling: bool,
}

impl Regression {
    /// The regression of `dataset` by trees of `primitives`, with the RMSE after linear scaling.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] if the set has a variable beyond the dataset's.
    pub fn new(primitives: PrimitiveSet<Math>, dataset: Dataset) -> Result<Self> {
        let variables = dataset.variables();
        if let Some(beyond) = primitives
            .primitives()
            .iter()
            .find_map(|p| match p.value() {
                Math::Variable(k) if usize::from(k) >= variables => Some(p.name().to_string()),
                _ => None,
            })
        {
            return Err(Error::InvalidSetting {
                setting: "primitives",
                reason: format!("the variable {beyond} is beyond the dataset's {variables}"),
            });
        }
        Ok(Self {
            primitives,
            dataset,
            metric: Metric::Rmse,
            linear_scaling: true,
        })
    }

    /// The error measure: [`Metric::Rmse`] by default.
    #[must_use]
    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Whether the error is the one of the tree's values after linear scaling, `a + b·f` with
    /// `a` and `b` fitted by least squares on the training sample: on by default.
    #[must_use]
    pub fn linear_scaling(mut self, on: bool) -> Self {
        self.linear_scaling = on;
        self
    }

    /// The primitive set.
    pub fn primitives(&self) -> &PrimitiveSet<Math> {
        &self.primitives
    }

    /// The dataset.
    pub fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    /// The tree's values at the points of `sample`, before scaling.
    ///
    /// # Panics
    ///
    /// If the tree isn't a tree of the set, or the sample has fewer variables than the set.
    pub fn values(&self, tree: &Tree, sample: &Sample) -> Vec<f64> {
        self.with_values(tree, sample, <[f64]>::to_vec)
    }

    /// The linear scaling of the tree, fitted on the training sample: [`Scaling::IDENTITY`] if
    /// linear scaling is off; `None` if a value of the tree on the training sample isn't finite.
    ///
    /// # Panics
    ///
    /// If the tree isn't a tree of the set.
    pub fn scaling(&self, tree: &Tree) -> Option<Scaling> {
        let training = &self.dataset.training;
        self.with_values(tree, training, |values| self.scaling_of(values, training))
    }

    /// The tree's predictions at the points of `sample`: its values after the scaling fitted on
    /// the training sample. `None` as for [`scaling`](Regression::scaling).
    ///
    /// # Panics
    ///
    /// As [`values`](Regression::values).
    pub fn predict(&self, tree: &Tree, sample: &Sample) -> Option<Vec<f64>> {
        let scaling = self.scaling(tree)?;
        let values = self.values(tree, sample);
        Some(self.scaled(scaling, &values).collect())
    }

    /// The error of the tree on `sample`, e.g. the test sample, after the scaling fitted on the
    /// training sample; `None` if a prediction isn't finite.
    ///
    /// # Panics
    ///
    /// As [`values`](Regression::values).
    pub fn error(&self, tree: &Tree, sample: &Sample) -> Option<f64> {
        let scaling = self.scaling(tree)?;
        self.with_values(tree, sample, |values| {
            self.error_of(self.scaled(scaling, values), sample.targets())
        })
    }

    /// The tree as text, with its scaling if linear scaling is on:
    /// `intercept + slope * (expression)`.
    ///
    /// # Panics
    ///
    /// If the tree isn't a tree of the set.
    pub fn display(&self, tree: &Tree) -> String {
        let expression = tree.display(&self.primitives).to_string();
        if !self.linear_scaling {
            return expression;
        }
        match self.scaling(tree) {
            Some(Scaling { intercept, slope }) => format!("{intercept} + {slope} * ({expression})"),
            None => expression,
        }
    }

    // `f` of the tree's values at the points of `sample`, in this thread's workspace
    fn with_values<T>(&self, tree: &Tree, sample: &Sample, f: impl FnOnce(&[f64]) -> T) -> T {
        thread_local! {
            static COLUMNS: RefCell<Columns> = RefCell::default();
        }
        COLUMNS.with_borrow_mut(|columns| {
            if columns.points() != sample.points() {
                *columns = Columns::new(sample.points());
            }
            let values = tree.evaluate_columns(&self.primitives, columns, |op, args, output| {
                op.apply_columns(args, sample.columns(), output);
            });
            f(values)
        })
    }

    fn scaling_of(&self, values: &[f64], training: &Sample) -> Option<Scaling> {
        if self.linear_scaling {
            Scaling::fit(values, training.targets())
        } else {
            values
                .iter()
                .all(|v| v.is_finite())
                .then_some(Scaling::IDENTITY)
        }
    }

    // the predictions from the values: the values themselves without linear scaling
    fn scaled<'a>(
        &self,
        scaling: Scaling,
        values: &'a [f64],
    ) -> impl Iterator<Item = f64> + use<'a> {
        let on = self.linear_scaling;
        values.iter().map(move |&v| {
            if on {
                scaling.intercept + scaling.slope * v
            } else {
                v
            }
        })
    }

    // the error of the predictions, None unless finite; sums in the order of the points
    fn error_of(&self, predictions: impl Iterator<Item = f64>, targets: &[f64]) -> Option<f64> {
        let mut sum = 0.0;
        for (p, &y) in predictions.zip(targets) {
            let e = p - y;
            sum += match self.metric {
                Metric::Rmse | Metric::Mse => e * e,
                Metric::Mae => e.abs(),
            };
        }
        let mean = sum / targets.len() as f64;
        let error = match self.metric {
            Metric::Rmse => mean.sqrt(),
            Metric::Mse | Metric::Mae => mean,
        };
        error.is_finite().then_some(error)
    }
}

impl FitnessFunction<Tree> for Regression {
    type Output = Option<f64>;

    /// The error on the training sample, `None` if a value isn't finite.
    fn evaluate(&self, tree: &Tree) -> Option<f64> {
        let training = &self.dataset.training;
        self.with_values(tree, training, |values| {
            let scaling = self.scaling_of(values, training)?;
            self.error_of(self.scaled(scaling, values), training.targets())
        })
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Metric::Rmse => "RMSE",
            Metric::Mse => "MSE",
            Metric::Mae => "MAE",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::problems::{
        Koza1, Koza2, Koza3, Nguyen1, Nguyen2, Nguyen3, Nguyen4, Nguyen5, Nguyen6, Nguyen7,
        Nguyen8, Nguyen9, Nguyen10, Nguyen11, Nguyen12, RegressionProblem,
    };
    use super::*;
    use crate::gp::Gp;
    use crate::rng::StreamRng;

    fn line(xs: &[f64], f: impl Fn(f64) -> f64) -> Dataset {
        let ys = xs.iter().map(|&x| f(x)).collect();
        Dataset::new(Sample::new(vec![xs.to_vec()], ys).unwrap())
    }

    fn xs() -> Vec<f64> {
        (0..=20).map(|i| f64::from(i) / 10.0 - 1.0).collect()
    }

    #[test]
    fn a_function_gives_the_same_bits_at_a_point_and_on_columns() {
        let values = [
            -3.5,
            -1.0,
            -0.0,
            0.0,
            0.25,
            1.0,
            2.0,
            700.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let variables = vec![values.to_vec()];
        for function in Math::FUNCTIONS {
            for &b in &values {
                let a_column = values.to_vec();
                let b_column = vec![b; values.len()];
                let mut output = vec![0.0; values.len()];
                function.apply_columns(&[&a_column, &b_column], &variables, &mut output);
                for (i, &a) in values.iter().enumerate() {
                    let point = function.apply(&[a, b], &[a]);
                    assert_eq!(
                        output[i].to_bits(),
                        point.to_bits(),
                        "{function:?}({a}, {b})"
                    );
                }
            }
        }
        let mut output = vec![0.0; values.len()];
        Math::Variable(0).apply_columns(&[], &variables, &mut output);
        assert_eq!(output[4], 0.25);
        assert_eq!(Math::Variable(1).apply(&[], &[1.0, 2.0]), 2.0);
    }

    #[test]
    fn the_functions_by_hand() {
        let at = |f: Math, a: f64, b: f64| f.apply(&[a, b], &[]);
        assert_eq!(at(Math::Aq, 3.0, 0.0), 3.0);
        assert_eq!(at(Math::Aq, 1.0, 1.0), 1.0 / 2.0_f64.sqrt());
        assert!(at(Math::Div, 1.0, 0.0).is_infinite());
        assert_eq!(at(Math::ProtectedDiv, 5.0, 0.0), 1.0);
        assert_eq!(at(Math::ProtectedDiv, 5.0, 2.0), 2.5);
        assert!(at(Math::Log, -1.0, 0.0).is_nan());
        assert_eq!(at(Math::ProtectedLog, 0.0, 0.0), 0.0);
        assert_eq!(at(Math::ProtectedLog, -1.0, 0.0), 0.0);
        assert_eq!(at(Math::ProtectedSqrt, -4.0, 0.0), 2.0);
        assert!(at(Math::Sqrt, -4.0, 0.0).is_nan());
        assert_eq!(at(Math::Cube, -2.0, 0.0), -8.0);
        assert_eq!(at(Math::Square, -3.0, 0.0), 9.0);
        assert_eq!(at(Math::Neg, 3.0, 0.0), -3.0);
        assert_eq!(at(Math::Inv, 4.0, 0.0), 0.25);
        assert_eq!(at(Math::Inv, 0.0, 0.0), f64::INFINITY);
        assert_eq!(at(Math::Abs, -3.0, 0.0), 3.0);
        for f in Math::FUNCTIONS {
            assert!(matches!(f.arity(), 1 | 2));
        }
        assert_eq!(Math::Variable(3).arity(), 0);
    }

    #[test]
    fn evaluation_of_random_trees_matches_the_stack_evaluator_to_the_bit() {
        let set = primitives(
            Math::FUNCTIONS,
            ["x", "y"],
            Some(Constants::uniform(-2.0..=2.0).unwrap()),
        )
        .unwrap();
        let gp = Gp::builder(set.clone()).build().unwrap();
        let mut rng = StreamRng::seed_from_u64(7);
        let points: Vec<[f64; 2]> = (0..30)
            .map(|i| [f64::from(i) / 7.0 - 2.0, f64::from(i % 5) - 2.0])
            .collect();
        let sample = Sample::from_points(&points, |p| p[0] + p[1]).unwrap();
        let regression = Regression::new(set.clone(), Dataset::new(sample.clone())).unwrap();
        let mut stack = Vec::new();
        for tree in gp.ramped_half_and_half(300, &mut rng).unwrap() {
            let columns = regression.values(&tree, &sample);
            for (point, &column) in points.iter().zip(&columns) {
                let value = tree.evaluate(
                    &set,
                    &mut stack,
                    |op, args: &[f64]| op.apply(args, point),
                    |_, constant| constant,
                );
                assert_eq!(value.to_bits(), column.to_bits(), "{}", tree.display(&set));
            }
        }
    }

    #[test]
    fn linear_scaling_recovers_the_intercept_and_the_slope() {
        let xs = xs();
        let set = primitives([Math::Add, Math::Mul], ["x"], None).unwrap();
        let x = set.parse("x").unwrap();
        // y = 2 + 3x: the tree x, scaled, fits exactly
        let regression = Regression::new(set.clone(), line(&xs, |x| 2.0 + 3.0 * x)).unwrap();
        let Scaling { intercept, slope } = regression.scaling(&x).unwrap();
        assert!((intercept - 2.0).abs() < 1e-15 && (slope - 3.0).abs() < 1e-15);
        assert!(regression.evaluate(&x).unwrap() < 1e-15);
        assert!(regression.display(&x).ends_with(" * (x)"));
        // without scaling, the tree's own error
        let plain = regression.clone().linear_scaling(false);
        assert_eq!(plain.scaling(&x), Some(Scaling::IDENTITY));
        assert!(plain.evaluate(&x).unwrap() > 1.0);
        assert_eq!(plain.display(&x), "x");
    }

    #[test]
    fn a_constant_tree_is_scaled_to_the_mean() {
        let xs = xs();
        let set = primitives([Math::Add], ["x"], Some(Constants::choice([1.0]).unwrap())).unwrap();
        let regression = Regression::new(set.clone(), line(&xs, |x| x * x)).unwrap();
        let one = set.parse("1.0").unwrap();
        let scaling = regression.scaling(&one).unwrap();
        assert_eq!(scaling.slope, 0.0);
        let mean = xs.iter().map(|x| x * x).sum::<f64>() / xs.len() as f64;
        assert_eq!(scaling.intercept, mean);
    }

    #[test]
    fn the_scaled_error_is_the_least_squares_minimum() {
        let xs = xs();
        let set = primitives([Math::Mul, Math::Sin], ["x"], None).unwrap();
        let regression = Regression::new(set.clone(), line(&xs, |x| x * x * x - x))
            .unwrap()
            .metric(Metric::Mse);
        let tree = set.parse("sin(mul(x, x))").unwrap();
        let best = regression.evaluate(&tree).unwrap();
        let Scaling { intercept, slope } = regression.scaling(&tree).unwrap();
        let values = regression.values(&tree, regression.dataset().training());
        let targets = regression.dataset().training().targets();
        let mse = |a: f64, b: f64| {
            values
                .iter()
                .zip(targets)
                .map(|(v, y)| (a + b * v - y) * (a + b * v - y))
                .sum::<f64>()
                / values.len() as f64
        };
        assert!((mse(intercept, slope) - best).abs() < 1e-15);
        for (da, db) in [(1e-3, 0.0), (-1e-3, 0.0), (0.0, 1e-3), (0.0, -1e-3)] {
            assert!(mse(intercept + da, slope + db) > best);
        }
    }

    #[test]
    fn the_metrics_by_hand() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let set = primitives([Math::Add], ["x"], None).unwrap();
        // y = x + (1, −1, 3, −3): errors of the tree x are 1, 1, 3, 3
        let offsets = [1.0, -1.0, 3.0, -3.0];
        let ys: Vec<f64> = xs.iter().zip(offsets).map(|(x, o)| x + o).collect();
        let dataset = Dataset::new(Sample::new(vec![xs], ys).unwrap());
        let x = set.parse("x").unwrap();
        let plain = Regression::new(set, dataset).unwrap().linear_scaling(false);
        assert_eq!(plain.clone().metric(Metric::Mse).evaluate(&x), Some(5.0));
        assert_eq!(
            plain.clone().metric(Metric::Rmse).evaluate(&x),
            Some(5.0_f64.sqrt())
        );
        assert_eq!(plain.metric(Metric::Mae).evaluate(&x), Some(2.0));
    }

    #[test]
    fn a_value_that_isnt_finite_makes_the_tree_invalid() {
        // x from 0: 1 / x is infinite at 0, the analytic quotient isn't
        let xs: Vec<f64> = (0..5).map(f64::from).collect();
        let set = primitives(
            [Math::Div, Math::Aq, Math::Log],
            ["x"],
            Some(Constants::choice([1.0]).unwrap()),
        )
        .unwrap();
        let regression = Regression::new(set.clone(), line(&xs, |x| x)).unwrap();
        for text in ["div(1.0, x)", "log(x)"] {
            let tree = set.parse(text).unwrap();
            assert_eq!(regression.evaluate(&tree), None, "{text}");
            assert_eq!(regression.scaling(&tree), None);
            assert_eq!(
                regression.predict(&tree, regression.dataset().training()),
                None
            );
            assert_eq!(
                regression.clone().linear_scaling(false).evaluate(&tree),
                None
            );
        }
        assert!(
            regression
                .evaluate(&set.parse("aq(1.0, x)").unwrap())
                .is_some()
        );
    }

    #[test]
    fn invalid_settings_are_errors() {
        assert!(primitives([Math::Add], Vec::<String>::new(), None).is_err());
        assert!(primitives([Math::Add], ["x", "x"], None).is_err());
        assert!(primitives([Math::Add], ["add"], None).is_err());
        assert!(primitives([Math::Variable(0)], ["x"], None).is_err());
        assert!(Sample::new(vec![], vec![1.0]).is_err());
        assert!(Sample::new(vec![vec![1.0]], vec![]).is_err());
        assert!(Sample::new(vec![vec![1.0, 2.0]], vec![1.0]).is_err());
        assert!(Sample::new(vec![vec![f64::NAN]], vec![1.0]).is_err());
        assert!(Sample::new(vec![vec![1.0]], vec![f64::INFINITY]).is_err());
        assert!(Sample::from_points([vec![1.0], vec![1.0, 2.0]], |_| 0.0).is_err());
        let one = Sample::new(vec![vec![1.0]], vec![1.0]).unwrap();
        let two = Sample::new(vec![vec![1.0], vec![2.0]], vec![1.0]).unwrap();
        assert!(Dataset::new(one.clone()).with_test(two).is_err());
        // a variable beyond the dataset's
        let set = primitives([Math::Add], ["x", "y"], None).unwrap();
        assert!(Regression::new(set, Dataset::new(one)).is_err());
    }

    #[test]
    fn the_koza_problems() {
        let problems: Vec<Box<dyn RegressionProblem>> = super::problems::all();
        assert_eq!(problems.len(), 15);
        for problem in problems
            .iter()
            .filter(|problem| problem.name().starts_with("Koza"))
        {
            let dataset = problem.dataset();
            assert_eq!(dataset.training().points(), 20);
            assert_eq!(dataset.test().unwrap().points(), 101);
            // the targets are the target function's
            let training = dataset.training();
            for i in 0..training.points() {
                let point = training.point(i);
                assert!((-1.0..=1.0).contains(&point[0]));
                assert_eq!(training.targets()[i], problem.target(&point));
            }
            assert!(!problem.formula().is_empty() && !problem.reference().is_empty());
        }
        assert_eq!(Koza1::new().target(&[2.0]), 30.0);
        assert_eq!(Koza2::new().target(&[2.0]), 18.0);
        assert_eq!(Koza3::new().target(&[2.0]), 36.0);
        // the same points on every platform: the first three training points, to the bit
        let first: Vec<u64> = (0..3)
            .map(|i| Koza1::new().dataset().training().point(i)[0].to_bits())
            .collect();
        assert_eq!(
            first,
            [
                4604259971976835272,
                4588306890474721152,
                4593998995362974144
            ],
            "0.6755432248660549, 0.056524711570573594, 0.13408649392029126"
        );
        // the exact formulas have no error
        let problem = Koza2::new();
        let set = problem.primitives();
        let tree = set
            .parse("add(sub(mul(mul(x, x), mul(mul(x, x), x)), add(mul(x, mul(x, x)), mul(x, mul(x, x)))), x)")
            .unwrap();
        assert!(problem.evaluate(&tree).unwrap() < 1e-15);
        let unscaled = problem.regression().clone().linear_scaling(false);
        assert!(unscaled.evaluate(&tree).unwrap() < 1e-15);
        assert!(
            unscaled
                .error(&tree, problem.dataset().test().unwrap())
                .unwrap()
                < 1e-15
        );
    }

    #[test]
    fn the_nguyen_problems() {
        let problems: Vec<Box<dyn RegressionProblem>> = super::problems::all();
        let nguyen: Vec<_> = problems
            .iter()
            .filter(|problem| problem.name().starts_with("Nguyen"))
            .collect();
        assert_eq!(nguyen.len(), 12);
        for (number, problem) in (1..).zip(&nguyen) {
            assert_eq!(problem.name(), format!("Nguyen-{number}"));
            let dataset = problem.dataset();
            let (training, test) = (dataset.training(), dataset.test().unwrap());
            let variables = if number <= 8 { 1 } else { 2 };
            let count = if variables == 1 { 20 } else { 100 };
            assert_eq!(training.variables(), variables);
            assert_eq!((training.points(), test.points()), (count, 5 * count));
            assert_eq!(problem.primitives().primitives().len(), 8 + variables);
            let range = match number {
                7 => 0.0..=2.0,
                8 => 0.0..=4.0,
                11 => 0.0..=1.0,
                _ => -1.0..=1.0,
            };
            // the targets are the target function's, at points in the range
            for sample in [training, test] {
                for i in 0..sample.points() {
                    let point = sample.point(i);
                    assert!(point.iter().all(|value| range.contains(value)));
                    assert_eq!(sample.targets()[i], problem.target(&point));
                }
            }
            // the training and test points differ
            assert_ne!(training.point(0), test.point(0));
        }
        let targets = [
            (Nguyen1::new().target(&[2.0]), 14.0),
            (Nguyen2::new().target(&[2.0]), 30.0),
            (Nguyen3::new().target(&[2.0]), 62.0),
            (Nguyen4::new().target(&[2.0]), 126.0),
            (Nguyen5::new().target(&[0.0]), -1.0),
            (Nguyen6::new().target(&[0.0]), 0.0),
            (Nguyen7::new().target(&[0.0]), 0.0),
            (Nguyen8::new().target(&[4.0]), 2.0),
            (Nguyen9::new().target(&[0.0, 0.0]), 0.0),
            (Nguyen10::new().target(&[0.0, 1.0]), 0.0),
            (Nguyen11::new().target(&[2.0, 3.0]), 8.0),
            (Nguyen12::new().target(&[2.0, 2.0]), 8.0),
        ];
        for (number, (target, expected)) in (1..).zip(targets) {
            assert_eq!(target, expected, "Nguyen-{number}");
        }
        // exact formulas in the set have no error, on the training and test points
        let exact: [(&dyn RegressionProblem, &str); 4] = [
            (
                &Nguyen5::new(),
                "sub(mul(sin(mul(x, x)), cos(x)), pdiv(x, x))",
            ),
            (
                &Nguyen7::new(),
                "add(plog(add(x, pdiv(x, x))), plog(add(mul(x, x), pdiv(x, x))))",
            ),
            (&Nguyen10::new(), "mul(sin(x), cos(y))"),
            (&Nguyen11::new(), "exp(mul(y, plog(x)))"),
        ];
        for (problem, formula) in exact {
            let tree = problem.primitives().parse(formula).unwrap();
            let regression = problem.regression();
            assert!(
                regression.evaluate(&tree).unwrap() < 1e-14,
                "{}",
                problem.name()
            );
            let test = problem.dataset().test().unwrap();
            assert!(
                regression.error(&tree, test).unwrap() < 1e-14,
                "{}",
                problem.name()
            );
        }
    }
}
