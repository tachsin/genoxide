//! The fitness functions of trees evaluated in Rust: symbolic regression (`gp::regression`: its
//! samples, datasets, `Regression` and test problems) and the Boolean problems (`gp::boolean`),
//! as Python objects; and [`TreeFitness`], the one of them a run evaluates without Python, with
//! the size as a second objective if asked.

use crate::errors::{setting, setting_named};
use crate::trees::{PyPrimitiveSet, PyTree, Set, from_logic, from_math, to_math};
use genoxide::Objective;
use genoxide::engine::FitnessFunction;
use genoxide::genome::Genome;
use genoxide::gp::boolean::{EvenParity, Multiplexer};
use genoxide::gp::regression::problems::{
    Koza1, Koza2, Koza3, Nguyen1, Nguyen2, Nguyen3, Nguyen4, Nguyen5, Nguyen6, Nguyen7, Nguyen8,
    Nguyen9, Nguyen10, Nguyen11, Nguyen12, RegressionProblem,
};
use genoxide::gp::regression::{self, Dataset, Math, Metric, Regression, Sample};
use genoxide::gp::{Constants, Tree};
use numpy::ndarray::Array2;
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike1, PyArrayLikeDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use std::sync::Arc;

type Result<T> = std::result::Result<T, String>;

fn value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

/// Points and their targets, for symbolic regression.
#[pyclass(frozen, module = "genoxide._genoxide", name = "Sample")]
pub struct PySample {
    pub sample: Sample,
}

#[pymethods]
impl PySample {
    /// The points of `x`, a row each with a value per variable (or a 1-D array of one variable),
    /// and their targets `y`.
    #[new]
    fn new(
        x: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
        y: PyArrayLike1<'_, f64, AllowTypeChange>,
    ) -> PyResult<Self> {
        let array = x.as_array();
        let columns: Vec<Vec<f64>> = match array.ndim() {
            1 => vec![array.iter().copied().collect()],
            2 => array
                .columns()
                .into_iter()
                .map(|column| column.iter().copied().collect())
                .collect(),
            _ => {
                return Err(value_error(format!(
                    "x is a 2-D array, a point per row, or a 1-D array of one variable, not of \
                     shape {:?}",
                    array.shape()
                )));
            }
        };
        let targets = y.as_array().to_vec();
        let sample = setting(Sample::new(columns, targets)).map_err(value_error)?;
        Ok(Self { sample })
    }

    /// The points, a row each with a value per variable.
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let (points, columns) = (self.sample.points(), self.sample.columns());
        Array2::from_shape_fn((points, columns.len()), |(point, k)| columns[k][point])
            .into_pyarray(py)
    }

    /// The targets.
    #[getter]
    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.sample.targets())
    }

    /// The number of variables.
    #[getter]
    fn variables(&self) -> usize {
        self.sample.variables()
    }

    /// The number of points.
    #[getter]
    fn points(&self) -> usize {
        self.sample.points()
    }

    /// The standard deviation of the targets, divided by the number of points.
    fn deviation(&self) -> f64 {
        self.sample.deviation()
    }

    fn __len__(&self) -> usize {
        self.sample.points()
    }

    fn __repr__(&self) -> String {
        format!(
            "Sample({} points of {} variables)",
            self.sample.points(),
            self.sample.variables()
        )
    }
}

/// A training sample, and a test sample the search never sees.
#[pyclass(frozen, module = "genoxide._genoxide", name = "Dataset")]
pub struct PyDataset {
    pub dataset: Dataset,
}

#[pymethods]
impl PyDataset {
    #[new]
    #[pyo3(signature = (training, test = None))]
    fn new(training: &PySample, test: Option<&PySample>) -> PyResult<Self> {
        let dataset = Dataset::new(training.sample.clone());
        let dataset = match test {
            Some(test) => setting(dataset.with_test(test.sample.clone())).map_err(value_error)?,
            None => dataset,
        };
        Ok(Self { dataset })
    }

    /// The training sample.
    #[getter]
    fn training(&self) -> PySample {
        PySample {
            sample: self.dataset.training().clone(),
        }
    }

    /// The test sample, or None.
    #[getter]
    fn test(&self) -> Option<PySample> {
        self.dataset.test().map(|sample| PySample {
            sample: sample.clone(),
        })
    }

    /// The number of variables.
    #[getter]
    fn variables(&self) -> usize {
        self.dataset.variables()
    }

    fn __repr__(&self) -> String {
        let test = match self.dataset.test() {
            Some(test) => format!(", {} test points", test.points()),
            None => String::new(),
        };
        format!(
            "Dataset({} training points{test})",
            self.dataset.training().points()
        )
    }
}

/// Ephemeral random constants, as the Python package describes them.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ConstantsConfig {
    Uniform { low: f64, high: f64 },
    Integers { low: i64, high: i64 },
    Choice { values: Vec<f64> },
    Normal { mean: f64, deviation: f64 },
}

/// Ephemeral random constants of their JSON description, as `gx.gp.Constants` gives it.
pub fn parse_constants(description: &str) -> PyResult<Constants> {
    let config: ConstantsConfig = serde_json::from_str(description)
        .map_err(|error| value_error(format!("invalid setting `constants`: {error}")))?;
    setting(match config {
        ConstantsConfig::Uniform { low, high } => Constants::uniform(low..=high),
        ConstantsConfig::Integers { low, high } => Constants::integers(low..=high),
        ConstantsConfig::Choice { values } => Constants::choice(values),
        ConstantsConfig::Normal { mean, deviation } => Constants::normal(mean, deviation),
    })
    .map_err(value_error)
}

/// A primitive set of symbolic regression: the functions by name, the variables, and the
/// constants (JSON, as `gx.gp.Constants` describes them) or none.
#[pyfunction]
#[pyo3(signature = (functions, variables, constants = None))]
pub fn regression_primitives(
    functions: Vec<String>,
    variables: Vec<String>,
    constants: Option<&str>,
) -> PyResult<PyPrimitiveSet> {
    let functions = functions
        .iter()
        .map(|name| {
            Math::FUNCTIONS
                .into_iter()
                .find(|function| function.name() == name)
                .ok_or_else(|| {
                    let names: Vec<&str> = Math::FUNCTIONS.iter().map(|f| f.name()).collect();
                    value_error(format!(
                        "invalid setting `functions`: no function {name:?}; the functions are {}",
                        names.join(", ")
                    ))
                })
        })
        .collect::<PyResult<Vec<Math>>>()?;
    let constants = constants.map(parse_constants).transpose()?;
    let set =
        setting(regression::primitives(functions, variables, constants)).map_err(value_error)?;
    Ok(PyPrimitiveSet::new(from_math(&set)))
}

fn metric(name: &str) -> Result<Metric> {
    match name {
        "rmse" => Ok(Metric::Rmse),
        "mse" => Ok(Metric::Mse),
        "mae" => Ok(Metric::Mae),
        _ => Err(format!(
            "metric is \"rmse\", \"mse\" or \"mae\", not {name:?}"
        )),
    }
}

fn metric_name(metric: Metric) -> &'static str {
    match metric {
        Metric::Mse => "mse",
        Metric::Mae => "mae",
        _ => "rmse",
    }
}

/// The fitness of symbolic regression: the error of a tree on the training sample.
#[pyclass(frozen, module = "genoxide._genoxide", name = "Regression")]
pub struct PyRegression {
    pub regression: Regression,
    pub set: Arc<Set>,
    metric: Metric,
    linear_scaling: bool,
}

impl PyRegression {
    fn build(set: Arc<Set>, regression: Regression, metric: Metric, linear_scaling: bool) -> Self {
        Self {
            regression: regression.metric(metric).linear_scaling(linear_scaling),
            set,
            metric,
            linear_scaling,
        }
    }
}

#[pymethods]
impl PyRegression {
    #[new]
    #[pyo3(signature = (primitives, dataset, *, metric = "rmse", linear_scaling = true))]
    fn new(
        primitives: &PyPrimitiveSet,
        dataset: &PyDataset,
        metric: &str,
        linear_scaling: bool,
    ) -> PyResult<Self> {
        let metric = self::metric(metric).map_err(value_error)?;
        let math = to_math(&primitives.set).ok_or_else(|| {
            value_error(
                "Regression takes a primitive set of gx.gp.regression's functions".to_string(),
            )
        })?;
        let regression =
            setting(Regression::new(math, dataset.dataset.clone())).map_err(value_error)?;
        Ok(Self::build(
            Arc::clone(&primitives.set),
            regression,
            metric,
            linear_scaling,
        ))
    }

    /// The error on the training sample, None if a value isn't finite.
    fn evaluate(&self, py: Python<'_>, tree: &PyTree) -> PyResult<Option<f64>> {
        let tree = tree.of(&self.set)?;
        Ok(py.detach(|| self.regression.evaluate(tree)))
    }

    fn __call__(&self, py: Python<'_>, tree: &PyTree) -> PyResult<Option<f64>> {
        self.evaluate(py, tree)
    }

    /// The error on `sample`, e.g. the test sample, after the scaling fitted on the training
    /// sample; None if a prediction isn't finite.
    fn error(&self, py: Python<'_>, tree: &PyTree, sample: &PySample) -> PyResult<Option<f64>> {
        let tree = tree.of(&self.set)?;
        check_variables(&self.set, sample)?;
        Ok(py.detach(|| self.regression.error(tree, &sample.sample)))
    }

    /// The tree's values at the points of `sample`, before scaling.
    fn values<'py>(
        &self,
        py: Python<'py>,
        tree: &PyTree,
        sample: &PySample,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let tree = tree.of(&self.set)?;
        check_variables(&self.set, sample)?;
        let values = py.detach(|| self.regression.values(tree, &sample.sample));
        Ok(PyArray1::from_vec(py, values))
    }

    /// The tree's predictions at the points of `sample`, after the scaling; None if a value on
    /// the training sample isn't finite.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        tree: &PyTree,
        sample: &PySample,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let tree = tree.of(&self.set)?;
        check_variables(&self.set, sample)?;
        let predictions = py.detach(|| self.regression.predict(tree, &sample.sample));
        Ok(predictions.map(|values| PyArray1::from_vec(py, values)))
    }

    /// The linear scaling of the tree, `(intercept, slope)`, fitted on the training sample;
    /// `(0.0, 1.0)` without linear scaling; None if a value isn't finite.
    fn scaling(&self, py: Python<'_>, tree: &PyTree) -> PyResult<Option<(f64, f64)>> {
        let tree = tree.of(&self.set)?;
        let scaling = py.detach(|| self.regression.scaling(tree));
        Ok(scaling.map(|scaling| (scaling.intercept, scaling.slope)))
    }

    /// The tree as text, with its scaling if linear scaling is on:
    /// `intercept + slope * (expression)`.
    fn display(&self, tree: &PyTree) -> PyResult<String> {
        let tree = tree.of(&self.set)?;
        Ok(self.regression.display(tree))
    }

    /// The primitive set.
    #[getter]
    fn primitives(&self) -> PyPrimitiveSet {
        PyPrimitiveSet {
            set: Arc::clone(&self.set),
        }
    }

    /// The dataset.
    #[getter]
    fn dataset(&self) -> PyDataset {
        PyDataset {
            dataset: self.regression.dataset().clone(),
        }
    }

    /// The error measure: "rmse", "mse" or "mae".
    #[getter]
    fn metric(&self) -> &'static str {
        metric_name(self.metric)
    }

    /// Whether the error is the one after linear scaling.
    #[getter]
    fn linear_scaling(&self) -> bool {
        self.linear_scaling
    }

    fn __repr__(&self) -> String {
        format!(
            "Regression(metric={:?}, linear_scaling={})",
            metric_name(self.metric),
            if self.linear_scaling { "True" } else { "False" }
        )
    }
}

// a sample with the variables of the set
fn check_variables(set: &Set, sample: &PySample) -> PyResult<()> {
    let variables = set
        .primitives()
        .iter()
        .filter_map(|primitive| match primitive.value() {
            crate::trees::Op::Math(Math::Variable(k)) => Some(usize::from(k) + 1),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    if sample.sample.variables() < variables {
        return Err(value_error(format!(
            "the sample has {} variables, the primitive set {variables}",
            sample.sample.variables()
        )));
    }
    Ok(())
}

/// A test problem of symbolic regression, by its name, e.g. "Koza-1".
fn regression_problem(name: &str) -> Option<Box<dyn RegressionProblem + Send>> {
    Some(match name {
        "Koza-1" => Box::new(Koza1::new()),
        "Koza-2" => Box::new(Koza2::new()),
        "Koza-3" => Box::new(Koza3::new()),
        "Nguyen-1" => Box::new(Nguyen1::new()),
        "Nguyen-2" => Box::new(Nguyen2::new()),
        "Nguyen-3" => Box::new(Nguyen3::new()),
        "Nguyen-4" => Box::new(Nguyen4::new()),
        "Nguyen-5" => Box::new(Nguyen5::new()),
        "Nguyen-6" => Box::new(Nguyen6::new()),
        "Nguyen-7" => Box::new(Nguyen7::new()),
        "Nguyen-8" => Box::new(Nguyen8::new()),
        "Nguyen-9" => Box::new(Nguyen9::new()),
        "Nguyen-10" => Box::new(Nguyen10::new()),
        "Nguyen-11" => Box::new(Nguyen11::new()),
        "Nguyen-12" => Box::new(Nguyen12::new()),
        _ => return None,
    })
}

/// A test problem of symbolic regression of `gp::regression::problems`, by its name.
#[pyclass(
    frozen,
    subclass,
    module = "genoxide._genoxide",
    name = "RegressionProblem"
)]
pub struct PyRegressionProblem {
    problem: Box<dyn RegressionProblem + Send>,
    regression: PyRegression,
}

#[pymethods]
impl PyRegressionProblem {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let problem = regression_problem(name)
            .ok_or_else(|| value_error(format!("no regression problem named {name:?}")))?;
        let set = Arc::new(from_math(problem.primitives()));
        let regression = PyRegression::build(set, problem.regression().clone(), Metric::Rmse, true);
        Ok(Self {
            problem,
            regression,
        })
    }

    /// The problem's name, e.g. "Koza-1".
    #[getter]
    fn name(&self) -> &'static str {
        self.problem.name()
    }

    /// The target as a formula, e.g. "x^4 + x^3 + x^2 + x".
    #[getter]
    fn formula(&self) -> &'static str {
        self.problem.formula()
    }

    /// The paper that defines the problem.
    #[getter]
    fn reference(&self) -> &'static str {
        self.problem.reference()
    }

    /// Its DOI or URL, or None.
    #[getter]
    fn reference_url(&self) -> Option<&'static str> {
        self.problem.reference_url()
    }

    /// The target at a point, a value per variable.
    fn target(&self, point: Vec<f64>) -> PyResult<f64> {
        let variables = self.problem.dataset().variables();
        if point.len() != variables {
            return Err(value_error(format!(
                "a point of {} has {variables} values, not {}",
                self.problem.name(),
                point.len()
            )));
        }
        Ok(self.problem.target(&point))
    }

    /// The paper's primitive set.
    fn primitives(&self) -> PyPrimitiveSet {
        self.regression.primitives()
    }

    /// The training and test samples.
    fn dataset(&self) -> PyDataset {
        self.regression.dataset()
    }

    /// The fitness function: the error of a tree on the training sample, by `metric`, after
    /// linear scaling or not.
    #[pyo3(signature = (*, metric = "rmse", linear_scaling = true))]
    fn regression(&self, metric: &str, linear_scaling: bool) -> PyResult<PyRegression> {
        let metric = self::metric(metric).map_err(value_error)?;
        Ok(PyRegression::build(
            Arc::clone(&self.regression.set),
            self.problem.regression().clone(),
            metric,
            linear_scaling,
        ))
    }

    /// The RMSE after linear scaling on the training sample, None if a value isn't finite.
    fn __call__(&self, py: Python<'_>, tree: &PyTree) -> PyResult<Option<f64>> {
        self.regression.evaluate(py, tree)
    }

    fn __repr__(&self) -> String {
        format!("{}()", self.problem.name().replace('-', ""))
    }
}

/// A Boolean problem of `gp::boolean`.
#[derive(Clone, Debug)]
pub enum Boolean {
    Multiplexer(Multiplexer),
    EvenParity(EvenParity),
}

impl Boolean {
    fn errors(&self, tree: &Tree) -> u64 {
        match self {
            Boolean::Multiplexer(problem) => problem.errors(tree),
            Boolean::EvenParity(problem) => problem.errors(tree),
        }
    }
}

/// A Boolean problem of `gp::boolean`: Koza's multiplexer ("multiplexer", with its address
/// bits) or even parity ("even_parity", with its inputs).
#[pyclass(
    frozen,
    subclass,
    module = "genoxide._genoxide",
    name = "BooleanProblem"
)]
pub struct PyBooleanProblem {
    problem: Boolean,
    set: Arc<Set>,
}

#[pymethods]
impl PyBooleanProblem {
    #[new]
    fn new(kind: &str, size: usize) -> PyResult<Self> {
        let (problem, set) = match kind {
            "multiplexer" => {
                let problem = setting_named(
                    Multiplexer::new(size),
                    &[("address_bits", "Multiplexer.address_bits")],
                )
                .map_err(value_error)?;
                let set = from_logic(problem.primitives());
                (Boolean::Multiplexer(problem), set)
            }
            "even_parity" => {
                let problem =
                    setting_named(EvenParity::new(size), &[("inputs", "EvenParity.inputs")])
                        .map_err(value_error)?;
                let set = from_logic(problem.primitives());
                (Boolean::EvenParity(problem), set)
            }
            _ => return Err(value_error(format!("no Boolean problem {kind:?}"))),
        };
        Ok(Self {
            problem,
            set: Arc::new(set),
        })
    }

    /// The paper's primitive set.
    fn primitives(&self) -> PyPrimitiveSet {
        PyPrimitiveSet {
            set: Arc::clone(&self.set),
        }
    }

    /// The number of inputs.
    #[getter]
    fn inputs(&self) -> usize {
        match &self.problem {
            Boolean::Multiplexer(problem) => problem.inputs(),
            Boolean::EvenParity(problem) => problem.inputs(),
        }
    }

    /// The number of cases of the truth table, 2^inputs.
    #[getter]
    fn cases(&self) -> u64 {
        match &self.problem {
            Boolean::Multiplexer(problem) => problem.cases(),
            Boolean::EvenParity(problem) => problem.cases(),
        }
    }

    /// The multiplexer's address bits, None for even parity.
    #[getter]
    fn address_bits(&self) -> Option<usize> {
        match &self.problem {
            Boolean::Multiplexer(problem) => Some(problem.address_bits()),
            Boolean::EvenParity(_) => None,
        }
    }

    /// The source of the problem.
    #[getter]
    fn reference(&self) -> &'static str {
        match &self.problem {
            Boolean::Multiplexer(problem) => problem.reference(),
            Boolean::EvenParity(problem) => problem.reference(),
        }
    }

    /// The right outputs, 64 cases per word: case `c` at bit `c % 64` of word `c / 64`.
    fn targets(&self) -> Vec<u64> {
        match &self.problem {
            Boolean::Multiplexer(problem) => problem.targets().to_vec(),
            Boolean::EvenParity(problem) => problem.targets().to_vec(),
        }
    }

    /// The tree's outputs, as `targets` lays them out.
    fn outputs(&self, tree: &PyTree) -> PyResult<Vec<u64>> {
        let tree = tree.of(&self.set)?;
        Ok(match &self.problem {
            Boolean::Multiplexer(problem) => problem.outputs(tree),
            Boolean::EvenParity(problem) => problem.outputs(tree),
        })
    }

    /// The number of cases the tree gets wrong: the fitness, 0 at the optimum.
    fn errors(&self, py: Python<'_>, tree: &PyTree) -> PyResult<u64> {
        let tree = tree.of(&self.set)?;
        Ok(py.detach(|| self.problem.errors(tree)))
    }

    fn __call__(&self, py: Python<'_>, tree: &PyTree) -> PyResult<f64> {
        Ok(self.errors(py, tree)? as f64)
    }
}

/// What a run of trees evaluates in Rust: symbolic regression or a Boolean problem, minimized,
/// and with `size`, the tree's number of nodes as a second objective, minimized too.
#[derive(Clone)]
pub struct TreeFitness {
    problem: TreeProblem,
    set: Arc<Set>,
    size: bool,
    name: &'static str,
}

#[derive(Clone)]
enum TreeProblem {
    Regression(Regression),
    Boolean(Boolean),
}

impl TreeFitness {
    /// The tree fitness that `object` is, if it's one: a `Regression`, a regression or Boolean
    /// problem, or `WithSize` of one.
    pub fn from_object(object: &Bound<'_, PyAny>) -> Option<Self> {
        if let Ok(regression) = object.cast::<PyRegression>() {
            let regression = regression.get();
            return Some(Self {
                problem: TreeProblem::Regression(regression.regression.clone()),
                set: Arc::clone(&regression.set),
                size: false,
                name: "Regression",
            });
        }
        if let Ok(problem) = object.cast::<PyRegressionProblem>() {
            let regression = &problem.get().regression;
            return Some(Self {
                problem: TreeProblem::Regression(regression.regression.clone()),
                set: Arc::clone(&regression.set),
                size: false,
                name: problem.get().problem.name(),
            });
        }
        if let Ok(problem) = object.cast::<PyBooleanProblem>() {
            let problem = problem.get();
            return Some(Self {
                problem: TreeProblem::Boolean(problem.problem.clone()),
                set: Arc::clone(&problem.set),
                size: false,
                name: match problem.problem {
                    Boolean::Multiplexer(_) => "Multiplexer",
                    Boolean::EvenParity(_) => "EvenParity",
                },
            });
        }
        if let Ok(with_size) = object.cast::<PyWithSize>() {
            return Some(with_size.get().fitness.clone());
        }
        None
    }

    /// The fitness of a tree: the error, or the cases wrong; None for an invalid tree.
    pub fn evaluate(&self, tree: &Tree) -> Option<f64> {
        match &self.problem {
            TreeProblem::Regression(regression) => regression.evaluate(tree),
            TreeProblem::Boolean(problem) => Some(problem.errors(tree) as f64),
        }
    }

    /// The objective values of a tree: the fitness, then the size with `WithSize`.
    pub fn values(&self, tree: &Tree) -> Option<Vec<f64>> {
        let fitness = self.evaluate(tree)?;
        Some(if self.size {
            vec![fitness, tree.len() as f64]
        } else {
            vec![fitness]
        })
    }

    /// Checks that the run's trees are of the fitness's primitive set, and its objectives the
    /// fitness's, minimized.
    pub fn check(&self, set: &Set, objectives: &[Objective]) -> Result<()> {
        let name = self.name;
        if set != self.set.as_ref() {
            return Err(format!(
                "{name} evaluates trees of its own primitive set: build the Gp from its primitives"
            ));
        }
        let count = if self.size { 2 } else { 1 };
        if objectives.len() != count {
            return Err(if self.size {
                format!(
                    "WithSize({name}) has 2 objectives, the error and the size: use Nsga2 with 2 \
                     objectives"
                )
            } else {
                format!("{name} has one objective: use a single-objective algorithm, or WithSize")
            });
        }
        if objectives.contains(&Objective::Maximize) {
            return Err(if self.size {
                format!(
                    "WithSize({name}) minimizes the error and the size: pass objectives=[\"minimize\", \"minimize\"]"
                )
            } else {
                format!("{name} minimizes its error: pass objective=\"minimize\"")
            });
        }
        Ok(())
    }
}

/// A fitness of trees with the size as a second objective: `(fitness, number of nodes)`, both
/// minimized, for NSGA-II.
#[pyclass(frozen, module = "genoxide._genoxide", name = "WithSize")]
pub struct PyWithSize {
    fitness: TreeFitness,
    inner: Py<PyAny>,
}

#[pymethods]
impl PyWithSize {
    #[new]
    fn new(fitness: &Bound<'_, PyAny>) -> PyResult<Self> {
        let inner = match TreeFitness::from_object(fitness) {
            Some(inner) if !inner.size => inner,
            _ => {
                return Err(value_error(format!(
                    "WithSize takes a gx.gp.regression.Regression, a regression problem or a \
                     Boolean problem, not {}",
                    fitness.repr()?
                )));
            }
        };
        Ok(Self {
            fitness: TreeFitness {
                size: true,
                ..inner
            },
            inner: fitness.clone().unbind(),
        })
    }

    /// The fitness whose objectives are the first.
    #[getter]
    fn fitness(&self, py: Python<'_>) -> Py<PyAny> {
        self.inner.clone_ref(py)
    }

    /// The objective values of a tree, `(fitness, size)`, or None for an invalid tree.
    fn __call__(&self, py: Python<'_>, tree: &PyTree) -> PyResult<Option<(f64, f64)>> {
        let tree = tree.of(&self.fitness.set)?;
        let values = py.detach(|| self.fitness.values(tree));
        Ok(values.map(|values| (values[0], values[1])))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("WithSize({})", self.inner.bind(py).repr()?))
    }
}
