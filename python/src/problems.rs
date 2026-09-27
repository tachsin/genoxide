//! The test problems of `genoxide::problems` and `genoxide::multi::problems`, as the Python
//! package describes them in JSON: their description, their evaluation, and the problem of a run
//! evaluated in Rust.

use crate::run::{WithObjectives, with_objectives};
use genoxide::engine::{FitnessFunction, IntoFitness};
use genoxide::genome::{Integer, Integers, Real, Reals, Representation};
use genoxide::multi::problems::{self as multi, DynMultiProblem, try_boxed};
use genoxide::problems::{
    self, Constraints, DynProblem, Optimum, Problem as _, cec2006, engineering,
};
use genoxide::{Fitness, Objective};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Deserialize;

/// A problem, as `_describe()` of a `gx.problems` class gives it.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Config {
    Sphere {
        dimensions: usize,
    },
    AxisParallelEllipsoid {
        dimensions: usize,
    },
    #[serde(rename = "schwefel_1_2")]
    Schwefel12 {
        dimensions: usize,
    },
    Rastrigin {
        dimensions: usize,
    },
    Rosenbrock {
        dimensions: usize,
    },
    Ackley {
        dimensions: usize,
    },
    Griewank {
        dimensions: usize,
    },
    #[serde(rename = "schwefel_2_26")]
    Schwefel226 {
        dimensions: usize,
    },
    Levy {
        dimensions: usize,
    },
    Zakharov {
        dimensions: usize,
    },
    StyblinskiTang {
        dimensions: usize,
    },
    Michalewicz {
        dimensions: usize,
    },
    Himmelblau {},
    Branin {},
    GoldsteinPrice {},
    SixHumpCamel {},
    G01 {},
    G02 {},
    G03 {
        tolerance: Option<f64>,
    },
    G04 {},
    G05 {
        tolerance: Option<f64>,
    },
    G06 {},
    WeldedBeam {},
    WeldedBeamRagsdell {},
    PressureVessel {},
    TensionCompressionSpring {},
    SpeedReducer {},
    GearTrain {},
    ThreeBarTruss {},
    CantileverBeam {},
    CarSideImpact {},
    #[serde(untagged)]
    Multi(MultiConfig),
}

/// A multi-objective problem, as `_describe()` of a `gx.problems` class gives it.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum MultiConfig {
    Zdt1 {
        variables: usize,
    },
    Zdt2 {
        variables: usize,
    },
    Zdt3 {
        variables: usize,
    },
    Zdt4 {
        variables: usize,
    },
    Zdt6 {
        variables: usize,
    },
    Dtlz1 {
        objectives: usize,
        variables: Option<usize>,
    },
    Dtlz2 {
        objectives: usize,
        variables: Option<usize>,
    },
    Dtlz3 {
        objectives: usize,
        variables: Option<usize>,
    },
    Dtlz4 {
        objectives: usize,
        variables: Option<usize>,
    },
    Schaffer1 {},
    Schaffer2 {},
    FonsecaFleming {
        variables: usize,
    },
    Kursawe {
        variables: usize,
    },
    Poloni {},
    Viennet1 {},
    Viennet2 {},
    Viennet3 {},
    Bnh {},
    Srn {},
    Tnk {},
    Osy {},
    Constr {},
}

// a size of at least `minimum`, or an error that names it; the constructors panic instead
fn at_least(value: usize, minimum: usize, name: &str, what: &str) -> Result<usize, String> {
    if value >= minimum {
        Ok(value)
    } else {
        Err(format!(
            "{name} needs at least {minimum} {what}, not {value}"
        ))
    }
}

impl MultiConfig {
    /// The number of objectives.
    pub fn objectives(&self) -> usize {
        match *self {
            Self::Dtlz1 { objectives, .. }
            | Self::Dtlz2 { objectives, .. }
            | Self::Dtlz3 { objectives, .. }
            | Self::Dtlz4 { objectives, .. } => objectives,
            Self::Viennet1 {} | Self::Viennet2 {} | Self::Viennet3 {} => 3,
            _ => 2,
        }
    }

    // the sizes, checked: an error, not the panic of the constructor
    fn check(&self) -> Result<(), String> {
        let variables = |variables: usize, minimum: usize, name: &str| {
            at_least(variables, minimum, name, "variables").map(|_| ())
        };
        match *self {
            Self::Zdt1 { variables: n } => variables(n, 2, "ZDT1"),
            Self::Zdt2 { variables: n } => variables(n, 2, "ZDT2"),
            Self::Zdt3 { variables: n } => variables(n, 2, "ZDT3"),
            Self::Zdt4 { variables: n } => variables(n, 2, "ZDT4"),
            Self::Zdt6 { variables: n } => variables(n, 2, "ZDT6"),
            Self::FonsecaFleming { variables: n } => variables(n, 1, "FON"),
            Self::Kursawe { variables: n } => variables(n, 2, "KUR"),
            Self::Dtlz1 {
                objectives,
                variables: n,
            }
            | Self::Dtlz2 {
                objectives,
                variables: n,
            }
            | Self::Dtlz3 {
                objectives,
                variables: n,
            }
            | Self::Dtlz4 {
                objectives,
                variables: n,
            } => {
                if !(2..=6).contains(&objectives) {
                    return Err(format!(
                        "DTLZ takes 2 to 6 objectives in Python, not {objectives}"
                    ));
                }
                match n {
                    Some(n) => variables(n, objectives, "DTLZ with this many objectives"),
                    None => Ok(()),
                }
            }
            _ => Ok(()),
        }
    }

    /// The problem, with `M` objectives: its number of objectives, checked before.
    pub fn build<const M: usize>(&self) -> Box<dyn DynMultiProblem<M>> {
        let problem = match *self {
            Self::Zdt1 { variables } => try_boxed::<_, 2, M>(multi::Zdt1::new(variables)),
            Self::Zdt2 { variables } => try_boxed::<_, 2, M>(multi::Zdt2::new(variables)),
            Self::Zdt3 { variables } => try_boxed::<_, 2, M>(multi::Zdt3::new(variables)),
            Self::Zdt4 { variables } => try_boxed::<_, 2, M>(multi::Zdt4::new(variables)),
            Self::Zdt6 { variables } => try_boxed::<_, 2, M>(multi::Zdt6::new(variables)),
            Self::Dtlz1 { variables, .. } => Some(multi::boxed(
                variables.map_or_else(multi::Dtlz1::<M>::default, multi::Dtlz1::<M>::new),
            )),
            Self::Dtlz2 { variables, .. } => Some(multi::boxed(
                variables.map_or_else(multi::Dtlz2::<M>::default, multi::Dtlz2::<M>::new),
            )),
            Self::Dtlz3 { variables, .. } => Some(multi::boxed(
                variables.map_or_else(multi::Dtlz3::<M>::default, multi::Dtlz3::<M>::new),
            )),
            Self::Dtlz4 { variables, .. } => Some(multi::boxed(
                variables.map_or_else(multi::Dtlz4::<M>::default, multi::Dtlz4::<M>::new),
            )),
            Self::Schaffer1 {} => try_boxed::<_, 2, M>(multi::Schaffer1),
            Self::Schaffer2 {} => try_boxed::<_, 2, M>(multi::Schaffer2),
            Self::FonsecaFleming { variables } => {
                try_boxed::<_, 2, M>(multi::FonsecaFleming::new(variables))
            }
            Self::Kursawe { variables } => try_boxed::<_, 2, M>(multi::Kursawe::new(variables)),
            Self::Poloni {} => try_boxed::<_, 2, M>(multi::Poloni),
            Self::Viennet1 {} => try_boxed::<_, 3, M>(multi::Viennet1),
            Self::Viennet2 {} => try_boxed::<_, 3, M>(multi::Viennet2),
            Self::Viennet3 {} => try_boxed::<_, 3, M>(multi::Viennet3),
            Self::Bnh {} => try_boxed::<_, 2, M>(multi::Bnh),
            Self::Srn {} => try_boxed::<_, 2, M>(multi::Srn),
            Self::Tnk {} => try_boxed::<_, 2, M>(multi::Tnk),
            Self::Osy {} => try_boxed::<_, 2, M>(multi::Osy),
            Self::Constr {} => try_boxed::<_, 2, M>(multi::Constr),
        };
        problem.expect("the number of objectives is the problem's")
    }
}

/// The name and the number of variables of the problem that `config` describes.
pub fn name_and_dimensions(config: MultiConfig) -> (&'static str, usize) {
    struct Describe(MultiConfig);

    impl WithObjectives for Describe {
        type Output = (&'static str, usize);

        fn with<const N: usize>(self) -> (&'static str, usize) {
            let problem = self.0.build::<N>();
            (problem.name(), problem.real().genome_len())
        }
    }

    with_objectives(config.objectives(), Describe(config)).expect("checked")
}

/// A problem that a run evaluates in Rust.
pub enum Problem {
    Single(Box<dyn DynProblem>),
    /// A single-objective problem on integer genomes.
    Integer(Box<dyn IntegerProblem>),
    Multi(MultiConfig),
}

/// A single-objective problem on [`Integer`] genomes, as a trait object: what [`DynProblem`] is
/// for real genomes.
pub trait IntegerProblem: Send + Sync {
    fn name(&self) -> &'static str;
    fn integer(&self) -> Integer;
    fn objective(&self) -> Objective;
    fn evaluate(&self, genome: &Integers) -> Fitness;
    fn optimum(&self) -> Option<Optimum<Integers>>;
    fn reference(&self) -> &'static str;
    fn reference_url(&self) -> Option<&'static str>;
}

impl<P> IntegerProblem for P
where
    P: problems::Problem<Representation = Integer> + Send + Sync,
{
    fn name(&self) -> &'static str {
        problems::Problem::name(self)
    }

    fn integer(&self) -> Integer {
        self.representation()
    }

    fn objective(&self) -> Objective {
        problems::Problem::objective(self)
    }

    fn evaluate(&self, genome: &Integers) -> Fitness {
        FitnessFunction::evaluate(self, genome)
            .into_fitness()
            .unwrap_or_else(|_| Fitness::invalid())
    }

    fn optimum(&self) -> Option<Optimum<Integers>> {
        problems::Problem::optimum(self)
    }

    fn reference(&self) -> &'static str {
        problems::Problem::reference(self)
    }

    fn reference_url(&self) -> Option<&'static str> {
        problems::Problem::reference_url(self)
    }
}

// the problem that `config` describes; a size below the minimum is an error, not the panic of
// the constructor
fn build(config: Config) -> Result<Problem, String> {
    let at_least = |dimensions: usize, minimum: usize, name: &str| {
        at_least(dimensions, minimum, name, "dimensions")
    };
    Ok(Problem::Single(match config {
        Config::Sphere { dimensions } => {
            problems::boxed(problems::Sphere::new(at_least(dimensions, 1, "Sphere")?))
        }
        Config::AxisParallelEllipsoid { dimensions } => problems::boxed(
            problems::AxisParallelEllipsoid::new(at_least(dimensions, 1, "AxisParallelEllipsoid")?),
        ),
        Config::Schwefel12 { dimensions } => problems::boxed(problems::Schwefel1_2::new(at_least(
            dimensions,
            1,
            "Schwefel1_2",
        )?)),
        Config::Rastrigin { dimensions } => problems::boxed(problems::Rastrigin::new(at_least(
            dimensions,
            1,
            "Rastrigin",
        )?)),
        Config::Rosenbrock { dimensions } => problems::boxed(problems::Rosenbrock::new(at_least(
            dimensions,
            2,
            "Rosenbrock",
        )?)),
        Config::Ackley { dimensions } => {
            problems::boxed(problems::Ackley::new(at_least(dimensions, 1, "Ackley")?))
        }
        Config::Griewank { dimensions } => problems::boxed(problems::Griewank::new(at_least(
            dimensions, 1, "Griewank",
        )?)),
        Config::Schwefel226 { dimensions } => problems::boxed(problems::Schwefel2_26::new(
            at_least(dimensions, 1, "Schwefel2_26")?,
        )),
        Config::Levy { dimensions } => {
            problems::boxed(problems::Levy::new(at_least(dimensions, 1, "Levy")?))
        }
        Config::Zakharov { dimensions } => problems::boxed(problems::Zakharov::new(at_least(
            dimensions, 1, "Zakharov",
        )?)),
        Config::StyblinskiTang { dimensions } => problems::boxed(problems::StyblinskiTang::new(
            at_least(dimensions, 1, "StyblinskiTang")?,
        )),
        Config::Michalewicz { dimensions } => problems::boxed(problems::Michalewicz::new(
            at_least(dimensions, 1, "Michalewicz")?,
        )),
        Config::Himmelblau {} => problems::boxed(problems::Himmelblau),
        Config::Branin {} => problems::boxed(problems::Branin),
        Config::GoldsteinPrice {} => problems::boxed(problems::GoldsteinPrice),
        Config::SixHumpCamel {} => problems::boxed(problems::SixHumpCamel),
        Config::G01 {} => problems::boxed(cec2006::G01),
        Config::G02 {} => problems::boxed(cec2006::G02),
        Config::G03 { tolerance } => {
            problems::boxed(cec2006::G03::with_tolerance(equality_tolerance(tolerance)?))
        }
        Config::G04 {} => problems::boxed(cec2006::G04),
        Config::G05 { tolerance } => {
            problems::boxed(cec2006::G05::with_tolerance(equality_tolerance(tolerance)?))
        }
        Config::G06 {} => problems::boxed(cec2006::G06),
        Config::WeldedBeam {} => problems::boxed(engineering::WeldedBeam),
        Config::WeldedBeamRagsdell {} => problems::boxed(engineering::WeldedBeamRagsdell),
        Config::PressureVessel {} => problems::boxed(engineering::PressureVessel),
        Config::TensionCompressionSpring {} => {
            problems::boxed(engineering::TensionCompressionSpring)
        }
        Config::SpeedReducer {} => problems::boxed(engineering::SpeedReducer),
        Config::GearTrain {} => return Ok(Problem::Integer(Box::new(engineering::GearTrain))),
        Config::ThreeBarTruss {} => problems::boxed(engineering::ThreeBarTruss),
        Config::CantileverBeam {} => problems::boxed(engineering::CantileverBeam),
        Config::CarSideImpact {} => problems::boxed(engineering::CarSideImpact),
        Config::Multi(config) => {
            config.check()?;
            return Ok(Problem::Multi(config));
        }
    }))
}

// an equality tolerance: the report's by default, and finite and at least 0 otherwise
fn equality_tolerance(tolerance: Option<f64>) -> Result<f64, String> {
    match tolerance {
        None => Ok(cec2006::EQUALITY_TOLERANCE),
        Some(tolerance) if tolerance.is_finite() && tolerance >= 0.0 => Ok(tolerance),
        Some(tolerance) => Err(format!(
            "the equality tolerance is a finite number of at least 0, not {tolerance}"
        )),
    }
}

/// The problem that `description` (JSON) describes.
pub fn parse(description: &str) -> PyResult<Problem> {
    let mut json = serde_json::Deserializer::from_str(description);
    let config: Config = serde_path_to_error::deserialize(&mut json).map_err(|error| {
        let (path, error) = (error.path().to_string(), error.into_inner());
        PyValueError::new_err(format!("invalid problem `{path}`: {error}"))
    })?;
    build(config).map_err(PyValueError::new_err)
}

// a genome per row, as numpy arrays of the problem's dimensions
fn rows(name: &str, real: &Real, genomes: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Reals>> {
    let genomes = genomes.as_array();
    let dimensions = real.genome_len();
    if genomes.ncols() != dimensions {
        return Err(PyValueError::new_err(format!(
            "{name} takes genomes of {dimensions} genes, not {}",
            genomes.ncols()
        )));
    }
    Ok(genomes
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().collect())
        .collect())
}

fn bounds(real: &Real) -> Vec<(f64, f64)> {
    real.bounds()
        .iter()
        .map(|range| (*range.start(), *range.end()))
        .collect()
}

// the rows of points of M values, as a 2-D numpy array
fn matrix<'py, const M: usize>(py: Python<'py>, points: &[[f64; M]]) -> Bound<'py, PyAny> {
    let values: Vec<f64> = points.iter().flatten().copied().collect();
    Array2::from_shape_vec((points.len(), M), values)
        .expect("a row per point")
        .into_pyarray(py)
        .into_any()
}

/// The description of the problem that `problem` (JSON) describes: its name, bounds (a pair per
/// gene), objectives ("minimize" or "maximize" each), number of constraints, reference and the
/// reference's URL; for a single-objective problem, its optimum (None, or its value, solutions a
/// row each, and whether it's proven); for a multi-objective one, its ideal and nadir points
/// (None if unknown).
#[pyfunction]
pub fn problem_info<'py>(py: Python<'py>, problem: &str) -> PyResult<Bound<'py, PyDict>> {
    let info = PyDict::new(py);
    match parse(problem)? {
        Problem::Single(problem) => {
            info.set_item("name", problem.name())?;
            info.set_item("genome", "real")?;
            info.set_item("bounds", bounds(&problem.real()))?;
            let objective = match problem.objective() {
                Objective::Maximize => "maximize",
                Objective::Minimize => "minimize",
            };
            info.set_item("objectives", vec![objective])?;
            let dimensions = problem.real().genome_len();
            info.set_item(
                "constraints",
                problem.constraints(&zeros(&problem.real())).len(),
            )?;
            match problem.optimum() {
                None => info.set_item("optimum", py.None())?,
                Some(optimum) => {
                    let solutions = optimum.solutions();
                    let genes: Vec<f64> =
                        solutions.iter().flat_map(|x| x.iter().copied()).collect();
                    let solutions = Array2::from_shape_vec((solutions.len(), dimensions), genes)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?;
                    let description = PyDict::new(py);
                    description.set_item("value", optimum.value())?;
                    description.set_item("solutions", solutions.into_pyarray(py))?;
                    description.set_item("proven", optimum.is_proven())?;
                    info.set_item("optimum", description)?;
                }
            }
            info.set_item("reference", problem.reference())?;
            info.set_item("reference_url", problem.reference_url())?;
        }
        Problem::Integer(problem) => {
            let integer = problem.integer();
            info.set_item("name", problem.name())?;
            info.set_item("genome", "integer")?;
            let pairs: Vec<(i64, i64)> = integer
                .bounds()
                .iter()
                .map(|range| (*range.start(), *range.end()))
                .collect();
            info.set_item("bounds", pairs)?;
            let objective = match problem.objective() {
                Objective::Maximize => "maximize",
                Objective::Minimize => "minimize",
            };
            info.set_item("objectives", vec![objective])?;
            info.set_item("constraints", 0)?;
            match problem.optimum() {
                None => info.set_item("optimum", py.None())?,
                Some(optimum) => {
                    let solutions = optimum.solutions();
                    let genes: Vec<i64> =
                        solutions.iter().flat_map(|x| x.iter().copied()).collect();
                    let solutions =
                        Array2::from_shape_vec((solutions.len(), integer.genome_len()), genes)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?;
                    let description = PyDict::new(py);
                    description.set_item("value", optimum.value())?;
                    description.set_item("solutions", solutions.into_pyarray(py))?;
                    description.set_item("proven", optimum.is_proven())?;
                    info.set_item("optimum", description)?;
                }
            }
            info.set_item("reference", problem.reference())?;
            info.set_item("reference_url", problem.reference_url())?;
        }
        Problem::Multi(config) => {
            let task = MultiInfo {
                py,
                config,
                info: &info,
            };
            run_with(config, task)??;
        }
    }
    Ok(info)
}

// runs `task` with the problem's number of objectives
fn run_with<T: WithObjectives>(config: MultiConfig, task: T) -> PyResult<T::Output> {
    with_objectives(config.objectives(), task).map_err(|count| {
        PyValueError::new_err(format!("problems take 2 to 6 objectives, not {count}"))
    })
}

struct MultiInfo<'a, 'py> {
    py: Python<'py>,
    config: MultiConfig,
    info: &'a Bound<'py, PyDict>,
}

impl WithObjectives for MultiInfo<'_, '_> {
    type Output = PyResult<()>;

    fn with<const N: usize>(self) -> PyResult<()> {
        let problem = self.config.build::<N>();
        let info = self.info;
        info.set_item("name", problem.name())?;
        info.set_item("genome", "real")?;
        info.set_item("bounds", bounds(&problem.real()))?;
        info.set_item("objectives", vec!["minimize"; N])?;
        info.set_item("constraints", problem.constraint_count())?;
        let point = |point: Option<[f64; N]>| point.map(|point| point.to_vec());
        info.set_item("ideal_point", point(problem.ideal_point()))?;
        info.set_item("nadir_point", point(problem.nadir_point()))?;
        info.set_item("optimum", self.py.None())?;
        info.set_item("reference", problem.reference())?;
        info.set_item("reference_url", problem.reference_url())?;
        Ok(())
    }
}

/// The values of `genomes`, a row each, for the problem that `problem` (JSON) describes: for a
/// single-objective problem, an array of scores; for a multi-objective one, a 2-D array with a
/// row of objective values per genome; and with constraints, a tuple of that and an array of
/// constraint violations. NaN for an invalid value.
#[pyfunction]
pub fn evaluate<'py>(
    py: Python<'py>,
    problem: &str,
    genomes: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyAny>> {
    match parse(problem)? {
        Problem::Single(problem) => {
            let genomes = rows(problem.name(), &problem.real(), &genomes)?;
            let constrained = !problem.constraints(&zeros(&problem.real())).is_empty();
            let (scores, violations): (Vec<f64>, Vec<f64>) = py.detach(|| {
                genomes
                    .iter()
                    .map(|genome| {
                        let fitness = problem.evaluate(genome);
                        match fitness.score() {
                            Some(score) => (score, fitness.violation()),
                            None => (f64::NAN, f64::NAN),
                        }
                    })
                    .unzip()
            });
            let scores = PyArray1::from_vec(py, scores).into_any();
            if constrained {
                (scores, PyArray1::from_vec(py, violations)).into_bound_py_any(py)
            } else {
                Ok(scores)
            }
        }
        Problem::Integer(problem) => {
            let genomes = integer_rows(problem.name(), &problem.integer(), &genomes)?;
            let scores: Vec<f64> = py.detach(|| {
                genomes
                    .iter()
                    .map(|genome| problem.evaluate(genome).score().unwrap_or(f64::NAN))
                    .collect()
            });
            Ok(PyArray1::from_vec(py, scores).into_any())
        }
        Problem::Multi(config) => run_with(
            config,
            MultiEvaluate {
                py,
                config,
                genomes,
            },
        )?,
    }
}

// a genome per row, as numpy arrays of the problem's dimensions whose values are whole numbers
fn integer_rows(
    name: &str,
    integer: &Integer,
    genomes: &PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<Integers>> {
    let genomes = genomes.as_array();
    let dimensions = integer.genome_len();
    if genomes.ncols() != dimensions {
        return Err(PyValueError::new_err(format!(
            "{name} takes genomes of {dimensions} genes, not {}",
            genomes.ncols()
        )));
    }
    genomes
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .map(|&gene| whole(name, gene))
                .collect::<PyResult<Integers>>()
        })
        .collect()
}

// a gene of an integer genome: a whole number that fits in an i64
fn whole(name: &str, gene: f64) -> PyResult<i64> {
    // −2^63 is the least value that fits, and 2^63 the first that doesn't
    const LIMIT: f64 = 9_223_372_036_854_775_808.0;
    if gene.fract() == 0.0 && (-LIMIT..LIMIT).contains(&gene) {
        Ok(gene as i64)
    } else {
        Err(PyValueError::new_err(format!(
            "{name} takes whole numbers as genes, not {gene}"
        )))
    }
}

// a genome of zeros for the problem's bounds, to count its constraints
fn zeros(real: &Real) -> Reals {
    Reals::from(vec![0.0; real.genome_len()])
}

struct MultiEvaluate<'py> {
    py: Python<'py>,
    config: MultiConfig,
    genomes: PyReadonlyArray2<'py, f64>,
}

impl<'py> WithObjectives for MultiEvaluate<'py> {
    type Output = PyResult<Bound<'py, PyAny>>;

    fn with<const N: usize>(self) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let problem = self.config.build::<N>();
        let genomes = rows(problem.name(), &problem.real(), &self.genomes)?;
        let (values, violations): (Vec<[f64; N]>, Vec<f64>) = py.detach(|| {
            genomes
                .iter()
                .map(|genome| {
                    let scores = problem.evaluate(genome);
                    match scores.values() {
                        Some(values) => (values, scores.violation()),
                        None => ([f64::NAN; N], f64::NAN),
                    }
                })
                .unzip()
        });
        let values = matrix(py, &values);
        if problem.constraint_count() > 0 {
            (values, PyArray1::from_vec(py, violations)).into_bound_py_any(py)
        } else {
            Ok(values)
        }
    }
}

/// The constraint values of `genome` for the problem that `problem` (JSON) describes: the
/// inequalities `g(x) <= 0`, then the equalities `h(x) = 0`.
#[pyfunction]
pub fn constraints<'py>(
    py: Python<'py>,
    problem: &str,
    genome: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let genome: Reals = genome.as_array().iter().copied().collect();
    let constraints = match parse(problem)? {
        Problem::Single(problem) => {
            check_length(problem.name(), &problem.real(), &genome)?;
            problem.constraints(&genome)
        }
        Problem::Integer(problem) => {
            let dimensions = problem.integer().genome_len();
            if genome.len() != dimensions {
                return Err(PyValueError::new_err(format!(
                    "{} takes genomes of {dimensions} genes, not {}",
                    problem.name(),
                    genome.len()
                )));
            }
            Constraints::none()
        }
        Problem::Multi(config) => run_with(
            config,
            MultiConstraints {
                config,
                genome: &genome,
            },
        )??,
    };
    let mut values = constraints.inequalities().to_vec();
    values.extend_from_slice(constraints.equalities());
    Ok(PyArray1::from_vec(py, values))
}

fn check_length(name: &str, real: &Real, genome: &Reals) -> PyResult<()> {
    let dimensions = real.genome_len();
    if genome.len() == dimensions {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} takes genomes of {dimensions} genes, not {}",
            genome.len()
        )))
    }
}

struct MultiConstraints<'a> {
    config: MultiConfig,
    genome: &'a Reals,
}

impl WithObjectives for MultiConstraints<'_> {
    type Output = PyResult<Constraints>;

    fn with<const N: usize>(self) -> PyResult<Constraints> {
        let problem = self.config.build::<N>();
        check_length(problem.name(), &problem.real(), self.genome)?;
        Ok(problem.constraints(self.genome))
    }
}

/// At least `points` points of the optimal front of the problem that `problem` (JSON) describes,
/// a row each, or None if the front isn't known.
#[pyfunction]
pub fn optimal_front<'py>(
    py: Python<'py>,
    problem: &str,
    points: usize,
) -> PyResult<Bound<'py, PyAny>> {
    // checked before allocating: a failed allocation would abort the interpreter
    if points as u128 > crate::run::MAX_POINTS {
        return Err(PyValueError::new_err(format!(
            "optimal_front takes at most 2^24 points, not {points}"
        )));
    }
    match parse(problem)? {
        Problem::Single(problem) => Err(PyValueError::new_err(format!(
            "{} has one objective, and an optimum instead of a front",
            problem.name()
        ))),
        Problem::Integer(problem) => Err(PyValueError::new_err(format!(
            "{} has one objective, and an optimum instead of a front",
            problem.name()
        ))),
        Problem::Multi(config) => run_with(config, MultiFront { py, config, points })?,
    }
}

struct MultiFront<'py> {
    py: Python<'py>,
    config: MultiConfig,
    points: usize,
}

impl<'py> WithObjectives for MultiFront<'py> {
    type Output = PyResult<Bound<'py, PyAny>>;

    fn with<const N: usize>(self) -> PyResult<Bound<'py, PyAny>> {
        let problem = self.config.build::<N>();
        let front = self.py.detach(|| problem.optimal_front(self.points));
        Ok(match front {
            Some(front) => matrix(self.py, &front),
            None => self.py.None().into_bound(self.py),
        })
    }
}

/// The design variables of `genome` for a problem with discrete variables that its genome rounds:
/// the pressure vessel and the speed reducer.
#[pyfunction]
pub fn design<'py>(
    py: Python<'py>,
    problem: &str,
    genome: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let genome: Reals = genome.as_array().iter().copied().collect();
    let mut json = serde_json::Deserializer::from_str(problem);
    let config: Config = serde_path_to_error::deserialize(&mut json)
        .map_err(|error| PyValueError::new_err(format!("invalid problem: {error}")))?;
    let design = match config {
        Config::PressureVessel {} => {
            let problem = engineering::PressureVessel;
            check_length(problem.name(), &problem.representation(), &genome)?;
            problem.design(&genome).to_vec()
        }
        Config::SpeedReducer {} => {
            let problem = engineering::SpeedReducer;
            check_length(problem.name(), &problem.representation(), &genome)?;
            problem.design(&genome).to_vec()
        }
        _ => {
            return Err(PyValueError::new_err(
                "only the problems with discrete variables have a design",
            ));
        }
    };
    Ok(PyArray1::from_vec(py, design))
}

/// The names of the problems of `genoxide::problems::all()`, in its order.
#[pyfunction]
pub fn problem_names() -> Vec<&'static str> {
    problems::all()
        .iter()
        .map(|problem| problem.name())
        .collect()
}

/// The names of the problems of `genoxide::multi::problems::all()` with `objectives` objectives,
/// 2 to 6, in its order.
#[pyfunction]
pub fn multi_problem_names(objectives: usize) -> PyResult<Vec<&'static str>> {
    struct Names;

    impl WithObjectives for Names {
        type Output = Vec<&'static str>;

        fn with<const N: usize>(self) -> Vec<&'static str> {
            multi::all::<N>()
                .iter()
                .map(|problem| problem.name())
                .collect()
        }
    }

    with_objectives(objectives, Names).map_err(|count| {
        PyValueError::new_err(format!("problems take 2 to 6 objectives, not {count}"))
    })
}
