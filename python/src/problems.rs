//! The test problems of `genoxide::problems`, as the Python package describes them in JSON: their
//! description, their evaluation, and the problem of a run evaluated in Rust.

use genoxide::Objective;
use genoxide::genome::{Reals, Representation};
use genoxide::problems::{self, DynProblem};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
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
}

// the problem that `config` describes; a size below the minimum is an error, not the panic of
// the constructor
fn build(config: Config) -> Result<Box<dyn DynProblem>, String> {
    let at_least = |dimensions: usize, minimum: usize, name: &str| {
        if dimensions >= minimum {
            Ok(dimensions)
        } else {
            Err(format!(
                "{name} needs at least {minimum} dimensions, not {dimensions}"
            ))
        }
    };
    Ok(match config {
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
    })
}

/// The problem that `description` (JSON) describes.
pub fn parse(description: &str) -> PyResult<Box<dyn DynProblem>> {
    let mut json = serde_json::Deserializer::from_str(description);
    let config: Config = serde_path_to_error::deserialize(&mut json).map_err(|error| {
        let (path, error) = (error.path().to_string(), error.into_inner());
        PyValueError::new_err(format!("invalid problem `{path}`: {error}"))
    })?;
    build(config).map_err(PyValueError::new_err)
}

// a genome per row, as numpy arrays of the problem's dimensions
fn rows(problem: &dyn DynProblem, genomes: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Reals>> {
    let genomes = genomes.as_array();
    let dimensions = problem.real().genome_len();
    if genomes.ncols() != dimensions {
        return Err(PyValueError::new_err(format!(
            "{} takes genomes of {dimensions} genes, not {}",
            problem.name(),
            genomes.ncols()
        )));
    }
    Ok(genomes
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().collect())
        .collect())
}

/// The description of the problem that `problem` (JSON) describes: its name, bounds (a pair per
/// gene), objective, optimum (None, or its value, solutions a row each, and whether it's
/// proven), reference and the reference's URL.
#[pyfunction]
pub fn problem_info<'py>(py: Python<'py>, problem: &str) -> PyResult<Bound<'py, PyDict>> {
    let problem = parse(problem)?;
    let info = PyDict::new(py);
    info.set_item("name", problem.name())?;
    let bounds: Vec<(f64, f64)> = problem
        .real()
        .bounds()
        .iter()
        .map(|range| (*range.start(), *range.end()))
        .collect();
    info.set_item("bounds", bounds)?;
    let objective = match problem.objective() {
        Objective::Maximize => "maximize",
        Objective::Minimize => "minimize",
    };
    info.set_item("objective", objective)?;
    match problem.optimum() {
        None => info.set_item("optimum", py.None())?,
        Some(optimum) => {
            let dimensions = problem.real().genome_len();
            let solutions = optimum.solutions();
            let genes: Vec<f64> = solutions.iter().flat_map(|x| x.iter().copied()).collect();
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
    Ok(info)
}

/// The scores of `genomes`, a row each, for the problem that `problem` (JSON) describes: NaN for
/// an invalid score.
#[pyfunction]
pub fn evaluate<'py>(
    py: Python<'py>,
    problem: &str,
    genomes: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let problem = parse(problem)?;
    let genomes = rows(problem.as_ref(), &genomes)?;
    let scores: Vec<f64> = py.detach(|| {
        genomes
            .iter()
            .map(|genome| problem.evaluate(genome).score().unwrap_or(f64::NAN))
            .collect()
    });
    Ok(PyArray1::from_vec(py, scores))
}

/// The names of the problems of `genoxide::problems::all()`, in its order.
#[pyfunction]
pub fn problem_names() -> Vec<&'static str> {
    problems::all()
        .iter()
        .map(|problem| problem.name())
        .collect()
}
