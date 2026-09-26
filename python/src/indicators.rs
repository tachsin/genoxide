//! The quality indicators of `genoxide::multi::indicator`, for fronts as numpy arrays.

use crate::run::{WithObjectives, with_objectives};
use genoxide::Objective;
use genoxide::multi::indicator as measure;
use numpy::PyReadonlyArray2;
use numpy::ndarray::ArrayView2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The indicator `kind` ("hypervolume", "igd", "gd", "igd_plus" or "spread") of `front`, a point
/// per row, against `reference`: the reference point (one row) for the hypervolume, the reference
/// front for the others. `objectives` has "minimize" or "maximize" per column, 2 to 6 of them.
#[pyfunction]
pub fn indicator(
    kind: &str,
    front: PyReadonlyArray2<'_, f64>,
    reference: PyReadonlyArray2<'_, f64>,
    objectives: Vec<String>,
) -> PyResult<f64> {
    let objectives = objectives
        .iter()
        .map(|objective| match objective.as_str() {
            "minimize" => Ok(Objective::Minimize),
            "maximize" => Ok(Objective::Maximize),
            other => Err(PyValueError::new_err(format!(
                "each objective is \"maximize\" or \"minimize\", not {other:?}"
            ))),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let task = Indicator {
        kind,
        front: front.as_array(),
        reference: reference.as_array(),
        objectives: &objectives,
    };
    with_objectives(objectives.len(), task)
        .unwrap_or_else(|count| {
            Err(format!(
                "the indicators take 2 to 6 objectives, not {count}"
            ))
        })
        .map_err(PyValueError::new_err)
}

// an indicator to compute with the number of objectives as a constant
struct Indicator<'a> {
    kind: &'a str,
    front: ArrayView2<'a, f64>,
    reference: ArrayView2<'a, f64>,
    objectives: &'a [Objective],
}

// the rows of `points`, each with `N` values
fn points<const N: usize>(
    points: ArrayView2<'_, f64>,
    name: &str,
) -> Result<Vec<[f64; N]>, String> {
    if points.ncols() != N {
        return Err(format!(
            "{name} has {} values per point, for {N} objectives",
            points.ncols()
        ));
    }
    Ok(points
        .rows()
        .into_iter()
        .map(|row| std::array::from_fn(|column| row[column]))
        .collect())
}

impl WithObjectives for Indicator<'_> {
    type Output = Result<f64, String>;

    fn with<const N: usize>(self) -> Result<f64, String> {
        let objectives: [Objective; N] =
            std::array::from_fn(|objective| self.objectives[objective]);
        let front = points::<N>(self.front, "the front")?;
        if self.kind == "hypervolume" {
            let reference = points::<N>(self.reference, "the reference point")?;
            let [point] = reference.as_slice() else {
                return Err("the hypervolume takes one reference point".to_string());
            };
            return Ok(measure::hypervolume(&front, point, &objectives));
        }
        let reference = points::<N>(self.reference, "the reference front")?;
        match self.kind {
            "igd" => Ok(measure::igd(&front, &reference)),
            "gd" => Ok(measure::gd(&front, &reference)),
            "igd_plus" => Ok(measure::igd_plus(&front, &reference, &objectives)),
            "spread" => Ok(measure::spread(&front, &reference, &objectives)),
            other => Err(format!("no indicator {other:?}")),
        }
    }
}
