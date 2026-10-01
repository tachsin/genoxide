//! What lets a Python program give the same results as a Rust one, to the bit, on every platform:
//! genoxide's portable math, and the random genomes of a seed.

use genoxide::StreamRng;
use genoxide::genome::{Real, Representation};
use genoxide::math;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::errors::genome_setting;

/// `function` of `genoxide::math`, by its name, of each value of `a`, or of each pair of values
/// of `a` and `b` for the functions of two arguments, in their order: `atan2(y, x)` is
/// `portable_math("atan2", y, x)`.
#[pyfunction]
#[pyo3(signature = (function, a, b = None))]
pub fn portable_math<'py>(
    py: Python<'py>,
    function: &str,
    a: PyReadonlyArray1<'py, f64>,
    b: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = a.as_array();
    let unary: Option<fn(f64) -> f64> = match function {
        "sin" => Some(math::sin),
        "cos" => Some(math::cos),
        "tan" => Some(math::tan),
        "asin" => Some(math::asin),
        "acos" => Some(math::acos),
        "atan" => Some(math::atan),
        "sinh" => Some(math::sinh),
        "cosh" => Some(math::cosh),
        "tanh" => Some(math::tanh),
        "exp" => Some(math::exp),
        "exp2" => Some(math::exp2),
        "expm1" => Some(math::exp_m1),
        "log" => Some(math::ln),
        "log1p" => Some(math::ln_1p),
        "log2" => Some(math::log2),
        "log10" => Some(math::log10),
        "cbrt" => Some(math::cbrt),
        _ => None,
    };
    if let Some(unary) = unary {
        return Ok(PyArray1::from_iter(py, x.iter().map(|&x| unary(x))));
    }
    let binary: fn(f64, f64) -> f64 = match function {
        "atan2" => math::atan2,
        "pow" => math::powf,
        "hypot" => math::hypot,
        _ => {
            return Err(PyValueError::new_err(format!(
                "genoxide.math has no function {function}"
            )));
        }
    };
    let y = b.ok_or_else(|| PyValueError::new_err(format!("{function} takes two arguments")))?;
    let y = y.as_array();
    if x.len() != y.len() {
        return Err(PyValueError::new_err(
            "the arguments have different lengths",
        ));
    }
    Ok(PyArray1::from_iter(
        py,
        x.iter().zip(y.iter()).map(|(&a, &b)| binary(a, b)),
    ))
}

/// A random genome of the real genome of `bounds`, a pair per gene, as Rust's
/// `real.random_genome(&mut StreamRng::seed_from_u64(seed))` gives it.
#[pyfunction]
pub fn random_real<'py>(
    py: Python<'py>,
    bounds: Vec<(f64, f64)>,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let real = genome_setting(
        Real::new(bounds.iter().map(|&(low, high)| low..=high)),
        "Real",
    )
    .map_err(PyValueError::new_err)?;
    let genome = real.random_genome(&mut StreamRng::seed_from_u64(seed));
    Ok(PyArray1::from_slice(py, &genome))
}
