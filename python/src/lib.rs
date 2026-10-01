//! The native module of the `genoxide` Python package. The package's Python code describes a run
//! in JSON; [`run::run`] builds it with genoxide and runs it with a Python fitness function, or a
//! test problem of `genoxide::problems` or `genoxide::multi::problems` evaluated in Rust.

#![forbid(unsafe_code)]

mod checkpoint;
mod config;
mod control;
mod errors;
mod fitness;
mod genes;
mod indicators;
mod networks;
mod operators;
mod portable;
mod problems;
mod run;
mod snapshot;
mod tasks;

use pyo3::prelude::*;

#[pymodule]
fn _genoxide(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run::run, module)?)?;
    module.add_class::<control::Running>()?;
    module.add_class::<snapshot::Snapshot>()?;
    module.add_function(wrap_pyfunction!(run::das_dennis, module)?)?;
    module.add_function(wrap_pyfunction!(problems::problem_info, module)?)?;
    module.add_function(wrap_pyfunction!(problems::evaluate, module)?)?;
    module.add_function(wrap_pyfunction!(problems::constraints, module)?)?;
    module.add_function(wrap_pyfunction!(problems::optimal_front, module)?)?;
    module.add_function(wrap_pyfunction!(problems::problem_names, module)?)?;
    module.add_function(wrap_pyfunction!(problems::design, module)?)?;
    module.add_function(wrap_pyfunction!(problems::multi_problem_names, module)?)?;
    module.add_function(wrap_pyfunction!(indicators::indicator, module)?)?;
    module.add_class::<networks::PyNetwork>()?;
    module.add_class::<networks::NetworkPolicy>()?;
    module.add_class::<tasks::PyTask>()?;
    module.add_function(wrap_pyfunction!(tasks::balance_evaluate, module)?)?;
    module.add("SUCCESS_STEPS", genoxide::problems::control::SUCCESS_STEPS)?;
    module.add("DAMPING_STEPS", genoxide::problems::control::DAMPING_STEPS)?;
    module.add(
        "GENERALIZATION_THRESHOLD",
        genoxide::problems::control::GENERALIZATION_THRESHOLD,
    )?;
    module.add_function(wrap_pyfunction!(portable::portable_math, module)?)?;
    module.add_function(wrap_pyfunction!(portable::random_real, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
