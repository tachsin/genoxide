//! The native module of the `genoxide` Python package. The package's Python code describes a run
//! in JSON; [`run::run`] builds it with genoxide and runs it with a Python fitness function, or a
//! test problem of `genoxide::problems` evaluated in Rust.

#![forbid(unsafe_code)]

mod config;
mod errors;
mod fitness;
mod genes;
mod indicators;
mod operators;
mod problems;
mod run;

use pyo3::prelude::*;

#[pymodule]
fn _genoxide(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run::run, module)?)?;
    module.add_function(wrap_pyfunction!(run::das_dennis, module)?)?;
    module.add_function(wrap_pyfunction!(problems::problem_info, module)?)?;
    module.add_function(wrap_pyfunction!(problems::evaluate, module)?)?;
    module.add_function(wrap_pyfunction!(problems::problem_names, module)?)?;
    module.add_function(wrap_pyfunction!(indicators::indicator, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
