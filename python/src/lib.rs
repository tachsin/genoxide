//! The native module of the `genoxide` Python package. The package's Python code describes a run
//! in JSON; [`run::run`] builds it with genoxide and runs it with a Python fitness function.

#![forbid(unsafe_code)]

mod config;
mod fitness;
mod genes;
mod operators;
mod run;

use pyo3::prelude::*;

#[pymodule]
fn _genoxide(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run::run, module)?)?;
    module.add_function(wrap_pyfunction!(run::das_dennis, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
