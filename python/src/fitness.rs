//! Python fitness functions: called with a genome as a numpy array, or with a generation as a
//! 2-D array (a batch). And the progress callback, called after every generation.
//!
//! An exception in the fitness function or the progress callback, or Ctrl+C, stops the run: the
//! first exception is kept, the abort flag is set, and the genomes left get an invalid fitness
//! without a call. The run raises the exception when it returns.

use crate::genes::{self, Genes};
use genoxide::Fitness;
use genoxide::engine::{FitnessFunction, IntoFitness, Progress};
use genoxide::multi::{IntoScores, MultiFitnessFunction, Scores};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

/// The fitness function and progress callback of a run, and what went wrong in them.
pub struct Shared {
    function: Py<PyAny>,
    batch: bool,
    on_generation: Option<Py<PyAny>>,
    error: Mutex<Option<PyErr>>,
    abort: Arc<AtomicBool>,
}

impl Shared {
    pub fn new(function: Py<PyAny>, batch: bool, on_generation: Option<Py<PyAny>>) -> Self {
        Self {
            function,
            batch,
            on_generation,
            error: Mutex::new(None),
            abort: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The flag that stops the run.
    pub fn abort_flag(&self) -> Arc<AtomicBool> {
        self.abort.clone()
    }

    fn aborted(&self) -> bool {
        self.abort.load(Ordering::Relaxed)
    }

    // keeps the first error and stops the run
    fn fail(&self, error: PyErr) {
        let mut slot = self.error.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(error);
        }
        self.abort.store(true, Ordering::Relaxed);
    }

    /// The error that stopped the run, if any.
    pub fn take_error(&self) -> Option<PyErr> {
        self.error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// After a generation, on the thread that runs the engine, the one that called `run`: stops
    /// the run on Ctrl+C, which Python handles there, and calls the progress callback with the
    /// generation, the evaluations, the seconds and `value` (the best fitness, or the size of
    /// the front). The callback returns False to stop the run.
    pub fn after_generation<V>(&self, progress: &Progress, value: V)
    where
        V: for<'py> IntoPyObject<'py>,
    {
        Python::attach(|py| {
            if let Err(error) = py.check_signals() {
                self.fail(error);
            }
            let Some(callback) = &self.on_generation else {
                return;
            };
            if self.aborted() {
                return;
            }
            let arguments = (
                progress.generation(),
                progress.evaluations(),
                progress.elapsed().as_secs_f64(),
                value,
            );
            let go_on = callback.bind(py).call1(arguments);
            match go_on.and_then(|go_on| go_on.extract::<bool>()) {
                Ok(true) => {}
                Ok(false) => self.abort.store(true, Ordering::Relaxed),
                Err(error) => self.fail(error),
            }
        });
    }

    // calls the function with `argument`, or keeps its error
    fn call<'py, T>(
        &self,
        py: Python<'py>,
        argument: PyResult<Bound<'py, PyAny>>,
        convert: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
    ) -> Option<T> {
        let result = argument
            .and_then(|argument| self.function.bind(py).call1((argument,)))
            .and_then(|result| convert(&result));
        result.map_err(|error| self.fail(error)).ok()
    }
}

fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .name()
        .map_or_else(|_| "?".to_string(), |name| name.to_string())
}

// a batch returned a score per genome
fn check_count(scores: usize, genomes: usize) -> PyResult<()> {
    if scores == genomes {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "the batch fitness function returned {scores} scores for {genomes} genomes"
        )))
    }
}

/// The score of a single-objective fitness function.
#[derive(Clone, Copy, Debug)]
pub enum Value {
    Score(f64),
    Constrained(f64, f64),
    Invalid,
}

impl IntoFitness for Value {
    fn into_fitness(self) -> genoxide::Result<Fitness> {
        match self {
            Self::Score(score) => score.into_fitness(),
            Self::Constrained(score, violation) => (score, violation).into_fitness(),
            Self::Invalid => Ok(Fitness::invalid()),
        }
    }
}

// a number, None (invalid) or (score, violation)
fn value(result: &Bound<'_, PyAny>) -> PyResult<Value> {
    if result.is_none() {
        return Ok(Value::Invalid);
    }
    if let Ok(tuple) = result.cast::<PyTuple>() {
        let (score, violation) = tuple.extract::<(f64, f64)>()?;
        return Ok(Value::Constrained(score, violation));
    }
    result.extract::<f64>().map(Value::Score).map_err(|_| {
        PyTypeError::new_err(format!(
            "a fitness function returns a number, None (an invalid solution) or a tuple (score, constraint violation), not {}",
            type_name(result)
        ))
    })
}

// scores and optional constraint violations, as float64 arrays: the Python package converts what
// a batch function returns
fn values(result: &Bound<'_, PyAny>, genomes: usize) -> PyResult<Vec<Value>> {
    let (scores, violations) =
        result.extract::<(PyReadonlyArray1<'_, f64>, Option<PyReadonlyArray1<'_, f64>>)>()?;
    let scores = scores.as_array();
    check_count(scores.len(), genomes)?;
    match violations {
        None => Ok(scores.iter().map(|&score| Value::Score(score)).collect()),
        Some(violations) => {
            let violations = violations.as_array();
            check_count(violations.len(), genomes)?;
            Ok(scores
                .iter()
                .zip(violations.iter())
                .map(|(&score, &violation)| Value::Constrained(score, violation))
                .collect())
        }
    }
}

/// A single-objective fitness function.
pub struct Single<'a>(pub &'a Shared);

impl<G: Genes> FitnessFunction<G> for Single<'_> {
    type Output = Value;

    fn evaluate(&self, genome: &G) -> Value {
        let shared = self.0;
        if shared.aborted() {
            return Value::Invalid;
        }
        Python::attach(|py| {
            let argument = Ok(genes::array(py, genome).into_any());
            shared.call(py, argument, value).unwrap_or(Value::Invalid)
        })
    }

    fn is_batch(&self) -> bool {
        self.0.batch
    }

    fn evaluate_batch(&self, genomes: &[&G]) -> Vec<Value> {
        let shared = self.0;
        if !shared.batch {
            return genomes
                .iter()
                .map(|genome| FitnessFunction::<G>::evaluate(self, genome))
                .collect();
        }
        if shared.aborted() {
            return vec![Value::Invalid; genomes.len()];
        }
        Python::attach(|py| {
            let argument = genes::matrix(py, genomes).map(Bound::into_any);
            shared
                .call(py, argument, |result| values(result, genomes.len()))
                .unwrap_or_else(|| vec![Value::Invalid; genomes.len()])
        })
    }
}

/// The scores of a multi-objective fitness function.
#[derive(Clone, Copy, Debug)]
pub enum MultiValue<const M: usize> {
    Scores([f64; M]),
    Constrained([f64; M], f64),
    Invalid,
}

impl<const M: usize> IntoScores<M> for MultiValue<M> {
    fn into_scores(self) -> genoxide::Result<Scores<M>> {
        match self {
            Self::Scores(scores) => scores.into_scores(),
            Self::Constrained(scores, violation) => (scores, violation).into_scores(),
            Self::Invalid => Ok(Scores::invalid()),
        }
    }
}

fn objectives<const M: usize>(scores: &Bound<'_, PyAny>) -> PyResult<[f64; M]> {
    let scores = scores.extract::<Vec<f64>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "a multi-objective fitness function returns a sequence of {M} numbers, None (an invalid solution) or a tuple (scores, constraint violation), not {}",
            type_name(scores)
        ))
    })?;
    <[f64; M]>::try_from(scores).map_err(|scores| {
        PyValueError::new_err(format!(
            "the fitness function returned {} scores, for {M} objectives",
            scores.len()
        ))
    })
}

// M numbers, None (invalid) or (M numbers, violation): a tuple of 2 whose first item isn't a
// number
fn multi_value<const M: usize>(result: &Bound<'_, PyAny>) -> PyResult<MultiValue<M>> {
    if result.is_none() {
        return Ok(MultiValue::Invalid);
    }
    if let Ok(tuple) = result.cast::<PyTuple>() {
        if tuple.len() == 2 {
            let first = tuple.get_item(0)?;
            if first.extract::<f64>().is_err() {
                let violation = tuple.get_item(1)?.extract::<f64>()?;
                return Ok(MultiValue::Constrained(objectives(&first)?, violation));
            }
        }
    }
    objectives(result).map(MultiValue::Scores)
}

// a float64 array of a row of scores per genome, and optional constraint violations: the Python
// package converts what a batch function returns
fn multi_values<const M: usize>(
    result: &Bound<'_, PyAny>,
    genomes: usize,
) -> PyResult<Vec<MultiValue<M>>> {
    let (scores, violations) =
        result.extract::<(PyReadonlyArray2<'_, f64>, Option<PyReadonlyArray1<'_, f64>>)>()?;
    let scores = scores.as_array();
    check_count(scores.nrows(), genomes)?;
    if scores.ncols() != M {
        return Err(PyValueError::new_err(format!(
            "the batch fitness function returned {} scores per genome, for {M} objectives",
            scores.ncols()
        )));
    }
    let rows = scores.rows().into_iter().map(|row| {
        let mut values = [0.0; M];
        for (value, &score) in values.iter_mut().zip(row.iter()) {
            *value = score;
        }
        values
    });
    match violations {
        None => Ok(rows.map(MultiValue::Scores).collect()),
        Some(violations) => {
            let violations = violations.as_array();
            check_count(violations.len(), genomes)?;
            Ok(rows
                .zip(violations.iter())
                .map(|(values, &violation)| MultiValue::Constrained(values, violation))
                .collect())
        }
    }
}

/// A multi-objective fitness function.
pub struct Multi<'a>(pub &'a Shared);

impl<G: Genes, const M: usize> MultiFitnessFunction<G, M> for Multi<'_> {
    type Output = MultiValue<M>;

    fn evaluate(&self, genome: &G) -> MultiValue<M> {
        let shared = self.0;
        if shared.aborted() {
            return MultiValue::Invalid;
        }
        Python::attach(|py| {
            let argument = Ok(genes::array(py, genome).into_any());
            shared
                .call(py, argument, multi_value)
                .unwrap_or(MultiValue::Invalid)
        })
    }

    fn is_batch(&self) -> bool {
        self.0.batch
    }

    fn evaluate_batch(&self, genomes: &[&G]) -> Vec<MultiValue<M>> {
        let shared = self.0;
        if !shared.batch {
            return genomes
                .iter()
                .map(|genome| MultiFitnessFunction::<G, M>::evaluate(self, genome))
                .collect();
        }
        if shared.aborted() {
            return vec![MultiValue::Invalid; genomes.len()];
        }
        Python::attach(|py| {
            let argument = genes::matrix(py, genomes).map(Bound::into_any);
            shared
                .call(py, argument, |result| multi_values(result, genomes.len()))
                .unwrap_or_else(|| vec![MultiValue::Invalid; genomes.len()])
        })
    }
}
