//! Python fitness functions: called with a genome as a numpy array, or with a generation as a
//! 2-D array (a batch); and the test problems of `genoxide::problems` and
//! `genoxide::multi::problems`, evaluated in Rust. And the progress callback, called after every
//! generation.
//!
//! An exception in the fitness function or the progress callback, or Ctrl+C, stops the run: the
//! first exception is kept, the abort flag is set, and the genomes left get an invalid fitness
//! without a call. The run raises the exception when it returns.
//!
//! Python runs signal handlers on the main thread only, the one that called `run` and runs the
//! engine. A parallel run's function runs on rayon's threads while that thread checks for Ctrl+C
//! every `SIGNAL_CHECK`, so Ctrl+C stops the run after the calls under way, not after the
//! generation.
//!
//! A batch function's matrix is reused: after a call, the run keeps it, and the next batch of as
//! many genomes gets it back with its genomes written into it, if its reference count shows that
//! Python kept no reference to it (a function that keeps it gets a new matrix next time, and what
//! it kept never changes). A large matrix is then allocated, and its memory touched for the first
//! time, once per run instead of once per generation: with 1000 genomes of 1000 genes, a
//! generation took a fifth less. A function of one genome gets a new array per call: writing a
//! genome into a kept array is only safe (without unsafe code) through numpy's borrow checking,
//! which cost as much as the new array.

use crate::genes::{GenomeContext, PyGenome};
use crate::problems::{IntegerProblem, MultiNative};
use crate::tasks::Balance;
use crate::tree_problems::TreeFitness;
use genoxide::Fitness;
use genoxide::engine::{BatchExtras, Extras, FitnessFunction, IntoFitness, Progress, Provided};
use genoxide::multi::{IntoScores, MultiFitnessFunction, Scores};
use genoxide::problems::DynProblem;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

// how often the thread that called `run` checks for Ctrl+C while a parallel run's calls go on
const SIGNAL_CHECK: Duration = Duration::from_millis(50);

/// Where a Python fitness function's gradient comes from, for L-BFGS-B: nowhere, a function of
/// its own (`gradient=g`), or the fitness function itself, which returns `(value, gradient)`
/// (`gradient=True`).
pub enum Gradient {
    None,
    Function(Py<PyAny>),
    Combined,
}

/// The fitness function and progress callback of a run, and what went wrong in them.
pub struct Shared {
    function: Py<PyAny>,
    gradient: Gradient,
    batch: bool,
    parallel: bool,
    on_generation: Option<Py<PyAny>>,
    error: Mutex<Option<PyErr>>,
    abort: Arc<AtomicBool>,
    // the matrix of the last batch, to write the next batch into
    matrix: Mutex<Option<Py<PyAny>>>,
    // what the genomes need to become Python objects
    context: GenomeContext,
}

impl Shared {
    pub fn new(
        function: Py<PyAny>,
        gradient: Gradient,
        batch: bool,
        parallel: bool,
        on_generation: Option<Py<PyAny>>,
        context: GenomeContext,
    ) -> Self {
        Self {
            function,
            gradient,
            batch,
            parallel,
            on_generation,
            error: Mutex::new(None),
            abort: Arc::new(AtomicBool::new(false)),
            matrix: Mutex::new(None),
            context,
        }
    }

    /// What the genomes need to become Python objects.
    pub fn context(&self) -> &GenomeContext {
        &self.context
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
    /// generation, the evaluations, the seconds and the values `state` makes (the best fitness
    /// or the size of the front, then copies of the population, made into arrays when read),
    /// only made for a callback. The callback returns False to stop the run.
    pub fn after_generation<S>(&self, progress: &Progress, state: S)
    where
        S: for<'py> FnOnce(Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>>,
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
            let go_on = state(py).and_then(|state| {
                let mut arguments = vec![
                    progress.generation().into_pyobject(py)?.into_any(),
                    progress.evaluations().into_pyobject(py)?.into_any(),
                    progress
                        .elapsed()
                        .as_secs_f64()
                        .into_pyobject(py)?
                        .into_any(),
                ];
                arguments.extend(state);
                callback.bind(py).call1(PyTuple::new(py, arguments)?)
            });
            match go_on.and_then(|go_on| go_on.extract::<bool>()) {
                Ok(true) => {}
                Ok(false) => self.abort.store(true, Ordering::Relaxed),
                Err(error) => self.fail(error),
            }
        });
    }

    /// Calls the control of a run through `call`, with the generation, the evaluations and the
    /// seconds, after the progress callback of the same generation. Not after an error, which
    /// stops the run. Returns false if the control raised an exception, which is kept.
    pub fn control<C>(&self, progress: &Progress, call: C) -> bool
    where
        C: for<'py> FnOnce(Python<'py>, Vec<Bound<'py, PyAny>>) -> PyResult<()>,
    {
        if self.failed() {
            return true;
        }
        Python::attach(|py| {
            let called = (|| {
                let arguments = vec![
                    progress.generation().into_pyobject(py)?.into_any(),
                    progress.evaluations().into_pyobject(py)?.into_any(),
                    progress
                        .elapsed()
                        .as_secs_f64()
                        .into_pyobject(py)?
                        .into_any(),
                ];
                call(py, arguments)
            })();
            called.map_err(|error| self.fail(error)).is_ok()
        })
    }

    /// Whether an error stops the run.
    pub fn failed(&self) -> bool {
        self.error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
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

    // calls the function of one genome with `genome`, unless the run is stopping
    fn call_genome<'py, G: PyGenome, T>(
        &self,
        py: Python<'py>,
        genome: &G,
        convert: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
    ) -> Option<T> {
        if self.aborted() {
            return None;
        }
        let argument = genome.object(py, &self.context);
        self.call(py, argument, convert)
    }

    // calls the batch function with `genomes`, a genome per row
    fn call_batch<'py, G: PyGenome, T>(
        &self,
        py: Python<'py>,
        genomes: &[&G],
        convert: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
    ) -> Option<T> {
        let reused = self
            .matrix
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .map(|matrix| matrix.into_bound(py))
            // the run's reference only: Python kept none
            .filter(|matrix| only_reference(matrix) && G::refill(matrix, genomes));
        let matrix = match reused {
            Some(matrix) => Ok(matrix),
            None => G::batch(py, genomes, &self.context),
        };
        let kept = matrix.as_ref().ok().cloned();
        let result = self.call(py, matrix, convert);
        if let Some(matrix) = kept {
            *self.matrix.lock().unwrap_or_else(PoisonError::into_inner) = Some(matrix.unbind());
        }
        result
    }

    // a generation's values from the function of one genome: called on this thread, the engine's,
    // attached to Python once; or with `parallel`, on rayon's threads, while this thread checks
    // for Ctrl+C. `invalid` for a genome not called, after an error or Ctrl+C.
    fn call_genomes<G: PyGenome, T: Copy + Send + Sync>(
        &self,
        genomes: &[&G],
        convert: for<'py> fn(&Bound<'py, PyAny>) -> PyResult<T>,
        invalid: T,
    ) -> Vec<T> {
        if !self.parallel {
            return Python::attach(|py| {
                genomes
                    .iter()
                    .map(|genome| self.call_genome(py, *genome, convert).unwrap_or(invalid))
                    .collect()
            });
        }
        std::thread::scope(|scope| {
            let (done, finished) = mpsc::channel();
            let calls = scope.spawn(move || {
                // in order, whatever the thread count
                let values: Vec<T> = genomes
                    .par_iter()
                    .map(|genome| {
                        Python::attach(|py| self.call_genome(py, *genome, convert))
                            .unwrap_or(invalid)
                    })
                    .collect();
                let _ = done.send(());
                values
            });
            // until the calls are done: Ctrl+C sets the abort flag, and the calls not yet started
            // are skipped
            while let Err(RecvTimeoutError::Timeout) = finished.recv_timeout(SIGNAL_CHECK) {
                Python::attach(|py| {
                    if let Err(error) = py.check_signals() {
                        self.fail(error);
                    }
                });
            }
            calls
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        })
    }
}

impl Shared {
    /// Whether the Python function gives a gradient.
    pub fn provides_gradient(&self) -> bool {
        !matches!(self.gradient, Gradient::None)
    }

    // the value of one genome and its gradient into `gradient`, a call of the function (and of
    // the gradient's function); `None` after an error or Ctrl+C
    fn call_with_gradient<G: PyGenome>(
        &self,
        py: Python<'_>,
        genome: &G,
        gradient: &mut [f64],
    ) -> Option<Value> {
        if self.aborted() {
            return None;
        }
        let argument = genome.object(py, &self.context);
        match &self.gradient {
            Gradient::Function(function) => {
                let argument = argument.map_err(|error| self.fail(error)).ok()?;
                let value = self.call(py, Ok(argument.clone()), value)?;
                let written = function
                    .bind(py)
                    .call1((argument,))
                    .and_then(|result| gradient_into(&result, gradient));
                written.map_err(|error| self.fail(error)).ok()?;
                Some(value)
            }
            Gradient::Combined => self.call(py, argument, |result| {
                value_with_gradient(result, Some(gradient))
            }),
            Gradient::None => self.call(py, argument, value),
        }
    }

    // the values of a batch and their gradients into `gradients`, a row per genome
    fn call_batch_with_gradients<G: PyGenome>(
        &self,
        py: Python<'_>,
        genomes: &[&G],
        gradients: &mut [f64],
    ) -> Option<Vec<Value>> {
        if self.aborted() {
            return None;
        }
        let matrix = G::batch(py, genomes, &self.context);
        match &self.gradient {
            Gradient::Function(function) => {
                let matrix = matrix.map_err(|error| self.fail(error)).ok()?;
                let values = self.call(py, Ok(matrix.clone()), |result| {
                    values(result, genomes.len())
                })?;
                let written = function
                    .bind(py)
                    .call1((matrix,))
                    .and_then(|result| gradient_rows(&result, gradients, genomes.len()));
                written.map_err(|error| self.fail(error)).ok()?;
                Some(values)
            }
            Gradient::Combined => self.call(py, matrix, |result| {
                values_with_gradients(result, genomes.len(), Some(gradients))
            }),
            Gradient::None => self.call(py, matrix, |result| values(result, genomes.len())),
        }
    }
}

// the value of `(value, gradient)`, the gradient written into `gradient` if it's given: what a
// function with `gradient=True` returns (the Python package makes the gradient a float64 array)
fn value_with_gradient(result: &Bound<'_, PyAny>, gradient: Option<&mut [f64]>) -> PyResult<Value> {
    let pair = result
        .cast::<PyTuple>()
        .ok()
        .filter(|tuple| tuple.len() == 2)
        .ok_or_else(|| {
            PyTypeError::new_err(format!(
                "with gradient=True, the fitness function returns a tuple (value, gradient), not {}",
                type_name(result)
            ))
        })?;
    let value = value(&pair.get_item(0)?)?;
    if let Some(gradient) = gradient {
        gradient_into(&pair.get_item(1)?, gradient)?;
    }
    Ok(value)
}

// `value_with_gradient` without the gradient: a function with `gradient=True` called where no
// gradient is wanted (finite differences)
fn value_without_gradient(result: &Bound<'_, PyAny>) -> PyResult<Value> {
    value_with_gradient(result, None)
}

// a gradient, a float64 array of a value per gene, into `gradient`
fn gradient_into(result: &Bound<'_, PyAny>, gradient: &mut [f64]) -> PyResult<()> {
    let array = result.extract::<PyReadonlyArray1<'_, f64>>()?;
    let array = array.as_array();
    if array.len() != gradient.len() {
        return Err(PyValueError::new_err(format!(
            "the gradient has {} values, for {} genes",
            array.len(),
            gradient.len()
        )));
    }
    for (gradient, &value) in gradient.iter_mut().zip(array.iter()) {
        *gradient = value;
    }
    Ok(())
}

// the gradients of a batch, a float64 array of a row per genome, into `gradients`
fn gradient_rows(result: &Bound<'_, PyAny>, gradients: &mut [f64], genomes: usize) -> PyResult<()> {
    let array = result.extract::<PyReadonlyArray2<'_, f64>>()?;
    let array = array.as_array();
    let genes = gradients.len().checked_div(genomes).unwrap_or(0);
    if array.nrows() != genomes || array.ncols() != genes {
        return Err(PyValueError::new_err(format!(
            "the gradients have shape ({}, {}), for {genomes} genomes of {genes} genes",
            array.nrows(),
            array.ncols()
        )));
    }
    for (gradient, &value) in gradients.iter_mut().zip(array.iter()) {
        *gradient = value;
    }
    Ok(())
}

// the scores and gradients of a batch with `gradient=True`: (scores, gradients), float64 arrays
// the Python package makes, the gradients written into `gradients` if they're given
fn values_with_gradients(
    result: &Bound<'_, PyAny>,
    genomes: usize,
    gradients: Option<&mut [f64]>,
) -> PyResult<Vec<Value>> {
    let (scores, rows) = result.extract::<(PyReadonlyArray1<'_, f64>, Bound<'_, PyAny>)>()?;
    let scores = scores.as_array();
    check_count(scores.len(), genomes)?;
    if let Some(gradients) = gradients {
        gradient_rows(&rows, gradients, genomes)?;
    }
    Ok(scores.iter().map(|&score| Value::Score(score)).collect())
}

// whether `object` is referenced by the caller only: Python kept no reference to it. (pyo3
// deprecates `get_refcnt` for `ffi::Py_REFCNT`, which is unsafe, and this crate has no unsafe
// code.)
#[allow(deprecated)]
fn only_reference(object: &Bound<'_, PyAny>) -> bool {
    object.get_refcnt() == 1
}

fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .name()
        .map_or_else(|_| "?".to_string(), |name| name.to_string())
}

// the error of a result that doesn't convert to numbers: a TypeError with `message`, caused by the
// conversion's TypeError. Any other error, e.g. an OverflowError for an int too large for a float
// or an exception in `__float__`, is the result's own, and raised as it is.
// the types of a tuple's items, e.g. "(float, NoneType)"
fn item_types(tuple: &Bound<'_, PyTuple>) -> String {
    let names: Vec<String> = tuple.iter().map(|item| type_name(&item)).collect();
    format!("({})", names.join(", "))
}

fn not_numbers(py: Python<'_>, error: PyErr, message: impl FnOnce() -> String) -> PyErr {
    if !error.is_instance_of::<PyTypeError>(py) {
        return error;
    }
    let wrong = PyTypeError::new_err(message());
    wrong.set_cause(py, Some(error));
    wrong
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
    /// The fitness of a test problem evaluated in Rust.
    Native(Fitness),
}

impl IntoFitness for Value {
    fn into_fitness(self) -> genoxide::Result<Fitness> {
        match self {
            Self::Score(score) => score.into_fitness(),
            Self::Constrained(score, violation) => (score, violation).into_fitness(),
            Self::Invalid => Ok(Fitness::invalid()),
            Self::Native(fitness) => Ok(fitness),
        }
    }
}

// a number, None (invalid) or (score, violation)
fn value(result: &Bound<'_, PyAny>) -> PyResult<Value> {
    if result.is_none() {
        return Ok(Value::Invalid);
    }
    if let Ok(tuple) = result.cast::<PyTuple>() {
        // PyO3's error for another length is a ValueError
        if tuple.len() != 2 {
            return Err(PyTypeError::new_err(format!(
                "a fitness function's tuple is (score, constraint violation), two numbers, not a tuple of length {}",
                tuple.len()
            )));
        }
        let (score, violation) = tuple.extract::<(f64, f64)>().map_err(|error| {
            not_numbers(result.py(), error, || {
                format!(
                    "a fitness function's tuple is (score, constraint violation), two numbers, not {}",
                    item_types(tuple)
                )
            })
        })?;
        return Ok(Value::Constrained(score, violation));
    }
    result.extract::<f64>().map(Value::Score).map_err(|error| {
        not_numbers(result.py(), error, || {
            format!(
                "a fitness function returns a number, None (an invalid solution) or a tuple (score, constraint violation), not {}",
                type_name(result)
            )
        })
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

/// A single-objective fitness function: the Python function of `Shared`, or a test problem of
/// `genoxide::problems`, evaluated in Rust without Python.
pub struct Single<'a> {
    pub shared: &'a Shared,
    pub problem: Option<Native<'a>>,
}

/// A single-objective test problem evaluated in Rust: on real or on integer genomes; or a
/// network's weights balancing poles; or a fitness of trees.
#[derive(Clone, Copy)]
pub enum Native<'a> {
    Real(&'a dyn DynProblem),
    Integer(&'a dyn IntegerProblem),
    Balance(&'a Balance),
    Tree(&'a TreeFitness),
}

impl<G: PyGenome> FitnessFunction<G> for Single<'_> {
    type Output = Value;

    fn evaluate(&self, genome: &G) -> Value {
        let shared = self.shared;
        if shared.aborted() {
            return Value::Invalid;
        }
        match self.problem {
            // the run checks that the genome is the problem's
            Some(Native::Real(problem)) => {
                return genome.reals().map_or(Value::Invalid, |genome| {
                    Value::Native(problem.evaluate(genome))
                });
            }
            Some(Native::Integer(problem)) => {
                return genome.integers().map_or(Value::Invalid, |genome| {
                    Value::Native(problem.evaluate(genome))
                });
            }
            Some(Native::Balance(balance)) => {
                return genome.reals().map_or(Value::Invalid, |weights| {
                    Value::Native(balance.evaluate(weights))
                });
            }
            Some(Native::Tree(fitness)) => {
                return genome
                    .tree()
                    .and_then(|tree| fitness.evaluate(tree))
                    .map_or(Value::Invalid, Value::Score);
            }
            None => {}
        }
        let convert = match shared.gradient {
            Gradient::Combined => value_without_gradient,
            _ => value,
        };
        Python::attach(|py| shared.call_genome(py, genome, convert)).unwrap_or(Value::Invalid)
    }

    // a Python function gets a generation at a time (`Shared::call_genomes`); a test problem is
    // evaluated a genome at a time, by the engine, in parallel if asked
    fn is_batch(&self) -> bool {
        self.problem.is_none()
    }

    // the analytic gradients of the smooth test problems, and a Python function's
    fn provides(&self) -> Provided {
        match self.problem {
            Some(Native::Real(problem)) => problem.provides(),
            Some(_) => Provided::NOTHING,
            None if self.shared.provides_gradient() => Provided::GRADIENT,
            None => Provided::NOTHING,
        }
    }

    fn evaluate_with(&self, genome: &G, extras: &mut Extras<'_>) -> Value {
        let shared = self.shared;
        let Some(gradient) = extras.gradient() else {
            return FitnessFunction::<G>::evaluate(self, genome);
        };
        if shared.aborted() {
            return Value::Invalid;
        }
        match self.problem {
            Some(Native::Real(problem)) => genome.reals().map_or(Value::Invalid, |genome| {
                Value::Native(problem.evaluate_with(genome, &mut Extras::with_gradient(gradient)))
            }),
            Some(_) => FitnessFunction::<G>::evaluate(self, genome),
            None => Python::attach(|py| shared.call_with_gradient(py, genome, gradient))
                .unwrap_or(Value::Invalid),
        }
    }

    // with gradients, a Python function is called a genome at a time on this thread, or once
    // with the batch
    fn evaluate_batch_with(&self, genomes: &[&G], extras: &mut BatchExtras<'_>) -> Vec<Value> {
        let shared = self.shared;
        let dimensions = extras.dimensions();
        let Some(gradients) = extras.gradients() else {
            return FitnessFunction::<G>::evaluate_batch(self, genomes);
        };
        if self.problem.is_some() || !shared.batch {
            return genomes
                .iter()
                .zip(gradients.chunks_mut(dimensions.max(1)))
                .map(|(genome, row)| {
                    FitnessFunction::<G>::evaluate_with(
                        self,
                        genome,
                        &mut Extras::with_gradient(row),
                    )
                })
                .collect();
        }
        if shared.aborted() || genomes.is_empty() {
            return vec![Value::Invalid; genomes.len()];
        }
        Python::attach(|py| shared.call_batch_with_gradients(py, genomes, gradients))
            .unwrap_or_else(|| vec![Value::Invalid; genomes.len()])
    }

    fn evaluate_batch(&self, genomes: &[&G]) -> Vec<Value> {
        let shared = self.shared;
        if !FitnessFunction::<G>::is_batch(self) {
            return genomes
                .iter()
                .map(|genome| FitnessFunction::<G>::evaluate(self, genome))
                .collect();
        }
        // no call after an error, or for a generation of copies, which inherit their fitness
        if shared.aborted() || genomes.is_empty() {
            return vec![Value::Invalid; genomes.len()];
        }
        let combined = matches!(shared.gradient, Gradient::Combined);
        if !shared.batch {
            let convert = if combined {
                value_without_gradient
            } else {
                value
            };
            return shared.call_genomes(genomes, convert, Value::Invalid);
        }
        Python::attach(|py| {
            shared
                .call_batch(py, genomes, |result| {
                    if combined {
                        values_with_gradients(result, genomes.len(), None)
                    } else {
                        values(result, genomes.len())
                    }
                })
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
    /// The scores of a test problem evaluated in Rust.
    Native(Scores<M>),
}

impl<const M: usize> IntoScores<M> for MultiValue<M> {
    fn into_scores(self) -> genoxide::Result<Scores<M>> {
        match self {
            Self::Scores(scores) => scores.into_scores(),
            Self::Constrained(scores, violation) => (scores, violation).into_scores(),
            Self::Invalid => Ok(Scores::invalid()),
            Self::Native(scores) => Ok(scores),
        }
    }
}

fn objectives<const M: usize>(scores: &Bound<'_, PyAny>) -> PyResult<[f64; M]> {
    let scores = scores.extract::<Vec<f64>>().map_err(|error| {
        not_numbers(scores.py(), error, || {
            format!(
                "a multi-objective fitness function returns a sequence of {M} numbers, None (an invalid solution) or a tuple (scores, constraint violation), not {}",
                type_name(scores)
            )
        })
    })?;
    <[f64; M]>::try_from(scores).map_err(|scores| {
        PyValueError::new_err(format!(
            "the fitness function returned {} scores, for {M} objectives",
            scores.len()
        ))
    })
}

// M numbers, None (invalid) or (M numbers, violation): a tuple of 2 whose first item isn't a
// number. A first item that is a number but fails to convert raises its error.
fn multi_value<const M: usize>(result: &Bound<'_, PyAny>) -> PyResult<MultiValue<M>> {
    if result.is_none() {
        return Ok(MultiValue::Invalid);
    }
    if let Ok(tuple) = result.cast::<PyTuple>()
        && tuple.len() == 2
    {
        let first = tuple.get_item(0)?;
        match first.extract::<f64>() {
            Ok(_) => {}
            Err(error) if error.is_instance_of::<PyTypeError>(result.py()) => {
                let violation = tuple.get_item(1)?.extract::<f64>().map_err(|error| {
                    not_numbers(result.py(), error, || {
                        format!(
                            "a multi-objective fitness function's tuple is (scores, constraint violation), the violation a number, not {}",
                            item_types(tuple)
                        )
                    })
                })?;
                return Ok(MultiValue::Constrained(objectives(&first)?, violation));
            }
            Err(error) => return Err(error),
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

/// A multi-objective fitness function: the Python function of `Shared`, or a test problem of
/// `genoxide::multi::problems` or a fitness of trees with their size, evaluated in Rust without
/// Python.
pub struct Multi<'a, const M: usize> {
    pub shared: &'a Shared,
    pub problem: Option<&'a MultiNative<M>>,
    pub tree: Option<&'a TreeFitness>,
}

impl<G: PyGenome, const M: usize> MultiFitnessFunction<G, M> for Multi<'_, M> {
    type Output = MultiValue<M>;

    fn evaluate(&self, genome: &G) -> MultiValue<M> {
        let shared = self.shared;
        if shared.aborted() {
            return MultiValue::Invalid;
        }
        if let Some(problem) = self.problem {
            // the run checks that the genome is the problem's
            return MultiValue::Native(problem.evaluate(genome));
        }
        if let Some(fitness) = self.tree {
            // the run checks that the objectives are the fitness's
            let values = genome.tree().and_then(|tree| fitness.values(tree));
            return values
                .and_then(|values| <[f64; M]>::try_from(values).ok())
                .map_or(MultiValue::Invalid, MultiValue::Scores);
        }
        Python::attach(|py| shared.call_genome(py, genome, multi_value))
            .unwrap_or(MultiValue::Invalid)
    }

    // as for `Single`
    fn is_batch(&self) -> bool {
        self.problem.is_none() && self.tree.is_none()
    }

    fn evaluate_batch(&self, genomes: &[&G]) -> Vec<MultiValue<M>> {
        let shared = self.shared;
        if !MultiFitnessFunction::<G, M>::is_batch(self) {
            return genomes
                .iter()
                .map(|genome| MultiFitnessFunction::<G, M>::evaluate(self, genome))
                .collect();
        }
        // no call after an error, or for a generation of copies, which inherit their scores
        if shared.aborted() || genomes.is_empty() {
            return vec![MultiValue::Invalid; genomes.len()];
        }
        if !shared.batch {
            return shared.call_genomes(genomes, multi_value::<M>, MultiValue::Invalid);
        }
        Python::attach(|py| {
            shared
                .call_batch(py, genomes, |result| multi_values(result, genomes.len()))
                .unwrap_or_else(|| vec![MultiValue::Invalid; genomes.len()])
        })
    }
}
