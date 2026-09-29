//! A population as the progress callbacks see it: copied after the generation into plain vectors
//! (the genomes' words, the scores and the violations), which is fast, and made into numpy arrays
//! only when Python reads them. A callback that doesn't read the population doesn't pay for its
//! arrays, and a progress object kept after its callback stays valid.

use crate::genes::Genes;
use genoxide::multi::Scores;
use genoxide::{Fitness, Individual};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::marker::PhantomData;

/// Genomes that become a 2-D array, a genome per row.
trait Matrix: Send + Sync {
    fn matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
}

/// Genomes of `length` genes, as their words, one after the other.
struct Packed<G: Genes> {
    words: Vec<G::Word>,
    rows: usize,
    length: usize,
    genome: PhantomData<fn() -> G>,
}

impl<G: Genes> Packed<G> {
    fn new<'a>(genomes: impl ExactSizeIterator<Item = &'a G>) -> Self
    where
        G: 'a,
    {
        let rows = genomes.len();
        let mut words = Vec::new();
        let mut length = 0;
        for (row, genome) in genomes.enumerate() {
            genome.push_words(&mut words);
            if row == 0 {
                length = genome.len();
                // every genome has as many words as the first
                words.reserve_exact((rows - 1) * words.len());
            }
        }
        Self {
            words,
            rows,
            length,
            genome: PhantomData,
        }
    }
}

impl<G: Genes> Matrix for Packed<G> {
    fn matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let per_row = self.words.len().checked_div(self.rows).unwrap_or(0);
        let mut genes = Vec::with_capacity(self.rows * self.length);
        for row in 0..self.rows {
            let words = &self.words[row * per_row..(row + 1) * per_row];
            G::push_genes_of(words, self.length, &mut genes);
        }
        let matrix = Array2::from_shape_vec((self.rows, self.length), genes).map_err(|error| {
            PyValueError::new_err(format!("genomes of different lengths: {error}"))
        })?;
        Ok(matrix.into_pyarray(py).into_any())
    }
}

/// Individuals after a generation: their genomes (the population's, not the front's), their
/// scores or objective values (NaN for an invalid solution) and their constraint violations (0
/// for a feasible solution, NaN for an invalid one).
#[pyclass(frozen, module = "genoxide._genoxide")]
pub struct Snapshot {
    genomes: Option<Box<dyn Matrix>>,
    values: Vec<f64>,
    // the number of objective values of a multi-objective run, a column each; None for a score
    objectives: Option<usize>,
    violations: Vec<f64>,
}

impl Snapshot {
    /// The population of a single-objective run.
    pub fn single<'a, G: Genes>(
        individuals: impl ExactSizeIterator<Item = &'a Individual<G>> + Clone,
    ) -> Self {
        let (values, violations) = individuals
            .clone()
            .map(|individual| {
                let fitness = individual.fitness().unwrap_or(Fitness::invalid());
                let score = fitness.score();
                (
                    score.unwrap_or(f64::NAN),
                    score.map_or(f64::NAN, |_| fitness.violation()),
                )
            })
            .unzip();
        Self {
            genomes: Some(Box::new(Packed::new(individuals.map(Individual::genome)))),
            values,
            objectives: None,
            violations,
        }
    }

    /// Individuals of a multi-objective run, with their genomes (the population) or without
    /// them (its front).
    pub fn multi<'a, G: Genes, const N: usize>(
        individuals: impl ExactSizeIterator<Item = &'a Individual<G, Scores<N>>> + Clone,
        with_genomes: bool,
    ) -> Self {
        let (values, violations) = objective_values(individuals.clone());
        Self {
            genomes: with_genomes.then(|| {
                Box::new(Packed::new(individuals.map(Individual::genome))) as Box<dyn Matrix>
            }),
            values,
            objectives: Some(N),
            violations,
        }
    }
}

#[pymethods]
impl Snapshot {
    /// The genomes, a row each; None for a front.
    fn genomes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.genomes
            .as_ref()
            .map(|genomes| genomes.matrix(py))
            .transpose()
    }

    /// The scores, a 1-D array; or the objective values, a row each.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.objectives {
            None => Ok(PyArray1::from_slice(py, &self.values).into_any()),
            Some(objectives) => {
                let shape = (self.violations.len(), objectives);
                let values = Array2::from_shape_vec(shape, self.values.clone())
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
                Ok(values.into_pyarray(py).into_any())
            }
        }
    }

    /// The constraint violations.
    fn violations<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.violations)
    }
}

/// The objective values of individuals, a row of `N` each one after the other, and their
/// constraint violations; NaN for an invalid solution.
pub fn objective_values<'a, G: Genes + 'a, const N: usize>(
    individuals: impl IntoIterator<Item = &'a Individual<G, Scores<N>>>,
) -> (Vec<f64>, Vec<f64>) {
    let mut objectives = Vec::new();
    let mut violations = Vec::new();
    for individual in individuals {
        let scores = individual.fitness();
        objectives.extend(
            scores
                .and_then(|scores| scores.values())
                .unwrap_or([f64::NAN; N]),
        );
        // no violation for an invalid solution: NaN, as 0 means feasible
        let valid = scores.filter(|scores| scores.is_valid());
        violations.push(valid.map_or(f64::NAN, |scores| scores.violation()));
    }
    (objectives, violations)
}
