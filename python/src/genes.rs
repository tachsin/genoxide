//! Genomes as numpy arrays: one genome as a 1-D array, a batch as a 2-D array with a genome per
//! row.

use genoxide::genome::{Bits, Genome, Integers, Order, Reals};
use numpy::ndarray::Array2;
use numpy::{Element, IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A genome whose genes go into numpy arrays.
pub trait Genes: Genome {
    /// The numpy type of a gene: `bool`, `float64` or `int64`.
    type Element: Element;

    /// Appends the genes to `genes`.
    fn push_genes(&self, genes: &mut Vec<Self::Element>);

    /// The genome as real numbers, if it is: what the test problems of `genoxide::problems`
    /// evaluate.
    fn reals(&self) -> Option<&Reals> {
        None
    }
}

impl Genes for Bits {
    type Element = bool;

    fn push_genes(&self, genes: &mut Vec<bool>) {
        genes.extend(self.iter());
    }
}

impl Genes for Reals {
    type Element = f64;

    fn push_genes(&self, genes: &mut Vec<f64>) {
        genes.extend_from_slice(self);
    }

    fn reals(&self) -> Option<&Reals> {
        Some(self)
    }
}

impl Genes for Integers {
    type Element = i64;

    fn push_genes(&self, genes: &mut Vec<i64>) {
        genes.extend_from_slice(self);
    }
}

impl Genes for Order {
    type Element = i64;

    // positions fit in an i64: a permutation of more than 2^63 elements can't be allocated
    fn push_genes(&self, genes: &mut Vec<i64>) {
        genes.extend(self.iter().map(|&position| position as i64));
    }
}

/// The genome as a 1-D array.
pub fn array<'py, G: Genes>(py: Python<'py>, genome: &G) -> Bound<'py, PyArray1<G::Element>> {
    let mut genes = Vec::with_capacity(genome.len());
    genome.push_genes(&mut genes);
    PyArray1::from_vec(py, genes)
}

/// The genomes as a 2-D array, a genome per row. They all have the same length: every genome
/// type of the package has a fixed length.
pub fn matrix<'py, G: Genes>(
    py: Python<'py>,
    genomes: &[&G],
) -> PyResult<Bound<'py, PyArray2<G::Element>>> {
    let length = genomes.first().map_or(0, |genome| genome.len());
    let mut genes = Vec::with_capacity(genomes.len() * length);
    for genome in genomes {
        genome.push_genes(&mut genes);
    }
    let matrix = Array2::from_shape_vec((genomes.len(), length), genes)
        .map_err(|error| PyValueError::new_err(format!("genomes of different lengths: {error}")))?;
    Ok(matrix.into_pyarray(py))
}
