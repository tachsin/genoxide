//! Genomes as numpy arrays: one genome as a 1-D array, a batch as a 2-D array with a genome per
//! row.

use genoxide::genome::{Bits, Genome, Integers, Order, Reals};
use numpy::ndarray::Array2;
use numpy::{Element, IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cell::RefCell;
use std::thread::LocalKey;

/// A genome whose genes go into numpy arrays.
pub trait Genes: Genome + 'static {
    /// The numpy type of a gene: `bool`, `float64` or `int64`.
    type Element: Element + Copy;

    /// What a genome is kept as in a [`crate::snapshot::Snapshot`], until its genes are read: its
    /// genes, or the 64-bit words of its bits, eight times smaller than their bools.
    type Word: Copy + Send + Sync + 'static;

    /// Appends the genes to `genes`.
    fn push_genes(&self, genes: &mut Vec<Self::Element>);

    /// Appends the genome's words to `words`.
    fn push_words(&self, words: &mut Vec<Self::Word>);

    /// Appends the genes of a genome of `length` genes, from its words, to `genes`.
    fn push_genes_of(words: &[Self::Word], length: usize, genes: &mut Vec<Self::Element>);

    /// The genome as a 1-D array.
    fn array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Self::Element>>;

    /// The genome as real numbers, if it is: what the test problems of `genoxide::problems`
    /// evaluate.
    fn reals(&self) -> Option<&Reals> {
        None
    }

    /// The genome as whole numbers, if it is: what the integer test problems evaluate.
    fn integers(&self) -> Option<&Integers> {
        None
    }

    /// The genome as bits, if it is: what the binary test problems evaluate.
    fn bits(&self) -> Option<&Bits> {
        None
    }
}

const WORD_BITS: usize = u64::BITS as usize;

// the 8 bits of each byte as bools, from the lowest bit
const BYTE_BITS: [[bool; 8]; 256] = {
    let mut table = [[false; 8]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut bit = 0;
        while bit < 8 {
            table[byte][bit] = byte >> bit & 1 == 1;
            bit += 1;
        }
        byte += 1;
    }
    table
};

impl Genes for Bits {
    type Element = bool;
    type Word = u64;

    fn push_genes(&self, genes: &mut Vec<bool>) {
        Self::push_genes_of(self.as_words(), self.len(), genes);
    }

    fn push_words(&self, words: &mut Vec<u64>) {
        words.extend_from_slice(self.as_words());
    }

    // a byte of bits at a time, from a table: bit by bit takes ten times as long
    fn push_genes_of(words: &[u64], length: usize, genes: &mut Vec<bool>) {
        let start = genes.len();
        genes.resize(start + length, false);
        let genes = &mut genes[start..];
        let (chunks, rest) = genes.as_chunks_mut::<WORD_BITS>();
        let full = chunks.len();
        for (chunk, word) in chunks.iter_mut().zip(words) {
            let (bytes, _) = chunk.as_chunks_mut::<8>();
            for (genes, byte) in bytes.iter_mut().zip(word.to_le_bytes()) {
                *genes = BYTE_BITS[usize::from(byte)];
            }
        }
        if let Some(&word) = words.get(full) {
            for (bit, gene) in rest.iter_mut().enumerate() {
                *gene = word >> bit & 1 == 1;
            }
        }
    }

    fn array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        through_buffer(py, &BOOLS, self)
    }

    fn bits(&self) -> Option<&Bits> {
        Some(self)
    }
}

impl Genes for Reals {
    type Element = f64;
    type Word = f64;

    fn push_genes(&self, genes: &mut Vec<f64>) {
        genes.extend_from_slice(self);
    }

    fn push_words(&self, words: &mut Vec<f64>) {
        words.extend_from_slice(self);
    }

    fn push_genes_of(words: &[f64], _: usize, genes: &mut Vec<f64>) {
        genes.extend_from_slice(words);
    }

    fn array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self)
    }

    fn reals(&self) -> Option<&Reals> {
        Some(self)
    }
}

impl Genes for Integers {
    type Element = i64;
    type Word = i64;

    fn push_genes(&self, genes: &mut Vec<i64>) {
        genes.extend_from_slice(self);
    }

    fn push_words(&self, words: &mut Vec<i64>) {
        words.extend_from_slice(self);
    }

    fn push_genes_of(words: &[i64], _: usize, genes: &mut Vec<i64>) {
        genes.extend_from_slice(words);
    }

    fn array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, self)
    }

    fn integers(&self) -> Option<&Integers> {
        Some(self)
    }
}

impl Genes for Order {
    type Element = i64;
    type Word = i64;

    // positions fit in an i64: a permutation of more than 2^63 elements can't be allocated
    fn push_genes(&self, genes: &mut Vec<i64>) {
        genes.extend(self.iter().map(|&position| position as i64));
    }

    fn push_words(&self, words: &mut Vec<i64>) {
        self.push_genes(words);
    }

    fn push_genes_of(words: &[i64], _: usize, genes: &mut Vec<i64>) {
        genes.extend_from_slice(words);
    }

    fn array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        through_buffer(py, &INTEGERS, self)
    }
}

thread_local! {
    // the genes of a genome that isn't a slice of them, before they're copied into an array: a
    // buffer per thread, reused. Numpy allocates the array's memory, which is faster than an
    // array made from a vector, kept alive by a Python object of its own.
    static BOOLS: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
    static INTEGERS: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
}

// the genome as a 1-D array, its genes put in `buffer` first
fn through_buffer<'py, G: Genes>(
    py: Python<'py>,
    buffer: &'static LocalKey<RefCell<Vec<G::Element>>>,
    genome: &G,
) -> Bound<'py, PyArray1<G::Element>> {
    buffer.with_borrow_mut(|genes| {
        genes.clear();
        genome.push_genes(genes);
        PyArray1::from_slice(py, genes)
    })
}

/// The genome as a 1-D array.
pub fn array<'py, G: Genes>(py: Python<'py>, genome: &G) -> Bound<'py, PyArray1<G::Element>> {
    genome.array(py)
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
