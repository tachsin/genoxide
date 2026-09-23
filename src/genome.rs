//! Genomes (the encoded solutions) and representations (the spaces they live in).
//!
//! - A [`Genome`] is the value that evolves, e.g. [`Bits`].
//! - A [`Representation`] describes the space of genomes, e.g. [`Binary`] with a length. It
//!   creates random genomes and validates genomes provided by users (e.g. seeds).
//!
//! | Representation | Genome | Genes |
//! |---|---|---|
//! | [`Binary`] | [`Bits`] | bits, packed 64 per word |
//! | [`Integer`] | [`Integers`] | `i64`, inclusive bounds per gene |
//! | [`Real`] | [`Reals`] | `f64`, inclusive finite bounds per gene |
//! | [`Permutation`] | [`Order`] | an ordering of `0..n` |

pub mod binary;
pub mod integer;
pub mod permutation;
pub mod real;

pub use binary::{Binary, Bits};
pub use integer::{Integer, Integers};
pub use permutation::{Order, Permutation};
pub use real::{Real, Reals};

use crate::{Result, StreamRng};
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Range;

/// An encoded solution: the value that evolves.
///
/// Genomes are compared and hashed by value, which is used to detect duplicates, e.g. for a hall
/// of fame of unique solutions or a fitness cache.
pub trait Genome: Clone + Debug + PartialEq + Eq + Hash + Send + Sync {
    /// The number of genes.
    fn len(&self) -> usize;

    /// Whether there are no genes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The space of genomes of a problem, e.g. [`Binary`] genomes of a given length.
pub trait Representation: Clone + Debug + Send + Sync {
    /// The genome type of this representation.
    type Genome: Genome;

    /// The number of genes of every genome.
    fn genome_len(&self) -> usize;

    /// A random genome, uniformly distributed over the space.
    fn random_genome(&self, rng: &mut StreamRng) -> Self::Genome;

    /// Checks that a genome belongs to this space, e.g. a genome provided as a seed.
    fn validate(&self, genome: &Self::Genome) -> Result<()>;
}

/// Genomes whose genes can be exchanged position by position with another genome of the same
/// representation, which point and uniform crossover need.
///
/// Permutations don't implement it: exchanging genes by position would create duplicates.
pub trait SwapGenes: Genome {
    /// Exchanges the genes in `range` with `other`.
    ///
    /// # Panics
    ///
    /// If the genomes have different lengths or the range is out of bounds.
    fn swap_range(&mut self, other: &mut Self, range: Range<usize>);

    /// Exchanges the gene at `index` with `other`.
    ///
    /// # Panics
    ///
    /// If the genomes have different lengths or the index is out of bounds.
    fn swap_gene(&mut self, other: &mut Self, index: usize) {
        self.swap_range(other, index..index + 1);
    }

    /// Exchanges each gene with `other` with probability `rate` (uniform crossover). `rate` is in
    /// `[0, 1]`.
    fn swap_uniform(&mut self, other: &mut Self, rate: f64, rng: &mut StreamRng) {
        let chance = crate::rng::Chance::new(rate);
        for index in 0..self.len() {
            if rng.chance(chance) {
                self.swap_gene(other, index);
            }
        }
    }
}
