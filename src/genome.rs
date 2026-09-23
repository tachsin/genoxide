//! Genomes (the encoded solutions) and representations (the spaces they live in).
//!
//! - A [`Genome`] is the value that evolves, e.g. [`Bits`].
//! - A [`Representation`] describes the space of genomes, e.g. [`Binary`] with a length. It
//!   creates random genomes and validates genomes provided by users (e.g. seeds).

pub mod binary;

pub use binary::{Binary, Bits};

use crate::{Result, StreamRng};
use std::fmt::Debug;
use std::hash::Hash;

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
