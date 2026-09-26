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
//! | [`Real`] | [`Reals`] | `f64`, inclusive bounds per gene, of finite width |
//! | [`Permutation`] | [`Order`] | an ordering of `0..n` |
//! | [`AdaptiveReal`] | [`AdaptiveReals`] | `f64` like [`Real`], plus a mutation step size |

pub mod adaptive;
pub mod binary;
pub mod integer;
pub mod permutation;
pub mod real;

pub use adaptive::{AdaptiveReal, AdaptiveReals};
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
///
/// Implement it, with a [`Genome`], for a space of your own; then implement
/// [`Crossover`](crate::operator::Crossover) and [`Mutate`](crate::operator::Mutate) for it.
///
/// ```
/// use genoxide::genome::{Genome, Representation};
/// use genoxide::{Error, Result, StreamRng};
/// use rand::RngExt;
///
/// // a word of lowercase letters
/// #[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// struct Word(Vec<u8>);
///
/// impl Genome for Word {
///     fn len(&self) -> usize {
///         self.0.len()
///     }
/// }
///
/// // the words of `len` letters
/// #[derive(Clone, Debug)]
/// struct Words {
///     len: usize,
/// }
///
/// impl Representation for Words {
///     type Genome = Word;
///
///     fn genome_len(&self) -> usize {
///         self.len
///     }
///
///     fn random_genome(&self, rng: &mut StreamRng) -> Word {
///         Word((0..self.len).map(|_| rng.random_range(b'a'..=b'z')).collect())
///     }
///
///     fn validate(&self, word: &Word) -> Result<()> {
///         if word.0.len() == self.len && word.0.iter().all(u8::is_ascii_lowercase) {
///             Ok(())
///         } else {
///             let reason = format!("not a word of {} lowercase letters", self.len);
///             Err(Error::InvalidGenome { reason })
///         }
///     }
/// }
///
/// let words = Words { len: 5 };
/// let word = words.random_genome(&mut StreamRng::seed_from_u64(0));
/// assert!(words.validate(&word).is_ok());
/// assert!(words.validate(&Word(b"Hello".to_vec())).is_err());
/// ```
pub trait Representation: Clone + Debug + Send + Sync {
    /// The genome type of this representation.
    type Genome: Genome;

    /// The number of genes of every genome.
    fn genome_len(&self) -> usize;

    /// A random genome, uniformly distributed over the space.
    fn random_genome(&self, rng: &mut StreamRng) -> Self::Genome;

    /// Checks that a genome belongs to this space, e.g. a genome provided as a seed.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidGenome`](crate::Error::InvalidGenome) for a genome outside the space.
    fn validate(&self, genome: &Self::Genome) -> Result<()>;
}

/// Genomes whose genes can be exchanged position by position with another genome of the same
/// representation, which point and uniform crossover need.
///
/// Permutations don't implement it: exchanging genes by position would create duplicates.
#[diagnostic::on_unimplemented(
    message = "the genes of `{Self}` can't be exchanged by position",
    label = "point and uniform crossover need `SwapGenes`",
    note = "for a permutation, exchanging genes by position would duplicate genes: use a permutation crossover such as `OrderCrossover`"
)]
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
    ///
    /// # Panics
    ///
    /// If the genomes have different lengths and a gene is exchanged ([`Bits`] checks the lengths
    /// first).
    fn swap_uniform(&mut self, other: &mut Self, rate: f64, rng: &mut StreamRng) {
        let chance = crate::rng::Chance::new(rate);
        rng.chosen(chance, self.len(), |_, index| self.swap_gene(other, index));
    }
}
