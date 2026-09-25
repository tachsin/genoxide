//! Permutation genomes: orderings of `0..n`.

use super::{Genome, Representation};
use crate::operator::check_size;
use crate::{Error, Result, StreamRng};
use std::ops::Deref;

/// A genome that is an ordering of `0..n`, e.g. the order of the cities of a tour.
///
/// It's always a permutation: it can be read like a slice, and changed only by swapping genes.
///
/// ```
/// use genoxide::genome::Order;
///
/// let mut tour = Order::new(vec![2, 0, 1])?;
/// tour.swap(0, 2);
/// assert_eq!(&tour[..], &[1, 0, 2]);
/// assert!(Order::new(vec![0, 0, 1]).is_err());
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Order {
    genes: Vec<usize>,
}

impl Order {
    /// The ordering `genes`, which must contain every number of `0..genes.len()` once.
    pub fn new(genes: Vec<usize>) -> Result<Self> {
        let mut seen = vec![false; genes.len()];
        for &gene in &genes {
            match seen.get_mut(gene) {
                Some(seen) if !*seen => *seen = true,
                _ => {
                    return Err(Error::InvalidGenome {
                        reason: format!(
                            "not a permutation of 0..{}: {gene} is out of range or repeated",
                            genes.len()
                        ),
                    });
                }
            }
        }
        Ok(Self { genes })
    }

    /// The ordering `0, 1, …, len - 1`.
    pub fn identity(len: usize) -> Self {
        Self {
            genes: (0..len).collect(),
        }
    }

    /// Swaps the genes at positions `a` and `b`.
    ///
    /// # Panics
    ///
    /// If a position is out of bounds.
    pub fn swap(&mut self, a: usize, b: usize) {
        self.genes.swap(a, b);
    }

    /// The genes, consuming the genome.
    pub fn into_vec(self) -> Vec<usize> {
        self.genes
    }

    // an ordering from genes that are known to be a permutation
    pub(crate) fn from_permutation(genes: Vec<usize>) -> Self {
        debug_assert!(
            Self::new(genes.clone()).is_ok(),
            "not a permutation: {genes:?}"
        );
        Self { genes }
    }

    // the genes, for changes that keep them a permutation (reverse, rotate, shuffle)
    pub(crate) fn genes_mut(&mut self) -> &mut [usize] {
        &mut self.genes
    }
}

impl Deref for Order {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        &self.genes
    }
}

impl Genome for Order {
    fn len(&self) -> usize {
        self.genes.len()
    }
}

/// Permutation genomes ([`Order`]) of `0..len`.
///
/// Point and uniform crossover don't apply: exchanging genes by position would create
/// duplicates. The [`permutation`](crate::operator::permutation) operators keep every child a
/// permutation, e.g. [`OrderCrossover`](crate::operator::OrderCrossover) with
/// [`InversionMutation`](crate::operator::InversionMutation).
///
/// ```
/// use genoxide::genome::{Permutation, Representation};
/// use genoxide::StreamRng;
///
/// let permutation = Permutation::new(5)?;
/// let genome = permutation.random_genome(&mut StreamRng::seed_from_u64(0));
/// assert!(permutation.validate(&genome).is_ok());
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Permutation {
    len: usize,
}

impl Permutation {
    /// Permutations of `0..len`, `len` between 1 and 2^24.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a length of 0 or above 2^24.
    pub fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::InvalidSetting {
                setting: "len",
                reason: "a permutation needs at least 1 gene".to_string(),
            });
        }
        Ok(Self {
            len: check_size("len", len)?,
        })
    }
}

impl Representation for Permutation {
    type Genome = Order;

    fn genome_len(&self) -> usize {
        self.len
    }

    fn random_genome(&self, rng: &mut StreamRng) -> Order {
        // Fisher-Yates
        let mut order = Order::identity(self.len);
        for position in (1..self.len).rev() {
            order.swap(position, rng.below(position + 1));
        }
        order
    }

    fn validate(&self, genome: &Order) -> Result<()> {
        // an Order is always a permutation, only the length can be wrong
        if genome.len() == self.len {
            Ok(())
        } else {
            Err(Error::InvalidGenome {
                reason: format!("expected {} genes, got {}", self.len, genome.len()),
            })
        }
    }
}

// validated like `new`
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Order {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename = "Order")]
        struct Raw {
            genes: Vec<usize>,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.genes).map_err(serde::de::Error::custom)
    }
}

// validated like `new`
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Permutation {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename = "Permutation")]
        struct Raw {
            len: usize,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.len).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn validation() {
        assert!(Permutation::new(0).is_err());
        assert!(Order::new(vec![1, 2]).is_err());
        assert!(Order::new(vec![]).is_ok());
        let permutation = Permutation::new(3).unwrap();
        assert!(permutation.validate(&Order::identity(3)).is_ok());
        assert!(permutation.validate(&Order::identity(2)).is_err());
    }

    #[test]
    fn random_is_uniform() {
        let permutation = Permutation::new(3).unwrap();
        let mut rng = StreamRng::seed_from_u64(0);
        let mut counts = std::collections::BTreeMap::new();
        for _ in 0..60_000 {
            *counts
                .entry(permutation.random_genome(&mut rng).into_vec())
                .or_insert(0) += 1;
        }
        assert_eq!(counts.len(), 6);
        assert!(
            counts.values().all(|&c| (9_500..10_500).contains(&c)),
            "{counts:?}"
        );
    }

    proptest! {
        #[test]
        fn random_genomes_are_permutations(len in 1usize..100, seed: u64) {
            let genome = Permutation::new(len).unwrap().random_genome(&mut StreamRng::seed_from_u64(seed));
            prop_assert!(Order::new(genome.into_vec()).is_ok());
        }
    }
}
