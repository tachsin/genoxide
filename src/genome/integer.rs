//! Integer genomes, bounded per gene.

use super::{Genome, Representation, SwapGenes};
use crate::{Error, Result, StreamRng};
use rand::Rng;
use std::ops::{Deref, DerefMut, Range, RangeInclusive};

/// A genome of `i64` genes.
///
/// It dereferences to a slice, so it can be read and changed like one.
///
/// ```
/// use genoxide::genome::Integers;
///
/// let mut genome = Integers::from(vec![3, -1, 4]);
/// genome[1] = 5;
/// assert_eq!(genome.iter().sum::<i64>(), 12);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Integers {
    genes: Vec<i64>,
}

impl Integers {
    /// The genes, consuming the genome.
    pub fn into_vec(self) -> Vec<i64> {
        self.genes
    }
}

impl From<Vec<i64>> for Integers {
    fn from(genes: Vec<i64>) -> Self {
        Self { genes }
    }
}

impl FromIterator<i64> for Integers {
    fn from_iter<I: IntoIterator<Item = i64>>(iter: I) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
    }
}

impl Deref for Integers {
    type Target = [i64];

    fn deref(&self) -> &[i64] {
        &self.genes
    }
}

impl DerefMut for Integers {
    fn deref_mut(&mut self) -> &mut [i64] {
        &mut self.genes
    }
}

impl Genome for Integers {
    fn len(&self) -> usize {
        self.genes.len()
    }
}

impl SwapGenes for Integers {
    fn swap_range(&mut self, other: &mut Self, range: Range<usize>) {
        assert_eq!(self.len(), other.len(), "genomes of different lengths");
        self.genes[range.clone()].swap_with_slice(&mut other.genes[range]);
    }
}

/// Integer genomes ([`Integers`]) with inclusive bounds per gene.
///
/// ```
/// use genoxide::genome::{Integer, Representation};
/// use genoxide::StreamRng;
///
/// // a die, a coin and a percentage
/// let integer = Integer::new([1..=6, 0..=1, 0..=100])?;
/// let genome = integer.random_genome(&mut StreamRng::seed_from_u64(0));
/// assert!((1..=6).contains(&genome[0]));
/// assert!(integer.validate(&genome).is_ok());
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Integer {
    bounds: Vec<RangeInclusive<i64>>,
    // the genes with more than one possible value
    #[cfg_attr(feature = "serde", serde(skip))]
    variable: Vec<usize>,
}

impl Integer {
    /// Integer genomes with these bounds, one per gene. At least one gene, and every range must
    /// contain a value.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for no bounds or an empty range.
    pub fn new<I: IntoIterator<Item = RangeInclusive<i64>>>(bounds: I) -> Result<Self> {
        let bounds: Vec<_> = bounds.into_iter().collect();
        if bounds.is_empty() {
            return Err(Error::InvalidSetting {
                setting: "bounds",
                reason: "an integer genome needs at least 1 gene".to_string(),
            });
        }
        if let Some((gene, range)) = bounds
            .iter()
            .enumerate()
            .find(|(_, range)| range.is_empty())
        {
            return Err(Error::InvalidSetting {
                setting: "bounds",
                reason: format!("the range of gene {gene} is empty: {range:?}"),
            });
        }
        let variable = bounds
            .iter()
            .enumerate()
            .filter(|(_, range)| range.start() < range.end())
            .map(|(gene, _)| gene)
            .collect();
        Ok(Self { bounds, variable })
    }

    /// Integer genomes of `len` genes, all with the same bounds.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a length of 0 or an empty range.
    pub fn uniform(len: usize, bounds: RangeInclusive<i64>) -> Result<Self> {
        Self::new(std::iter::repeat_n(bounds, len))
    }

    /// The bounds, one per gene.
    pub fn bounds(&self) -> &[RangeInclusive<i64>] {
        &self.bounds
    }

    pub(crate) fn variable_genes(&self) -> &[usize] {
        &self.variable
    }
}

// the number of values in `range` minus 1, which always fits in a u64
fn span_minus_one(range: &RangeInclusive<i64>) -> u64 {
    range.end().abs_diff(*range.start())
}

// the value at `offset` from the start of `range`
fn at_offset(range: &RangeInclusive<i64>, offset: u64) -> i64 {
    range.start().wrapping_add_unsigned(offset)
}

// a uniformly random value in `range`
pub(crate) fn random_in(range: &RangeInclusive<i64>, rng: &mut StreamRng) -> i64 {
    let offset = match span_minus_one(range).checked_add(1) {
        Some(span) => rng.below_u64(span),
        // the whole i64 range
        None => rng.next_u64(),
    };
    at_offset(range, offset)
}

// a uniformly random value in `range` other than `current`, which is in `range`; the range has
// at least 2 values
pub(crate) fn random_other_in(
    range: &RangeInclusive<i64>,
    current: i64,
    rng: &mut StreamRng,
) -> i64 {
    let offset = rng.below_u64(span_minus_one(range));
    let current_offset = current.abs_diff(*range.start());
    at_offset(
        range,
        if offset >= current_offset {
            offset + 1
        } else {
            offset
        },
    )
}

impl Representation for Integer {
    type Genome = Integers;

    fn genome_len(&self) -> usize {
        self.bounds.len()
    }

    fn random_genome(&self, rng: &mut StreamRng) -> Integers {
        self.bounds
            .iter()
            .map(|range| random_in(range, rng))
            .collect()
    }

    fn validate(&self, genome: &Integers) -> Result<()> {
        if genome.len() != self.bounds.len() {
            return Err(Error::InvalidGenome {
                reason: format!("expected {} genes, got {}", self.bounds.len(), genome.len()),
            });
        }
        match genome
            .iter()
            .zip(&self.bounds)
            .position(|(gene, range)| !range.contains(gene))
        {
            Some(gene) => Err(Error::InvalidGenome {
                reason: format!(
                    "gene {gene} is {}, outside {:?}",
                    genome[gene], self.bounds[gene]
                ),
            }),
            None => Ok(()),
        }
    }
}

// validated like `new`
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Integer {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename = "Integer")]
        struct Raw {
            bounds: Vec<RangeInclusive<i64>>,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.bounds).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn validation() {
        assert!(Integer::new([]).is_err());
        #[allow(clippy::reversed_empty_ranges)]
        let empty = 1..=0;
        assert!(Integer::new([0..=1, empty]).is_err());
        assert!(Integer::uniform(0, 0..=1).is_err());
        let integer = Integer::new([0..=1, 5..=5]).unwrap();
        assert_eq!(integer.variable_genes(), &[0]);
        assert!(integer.validate(&Integers::from(vec![1, 5])).is_ok());
        assert!(integer.validate(&Integers::from(vec![2, 5])).is_err());
        assert!(integer.validate(&Integers::from(vec![1])).is_err());
    }

    #[test]
    fn random_is_uniform() {
        let range = -3..=3;
        let mut rng = StreamRng::seed_from_u64(0);
        let mut counts = [0usize; 7];
        for _ in 0..70_000 {
            counts[(random_in(&range, &mut rng) + 3) as usize] += 1;
        }
        assert!(
            counts.iter().all(|&c| (9_500..10_500).contains(&c)),
            "{counts:?}"
        );
    }

    #[test]
    fn extreme_ranges() {
        let mut rng = StreamRng::seed_from_u64(0);
        let full = i64::MIN..=i64::MAX;
        let values: Vec<i64> = (0..100).map(|_| random_in(&full, &mut rng)).collect();
        assert!(values.iter().any(|&v| v < 0) && values.iter().any(|&v| v > 0));
        for current in [i64::MIN, 0, i64::MAX] {
            assert_ne!(random_other_in(&full, current, &mut rng), current);
        }
        let top = i64::MAX - 1..=i64::MAX;
        assert_eq!(random_other_in(&top, i64::MAX, &mut rng), i64::MAX - 1);
        assert_eq!(random_other_in(&top, i64::MAX - 1, &mut rng), i64::MAX);
    }

    fn any_range() -> impl Strategy<Value = RangeInclusive<i64>> {
        prop_oneof![
            (-10i64..10, 0i64..10).prop_map(|(start, len)| start..=start + len),
            (any::<i64>(), any::<i64>()).prop_map(|(a, b)| a.min(b)..=a.max(b)),
        ]
    }

    proptest! {
        #[test]
        fn random_genomes_are_valid(bounds in prop::collection::vec(any_range(), 1..20), seed: u64) {
            let integer = Integer::new(bounds).unwrap();
            let genome = integer.random_genome(&mut StreamRng::seed_from_u64(seed));
            prop_assert!(integer.validate(&genome).is_ok());
        }

        #[test]
        fn random_other_differs(range in any_range(), seed: u64) {
            prop_assume!(range.start() < range.end());
            let mut rng = StreamRng::seed_from_u64(seed);
            let current = random_in(&range, &mut rng);
            let other = random_other_in(&range, current, &mut rng);
            prop_assert!(range.contains(&other));
            prop_assert_ne!(other, current);
        }

        #[test]
        fn swap_range(a in prop::collection::vec(any::<i64>(), 1..20), seed: u64) {
            let b: Vec<i64> = a.iter().map(|gene| gene.wrapping_add(1)).collect();
            let (mut x, mut y) = (Integers::from(a.clone()), Integers::from(b.clone()));
            let mut rng = StreamRng::seed_from_u64(seed);
            let start = rng.below(a.len());
            let end = start + rng.below(a.len() - start + 1);
            x.swap_range(&mut y, start..end);
            for gene in 0..a.len() {
                let swapped = (start..end).contains(&gene);
                prop_assert_eq!(x[gene], if swapped { b[gene] } else { a[gene] });
                prop_assert_eq!(y[gene], if swapped { a[gene] } else { b[gene] });
            }
        }
    }
}
