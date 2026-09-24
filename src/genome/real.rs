//! Real-valued genomes, bounded per gene.

use super::{Genome, Representation, SwapGenes};
use crate::{Error, Result, StreamRng};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut, Range, RangeInclusive};

/// A genome of `f64` genes.
///
/// It dereferences to a slice, so it can be read and changed like one. Genomes are compared and
/// hashed by the bits of their genes, so `0.0` and `-0.0` are different genes, and NaN equals
/// NaN.
///
/// ```
/// use genoxide::genome::Reals;
///
/// let genome = Reals::from(vec![0.5, -1.5]);
/// let norm = genome.iter().map(|x| x * x).sum::<f64>().sqrt();
/// assert!(norm > 1.5);
/// ```
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Reals {
    genes: Vec<f64>,
}

impl Reals {
    /// The genes, consuming the genome.
    pub fn into_vec(self) -> Vec<f64> {
        self.genes
    }
}

impl From<Vec<f64>> for Reals {
    fn from(genes: Vec<f64>) -> Self {
        Self { genes }
    }
}

impl FromIterator<f64> for Reals {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
    }
}

impl Deref for Reals {
    type Target = [f64];

    fn deref(&self) -> &[f64] {
        &self.genes
    }
}

impl DerefMut for Reals {
    fn deref_mut(&mut self) -> &mut [f64] {
        &mut self.genes
    }
}

impl PartialEq for Reals {
    fn eq(&self, other: &Self) -> bool {
        self.genes.len() == other.genes.len()
            && self
                .genes
                .iter()
                .zip(&other.genes)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for Reals {}

impl Hash for Reals {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.genes.len().hash(state);
        for gene in &self.genes {
            gene.to_bits().hash(state);
        }
    }
}

impl Genome for Reals {
    fn len(&self) -> usize {
        self.genes.len()
    }
}

impl SwapGenes for Reals {
    fn swap_range(&mut self, other: &mut Self, range: Range<usize>) {
        assert_eq!(self.len(), other.len(), "genomes of different lengths");
        self.genes[range.clone()].swap_with_slice(&mut other.genes[range]);
    }
}

/// Real-valued genomes ([`Reals`]) with inclusive, finite bounds per gene.
///
/// ```
/// use genoxide::genome::{Real, Representation};
/// use genoxide::StreamRng;
///
/// let real = Real::uniform(10, -5.12..=5.12)?;
/// let genome = real.random_genome(&mut StreamRng::seed_from_u64(0));
/// assert!(genome.iter().all(|x| (-5.12..=5.12).contains(x)));
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Real {
    bounds: Vec<RangeInclusive<f64>>,
    // the genes with more than one possible value
    #[cfg_attr(feature = "serde", serde(skip))]
    variable: Vec<usize>,
}

impl Real {
    /// Real-valued genomes with these bounds, one per gene. At least one gene, and every range
    /// must be finite and contain a value.
    pub fn new<I: IntoIterator<Item = RangeInclusive<f64>>>(bounds: I) -> Result<Self> {
        let bounds: Vec<_> = bounds.into_iter().collect();
        if bounds.is_empty() {
            return Err(Error::InvalidSetting {
                setting: "bounds",
                reason: "a real-valued genome needs at least 1 gene".to_string(),
            });
        }
        for (gene, range) in bounds.iter().enumerate() {
            let invalid = |problem: &str| {
                Err(Error::InvalidSetting {
                    setting: "bounds",
                    reason: format!("the range of gene {gene} {problem}: {range:?}"),
                })
            };
            if range.is_empty() {
                // also NaN bounds
                return invalid("is empty");
            }
            if !(range.end() - range.start()).is_finite() {
                return invalid("is not finite");
            }
        }
        let variable = bounds
            .iter()
            .enumerate()
            .filter(|(_, range)| range.start() < range.end())
            .map(|(gene, _)| gene)
            .collect();
        Ok(Self { bounds, variable })
    }

    /// Real-valued genomes of `len` genes, all with the same bounds.
    pub fn uniform(len: usize, bounds: RangeInclusive<f64>) -> Result<Self> {
        Self::new(std::iter::repeat_n(bounds, len))
    }

    /// The bounds, one per gene.
    pub fn bounds(&self) -> &[RangeInclusive<f64>] {
        &self.bounds
    }

    pub(crate) fn variable_genes(&self) -> &[usize] {
        &self.variable
    }
}

// a uniformly random value in `range`, which is finite
pub(crate) fn random_in(range: &RangeInclusive<f64>, rng: &mut StreamRng) -> f64 {
    let (start, end) = (*range.start(), *range.end());
    // rounding can reach the end, never pass it
    (start + rng.unit_f64() * (end - start)).min(end)
}

// a uniformly random value in `range` other than `current`; the range has more than one value
pub(crate) fn random_other_in(
    range: &RangeInclusive<f64>,
    current: f64,
    rng: &mut StreamRng,
) -> f64 {
    // a repeat has probability 0 for any but the tiniest ranges, so this almost never loops
    for _ in 0..64 {
        let value = random_in(range, rng);
        if value != current {
            return value;
        }
    }
    if current == *range.start() {
        *range.end()
    } else {
        *range.start()
    }
}

impl Representation for Real {
    type Genome = Reals;

    fn genome_len(&self) -> usize {
        self.bounds.len()
    }

    fn random_genome(&self, rng: &mut StreamRng) -> Reals {
        self.bounds
            .iter()
            .map(|range| random_in(range, rng))
            .collect()
    }

    fn validate(&self, genome: &Reals) -> Result<()> {
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
impl<'de> serde::Deserialize<'de> for Real {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename = "Real")]
        struct Raw {
            bounds: Vec<RangeInclusive<f64>>,
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
        assert!(Real::new([]).is_err());
        assert!(Real::new([0.0..=f64::NAN]).is_err());
        assert!(Real::new([1.0..=0.0]).is_err());
        assert!(Real::new([0.0..=f64::INFINITY]).is_err());
        assert!(Real::new([-f64::MAX..=f64::MAX]).is_err());
        let real = Real::new([0.0..=1.0, 2.0..=2.0]).unwrap();
        assert_eq!(real.variable_genes(), &[0]);
        assert!(real.validate(&Reals::from(vec![1.0, 2.0])).is_ok());
        assert!(real.validate(&Reals::from(vec![f64::NAN, 2.0])).is_err());
        assert!(real.validate(&Reals::from(vec![0.5])).is_err());
    }

    #[test]
    fn random_is_uniform() {
        let range = 0.0..=7.0;
        let mut rng = StreamRng::seed_from_u64(0);
        let mut counts = [0usize; 7];
        for _ in 0..70_000 {
            counts[(random_in(&range, &mut rng) as usize).min(6)] += 1;
        }
        assert!(
            counts.iter().all(|&c| (9_500..10_500).contains(&c)),
            "{counts:?}"
        );
    }

    #[test]
    fn tiny_range() {
        // 1.0 and the next float
        let range = 1.0..=f64::from_bits(1.0f64.to_bits() + 1);
        let mut rng = StreamRng::seed_from_u64(0);
        for current in [*range.start(), *range.end()] {
            assert_ne!(random_other_in(&range, current, &mut rng), current);
        }
    }

    #[test]
    fn equality_is_by_bits() {
        assert_ne!(Reals::from(vec![0.0]), Reals::from(vec![-0.0]));
        assert_eq!(Reals::from(vec![f64::NAN]), Reals::from(vec![f64::NAN]));
    }

    fn any_range() -> impl Strategy<Value = RangeInclusive<f64>> {
        (-1e6..1e6f64, 0.0..1e6f64).prop_map(|(start, len)| start..=start + len)
    }

    proptest! {
        #[test]
        fn random_genomes_are_valid(bounds in prop::collection::vec(any_range(), 1..20), seed: u64) {
            let real = Real::new(bounds).unwrap();
            let genome = real.random_genome(&mut StreamRng::seed_from_u64(seed));
            prop_assert!(real.validate(&genome).is_ok());
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
    }
}
