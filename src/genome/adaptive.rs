//! Real-valued genomes with a step size that evolves with them (self-adaptation).

use super::{Genome, Real, Reals, Representation, SwapGenes};
use crate::{Error, Result, StreamRng};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut, Range};

/// A genome of `f64` genes with its own mutation step size.
///
/// The step size is a fraction of each gene's range, like the `sigma` of
/// [`GaussianMutation`](crate::operator::GaussianMutation), and
/// [`SelfAdaptiveMutation`](crate::operator::SelfAdaptiveMutation) changes it along with the
/// genes. It dereferences to the genes, so a fitness function reads it like a slice.
///
/// ```
/// use genoxide::genome::{AdaptiveReals, Reals};
///
/// let genome = AdaptiveReals::new(Reals::from(vec![0.5, -1.0]), 0.1);
/// assert_eq!(genome.iter().sum::<f64>(), -0.5);
/// assert_eq!(genome.step(), 0.1);
/// ```
#[derive(Clone, Debug, Default)]
pub struct AdaptiveReals {
    genes: Reals,
    step: f64,
}

impl AdaptiveReals {
    /// A genome with these genes and step size.
    pub fn new(genes: Reals, step: f64) -> Self {
        Self { genes, step }
    }

    /// The genes.
    pub fn genes(&self) -> &Reals {
        &self.genes
    }

    /// The step size, a fraction of each gene's range.
    pub fn step(&self) -> f64 {
        self.step
    }

    /// The genes and the step size, consuming the genome.
    pub fn into_parts(self) -> (Reals, f64) {
        (self.genes, self.step)
    }

    pub(crate) fn set_step(&mut self, step: f64) {
        self.step = step;
    }
}

impl Deref for AdaptiveReals {
    type Target = [f64];

    fn deref(&self) -> &[f64] {
        &self.genes
    }
}

impl DerefMut for AdaptiveReals {
    fn deref_mut(&mut self) -> &mut [f64] {
        &mut self.genes
    }
}

impl PartialEq for AdaptiveReals {
    fn eq(&self, other: &Self) -> bool {
        self.genes == other.genes && self.step.to_bits() == other.step.to_bits()
    }
}

impl Eq for AdaptiveReals {}

impl Hash for AdaptiveReals {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.genes.hash(state);
        self.step.to_bits().hash(state);
    }
}

impl Genome for AdaptiveReals {
    fn len(&self) -> usize {
        self.genes.len()
    }
}

impl SwapGenes for AdaptiveReals {
    /// Exchanges the genes in `range`; each genome keeps its own step size.
    fn swap_range(&mut self, other: &mut Self, range: Range<usize>) {
        self.genes.swap_range(&mut other.genes, range);
    }
}

/// Real-valued genomes with a step size each ([`AdaptiveReals`]), for
/// [`SelfAdaptiveMutation`](crate::operator::SelfAdaptiveMutation).
///
/// The genes are those of a [`Real`] representation. Random genomes start with the initial step
/// size.
///
/// ```
/// use genoxide::genome::{AdaptiveReal, Real, Representation};
/// use genoxide::StreamRng;
///
/// let adaptive = AdaptiveReal::new(Real::uniform(5, -5.0..=5.0)?, 0.3)?;
/// let genome = adaptive.random_genome(&mut StreamRng::seed_from_u64(0));
/// assert_eq!(genome.step(), 0.3);
/// assert!(adaptive.validate(&genome).is_ok());
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveReal {
    real: Real,
    initial_step: f64,
}

impl AdaptiveReal {
    /// Genomes of `real` with a step size, starting at `initial_step` (positive and finite, a
    /// fraction of each gene's range; e.g. 0.3 to start broad).
    pub fn new(real: Real, initial_step: f64) -> Result<Self> {
        if !(initial_step > 0.0 && initial_step.is_finite()) {
            return Err(Error::InvalidSetting {
                setting: "initial_step",
                reason: format!("must be positive and finite, got {initial_step}"),
            });
        }
        Ok(Self { real, initial_step })
    }

    /// The representation of the genes.
    pub fn real(&self) -> &Real {
        &self.real
    }

    /// The step size of random genomes.
    pub fn initial_step(&self) -> f64 {
        self.initial_step
    }
}

impl Representation for AdaptiveReal {
    type Genome = AdaptiveReals;

    fn genome_len(&self) -> usize {
        self.real.genome_len()
    }

    fn random_genome(&self, rng: &mut StreamRng) -> AdaptiveReals {
        AdaptiveReals::new(self.real.random_genome(rng), self.initial_step)
    }

    fn validate(&self, genome: &AdaptiveReals) -> Result<()> {
        self.real.validate(genome.genes())?;
        if genome.step() > 0.0 && genome.step().is_finite() {
            Ok(())
        } else {
            Err(Error::InvalidGenome {
                reason: format!(
                    "the step size must be positive and finite, got {}",
                    genome.step()
                ),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation() {
        let real = Real::uniform(2, 0.0..=1.0).unwrap();
        assert!(AdaptiveReal::new(real.clone(), 0.0).is_err());
        assert!(AdaptiveReal::new(real.clone(), f64::INFINITY).is_err());
        let adaptive = AdaptiveReal::new(real, 0.2).unwrap();
        let genes = Reals::from(vec![0.5, 0.5]);
        assert!(
            adaptive
                .validate(&AdaptiveReals::new(genes.clone(), 0.1))
                .is_ok()
        );
        assert!(
            adaptive
                .validate(&AdaptiveReals::new(genes.clone(), 0.0))
                .is_err()
        );
        assert!(
            adaptive
                .validate(&AdaptiveReals::new(Reals::from(vec![2.0, 0.5]), 0.1))
                .is_err()
        );
    }

    #[test]
    fn crossover_keeps_the_step_sizes() {
        let mut a = AdaptiveReals::new(Reals::from(vec![0.0; 4]), 0.1);
        let mut b = AdaptiveReals::new(Reals::from(vec![1.0; 4]), 0.2);
        a.swap_range(&mut b, 1..3);
        assert_eq!(&a[..], &[0.0, 1.0, 1.0, 0.0]);
        assert_eq!((a.step(), b.step()), (0.1, 0.2));
    }
}
