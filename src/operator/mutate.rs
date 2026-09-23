//! Mutation: random changes to a genome.

use super::{Mutate, check_rate};
use crate::genome::Binary;
use crate::genome::Bits;
use crate::rng::Chance;
use crate::{Error, Result, StreamRng};

/// Bit-flip mutation for [`Binary`] genomes: flips each bit with a probability, or exactly `n`
/// bits. The genome always changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BitFlip {
    mode: Mode,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    PerGene { rate: f64, chance: Chance },
    Count(usize),
}

impl BitFlip {
    /// Flips each bit with probability `rate` (greater than 0 and at most 1). If no bit is flipped,
    /// one random bit is flipped, so the genome always changes. A common choice is `1 / length`.
    pub fn per_gene(rate: f64) -> Result<Self> {
        let rate = check_rate("bit_flip_rate", rate)?;
        Ok(Self {
            mode: Mode::PerGene {
                rate,
                chance: Chance::new(rate),
            },
        })
    }

    /// Flips exactly `count` distinct random bits (at least 1, at most the genome length).
    pub fn count(count: usize) -> Result<Self> {
        if count == 0 {
            return Err(Error::InvalidSetting {
                setting: "bit_flip_count",
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            mode: Mode::Count(count),
        })
    }
}

impl Mutate<Binary> for BitFlip {
    fn mutate(&self, _representation: &Binary, genome: &mut Bits, rng: &mut StreamRng) {
        let len = genome.len();
        if len == 0 {
            return;
        }
        match self.mode {
            Mode::PerGene { chance, .. } => {
                let mut flipped = false;
                for index in 0..len {
                    if rng.chance(chance) {
                        genome.flip(index);
                        flipped = true;
                    }
                }
                if !flipped {
                    genome.flip(rng.below(len));
                }
            }
            Mode::Count(count) => {
                for index in rng.sample_distinct(count.min(len), len) {
                    genome.flip(index);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Representation;
    use proptest::prelude::*;

    #[test]
    fn validation() {
        assert!(BitFlip::per_gene(0.0).is_err());
        assert!(BitFlip::per_gene(1.5).is_err());
        assert!(BitFlip::count(0).is_err());
    }

    #[test]
    fn per_gene_rate() {
        let binary = Binary::new(1_000).unwrap();
        let mutation = BitFlip::per_gene(0.05).unwrap();
        let mut rng = StreamRng::seed_from_u64(0);
        let mut flipped = 0;
        for _ in 0..100 {
            let mut genome = Bits::zeros(1_000);
            mutation.mutate(&binary, &mut genome, &mut rng);
            flipped += genome.count_ones();
        }
        let rate = flipped as f64 / 100_000.0;
        assert!((rate - 0.05).abs() < 0.003, "rate {rate}");
    }

    proptest! {
        #[test]
        fn always_changes(len in 1usize..200, rate in 0.0001..=1.0f64, seed: u64) {
            let binary = Binary::new(len).unwrap();
            let mut rng = StreamRng::seed_from_u64(seed);
            let original = binary.random_genome(&mut rng);
            let mut genome = original.clone();
            BitFlip::per_gene(rate).unwrap().mutate(&binary, &mut genome, &mut rng);
            prop_assert_ne!(&genome, &original);
            prop_assert!(binary.validate(&genome).is_ok());
        }

        #[test]
        fn count_flips_exactly(len in 1usize..200, count in 1usize..20, seed: u64) {
            let binary = Binary::new(len).unwrap();
            let mut genome = Bits::zeros(len);
            BitFlip::count(count).unwrap().mutate(&binary, &mut genome, &mut StreamRng::seed_from_u64(seed));
            prop_assert_eq!(genome.count_ones(), count.min(len));
        }
    }
}
