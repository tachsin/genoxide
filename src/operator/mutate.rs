//! Mutation: random changes to a genome.

use super::{Mutate, check_rate};
use crate::genome::{Binary, Bits, Integer, Integers, Order, Permutation, Real, Reals};
use crate::rng::Chance;
use crate::{Error, Result, StreamRng};

/// Which genes a mutation changes: each with a probability, or `n` (all if there are fewer).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    PerGene { chance: Chance },
    Count(usize),
}

impl Mode {
    fn per_gene(setting: &'static str, rate: f64) -> Result<Self> {
        Ok(Mode::PerGene {
            chance: Chance::new(check_rate(setting, rate)?),
        })
    }

    fn count(setting: &'static str, count: usize) -> Result<Self> {
        if count == 0 {
            return Err(Error::InvalidSetting {
                setting,
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Mode::Count(count))
    }

    // calls `change` with the genes to change, out of `genes` candidates: at least one
    fn apply(
        self,
        genes: usize,
        gene: impl Fn(usize) -> usize,
        rng: &mut StreamRng,
        mut change: impl FnMut(usize, &mut StreamRng),
    ) {
        if genes == 0 {
            return;
        }
        match self {
            Mode::PerGene { chance } => {
                let mut changed = false;
                for candidate in 0..genes {
                    if rng.chance(chance) {
                        change(gene(candidate), rng);
                        changed = true;
                    }
                }
                if !changed {
                    let candidate = rng.below(genes);
                    change(gene(candidate), rng);
                }
            }
            Mode::Count(count) => {
                for candidate in rng.sample_distinct(count.min(genes), genes) {
                    change(gene(candidate), rng);
                }
            }
        }
    }
}

/// Bit-flip mutation for [`Binary`] genomes: flips each bit with a probability, or `n` bits. The
/// genome always changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BitFlip {
    mode: Mode,
}

impl BitFlip {
    /// Flips each bit with probability `rate` (greater than 0 and at most 1). If no bit is flipped,
    /// one random bit is flipped, so the genome always changes. A common choice is `1 / length`.
    pub fn per_gene(rate: f64) -> Result<Self> {
        Ok(Self {
            mode: Mode::per_gene("bit_flip_rate", rate)?,
        })
    }

    /// Flips `count` distinct random bits, `count` at least 1. A genome with fewer bits has all of
    /// them flipped: the operator doesn't know the genome length when it's created.
    pub fn count(count: usize) -> Result<Self> {
        Ok(Self {
            mode: Mode::count("bit_flip_count", count)?,
        })
    }
}

impl Mutate<Binary> for BitFlip {
    fn mutate(&self, _representation: &Binary, genome: &mut Bits, rng: &mut StreamRng) {
        self.mode
            .apply(genome.len(), |bit| bit, rng, |bit, _| genome.flip(bit));
    }
}

/// Uniform mutation for [`Integer`] and [`Real`] genomes: gives genes a new random value within
/// their bounds, different from the current one. Each gene with a probability, or `n` genes.
///
/// The genome always changes. Genes whose bounds allow a single value are never changed.
///
/// ```
/// use genoxide::genome::{Integer, Integers, Representation};
/// use genoxide::operator::{Mutate, UniformMutation};
/// use genoxide::StreamRng;
///
/// let integer = Integer::uniform(5, 0..=9)?;
/// let mut genome = Integers::from(vec![0; 5]);
/// UniformMutation::count(2)?.mutate(&integer, &mut genome, &mut StreamRng::seed_from_u64(0));
/// assert_eq!(genome.iter().filter(|&&gene| gene != 0).count(), 2);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformMutation {
    mode: Mode,
}

impl UniformMutation {
    /// Changes each gene with probability `rate` (greater than 0 and at most 1). If no gene is
    /// changed, one random gene is changed, so the genome always changes. A common choice is
    /// `1 / length`.
    pub fn per_gene(rate: f64) -> Result<Self> {
        Ok(Self {
            mode: Mode::per_gene("uniform_mutation_rate", rate)?,
        })
    }

    /// Changes `count` distinct random genes, `count` at least 1. If fewer genes can change, all
    /// of them are changed: the operator doesn't know the representation when it's created.
    pub fn count(count: usize) -> Result<Self> {
        Ok(Self {
            mode: Mode::count("uniform_mutation_count", count)?,
        })
    }
}

impl Mutate<Integer> for UniformMutation {
    fn mutate(&self, representation: &Integer, genome: &mut Integers, rng: &mut StreamRng) {
        let variable = representation.variable_genes();
        let bounds = representation.bounds();
        self.mode.apply(
            variable.len(),
            |candidate| variable[candidate],
            rng,
            |gene, rng| {
                genome[gene] =
                    crate::genome::integer::random_other_in(&bounds[gene], genome[gene], rng);
            },
        );
    }
}

impl Mutate<Real> for UniformMutation {
    fn mutate(&self, representation: &Real, genome: &mut Reals, rng: &mut StreamRng) {
        let variable = representation.variable_genes();
        let bounds = representation.bounds();
        self.mode.apply(
            variable.len(),
            |candidate| variable[candidate],
            rng,
            |gene, rng| {
                genome[gene] =
                    crate::genome::real::random_other_in(&bounds[gene], genome[gene], rng);
            },
        );
    }
}

/// Swap mutation for [`Permutation`] genomes: exchanges the genes of `count` random pairs of
/// positions, 1 by default.
///
/// The pairs are disjoint, so the swaps never undo each other and the genome always changes
/// (unless it has a single gene). At most `length / 2` pairs are swapped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapMutation {
    count: usize,
}

impl SwapMutation {
    /// Swaps one pair of positions.
    pub fn new() -> Self {
        Self { count: 1 }
    }

    /// Swaps `count` disjoint pairs of positions, at least 1.
    pub fn count(count: usize) -> Result<Self> {
        if count == 0 {
            return Err(Error::InvalidSetting {
                setting: "swap_count",
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { count })
    }
}

impl Default for SwapMutation {
    fn default() -> Self {
        Self::new()
    }
}

impl Mutate<Permutation> for SwapMutation {
    fn mutate(&self, _representation: &Permutation, genome: &mut Order, rng: &mut StreamRng) {
        let len = genome.len();
        if self.count == 1 {
            if len < 2 {
                return;
            }
            let a = rng.below(len);
            let b = rng.below(len - 1);
            genome.swap(a, if b >= a { b + 1 } else { b });
            return;
        }
        let mut positions = rng.sample_distinct(2 * self.count.min(len / 2), len);
        // random pairs: shuffle the positions (Fisher-Yates), then pair them up
        for position in (1..positions.len()).rev() {
            positions.swap(position, rng.below(position + 1));
        }
        for pair in positions.chunks_exact(2) {
            genome.swap(pair[0], pair[1]);
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
        assert!(UniformMutation::per_gene(0.0).is_err());
        assert!(UniformMutation::count(0).is_err());
        assert!(SwapMutation::count(0).is_err());
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

    #[test]
    fn fixed_genes_never_change() {
        let integer = Integer::new([0..=0, 0..=5, 3..=3]).unwrap();
        let real = Real::new([0.0..=0.0, 0.0..=1.0]).unwrap();
        let mut rng = StreamRng::seed_from_u64(0);
        for _ in 0..100 {
            let mut genome = Integers::from(vec![0, 0, 3]);
            UniformMutation::count(3)
                .unwrap()
                .mutate(&integer, &mut genome, &mut rng);
            assert_eq!((genome[0], genome[2]), (0, 3));
            assert_ne!(genome[1], 0);
            let mut genome = Reals::from(vec![0.0, 0.5]);
            UniformMutation::per_gene(1.0)
                .unwrap()
                .mutate(&real, &mut genome, &mut rng);
            assert_eq!(genome[0], 0.0);
            assert_ne!(genome[1], 0.5);
        }
    }

    #[test]
    fn single_value_space_is_left_alone() {
        let integer = Integer::uniform(1, 4..=4).unwrap();
        let mut genome = Integers::from(vec![4]);
        let mutation = UniformMutation::per_gene(0.5).unwrap();
        mutation.mutate(&integer, &mut genome, &mut StreamRng::seed_from_u64(0));
        assert_eq!(genome[0], 4);
        let mut order = Order::identity(1);
        SwapMutation::new().mutate(
            &Permutation::new(1).unwrap(),
            &mut order,
            &mut StreamRng::seed_from_u64(0),
        );
        assert_eq!(order, Order::identity(1));
    }

    fn changed<T: PartialEq>(a: &[T], b: &[T]) -> usize {
        a.iter().zip(b).filter(|(x, y)| x != y).count()
    }

    proptest! {
        #[test]
        fn bit_flip_always_changes(len in 1usize..200, rate in 0.0001..=1.0f64, seed: u64) {
            let binary = Binary::new(len).unwrap();
            let mut rng = StreamRng::seed_from_u64(seed);
            let original = binary.random_genome(&mut rng);
            let mut genome = original.clone();
            BitFlip::per_gene(rate).unwrap().mutate(&binary, &mut genome, &mut rng);
            prop_assert_ne!(&genome, &original);
            prop_assert!(binary.validate(&genome).is_ok());
        }

        #[test]
        fn bit_flip_count_is_exact(len in 1usize..200, count in 1usize..20, seed: u64) {
            let binary = Binary::new(len).unwrap();
            let mut genome = Bits::zeros(len);
            BitFlip::count(count).unwrap().mutate(&binary, &mut genome, &mut StreamRng::seed_from_u64(seed));
            prop_assert_eq!(genome.count_ones(), count.min(len));
        }

        #[test]
        fn uniform_integer(len in 1usize..50, width in 1i64..10, count in 1usize..10, rate in 0.0001..=1.0f64, seed: u64) {
            let integer = Integer::uniform(len, -width..=width).unwrap();
            let mut rng = StreamRng::seed_from_u64(seed);
            let original = integer.random_genome(&mut rng);
            let mut genome = original.clone();
            UniformMutation::per_gene(rate).unwrap().mutate(&integer, &mut genome, &mut rng);
            prop_assert!(changed(&genome, &original) >= 1);
            prop_assert!(integer.validate(&genome).is_ok());
            let mut genome = original.clone();
            UniformMutation::count(count).unwrap().mutate(&integer, &mut genome, &mut rng);
            prop_assert_eq!(changed(&genome, &original), count.min(len));
            prop_assert!(integer.validate(&genome).is_ok());
        }

        #[test]
        fn uniform_real(len in 1usize..50, count in 1usize..10, rate in 0.0001..=1.0f64, seed: u64) {
            let real = Real::uniform(len, -1.0..=1.0).unwrap();
            let mut rng = StreamRng::seed_from_u64(seed);
            let original = real.random_genome(&mut rng);
            let mut genome = original.clone();
            UniformMutation::per_gene(rate).unwrap().mutate(&real, &mut genome, &mut rng);
            prop_assert!(changed(&genome, &original) >= 1);
            prop_assert!(real.validate(&genome).is_ok());
            let mut genome = original.clone();
            UniformMutation::count(count).unwrap().mutate(&real, &mut genome, &mut rng);
            prop_assert_eq!(changed(&genome, &original), count.min(len));
            prop_assert!(real.validate(&genome).is_ok());
        }

        #[test]
        fn swap(len in 2usize..50, count in 1usize..10, seed: u64) {
            let permutation = Permutation::new(len).unwrap();
            let mut rng = StreamRng::seed_from_u64(seed);
            let original = permutation.random_genome(&mut rng);
            let mut genome = original.clone();
            SwapMutation::count(count).unwrap().mutate(&permutation, &mut genome, &mut rng);
            prop_assert_eq!(changed(&genome, &original), 2 * count.min(len / 2));
            prop_assert!(Order::new(genome.into_vec()).is_ok());
        }
    }
}
