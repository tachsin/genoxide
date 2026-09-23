//! Mutation: random changes to a genome.

use super::{Mutate, check_rate};
use crate::genome::{Binary, Bits, Integer, Integers, Order, Permutation, Real, Reals};
use crate::math::pow;
use crate::rng::Chance;
use crate::{Error, Result, StreamRng};
use std::ops::RangeInclusive;

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
                rng.chosen(chance, genes, |rng, candidate| {
                    change(gene(candidate), rng);
                    changed = true;
                });
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

// a positive, finite value for `setting`
fn check_positive(setting: &'static str, value: f64) -> Result<f64> {
    if value > 0.0 && value.is_finite() {
        Ok(value)
    } else {
        Err(Error::InvalidSetting {
            setting,
            reason: format!("must be positive and finite, got {value}"),
        })
    }
}

// `value` mirrored at the bounds until it's inside them, as light reflects between two mirrors
fn reflect(value: f64, range: &RangeInclusive<f64>) -> f64 {
    let (start, end) = (*range.start(), *range.end());
    if (start..=end).contains(&value) {
        return value;
    }
    let width = end - start;
    let offset = (value - start).rem_euclid(2.0 * width);
    let offset = if offset > width {
        2.0 * width - offset
    } else {
        offset
    };
    (start + offset).clamp(start, end)
}

// a new value for a gene in `range`, other than `current`: `propose` until it differs (almost
// always the first time), then a uniform value
fn changed_gene(
    range: &RangeInclusive<f64>,
    current: f64,
    rng: &mut StreamRng,
    mut propose: impl FnMut(&mut StreamRng) -> f64,
) -> f64 {
    for _ in 0..64 {
        let value = propose(rng);
        if value != current && range.contains(&value) {
            return value;
        }
    }
    crate::genome::real::random_other_in(range, current, rng)
}

/// Gaussian mutation for [`Real`] genomes: adds normal noise to genes, each gene with a
/// probability or `n` genes.
///
/// The standard deviation is `sigma` times the width of the gene's range, so one `sigma` fits
/// genes with different bounds. A value that leaves the bounds is mirrored back in, which keeps
/// the bounds reachable without piling values up on them. Small steps make it the operator for
/// fine-tuning, e.g. `sigma` 0.01 to 0.1.
///
/// The genome always changes. Genes whose bounds allow a single value are never changed.
///
/// ```
/// use genoxide::genome::{Real, Reals, Representation};
/// use genoxide::operator::{GaussianMutation, Mutate};
/// use genoxide::StreamRng;
///
/// let real = Real::uniform(3, -1.0..=1.0)?;
/// let mut genome = Reals::from(vec![0.0; 3]);
/// GaussianMutation::count(1, 0.05)?.mutate(&real, &mut genome, &mut StreamRng::seed_from_u64(0));
/// let changed: Vec<f64> = genome.iter().copied().filter(|&gene| gene != 0.0).collect();
/// assert_eq!(changed.len(), 1);
/// assert!(changed[0].abs() < 0.6); // 6 standard deviations of 0.1
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianMutation {
    mode: Mode,
    sigma: f64,
}

impl GaussianMutation {
    /// Mutates each gene with probability `rate` (greater than 0 and at most 1), with a standard
    /// deviation of `sigma` (positive) times the gene's range. If no gene is picked, one random
    /// gene is, so the genome always changes.
    pub fn per_gene(rate: f64, sigma: f64) -> Result<Self> {
        Ok(Self {
            mode: Mode::per_gene("gaussian_mutation_rate", rate)?,
            sigma: check_positive("gaussian_sigma", sigma)?,
        })
    }

    /// Mutates `count` distinct random genes (at least 1; all genes that can change if there are
    /// fewer), with a standard deviation of `sigma` (positive) times the gene's range.
    pub fn count(count: usize, sigma: f64) -> Result<Self> {
        Ok(Self {
            mode: Mode::count("gaussian_mutation_count", count)?,
            sigma: check_positive("gaussian_sigma", sigma)?,
        })
    }

    /// The standard deviation, as a fraction of each gene's range.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
}

impl Mutate<Real> for GaussianMutation {
    fn mutate(&self, representation: &Real, genome: &mut Reals, rng: &mut StreamRng) {
        let variable = representation.variable_genes();
        let bounds = representation.bounds();
        self.mode.apply(
            variable.len(),
            |candidate| variable[candidate],
            rng,
            |gene, rng| {
                let range = &bounds[gene];
                let current = genome[gene];
                let deviation = self.sigma * (range.end() - range.start());
                genome[gene] = changed_gene(range, current, rng, |rng| {
                    reflect(current + deviation * rng.normal(), range)
                });
            },
        );
    }
}

/// Polynomial mutation for [`Real`] genomes: Deb's bounded polynomial mutation, as in NSGA-II and
/// pymoo, each gene with a probability or `n` genes.
///
/// The distribution index `eta` sets the step size: the larger, the closer the new value to the
/// old one. Common values are 20 (the NSGA-II default) and 5 to 100. Steps shrink near the bounds,
/// so the new value is always within them.
///
/// The genome always changes. Genes whose bounds allow a single value are never changed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolynomialMutation {
    mode: Mode,
    eta: f64,
}

impl PolynomialMutation {
    /// Mutates each gene with probability `rate` (greater than 0 and at most 1), with
    /// distribution index `eta` (0 or more). If no gene is picked, one random gene is, so the
    /// genome always changes. A common choice is `rate` `1 / length`, `eta` 20.
    pub fn per_gene(rate: f64, eta: f64) -> Result<Self> {
        Ok(Self {
            mode: Mode::per_gene("polynomial_mutation_rate", rate)?,
            eta: check_eta(eta)?,
        })
    }

    /// Mutates `count` distinct random genes (at least 1; all genes that can change if there are
    /// fewer), with distribution index `eta` (0 or more).
    pub fn count(count: usize, eta: f64) -> Result<Self> {
        Ok(Self {
            mode: Mode::count("polynomial_mutation_count", count)?,
            eta: check_eta(eta)?,
        })
    }

    /// The distribution index.
    pub fn eta(&self) -> f64 {
        self.eta
    }
}

fn check_eta(eta: f64) -> Result<f64> {
    if eta >= 0.0 && eta.is_finite() {
        Ok(eta)
    } else {
        Err(Error::InvalidSetting {
            setting: "polynomial_eta",
            reason: format!("must be 0 or more and finite, got {eta}"),
        })
    }
}

// Deb's bounded polynomial mutation of `current` in `range`
fn polynomial(current: f64, range: &RangeInclusive<f64>, eta: f64, rng: &mut StreamRng) -> f64 {
    let (start, end) = (*range.start(), *range.end());
    let width = end - start;
    let random = rng.unit_f64();
    let power = 1.0 / (eta + 1.0);
    let step = if random < 0.5 {
        let complement = 1.0 - (current - start) / width;
        let value = 2.0 * random + (1.0 - 2.0 * random) * pow(complement.max(0.0), eta + 1.0);
        pow(value, power) - 1.0
    } else {
        let complement = 1.0 - (end - current) / width;
        let value =
            2.0 * (1.0 - random) + 2.0 * (random - 0.5) * pow(complement.max(0.0), eta + 1.0);
        1.0 - pow(value, power)
    };
    (current + step * width).clamp(start, end)
}

impl Mutate<Real> for PolynomialMutation {
    fn mutate(&self, representation: &Real, genome: &mut Reals, rng: &mut StreamRng) {
        let variable = representation.variable_genes();
        let bounds = representation.bounds();
        self.mode.apply(
            variable.len(),
            |candidate| variable[candidate],
            rng,
            |gene, rng| {
                let range = &bounds[gene];
                let current = genome[gene];
                genome[gene] = changed_gene(range, current, rng, |rng| {
                    polynomial(current, range, self.eta, rng)
                });
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

    #[test]
    fn real_mutation_validation() {
        assert!(GaussianMutation::per_gene(0.1, 0.0).is_err());
        assert!(GaussianMutation::per_gene(0.1, f64::INFINITY).is_err());
        assert!(GaussianMutation::count(0, 0.1).is_err());
        assert!(PolynomialMutation::per_gene(0.1, -1.0).is_err());
        assert!(PolynomialMutation::per_gene(0.1, f64::NAN).is_err());
        assert!(PolynomialMutation::count(1, 0.0).is_ok());
    }

    #[test]
    fn reflect_mirrors_into_the_bounds() {
        let range = 0.0..=1.0;
        assert_eq!(reflect(0.25, &range), 0.25);
        assert_eq!(reflect(1.25, &range), 0.75);
        assert_eq!(reflect(-0.25, &range), 0.25);
        assert_eq!(reflect(2.25, &range), 0.25);
        assert_eq!(reflect(-1.25, &range), 0.75);
        assert_eq!(reflect(3.0, &range), 1.0);
    }

    // (mean, standard deviation) of the new value of gene 0, from `current`, over many mutations
    fn step_statistics(mutation: &impl Mutate<Real>, current: f64) -> (f64, f64) {
        let real = Real::uniform(1, 0.0..=10.0).unwrap();
        let mut rng = StreamRng::seed_from_u64(0);
        let values: Vec<f64> = (0..50_000)
            .map(|_| {
                let mut genome = Reals::from(vec![current]);
                mutation.mutate(&real, &mut genome, &mut rng);
                genome[0]
            })
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        (mean, variance.sqrt())
    }

    #[test]
    fn gaussian_steps_have_the_standard_deviation() {
        // sigma 0.02 of a range of 10: standard deviation 0.2, far from the bounds
        let (mean, deviation) = step_statistics(&GaussianMutation::count(1, 0.02).unwrap(), 5.0);
        assert!((mean - 5.0).abs() < 0.005, "mean {mean}");
        assert!((deviation - 0.2).abs() < 0.005, "deviation {deviation}");
    }

    #[test]
    fn polynomial_steps_are_centered_and_shrink_with_eta() {
        let spread = |eta| {
            let (mean, deviation) =
                step_statistics(&PolynomialMutation::count(1, eta).unwrap(), 5.0);
            assert!((mean - 5.0).abs() < 0.02, "eta {eta}: mean {mean}");
            deviation
        };
        let spreads = [spread(1.0), spread(5.0), spread(20.0), spread(100.0)];
        assert!(
            spreads.windows(2).all(|pair| pair[0] > pair[1]),
            "{spreads:?}"
        );
        // near a bound, steps stay inside
        let (mean, _) = step_statistics(&PolynomialMutation::count(1, 20.0).unwrap(), 0.01);
        assert!(mean > 0.01, "mean {mean}");
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
        fn gaussian_and_polynomial(len in 1usize..30, count in 1usize..10, rate in 0.0001..=1.0f64, sigma in 0.001..2.0f64, eta in 0.0..200.0f64, seed: u64) {
            // one fixed gene, which never changes
            let real = Real::new(std::iter::once(3.0..=3.0).chain(std::iter::repeat_n(-1.0..=1.0, len))).unwrap();
            let mut rng = StreamRng::seed_from_u64(seed);
            let original = real.random_genome(&mut rng);
            let check = |genome: Reals, exactly: Option<usize>| -> Result<(), TestCaseError> {
                prop_assert!(real.validate(&genome).is_ok());
                prop_assert_eq!(genome[0], 3.0);
                let changes = changed(&genome[..], &original[..]);
                match exactly {
                    Some(count) => prop_assert_eq!(changes, count),
                    None => prop_assert!(changes >= 1),
                }
                Ok(())
            };
            let mut genome = original.clone();
            GaussianMutation::per_gene(rate, sigma).unwrap().mutate(&real, &mut genome, &mut rng);
            check(genome, None)?;
            let mut genome = original.clone();
            GaussianMutation::count(count, sigma).unwrap().mutate(&real, &mut genome, &mut rng);
            check(genome, Some(count.min(len)))?;
            let mut genome = original.clone();
            PolynomialMutation::per_gene(rate, eta).unwrap().mutate(&real, &mut genome, &mut rng);
            check(genome, None)?;
            let mut genome = original.clone();
            PolynomialMutation::count(count, eta).unwrap().mutate(&real, &mut genome, &mut rng);
            check(genome, Some(count.min(len)))?;
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
