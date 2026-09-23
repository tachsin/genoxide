//! Crossover: recombining two genomes.

use super::{Crossover, check_rate};
use crate::genome::{Genome, Representation, SwapGenes};
use crate::{Error, Result, StreamRng};

/// k-point crossover: the genomes are cut at `k` random points and every other segment is
/// exchanged.
///
/// The points are only between genes, so the children always keep the first gene of their own
/// parent: a crossover never just exchanges the whole genomes. Genomes shorter than `k + 1` genes
/// use every possible point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointCrossover {
    points: usize,
}

impl PointCrossover {
    /// One-point crossover.
    pub fn one_point() -> Self {
        Self { points: 1 }
    }

    /// Two-point crossover.
    pub fn two_point() -> Self {
        Self { points: 2 }
    }

    /// k-point crossover, `points` at least 1.
    pub fn k_point(points: usize) -> Result<Self> {
        if points == 0 {
            return Err(Error::InvalidSetting {
                setting: "crossover_points",
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { points })
    }

    /// The number of points.
    pub fn points(&self) -> usize {
        self.points
    }
}

impl<R> Crossover<R> for PointCrossover
where
    R: Representation,
    R::Genome: SwapGenes,
{
    fn crossover(
        &self,
        _representation: &R,
        a: &mut R::Genome,
        b: &mut R::Genome,
        rng: &mut StreamRng,
    ) {
        let len = a.len();
        if len < 2 {
            return;
        }
        // points in 1..len, between genes
        let cuts: Vec<usize> = rng
            .sample_distinct(self.points.min(len - 1), len - 1)
            .into_iter()
            .map(|index| index + 1)
            .collect();
        for segment in cuts.chunks(2) {
            let end = segment.get(1).copied().unwrap_or(len);
            a.swap_range(b, segment[0]..end);
        }
    }
}

/// Uniform crossover: every gene is exchanged with probability `rate`, 0.5 by default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformCrossover {
    rate: f64,
}

impl UniformCrossover {
    /// Uniform crossover exchanging each gene with probability 0.5.
    pub fn new() -> Self {
        Self { rate: 0.5 }
    }

    /// Uniform crossover exchanging each gene with probability `rate`, greater than 0 and less
    /// than 1.
    pub fn with_rate(rate: f64) -> Result<Self> {
        if rate < 1.0 {
            Ok(Self {
                rate: check_rate("uniform_crossover_rate", rate)?,
            })
        } else {
            Err(Error::InvalidSetting {
                setting: "uniform_crossover_rate",
                reason: format!("must be less than 1 (1 exchanges the whole genomes), got {rate}"),
            })
        }
    }

    /// The probability that a gene is exchanged.
    pub fn rate(&self) -> f64 {
        self.rate
    }
}

impl Default for UniformCrossover {
    fn default() -> Self {
        Self::new()
    }
}

impl<R> Crossover<R> for UniformCrossover
where
    R: Representation,
    R::Genome: SwapGenes,
{
    fn crossover(
        &self,
        _representation: &R,
        a: &mut R::Genome,
        b: &mut R::Genome,
        rng: &mut StreamRng,
    ) {
        a.swap_uniform(b, self.rate, rng);
    }
}

/// No crossover: the children are copies of their parents, for algorithms that only mutate,
/// e.g. (μ+λ). It fits every representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoCrossover;

impl<R: Representation> Crossover<R> for NoCrossover {
    fn crossover(
        &self,
        _representation: &R,
        _a: &mut R::Genome,
        _b: &mut R::Genome,
        _rng: &mut StreamRng,
    ) {
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{Binary, Bits};
    use proptest::prelude::*;

    fn parents(len: usize) -> (Bits, Bits) {
        (Bits::zeros(len), Bits::ones(len))
    }

    // positions where child a has the gene of parent b (parent a is zeros, parent b is ones)
    fn exchanged(child: &Bits) -> Vec<bool> {
        child.iter().collect()
    }

    fn switches(genes: &[bool]) -> usize {
        genes.windows(2).filter(|pair| pair[0] != pair[1]).count()
    }

    #[test]
    fn validation() {
        assert!(PointCrossover::k_point(0).is_err());
        assert!(UniformCrossover::with_rate(0.0).is_err());
        assert!(UniformCrossover::with_rate(1.0).is_err());
        assert!(UniformCrossover::with_rate(0.3).is_ok());
    }

    #[test]
    fn uniform_rate() {
        let binary = Binary::new(1_000).unwrap();
        let mut rng = StreamRng::seed_from_u64(0);
        for rate in [0.5, 0.2] {
            let crossover = UniformCrossover::with_rate(rate).unwrap();
            let mut total = 0;
            for _ in 0..100 {
                let (mut a, mut b) = parents(1_000);
                crossover.crossover(&binary, &mut a, &mut b, &mut rng);
                total += a.count_ones();
            }
            let fraction = total as f64 / 100_000.0;
            assert!(
                (fraction - rate).abs() < 0.01,
                "rate {rate}: exchanged {fraction}"
            );
        }
    }

    proptest! {
        #[test]
        fn point_crossover(len in 1usize..200, points in 1usize..6, seed: u64) {
            let binary = Binary::new(len).unwrap();
            let crossover = PointCrossover::k_point(points).unwrap();
            let (mut a, mut b) = parents(len);
            crossover.crossover(&binary, &mut a, &mut b, &mut StreamRng::seed_from_u64(seed));
            let genes = exchanged(&a);
            // complementary children
            prop_assert!(a.iter().zip(b.iter()).all(|(x, y)| x != y));
            // never a whole genome exchange: the first gene stays
            prop_assert_eq!(a.get(0), Some(false));
            // exactly min(k, len - 1) cut points
            prop_assert_eq!(switches(&genes), points.min(len - 1));
        }

        #[test]
        fn uniform_crossover_is_complementary(len in 1usize..200, seed: u64) {
            let binary = Binary::new(len).unwrap();
            let (mut a, mut b) = parents(len);
            UniformCrossover::new().crossover(&binary, &mut a, &mut b, &mut StreamRng::seed_from_u64(seed));
            prop_assert!(a.iter().zip(b.iter()).all(|(x, y)| x != y));
            prop_assert_eq!(a.count_ones() + b.count_ones(), len);
        }
    }
}
