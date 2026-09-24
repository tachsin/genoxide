//! Crossover: recombining two genomes.

use super::{Crossover, check_rate};
use crate::genome::{Genome, Real, Reals, Representation, SwapGenes};
use crate::math::pow;
use crate::rng::Chance;
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

/// Simulated binary crossover (SBX) for [`Real`] genomes: Deb and Agrawal's bounded version, as in
/// NSGA-II and pymoo.
///
/// Each gene is recombined with probability one half. The two child values are spread
/// symmetrically around the parents' mean, like one-point crossover spreads bit strings, with the
/// distribution index `eta` setting how far: the larger, the closer the children to their
/// parents. Common values are 15 to 20 (and 20 for polynomial mutation, its usual partner). The
/// children always stay within the bounds, and then each gene goes to either child with probability
/// one half.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimulatedBinaryCrossover {
    eta: f64,
}

impl SimulatedBinaryCrossover {
    /// SBX with distribution index `eta`, 0 or more and finite.
    pub fn new(eta: f64) -> Result<Self> {
        if eta >= 0.0 && eta.is_finite() {
            Ok(Self { eta })
        } else {
            Err(Error::InvalidSetting {
                setting: "sbx_eta",
                reason: format!("must be 0 or more and finite, got {eta}"),
            })
        }
    }

    /// The distribution index.
    pub fn eta(&self) -> f64 {
        self.eta
    }
}

// the spread factor of SBX for one side, from `beta` (1 or more) and a random number in [0, 1)
fn sbx_spread(beta: f64, random: f64, eta: f64) -> f64 {
    // beta is infinite when the parents are extremely close compared to the bounds: beta^-(eta+1)
    // is then 0
    let alpha = if beta.is_finite() {
        2.0 - pow(beta, -(eta + 1.0))
    } else {
        2.0
    };
    if random <= 1.0 / alpha {
        pow(random * alpha, 1.0 / (eta + 1.0))
    } else {
        pow(1.0 / (2.0 - random * alpha), 1.0 / (eta + 1.0))
    }
}

impl Crossover<Real> for SimulatedBinaryCrossover {
    fn crossover(&self, representation: &Real, a: &mut Reals, b: &mut Reals, rng: &mut StreamRng) {
        for (gene, range) in representation.bounds().iter().enumerate() {
            if !rng.chance(Chance::Half) {
                continue;
            }
            let (x, y) = (a[gene], b[gene]);
            // equal genes have nothing to spread
            if (x - y).abs() <= 1e-14 {
                continue;
            }
            let (low, high) = (x.min(y), x.max(y));
            let (start, end) = (*range.start(), *range.end());
            let random = rng.unit_f64();
            let spread_low = sbx_spread(1.0 + 2.0 * (low - start) / (high - low), random, self.eta);
            let spread_high = sbx_spread(1.0 + 2.0 * (end - high) / (high - low), random, self.eta);
            let first = (0.5 * ((low + high) - spread_low * (high - low))).clamp(start, end);
            let second = (0.5 * ((low + high) + spread_high * (high - low))).clamp(start, end);
            if rng.chance(Chance::Half) {
                (a[gene], b[gene]) = (second, first);
            } else {
                (a[gene], b[gene]) = (first, second);
            }
        }
    }
}

/// Blend crossover (BLX-α) for [`Real`] genomes: each child gene is uniformly random in the
/// interval of the parents' genes, widened by `alpha` times its width on each side and limited to
/// the bounds.
///
/// With `alpha` 0 the children stay between their parents, which narrows the population. About
/// 0.366 keeps the spread of the population on average, and 0.5, the common choice, widens it a
/// little (by about 17% in variance per generation, before selection).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendCrossover {
    alpha: f64,
}

impl BlendCrossover {
    /// BLX-α with `alpha`, 0 or more and finite.
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha >= 0.0 && alpha.is_finite() {
            Ok(Self { alpha })
        } else {
            Err(Error::InvalidSetting {
                setting: "blend_alpha",
                reason: format!("must be 0 or more and finite, got {alpha}"),
            })
        }
    }

    /// How far the interval is widened, as a fraction of its width.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

impl Crossover<Real> for BlendCrossover {
    fn crossover(&self, representation: &Real, a: &mut Reals, b: &mut Reals, rng: &mut StreamRng) {
        for (gene, range) in representation.bounds().iter().enumerate() {
            let (x, y) = (a[gene], b[gene]);
            let widening = self.alpha * (x - y).abs();
            let low = (x.min(y) - widening).max(*range.start());
            let high = (x.max(y) + widening).min(*range.end());
            let interval = low..=high;
            a[gene] = crate::genome::real::random_in(&interval, rng);
            b[gene] = crate::genome::real::random_in(&interval, rng);
        }
    }
}

/// Arithmetic (whole) crossover for [`Real`] genomes: the children are weighted averages of their
/// parents, `w a + (1 - w) b` and `(1 - w) a + w b`.
///
/// The weight is random for every crossover by default, or fixed. The children are always between
/// their parents, so this narrows the population: pair it with a mutation that explores.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArithmeticCrossover {
    weight: Option<f64>,
}

impl ArithmeticCrossover {
    /// Arithmetic crossover with a random weight in [0, 1) for every crossover.
    pub fn new() -> Self {
        Self { weight: None }
    }

    /// Arithmetic crossover with a fixed `weight`, greater than 0 and less than 1, but not 0.5,
    /// which would give two identical children.
    pub fn with_weight(weight: f64) -> Result<Self> {
        if weight > 0.0 && weight < 1.0 && weight != 0.5 {
            Ok(Self {
                weight: Some(weight),
            })
        } else {
            Err(Error::InvalidSetting {
                setting: "arithmetic_weight",
                reason: format!("must be greater than 0, less than 1 and not 0.5, got {weight}"),
            })
        }
    }

    /// The fixed weight, or `None` for a random one.
    pub fn weight(&self) -> Option<f64> {
        self.weight
    }
}

impl Default for ArithmeticCrossover {
    fn default() -> Self {
        Self::new()
    }
}

impl Crossover<Real> for ArithmeticCrossover {
    fn crossover(&self, representation: &Real, a: &mut Reals, b: &mut Reals, rng: &mut StreamRng) {
        let weight = self.weight.unwrap_or_else(|| rng.unit_f64());
        for (gene, range) in representation.bounds().iter().enumerate() {
            let (x, y) = (a[gene], b[gene]);
            // rounding can leave a convex combination a hair outside the bounds
            a[gene] = (weight * x + (1.0 - weight) * y).clamp(*range.start(), *range.end());
            b[gene] = ((1.0 - weight) * x + weight * y).clamp(*range.start(), *range.end());
        }
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
    fn real_crossover_validation() {
        assert!(SimulatedBinaryCrossover::new(-1.0).is_err());
        assert!(SimulatedBinaryCrossover::new(f64::NAN).is_err());
        assert!(BlendCrossover::new(-0.1).is_err());
        assert!(ArithmeticCrossover::with_weight(0.5).is_err());
        assert!(ArithmeticCrossover::with_weight(1.0).is_err());
        assert!(ArithmeticCrossover::with_weight(0.3).is_ok());
    }

    #[test]
    fn sbx_spread_shrinks_with_eta() {
        // the average distance of a child gene from its nearest parent gene
        let distance = |eta| {
            let real = Real::uniform(1, -100.0..=100.0).unwrap();
            let crossover = SimulatedBinaryCrossover::new(eta).unwrap();
            let mut rng = StreamRng::seed_from_u64(0);
            let mut total = 0.0;
            for _ in 0..20_000 {
                let (mut a, mut b) = (Reals::from(vec![-1.0]), Reals::from(vec![1.0]));
                crossover.crossover(&real, &mut a, &mut b, &mut rng);
                total += (a[0].abs() - 1.0).abs() + (b[0].abs() - 1.0).abs();
            }
            total / 40_000.0
        };
        let distances = [
            distance(0.0),
            distance(2.0),
            distance(20.0),
            distance(200.0),
        ];
        // huge bounds and very close parents: no overflow
        let real = Real::uniform(1, -1e300..=1e300).unwrap();
        let (mut a, mut b) = (Reals::from(vec![0.0]), Reals::from(vec![1e-13]));
        let mut rng = StreamRng::seed_from_u64(0);
        for _ in 0..100 {
            SimulatedBinaryCrossover::new(15.0)
                .unwrap()
                .crossover(&real, &mut a, &mut b, &mut rng);
            assert!(real.validate(&a).is_ok() && real.validate(&b).is_ok());
        }
        assert!(
            distances.windows(2).all(|pair| pair[0] > pair[1]),
            "{distances:?}"
        );
        assert!(distances[3] < 0.01, "{distances:?}");
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

    fn real_parents(len: usize, seed: u64) -> (Real, Reals, Reals, StreamRng) {
        let real = Real::new((0..len).map(|gene| -(gene as f64)..=gene as f64 + 1.0)).unwrap();
        let mut rng = StreamRng::seed_from_u64(seed);
        let a = real.random_genome(&mut rng);
        let b = real.random_genome(&mut rng);
        (real, a, b, rng)
    }

    proptest! {
        #[test]
        fn real_crossovers_stay_in_bounds(len in 1usize..30, eta in 0.0..100.0f64, alpha in 0.0..2.0f64, seed: u64) {
            let (real, a, b, mut rng) = real_parents(len, seed);
            let (mut x, mut y) = (a.clone(), b.clone());
            SimulatedBinaryCrossover::new(eta).unwrap().crossover(&real, &mut x, &mut y, &mut rng);
            prop_assert!(real.validate(&x).is_ok() && real.validate(&y).is_ok());
            // genes are only recombined per gene: each child gene is between the bounds, and equal
            // parent genes stay
            let (mut x, mut y) = (a.clone(), a.clone());
            SimulatedBinaryCrossover::new(eta).unwrap().crossover(&real, &mut x, &mut y, &mut rng);
            prop_assert_eq!(&x, &a);
            prop_assert_eq!(&y, &a);

            let (mut x, mut y) = (a.clone(), b.clone());
            BlendCrossover::new(alpha).unwrap().crossover(&real, &mut x, &mut y, &mut rng);
            prop_assert!(real.validate(&x).is_ok() && real.validate(&y).is_ok());
            for gene in 0..len {
                let widening = alpha * (a[gene] - b[gene]).abs();
                for child in [x[gene], y[gene]] {
                    prop_assert!(child >= a[gene].min(b[gene]) - widening);
                    prop_assert!(child <= a[gene].max(b[gene]) + widening);
                }
            }

            let (mut x, mut y) = (a.clone(), b.clone());
            ArithmeticCrossover::new().crossover(&real, &mut x, &mut y, &mut rng);
            prop_assert!(real.validate(&x).is_ok() && real.validate(&y).is_ok());
            for gene in 0..len {
                // the children are between the parents, and keep their sum
                for child in [x[gene], y[gene]] {
                    // up to rounding
                    prop_assert!(child >= a[gene].min(b[gene]) - 1e-12);
                    prop_assert!(child <= a[gene].max(b[gene]) + 1e-12);
                }
                prop_assert!((x[gene] + y[gene] - a[gene] - b[gene]).abs() < 1e-9);
            }
        }

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
