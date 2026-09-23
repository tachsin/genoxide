//! Portable, seedable random number generation with independent streams.

use crate::math::log;
use rand::{Rng, SeedableRng, TryRng};
use rand_chacha::ChaCha8Rng;
use std::convert::Infallible;

/// The random number generator used throughout genoxide.
///
/// - **Portable:** the same seed gives the same numbers on every platform and every genoxide
///   version with the same major version (ChaCha8, checked against fixed values in the tests).
/// - **Independent streams:** [`derive`](StreamRng::derive) gives a generator for a run, island
///   or worker that depends only on this generator's seed and an id. It doesn't depend on how many
///   numbers were drawn before, so parallel work stays reproducible.
///
/// It implements rand's [`Rng`], so rand's distributions and [`RngExt`](rand::RngExt) methods
/// can be used with it.
///
/// ```
/// use genoxide::StreamRng;
/// use rand::RngExt;
///
/// let mut rng = StreamRng::seed_from_u64(42);
/// let a: f64 = rng.random();
/// assert_eq!(a, StreamRng::seed_from_u64(42).random::<f64>());
///
/// // independent generators for two parallel runs
/// let mut run_0 = rng.derive(0);
/// let mut run_1 = rng.derive(1);
/// assert_ne!(run_0.random::<u64>(), run_1.random::<u64>());
/// ```
#[derive(Clone, Debug)]
pub struct StreamRng {
    inner: ChaCha8Rng,
}

impl StreamRng {
    /// A generator seeded with `seed`: the same seed always gives the same numbers.
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// A generator seeded from the operating system's randomness, so not reproducible.
    pub fn from_entropy() -> Self {
        Self {
            inner: ChaCha8Rng::from_rng(&mut rand::rng()),
        }
    }

    /// An independent generator derived from this generator's seed and `id`.
    ///
    /// The result depends only on the seed and `id`, not on how many numbers this generator has
    /// produced. Different ids give independent generators, and derived generators can be derived
    /// again (e.g. run → island → worker) without collisions.
    pub fn derive(&self, id: u64) -> Self {
        // The child seed is read from stream `id` of the parent seed, far beyond the part of the
        // key stream the parent itself uses (stream 0, from word 0), so it never overlaps the
        // parent's output. A new seed (instead of only a new stream) keeps grandchildren distinct.
        let mut source = ChaCha8Rng::from_seed(self.inner.get_seed());
        source.set_stream(id);
        source.set_word_pos(DERIVE_WORD_POS);
        let mut seed = [0u8; 32];
        source.fill_bytes(&mut seed);
        Self {
            inner: ChaCha8Rng::from_seed(seed),
        }
    }
}

// 2^64 words (of 32 bits) into a stream: unreachable by normal use of a generator
const DERIVE_WORD_POS: u128 = 1 << 64;

// Sampling used by genoxide itself. Only integer arithmetic and exact conversions, so the results
// are the same on every platform, and don't change when rand changes its sampling algorithms.
impl StreamRng {
    /// A uniformly random integer in `0..n`. `n` must not be 0.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        self.below_u64(n as u64) as usize
    }

    /// A uniformly random integer in `0..n` (Lemire's method, unbiased). `n` must not be 0.
    pub(crate) fn below_u64(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0, "below(0)");
        let mut product = u128::from(self.next_u64()) * u128::from(n);
        if (product as u64) < n {
            let threshold = n.wrapping_neg() % n;
            while (product as u64) < threshold {
                product = u128::from(self.next_u64()) * u128::from(n);
            }
        }
        (product >> 64) as u64
    }

    /// A uniformly random `f64` in `[0, 1)`, with 53 random bits.
    pub(crate) fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// A standard normal random number (mean 0, standard deviation 1).
    ///
    /// Marsaglia's polar method: only `ln` and `sqrt`, no trigonometry, so the same on every
    /// platform.
    pub(crate) fn normal(&mut self) -> f64 {
        loop {
            let u = 2.0 * self.unit_f64() - 1.0;
            let v = 2.0 * self.unit_f64() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                return u * (-2.0 * log(s) / s).sqrt();
            }
        }
    }

    /// `true` with probability `chance`.
    pub(crate) fn chance(&mut self, chance: Chance) -> bool {
        match chance {
            Chance::Never => false,
            Chance::Always => true,
            Chance::Half => self.next_u64() < 1 << 63,
            Chance::Threshold(threshold) | Chance::Skip { threshold, .. } => {
                self.next_u64() < threshold
            }
        }
    }

    /// Calls `chosen` with each of `0..n` that is chosen with probability `chance`, independently
    /// and in ascending order: the same distribution as a [`chance`](StreamRng::chance) per
    /// index, with far fewer random numbers.
    ///
    /// - A chance below 0.4 skips ahead to the next chosen index: the number of skipped indices
    ///   is geometric, sampled with one random number (and a portable logarithm) per chosen index.
    /// - A chance of exactly one half takes 64 indices from each random number.
    /// - Other chances draw one random number per index.
    ///
    /// `chosen` gets the generator back, to draw random numbers of its own.
    pub(crate) fn chosen(
        &mut self,
        chance: Chance,
        n: usize,
        mut chosen: impl FnMut(&mut Self, usize),
    ) {
        match chance {
            Chance::Never => {}
            Chance::Always => (0..n).for_each(|index| chosen(self, index)),
            Chance::Half => {
                for start in (0..n).step_by(64) {
                    let mut bits = self.next_u64();
                    while bits != 0 {
                        let index = start + bits.trailing_zeros() as usize;
                        if index >= n {
                            break;
                        }
                        chosen(self, index);
                        bits &= bits - 1;
                    }
                }
            }
            Chance::Threshold(threshold) => {
                for index in 0..n {
                    if self.next_u64() < threshold {
                        chosen(self, index);
                    }
                }
            }
            Chance::Skip { log_complement, .. } => {
                let mut index = 0;
                while index < n {
                    // failures before the next success: floor(ln(u) / ln(1 - p)), u in (0, 1]
                    let skip = log(1.0 - self.unit_f64()) / log_complement;
                    if skip >= (n - index) as f64 {
                        break;
                    }
                    index += skip as usize;
                    chosen(self, index);
                    index += 1;
                }
            }
        }
    }

    /// `k` distinct integers from `0..n`, in ascending order (Floyd's algorithm). `k <= n`.
    pub(crate) fn sample_distinct(&mut self, k: usize, n: usize) -> Vec<usize> {
        debug_assert!(k <= n, "sample_distinct({k}, {n})");
        let mut selected = std::collections::BTreeSet::new();
        for j in (n - k)..n {
            let candidate = self.below(j + 1);
            if !selected.insert(candidate) {
                selected.insert(j);
            }
        }
        selected.into_iter().collect()
    }
}

/// A probability as an integer threshold, for Bernoulli trials without floating point, with what
/// [`StreamRng::chosen`] needs to choose among many indices quickly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Chance {
    Never,
    Always,
    // exactly one half: any single bit of a random u64
    Half,
    // true when a random u64 is below it
    Threshold(u64),
    // below SKIP_BELOW: chosen indices are found by skipping ahead, with ln(1 - p)
    Skip { threshold: u64, log_complement: f64 },
}

// Skipping ahead costs a random number and a logarithm per chosen index, testing a threshold a
// random number per index. Measured on bit-flip mutation (1000 genes, x86-64), skipping is faster
// below about 0.45: 42 times at 0.001, 1.2 times at 0.3.
const SKIP_BELOW: f64 = 0.4;

impl Chance {
    /// The chance of `probability`, which must be in `[0, 1]` (checked by the callers).
    pub(crate) fn new(probability: f64) -> Self {
        debug_assert!(
            (0.0..=1.0).contains(&probability),
            "probability {probability}"
        );
        // exact: the product is below 2^64, and the conversion truncates
        let threshold = || (probability * 18_446_744_073_709_551_616.0) as u64;
        if probability <= 0.0 {
            Chance::Never
        } else if probability >= 1.0 {
            Chance::Always
        } else if probability == 0.5 {
            Chance::Half
        } else if probability < SKIP_BELOW {
            // ln(1 - p), accurate for tiny p too, where 1 - p rounds to 1
            let log_complement = if probability < 1e-8 {
                -probability - probability * probability / 2.0
            } else {
                log(1.0 - probability)
            };
            Chance::Skip {
                threshold: threshold(),
                log_complement,
            }
        } else {
            Chance::Threshold(threshold())
        }
    }
}

impl TryRng for StreamRng {
    type Error = Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.inner.next_u32())
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.inner.next_u64())
    }

    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        self.inner.fill_bytes(dst);
        Ok(())
    }
}

impl SeedableRng for StreamRng {
    type Seed = [u8; 32];

    fn from_seed(seed: Self::Seed) -> Self {
        Self {
            inner: ChaCha8Rng::from_seed(seed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    // explicit, as the proptest prelude also exports an `Rng`
    use rand::Rng;

    fn first_u64s(mut rng: StreamRng) -> [u64; 3] {
        [rng.next_u64(), rng.next_u64(), rng.next_u64()]
    }

    /// Fixed values: these must never change for the same major version, on any platform.
    #[test]
    fn portable_values() {
        assert_eq!(
            first_u64s(StreamRng::seed_from_u64(42)),
            PORTABLE_SEED_42,
            "the numbers for a seed changed, which breaks reproducibility"
        );
        assert_eq!(
            first_u64s(StreamRng::seed_from_u64(42).derive(7)),
            PORTABLE_SEED_42_DERIVE_7,
            "the numbers for a derived stream changed, which breaks reproducibility"
        );
    }

    const PORTABLE_SEED_42: [u64; 3] = [
        12578764544318200737,
        17529487244874322312,
        7886285670807131020,
    ];
    const PORTABLE_SEED_42_DERIVE_7: [u64; 3] = [
        1169889660524673768,
        4703768454804564706,
        15961940369642500281,
    ];

    #[test]
    fn derive_is_independent_of_the_parent_position() {
        let parent = StreamRng::seed_from_u64(1);
        let mut used_parent = parent.clone();
        for _ in 0..1000 {
            used_parent.next_u64();
        }
        assert_eq!(
            first_u64s(parent.derive(3)),
            first_u64s(used_parent.derive(3))
        );
    }

    #[test]
    fn derive_does_not_collide() {
        let rng = StreamRng::seed_from_u64(1);
        let parent = first_u64s(rng.clone());
        let child_0 = first_u64s(rng.derive(0));
        let child_1 = first_u64s(rng.derive(1));
        let grandchild = first_u64s(rng.derive(0).derive(1));
        let all = [parent, child_0, child_1, grandchild];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "streams {i} and {j} collide");
            }
        }
    }

    #[test]
    fn portable_sampling_values() {
        let mut rng = StreamRng::seed_from_u64(42);
        let values = (
            rng.below(10),
            rng.below(1_000_000),
            rng.unit_f64(),
            rng.chance(Chance::new(0.5)),
            rng.sample_distinct(3, 10),
        );
        let expected = (6, 950275, 0.4275164028565197, false, vec![1, 2, 3]);
        assert_eq!(
            values, expected,
            "sampling changed, which breaks reproducibility"
        );
    }

    #[test]
    fn sample_distinct_is_uniform() {
        let mut rng = StreamRng::seed_from_u64(0);
        let mut counts = [0usize; 10];
        for _ in 0..30_000 {
            for index in rng.sample_distinct(3, 10) {
                counts[index] += 1;
            }
        }
        // each index is selected with probability 3/10: 9000 times
        assert!(
            counts.iter().all(|&c| (8_600..9_400).contains(&c)),
            "{counts:?}"
        );
    }

    #[test]
    fn chance_edges() {
        let mut rng = StreamRng::seed_from_u64(0);
        assert_eq!(Chance::new(0.0), Chance::Never);
        assert_eq!(Chance::new(1.0), Chance::Always);
        assert!((0..1000).all(|_| !rng.chance(Chance::new(0.0))));
        assert!((0..1000).all(|_| rng.chance(Chance::new(1.0))));
    }

    #[test]
    fn chance_frequency() {
        let mut rng = StreamRng::seed_from_u64(0);
        let hits = (0..100_000)
            .filter(|_| rng.chance(Chance::new(0.3)))
            .count();
        assert!((29_000..31_000).contains(&hits), "hits {hits}");
    }

    #[test]
    fn below_is_uniform() {
        let mut rng = StreamRng::seed_from_u64(0);
        let mut counts = [0usize; 7];
        for _ in 0..70_000 {
            counts[rng.below(7)] += 1;
        }
        assert!(
            counts.iter().all(|&c| (9_500..10_500).contains(&c)),
            "{counts:?}"
        );
    }

    #[test]
    fn normal_is_standard() {
        let mut rng = StreamRng::seed_from_u64(0);
        let samples: Vec<f64> = (0..200_000).map(|_| rng.normal()).collect();
        let count = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / count;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count;
        let within_one = samples.iter().filter(|x| x.abs() < 1.0).count() as f64 / count;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((variance - 1.0).abs() < 0.01, "variance {variance}");
        // P(|X| < 1) of a standard normal
        assert!((within_one - 0.6827).abs() < 0.005, "{within_one}");
        let mut rng = StreamRng::seed_from_u64(42);
        let values = [rng.normal(), rng.normal(), rng.normal()];
        assert_eq!(
            values.map(f64::to_bits),
            [
                4593777358611831395,  // 0.12793483831474636
                13830986477247399585, // -1.095928063849364
                13825395772567733866, // -0.46363556300011644
            ],
            "normal numbers changed, which breaks reproducibility"
        );
    }

    // (number chosen per run, all chosen indices) over `runs` runs
    fn chosen_runs(probability: f64, n: usize, runs: usize) -> (Vec<usize>, Vec<usize>) {
        let mut rng = StreamRng::seed_from_u64(0);
        let chance = Chance::new(probability);
        let mut per_run = Vec::new();
        let mut all = Vec::new();
        for _ in 0..runs {
            let mut chosen: Vec<usize> = Vec::new();
            rng.chosen(chance, n, |_, index| chosen.push(index));
            assert!(chosen.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(chosen.iter().all(|&index| index < n));
            per_run.push(chosen.len());
            all.extend(chosen);
        }
        (per_run, all)
    }

    #[test]
    fn chosen_edges() {
        assert_eq!(chosen_runs(0.0, 10, 5).1, Vec::<usize>::new());
        assert_eq!(chosen_runs(1.0, 3, 2).1, vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(chosen_runs(0.01, 0, 5).1, Vec::<usize>::new());
        assert!(matches!(Chance::new(0.01), Chance::Skip { .. }));
        assert!(matches!(Chance::new(1e-12), Chance::Skip { .. }));
        assert!(matches!(Chance::new(0.2), Chance::Skip { .. }));
        assert_eq!(Chance::new(0.5), Chance::Half);
        assert!(matches!(Chance::new(0.7), Chance::Threshold(_)));
    }

    #[test]
    fn chosen_is_binomial() {
        // skipping, half (word bits) and threshold, and a length that isn't a multiple of 64
        for (probability, n) in [
            (0.001, 1000),
            (0.01, 1000),
            (0.2, 300),
            (0.39, 300),
            (0.5, 100),
            (0.8, 300),
        ] {
            let runs = 4000;
            let (per_run, all) = chosen_runs(probability, n, runs);
            // the number chosen per run: mean n p, variance n p (1 - p)
            let mean = per_run.iter().sum::<usize>() as f64 / runs as f64;
            let variance = per_run
                .iter()
                .map(|&count| (count as f64 - mean).powi(2))
                .sum::<f64>()
                / runs as f64;
            let expected_mean = n as f64 * probability;
            let expected_variance = expected_mean * (1.0 - probability);
            let standard_error = (expected_variance / runs as f64).sqrt();
            assert!(
                (mean - expected_mean).abs() < 5.0 * standard_error,
                "p {probability}: mean {mean}, expected {expected_mean}"
            );
            assert!(
                (variance / expected_variance - 1.0).abs() < 0.15,
                "p {probability}: variance {variance}, expected {expected_variance}"
            );
            // no position is favored: each quarter of the indices gets a quarter
            let total = all.len() as f64;
            for quarter in 0..4 {
                let count = all
                    .iter()
                    .filter(|&&index| index * 4 / n == quarter)
                    .count() as f64;
                let expected = total / 4.0;
                assert!(
                    (count - expected).abs() < 5.0 * (expected * 0.75).sqrt(),
                    "p {probability}: quarter {quarter} got {count} of {total}"
                );
            }
        }
    }

    /// Fixed values: these must never change for the same major version, on any platform.
    #[test]
    fn portable_log_and_chosen_values() {
        let logs = [log(0.5), log(1e-300), log(0.999), log(1.0 - 2f64.powi(-53))];
        assert_eq!(
            logs.map(f64::to_bits),
            [
                13827790571168217583, // -0.6931471805599453
                13872659378474614709, // -690.7755278982137
                13785628853153536883, // -0.0010005003335835344
                13591863675404156928, // -1.1102230246251565e-16
            ],
            "the portable logarithm changed, which breaks reproducibility"
        );
        let mut rng = StreamRng::seed_from_u64(42);
        let mut chosen = |probability: f64, n: usize| {
            let mut indices = Vec::new();
            rng.chosen(Chance::new(probability), n, |_, index| indices.push(index));
            indices
        };
        // skipping, half and threshold
        let values = [
            chosen(0.01, 1000),
            chosen(0.5, 70),
            chosen(0.2, 20),
            chosen(0.7, 10),
        ];
        let expected: [Vec<usize>; 4] = [
            vec![113, 412, 468, 567, 601, 618, 655, 818, 965, 993],
            vec![
                0, 1, 5, 7, 8, 9, 11, 12, 14, 19, 21, 22, 23, 29, 30, 32, 33, 35, 36, 39, 41, 44,
                47, 50, 51, 52, 54, 55, 57, 58, 61, 62, 63, 64, 66, 67, 69,
            ],
            vec![4, 10, 11, 15, 16, 19],
            vec![0, 1, 2, 3, 4, 5, 6, 9],
        ];
        assert_eq!(
            values, expected,
            "sampling changed, which breaks reproducibility"
        );
    }

    #[test]
    fn from_entropy_differs() {
        assert_ne!(
            first_u64s(StreamRng::from_entropy()),
            first_u64s(StreamRng::from_entropy())
        );
    }

    proptest! {
        #[test]
        fn same_seed_same_numbers(seed: u64, id: u64) {
            prop_assert_eq!(
                first_u64s(StreamRng::seed_from_u64(seed)),
                first_u64s(StreamRng::seed_from_u64(seed))
            );
            prop_assert_eq!(
                first_u64s(StreamRng::seed_from_u64(seed).derive(id)),
                first_u64s(StreamRng::seed_from_u64(seed).derive(id))
            );
        }

        #[test]
        fn sampling_is_in_range(seed: u64, n in 1usize..1000, k_fraction in 0.0..=1.0f64) {
            let mut rng = StreamRng::seed_from_u64(seed);
            prop_assert!(rng.below(n) < n);
            let unit = rng.unit_f64();
            prop_assert!((0.0..1.0).contains(&unit));
            let k = (k_fraction * n as f64) as usize;
            let sample = rng.sample_distinct(k, n);
            prop_assert_eq!(sample.len(), k);
            prop_assert!(sample.windows(2).all(|pair| pair[0] < pair[1]));
            prop_assert!(sample.iter().all(|&i| i < n));
        }

        #[test]
        fn different_ids_give_different_streams(seed: u64, id: u64) {
            let rng = StreamRng::seed_from_u64(seed);
            prop_assert_ne!(
                first_u64s(rng.derive(id)),
                first_u64s(rng.derive(id.wrapping_add(1)))
            );
        }
    }
}
