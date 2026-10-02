//! NK landscapes, generated from a seed.

use super::{MAX_BITS, Optimum, Problem, binary, check_length};
use crate::engine::FitnessFunction;
use crate::genome::{Binary, Bits};
use crate::{Error, Objective, Result, StreamRng};
use rand::Rng;

// the stream of the generator that draws a landscape from its seed
const NK_STREAM: u64 = 3;
// the most table entries, N 2^(K+1): 128 MiB of them
const MAX_ENTRIES: usize = 1 << 24;
// the most steps `optimum` takes, exhaustively or by dynamic programming
const MAX_WORK: u128 = 1 << 32;
// a contribution c stands for c / 2^53
const UNIT: f64 = 1.0 / (1u64 << 53) as f64;

/// Which K sites bear on each site of an [`NkLandscape`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Neighborhood {
    /// Its flanking sites, on a circle: K/2 on each side for an even K (Kauffman and
    /// Weinberger's Table 1), and for an odd K, (K + 1)/2 after it and (K − 1)/2 before it.
    Adjacent,
    /// K distinct other sites drawn at random, uniformly, for each site (their Table 2).
    #[default]
    Random,
}

/// An NK landscape: N bits, each contributing a value that depends on its own bit and on K
/// others, drawn at random, the fitness their mean.
///
/// Site i contributes `wᵢ`, read from a table of 2^(K+1) values, one for each combination of the
/// bits of i and of the K sites that bear on it, its [`neighbors`](NkLandscape::neighbors). Each
/// value is drawn independently and uniformly from (0, 1), and the fitness is
/// `W = (1/N) Σ wᵢ` (equation 1). K tunes the landscape: at K = 0 the bits are independent and
/// there is a single optimum; as K grows to N − 1 the landscape becomes rugged, with more local
/// optima and lower ones, and at K = N − 1 it's uncorrelated: every one-bit change gives a new
/// random fitness.
///
/// The neighbors and the tables are drawn from `seed` with genoxide's portable
/// [`StreamRng`]: the same seed gives the same landscape on every platform. A contribution is
/// an odd multiple of 2^−53, `(2m + 1) / 2^53` with m drawn from 52 random bits, so it's in
/// (0, 1); the fitness adds them up exactly, as integers, and divides the sum by 2^53 N once, so
/// it's the same to the bit on every platform, and two strings tie only when their sums are equal.
///
/// The optimum isn't known in closed form: [`optimum`](Problem::optimum) computes it, by dynamic
/// programming over the circle for [`Adjacent`](Neighborhood::Adjacent) neighborhoods, in
/// O(N 4^K) steps, or by evaluating every string, changed one bit at a time in Gray code order,
/// in O(2^N K) steps, whichever is less work; and gives `None` when both take more than 2^32
/// steps. The work is done at each call.
///
/// Kauffman, S. A. and Weinberger, E. D. (1989). The NK model of rugged fitness landscapes and
/// its application to maturation of the immune response. *Journal of Theoretical Biology*
/// 141(2): 211-245, the section "The NK model" (two states per site, uniform fitness
/// contributions, equation 1) and Tables 1 and 2 (adjacent and random neighborhoods).
///
/// ```
/// use genoxide::prelude::*;
/// use genoxide::problems::Problem;
/// use genoxide::problems::binary::{Neighborhood, NkLandscape};
///
/// let landscape = NkLandscape::new(16, 2, Neighborhood::Adjacent, 7)?;
/// let optimum = landscape.optimum().expect("small enough");
/// let ga = Ga::builder(landscape.representation())
///     .population_size(50)
///     .select(Tournament::new(2)?)
///     .crossover(UniformCrossover::new())
///     .mutate(BitFlip::per_gene(1.0 / 16.0)?)
///     .seed(1)
///     .build()?;
/// let outcome = Engine::new(ga, landscape)
///     .stop_when(Stop::target(optimum.value()).or(Stop::generations(500)))
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Target);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NkLandscape {
    n: usize,
    k: usize,
    neighborhood: Neighborhood,
    seed: u64,
    // the K neighbors of each site, a row per site
    neighbors: Vec<usize>,
    // the 2^(K+1) contributions of each site, a row per site, as odd integers below 2^53
    tables: Vec<u64>,
}

impl NkLandscape {
    /// A landscape of `n` bits, each bearing on its contribution with `k` others chosen by
    /// `neighborhood`, the neighbors and contributions drawn from `seed`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] if `n` is 0 or above 2^24, `k` isn't below `n`, or the tables
    /// would have more than 2^24 values in all (N 2^(K+1)).
    pub fn new(n: usize, k: usize, neighborhood: Neighborhood, seed: u64) -> Result<Self> {
        if !(1..=MAX_BITS).contains(&n) {
            return Err(Error::InvalidSetting {
                setting: "n",
                reason: format!("must be between 1 and 2^24, got {n}"),
            });
        }
        if k >= n {
            return Err(Error::InvalidSetting {
                setting: "k",
                reason: format!("must be below n = {n}, got {k}"),
            });
        }
        let entries = u32::try_from(k + 1)
            .ok()
            .and_then(|bits| 1usize.checked_shl(bits))
            .and_then(|table| table.checked_mul(n));
        if entries.is_none_or(|entries| entries > MAX_ENTRIES) {
            return Err(Error::InvalidSetting {
                setting: "k",
                reason: format!(
                    "the tables of N = {n} sites with K = {k} would have N 2^(K+1) values, more \
                     than 2^24"
                ),
            });
        }
        let mut rng = StreamRng::seed_from_u64(seed).derive(NK_STREAM);
        let mut neighbors = Vec::with_capacity(n * k);
        for site in 0..n {
            match neighborhood {
                Neighborhood::Adjacent => {
                    let (before, after) = (k / 2, k - k / 2);
                    neighbors.extend((1..=before).rev().map(|d| (site + n - d) % n));
                    neighbors.extend((1..=after).map(|d| (site + d) % n));
                }
                Neighborhood::Random => {
                    // k of the n − 1 other sites, in ascending order
                    let others = rng.sample_distinct(k, n - 1);
                    neighbors.extend(others.into_iter().map(|j| if j < site { j } else { j + 1 }));
                }
            }
        }
        let tables = (0..n << (k + 1))
            .map(|_| ((rng.next_u64() >> 12) << 1) | 1)
            .collect();
        Ok(Self {
            n,
            k,
            neighborhood,
            seed,
            neighbors,
            tables,
        })
    }

    /// The number of bits, N.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The number of other sites that bear on each site, K.
    pub fn k(&self) -> usize {
        self.k
    }

    /// How the neighbors were chosen.
    pub fn neighborhood(&self) -> Neighborhood {
        self.neighborhood
    }

    /// The seed the landscape was drawn from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The K sites that bear on site `site`, in the order of the bits of its table's index.
    ///
    /// # Panics
    ///
    /// If `site` isn't below N.
    pub fn neighbors(&self, site: usize) -> &[usize] {
        assert!(site < self.n, "site {site} of {} sites", self.n);
        &self.neighbors[site * self.k..(site + 1) * self.k]
    }

    /// The contribution of site `site` when it and its neighbors have the bits of `index`: the
    /// site's own bit is bit K of the index (the highest), and its j-th neighbor's is bit
    /// K − 1 − j.
    ///
    /// # Panics
    ///
    /// If `site` isn't below N or `index` isn't below 2^(K+1).
    pub fn contribution(&self, site: usize, index: usize) -> f64 {
        assert!(site < self.n, "site {site} of {} sites", self.n);
        assert!(index < 1 << (self.k + 1), "an index has K + 1 bits");
        self.tables[(site << (self.k + 1)) + index] as f64 * UNIT
    }

    // the sum of the contributions of `x`, as integers
    fn sum(&self, x: &Bits) -> u128 {
        let words = x.as_words();
        let bit = |i: usize| (words[i / 64] >> (i % 64) & 1) as usize;
        let mut sum = 0;
        for site in 0..self.n {
            let mut index = bit(site);
            for &neighbor in self.neighbors(site) {
                index = index << 1 | bit(neighbor);
            }
            sum += u128::from(self.tables[(site << (self.k + 1)) + index]);
        }
        sum
    }

    // the fitness of a sum of contributions
    fn fitness(&self, sum: u128) -> f64 {
        sum as f64 * UNIT / self.n as f64
    }

    // the best string, by evaluating every string in Gray code order, a bit flipped at a time
    fn exhaustive(&self) -> Bits {
        let (n, k) = (self.n, self.k);
        // the sites whose index has bit `j`, and where
        let mut dependents: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for site in 0..n {
            dependents[site].push((site, k));
            for (j, &neighbor) in self.neighbors(site).iter().enumerate() {
                dependents[neighbor].push((site, k - 1 - j));
            }
        }
        let entry = |site: usize, index: usize| u128::from(self.tables[(site << (k + 1)) + index]);
        let mut indices = vec![0usize; n];
        let mut sum: u128 = (0..n).map(|site| entry(site, 0)).sum();
        let (mut best, mut best_string, mut string) = (sum, 0u64, 0u64);
        for step in 1u64..1 << n {
            let flipped = step.trailing_zeros() as usize;
            string ^= 1 << flipped;
            for &(site, position) in &dependents[flipped] {
                sum -= entry(site, indices[site]);
                indices[site] ^= 1 << position;
                sum += entry(site, indices[site]);
            }
            if sum > best {
                (best, best_string) = (sum, string);
            }
        }
        (0..n).map(|i| best_string >> i & 1 == 1).collect()
    }

    // the best string by dynamic programming, for adjacent neighborhoods: the first K bits fixed
    // in turn, then the others chosen one by one, the state being the last K bits
    fn dynamic(&self) -> Bits {
        let (n, k) = (self.n, self.k);
        let (before, after) = (k / 2, k - k / 2);
        let mut best: Option<(u128, usize, usize)> = None;
        for prefix in 0..1usize << k {
            let (sums, _) = self.dynamic_from(prefix, false);
            for (state, sum) in sums.into_iter().enumerate() {
                let Some(sum) = sum else { continue };
                let total = sum + self.wrapped(prefix, state, before, after);
                if best.is_none_or(|(best, _, _)| total > best) {
                    best = Some((total, prefix, state));
                }
            }
        }
        let (_, prefix, state) = best.expect("a string");
        let (_, dropped) = self.dynamic_from(prefix, true);
        // back from the last state, each step giving the bit it dropped
        let mut x = vec![false; n];
        for (i, bit) in x.iter_mut().take(k).enumerate() {
            *bit = prefix >> (k - 1 - i) & 1 == 1;
        }
        let mut state = state;
        for p in (k..n).rev() {
            let bit = dropped[(p - k) * (1 << k) + state];
            if k == 0 {
                // no state: the window is the bit itself
                x[p] = bit;
                continue;
            }
            x[p] = state & 1 == 1;
            state = (state >> 1) | (usize::from(bit) << (k - 1));
        }
        x.into_iter().collect()
    }

    // the best sums of the sites that don't wrap around the circle, for each final state (the
    // last K bits, the last one lowest), from `prefix` (the first K bits, the first one highest);
    // and, if `record`, the bit dropped at each step and new state, to go back
    fn dynamic_from(&self, prefix: usize, record: bool) -> (Vec<Option<u128>>, Vec<bool>) {
        let (n, k) = (self.n, self.k);
        let after = k - k / 2;
        let states = 1usize << k;
        let mask = states - 1;
        // a window of K + 1 bits x_{p−K} … x_p, x_p lowest, as the index of site p − after
        let index = |window: usize| {
            let own = window >> after & 1;
            let left = window >> (after + 1);
            let right = window & ((1 << after) - 1);
            own << k | left << after | right
        };
        let mut sums: Vec<Option<u128>> = vec![None; states];
        sums[prefix] = Some(0);
        let mut dropped = if record {
            vec![false; (n - k) * states]
        } else {
            Vec::new()
        };
        let mut next = vec![None; states];
        for p in k..n {
            next.fill(None);
            let site = p - after;
            let table = &self.tables[site << (k + 1)..(site + 1) << (k + 1)];
            for (state, sum) in sums.iter().enumerate() {
                let Some(sum) = sum else { continue };
                for bit in 0..2 {
                    let window = state << 1 | bit;
                    let total = sum + u128::from(table[index(window)]);
                    let new = window & mask;
                    if next[new].is_none_or(|best| total > best) {
                        next[new] = Some(total);
                        if record {
                            dropped[(p - k) * states + new] = window >> k & 1 == 1;
                        }
                    }
                }
            }
            std::mem::swap(&mut sums, &mut next);
        }
        (sums, dropped)
    }

    // the contributions of the K sites whose windows wrap around the circle, from the first K
    // bits and the last K
    fn wrapped(&self, prefix: usize, state: usize, before: usize, after: usize) -> u128 {
        let (n, k) = (self.n, self.k);
        let bit = |position: usize| {
            if position < k {
                prefix >> (k - 1 - position) & 1
            } else {
                state >> (n - 1 - position) & 1
            }
        };
        let mut sum = 0;
        for site in (0..before).chain(n - after..n) {
            let mut index = bit(site);
            for &neighbor in self.neighbors(site) {
                index = index << 1 | bit(neighbor);
            }
            sum += u128::from(self.tables[(site << (k + 1)) + index]);
        }
        sum
    }
}

impl FitnessFunction<Bits> for NkLandscape {
    type Output = f64;

    /// The mean contribution of the sites of `x`, W.
    ///
    /// # Panics
    ///
    /// If `x` doesn't have N bits.
    fn evaluate(&self, x: &Bits) -> f64 {
        check_length("NkLandscape", x, self.n);
        self.fitness(self.sum(x))
    }
}

impl Problem for NkLandscape {
    type Representation = Binary;

    fn name(&self) -> &'static str {
        "NkLandscape"
    }

    fn representation(&self) -> Binary {
        binary(self.n)
    }

    fn objective(&self) -> Objective {
        Objective::Maximize
    }

    /// The global maximum and a string that reaches it (the first found, if several tie), by
    /// dynamic programming or exhaustive search, whichever takes fewer steps; `None` if both take
    /// more than 2^32.
    fn optimum(&self) -> Option<Optimum<Bits>> {
        let (n, k) = (self.n as u128, self.k as u32);
        let exhaustive = (n < 64).then(|| (1u128 << n) * u128::from(k + 1));
        let dynamic = (self.neighborhood == Neighborhood::Adjacent && k < 40)
            .then(|| n * (1u128 << (2 * k + 1)));
        let solution = match (exhaustive, dynamic) {
            (Some(e), Some(d)) if d < e && d <= MAX_WORK => self.dynamic(),
            (Some(e), _) if e <= MAX_WORK => self.exhaustive(),
            (_, Some(d)) if d <= MAX_WORK => self.dynamic(),
            _ => return None,
        };
        Some(Optimum::proven(self.evaluate(&solution), vec![solution]))
    }

    fn reference(&self) -> &'static str {
        "Kauffman, S. A. and Weinberger, E. D. (1989). The NK model of rugged fitness landscapes \
         and its application to maturation of the immune response. Journal of Theoretical \
         Biology 141(2): 211-245."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1016/S0022-5193(89)80019-0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Representation;

    // the best fitness by evaluating every string directly
    fn brute_force(landscape: &NkLandscape) -> f64 {
        let n = landscape.n();
        (0u64..1 << n)
            .map(|string| {
                let x: Bits = (0..n).map(|i| string >> i & 1 == 1).collect();
                landscape.evaluate(&x)
            })
            .fold(f64::NEG_INFINITY, f64::max)
    }

    #[test]
    fn the_fitness_is_the_mean_of_the_tables() {
        let landscape = NkLandscape::new(6, 2, Neighborhood::Random, 4).unwrap();
        let x: Bits = "101100".chars().map(|bit| bit == '1').collect();
        let mut sum = 0.0;
        for site in 0..6 {
            let mut index = usize::from(x.get(site).unwrap());
            for &neighbor in landscape.neighbors(site) {
                assert_ne!(neighbor, site);
                index = index << 1 | usize::from(x.get(neighbor).unwrap());
            }
            let w = landscape.contribution(site, index);
            assert!(w > 0.0 && w < 1.0);
            sum += w;
        }
        assert!((landscape.evaluate(&x) - sum / 6.0).abs() < 1e-15);
    }

    #[test]
    fn adjacent_neighbors_flank_each_site_on_a_circle() {
        let landscape = NkLandscape::new(8, 4, Neighborhood::Adjacent, 0).unwrap();
        assert_eq!(landscape.neighbors(0), [6, 7, 1, 2]);
        assert_eq!(landscape.neighbors(5), [3, 4, 6, 7]);
        let odd = NkLandscape::new(8, 3, Neighborhood::Adjacent, 0).unwrap();
        assert_eq!(odd.neighbors(7), [6, 0, 1]);
        let none = NkLandscape::new(3, 0, Neighborhood::Adjacent, 0).unwrap();
        assert!(none.neighbors(1).is_empty());
    }

    #[test]
    fn random_neighbors_are_distinct_others() {
        let landscape = NkLandscape::new(20, 7, Neighborhood::Random, 9).unwrap();
        for site in 0..20 {
            let neighbors = landscape.neighbors(site);
            assert_eq!(neighbors.len(), 7);
            assert!(!neighbors.contains(&site));
            assert!(neighbors.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    // a landscape is the same on every platform: fixed values, to the bit
    #[test]
    fn landscapes_are_reproducible_from_their_seed() {
        let landscape = NkLandscape::new(10, 3, Neighborhood::Random, 42).unwrap();
        assert_eq!(
            landscape,
            NkLandscape::new(10, 3, Neighborhood::Random, 42).unwrap()
        );
        assert_ne!(
            landscape,
            NkLandscape::new(10, 3, Neighborhood::Random, 43).unwrap()
        );
        assert_eq!(landscape.neighbors(0), [2, 6, 7]);
        assert_eq!(landscape.neighbors(9), [4, 5, 8]);
        assert_eq!(
            landscape.contribution(0, 0).to_bits(),
            0x3fd2_08b8_6307_d686
        );
        assert_eq!(
            landscape.contribution(9, 15).to_bits(),
            0x3feb_fb4d_db16_b0c9
        );
        let x: Bits = "1100101101".chars().map(|bit| bit == '1').collect();
        assert_eq!(landscape.evaluate(&x).to_bits(), 0x3fd5_5624_43ce_9d63);
    }

    #[test]
    fn the_optimum_is_the_best_string() {
        for (n, k, neighborhood, seed) in [
            (1, 0, Neighborhood::Random, 0),
            (5, 0, Neighborhood::Adjacent, 1),
            (8, 1, Neighborhood::Adjacent, 2),
            (9, 2, Neighborhood::Adjacent, 3),
            (10, 3, Neighborhood::Adjacent, 4),
            (7, 6, Neighborhood::Adjacent, 5),
            (12, 4, Neighborhood::Random, 6),
            (11, 10, Neighborhood::Random, 7),
        ] {
            let landscape = NkLandscape::new(n, k, neighborhood, seed).unwrap();
            let optimum = landscape.optimum().unwrap();
            assert!(optimum.is_proven());
            assert_eq!(optimum.value(), brute_force(&landscape), "{n} {k}");
            assert_eq!(landscape.evaluate(&optimum.solutions()[0]), optimum.value());
            // dynamic programming and exhaustive search agree
            if neighborhood == Neighborhood::Adjacent {
                let dynamic = landscape.dynamic();
                let exhaustive = landscape.exhaustive();
                assert_eq!(
                    landscape.sum(&dynamic),
                    landscape.sum(&exhaustive),
                    "{n} {k}"
                );
            }
        }
    }

    #[test]
    fn large_adjacent_landscapes_are_solved_by_dynamic_programming() {
        let landscape = NkLandscape::new(200, 4, Neighborhood::Adjacent, 1).unwrap();
        let optimum = landscape.optimum().unwrap();
        // no string near it is better
        let best = optimum.solutions()[0].clone();
        for i in 0..200 {
            let mut x = best.clone();
            x.flip(i);
            assert!(landscape.evaluate(&x) <= optimum.value());
        }
        let mut rng = StreamRng::seed_from_u64(1);
        for _ in 0..1_000 {
            let x = Binary::new(200).unwrap().random_genome(&mut rng);
            assert!(landscape.evaluate(&x) < optimum.value());
        }
        // random neighborhoods of 200 bits are beyond exhaustive search
        let random = NkLandscape::new(200, 4, Neighborhood::Random, 1).unwrap();
        assert!(random.optimum().is_none());
    }

    #[test]
    fn invalid_landscapes_are_errors() {
        assert!(NkLandscape::new(0, 0, Neighborhood::Random, 0).is_err());
        assert!(NkLandscape::new(5, 5, Neighborhood::Random, 0).is_err());
        assert!(NkLandscape::new(100, 30, Neighborhood::Random, 0).is_err());
        assert!(NkLandscape::new((1 << 24) + 1, 0, Neighborhood::Random, 0).is_err());
        assert!(NkLandscape::new(5, 4, Neighborhood::Adjacent, 0).is_ok());
    }
}
