//! Parent selection.

use super::{MAX_SIZE, Select, check_rate};
use crate::genome::Genome;
use crate::rng::Chance;
use crate::{Error, Fitness, Objective, Population, Result, StreamRng};

fn fitness_of<G: Genome>(population: &Population<G>, index: usize) -> Fitness {
    population[index].fitness().unwrap_or(Fitness::invalid())
}

// A number in the order of `objective.compare`: the larger, the better, and equal for equal
// fitness values, so comparing keys gives the same selections and ties as comparing fitness
// values, in a few instructions. Invalid (and unevaluated) is 0. Otherwise the high half is the
// violation's bits, inverted: a violation is 0 or more, never -0 or NaN, so its bits are in its
// order. The low half is the score's bits in the order of `f64::total_cmp`, inverted to minimize.
fn sort_key(fitness: Option<Fitness>, objective: Objective) -> u128 {
    let Some((score, violation)) =
        fitness.and_then(|fitness| Some((fitness.score()?, fitness.violation())))
    else {
        return 0;
    };
    let bits = score.to_bits();
    let ordered = if bits >> 63 == 1 {
        !bits
    } else {
        bits | 1 << 63
    };
    let ordered = match objective {
        Objective::Maximize => ordered,
        Objective::Minimize => !ordered,
    };
    u128::from(!violation.to_bits()) << 64 | u128::from(ordered)
}

// the sort key of every individual
fn sort_keys<G: Genome>(population: &Population<G>, objective: Objective) -> Vec<u128> {
    population
        .iter()
        .map(|individual| sort_key(individual.fitness(), objective))
        .collect()
}

/// Tournament selection: the best of `size` individuals drawn at random (with replacement).
///
/// Larger tournaments give more selection pressure. A size of 1 is random selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tournament {
    size: usize,
}

impl Tournament {
    /// Tournaments of `size` individuals, at least 1 and at most 2^24, the largest population
    /// size.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a size of 0 or above 2^24.
    pub fn new(size: usize) -> Result<Self> {
        if size == 0 || size > MAX_SIZE {
            return Err(Error::InvalidSetting {
                setting: "tournament_size",
                reason: format!("must be between 1 and {MAX_SIZE}, got {size}"),
            });
        }
        Ok(Self { size })
    }

    /// The tournament size.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Select for Tournament {
    fn select<G: Genome>(
        &self,
        population: &Population<G>,
        objective: Objective,
        count: usize,
        rng: &mut StreamRng,
    ) -> Vec<usize> {
        let len = population.len();
        if len == 0 {
            return Vec::new();
        }
        if count.saturating_mul(self.size - 1) < len {
            // fewer comparisons than individuals: fitness values, compared as needed
            return (0..count)
                .map(|_| {
                    let mut winner = rng.below(len);
                    for _ in 1..self.size {
                        let contender = rng.below(len);
                        if objective.is_better(
                            fitness_of(population, contender),
                            fitness_of(population, winner),
                        ) {
                            winner = contender;
                        }
                    }
                    winner
                })
                .collect();
        }
        // the same tournaments, with a key per individual computed once
        let keys = sort_keys(population, objective);
        (0..count)
            .map(|_| {
                let mut winner = rng.below(len);
                let mut best = keys[winner];
                for _ in 1..self.size {
                    let contender = rng.below(len);
                    if keys[contender] > best {
                        (winner, best) = (contender, keys[contender]);
                    }
                }
                winner
            })
            .collect()
    }
}

// Selection weights proportional to how much better than the worst feasible individual an
// individual is. Invalid and infeasible individuals get weight 0. None if no individual has a
// positive weight.
fn proportional_weights<G: Genome>(
    population: &Population<G>,
    objective: Objective,
) -> Option<Vec<f64>> {
    let mut scores: Vec<Option<f64>> = (0..population.len())
        .map(|index| {
            let fitness = fitness_of(population, index);
            fitness.is_feasible().then(|| fitness.score()).flatten()
        })
        .collect();
    // Huge finite scores could make a difference to the worst, or the sum of the weights,
    // overflow. Scaling every score by a power of two keeps their proportions exact and makes
    // both impossible: each weight is then at most twice the largest score.
    let limit = f64::MAX / (4.0 * population.len() as f64);
    let largest = scores
        .iter()
        .flatten()
        .filter(|score| score.is_finite())
        .fold(0.0, |largest: f64, score| largest.max(score.abs()));
    if largest >= limit {
        let mut scale = 1.0;
        while largest * scale >= limit {
            scale *= 1.0 / (1u64 << 32) as f64;
        }
        for score in scores.iter_mut().flatten() {
            *score *= scale;
        }
    }
    let worst = scores.iter().flatten().copied().reduce(|a, b| {
        if objective.is_better(Fitness::new(a), Fitness::new(b)) {
            b
        } else {
            a
        }
    })?;
    let weights: Vec<f64> = scores
        .iter()
        .map(|score| match (score, objective) {
            (None, _) => 0.0,
            (Some(score), Objective::Maximize) => score - worst,
            (Some(score), Objective::Minimize) => worst - score,
        })
        .collect();
    if weights.iter().all(|weight| weight.is_finite()) {
        return (weights.iter().sum::<f64>() > 0.0).then_some(weights);
    }
    let best = scores.iter().flatten().copied().reduce(|a, b| {
        if objective.is_better(Fitness::new(b), Fitness::new(a)) {
            b
        } else {
            a
        }
    })?;
    Some(if best.is_infinite() {
        // an infinitely good score: only the best individuals get an (equal) weight
        scores
            .iter()
            .map(|score| if *score == Some(best) { 1.0 } else { 0.0 })
            .collect()
    } else {
        // an infinitely bad score: its individuals get no weight, the others an equal one, the
        // limit of a finite worst score going to infinity
        scores
            .iter()
            .map(|score| match score {
                Some(score) if *score != worst => 1.0,
                _ => 0.0,
            })
            .collect()
    })
}

// Cumulative weights, and the index of the last positive weight (the fallback for rounding).
fn cumulative(weights: &[f64]) -> (Vec<f64>, usize) {
    let mut total = 0.0;
    let cumulative = weights
        .iter()
        .map(|weight| {
            total += weight;
            total
        })
        .collect();
    let last_positive = weights
        .iter()
        .rposition(|&weight| weight > 0.0)
        .unwrap_or(0);
    (cumulative, last_positive)
}

// The index whose cumulative interval contains `point` (skips zero weights).
fn pick(cumulative: &[f64], last_positive: usize, point: f64) -> usize {
    cumulative
        .partition_point(|&sum| sum <= point)
        .min(last_positive)
}

fn uniform_indices(population_len: usize, count: usize, rng: &mut StreamRng) -> Vec<usize> {
    if population_len == 0 {
        return Vec::new();
    }
    (0..count).map(|_| rng.below(population_len)).collect()
}

/// Roulette wheel (fitness proportionate) selection.
///
/// The chance of an individual is proportional to how much better it is than the worst
/// individual, so scores don't need to be positive. Invalid and infeasible individuals are never
/// selected, unless every individual is equal, invalid or infeasible, in which case the selection
/// is uniform: with constraints, prefer [`Tournament`] or [`Rank`], which follow the feasibility
/// rules.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Roulette;

impl Select for Roulette {
    fn select<G: Genome>(
        &self,
        population: &Population<G>,
        objective: Objective,
        count: usize,
        rng: &mut StreamRng,
    ) -> Vec<usize> {
        let Some(weights) = proportional_weights(population, objective) else {
            return uniform_indices(population.len(), count, rng);
        };
        let (cumulative, last_positive) = cumulative(&weights);
        let total = *cumulative.last().expect("not empty");
        (0..count)
            .map(|_| pick(&cumulative, last_positive, rng.unit_f64() * total))
            .collect()
    }
}

/// Stochastic universal sampling: roulette wheel selection with evenly spaced pointers.
///
/// It uses the same chances as [`Roulette`], with less spread: an individual with an expected
/// number of selections of e.g. 2.4 is selected 2 or 3 times. The selected individuals come in a
/// random order, as Baker's method shuffles them before mating: the pointers find them in
/// population order, and parents taken in pairs would otherwise often be the same individual.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StochasticUniversalSampling;

impl Select for StochasticUniversalSampling {
    fn select<G: Genome>(
        &self,
        population: &Population<G>,
        objective: Objective,
        count: usize,
        rng: &mut StreamRng,
    ) -> Vec<usize> {
        let Some(weights) = proportional_weights(population, objective) else {
            return uniform_indices(population.len(), count, rng);
        };
        if count == 0 {
            return Vec::new();
        }
        let (cumulative, last_positive) = cumulative(&weights);
        let total = *cumulative.last().expect("not empty");
        let spacing = total / count as f64;
        let start = rng.unit_f64() * spacing;
        let mut picks: Vec<usize> = (0..count)
            .map(|i| pick(&cumulative, last_positive, start + i as f64 * spacing))
            .collect();
        // Fisher-Yates
        for i in (1..picks.len()).rev() {
            picks.swap(i, rng.below(i + 1));
        }
        picks
    }
}

/// Linear rank selection: the chance depends on the rank, not on the score.
///
/// With `pressure` `s` (from 1 to 2), the best individual is expected to be selected `s` times as
/// often as the average one and the worst `2 - s` times. Equal fitness values share their rank.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rank {
    pressure: f64,
}

impl Rank {
    /// Linear ranking with selection `pressure` from 1 (uniform) to 2.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a pressure outside [1, 2].
    pub fn new(pressure: f64) -> Result<Self> {
        if (1.0..=2.0).contains(&pressure) {
            Ok(Self { pressure })
        } else {
            Err(Error::InvalidSetting {
                setting: "rank_pressure",
                reason: format!("must be between 1 and 2, got {pressure}"),
            })
        }
    }
}

impl Default for Rank {
    /// A selection pressure of 1.5.
    fn default() -> Self {
        Self { pressure: 1.5 }
    }
}

impl Select for Rank {
    fn select<G: Genome>(
        &self,
        population: &Population<G>,
        objective: Objective,
        count: usize,
        rng: &mut StreamRng,
    ) -> Vec<usize> {
        let n = population.len();
        if n < 2 {
            return uniform_indices(n, count, rng);
        }
        // worst first, stable
        let keys = sort_keys(population, objective);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&index| keys[index]);
        // average rank (0 = worst) for ties
        let mut ranks = vec![0.0; n];
        let mut start = 0;
        while start < n {
            let key = keys[order[start]];
            let end = (start..n).find(|&i| keys[order[i]] != key).unwrap_or(n);
            let average = (start + end - 1) as f64 / 2.0;
            for &index in &order[start..end] {
                ranks[index] = average;
            }
            start = end;
        }
        let s = self.pressure;
        let weights: Vec<f64> = ranks
            .iter()
            .map(|rank| (2.0 - s) + 2.0 * (s - 1.0) * rank / (n - 1) as f64)
            .collect();
        if weights.iter().all(|&weight| weight <= 0.0) {
            return uniform_indices(n, count, rng);
        }
        let (cumulative, last_positive) = cumulative(&weights);
        let total = *cumulative.last().expect("not empty");
        (0..count)
            .map(|_| pick(&cumulative, last_positive, rng.unit_f64() * total))
            .collect()
    }
}

/// Truncation selection: uniformly random from the best `fraction` of the population, its best
/// ⌈fraction × size⌉ individuals (at least 1). A product within a few units of the last place of
/// a whole number counts as that number: 0.07 of 100 is 7, although `0.07 * 100.0` is
/// 7.000000000000001 in floating point.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Truncation {
    fraction: f64,
}

impl Truncation {
    /// Selects from the best `fraction` of the population, greater than 0 and at most 1.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a fraction outside (0, 1].
    pub fn new(fraction: f64) -> Result<Self> {
        Ok(Self {
            fraction: check_rate("truncation_fraction", fraction)?,
        })
    }
}

impl Select for Truncation {
    fn select<G: Genome>(
        &self,
        population: &Population<G>,
        objective: Objective,
        count: usize,
        rng: &mut StreamRng,
    ) -> Vec<usize> {
        let n = population.len();
        if n == 0 {
            return Vec::new();
        }
        // best first, stable
        let keys = sort_keys(population, objective);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| keys[b].cmp(&keys[a]));
        let top = truncation_size(self.fraction, n);
        (0..count).map(|_| order[rng.below(top)]).collect()
    }
}

// ⌈fraction × n⌉, from 1 to n, without the floating-point error of the product: 0.07 × 100 is
// 7.000000000000001, whose ceiling would add a whole individual
fn truncation_size(fraction: f64, n: usize) -> usize {
    let product = fraction * n as f64;
    let nearest = product.round();
    let size = if (product - nearest).abs() <= 4.0 * f64::EPSILON * nearest {
        nearest
    } else {
        product.ceil()
    };
    (size as usize).clamp(1, n)
}

/// Uniformly random selection, without selection pressure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RandomSelection;

impl Select for RandomSelection {
    fn select<G: Genome>(
        &self,
        population: &Population<G>,
        _objective: Objective,
        count: usize,
        rng: &mut StreamRng,
    ) -> Vec<usize> {
        uniform_indices(population.len(), count, rng)
    }
}

// `count` tournaments of `size` contestants drawn from `0..len` with replacement: the winner of
// each is the contestant that `better` prefers to the winner so far, so ties go to the first
fn tournaments(
    len: usize,
    count: usize,
    size: usize,
    rng: &mut StreamRng,
    mut better: impl FnMut(usize, usize) -> bool,
) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    (0..count)
        .map(|_| {
            let mut winner = rng.below(len);
            for _ in 1..size {
                let contender = rng.below(len);
                if better(contender, winner) {
                    winner = contender;
                }
            }
            winner
        })
        .collect()
}

fn check_tournament_size(setting: &'static str, size: usize) -> Result<usize> {
    if size == 0 || size > MAX_SIZE {
        return Err(Error::InvalidSetting {
            setting,
            reason: format!("must be between 1 and {MAX_SIZE}, got {size}"),
        });
    }
    Ok(size)
}

/// Lexicographic parsimony pressure (Luke and Panait 2002): tournament selection in which, of
/// individuals of equal fitness, the smaller wins.
///
/// Size is [`Genome::len`]: a tree's number of nodes ([`gp::Tree`](crate::gp::Tree)). Fitness
/// comes first, so the pressure toward small genomes acts only among equals: strong where
/// fitness values repeat (Boolean problems, counts of cases), little with continuous fitness.
/// Of individuals equal in fitness and size, the first drawn wins, which is a random one, as in
/// the paper. With genomes of one length it is [`Tournament`]. Luke and Panait used tournaments
/// of 2.
///
/// For continuous fitness, [`ratio_buckets`](LexicographicTournament::ratio_buckets) (the
/// paper's ratio bucketing) makes near values equal: the population sorted by fitness is cut
/// into buckets, and a tournament compares the buckets, then sizes.
///
/// ```
/// use genoxide::prelude::*;
///
/// let select = LexicographicTournament::new(2)?;
/// assert_eq!(select.size(), 2);
/// assert!(LexicographicTournament::new(0).is_err());
/// let bucketed = select.ratio_buckets(0.5)?; // the worst half, the worst half of the rest, ...
/// assert_eq!(bucketed.bucket_ratio(), Some(0.5));
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LexicographicTournament {
    size: usize,
    bucket_ratio: Option<f64>,
}

impl LexicographicTournament {
    /// Tournaments of `size` individuals, at least 1 and at most 2^24 (2 in Luke and Panait's
    /// experiments).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] (setting `tournament_size`) for a size of 0 or above 2^24.
    pub fn new(size: usize) -> Result<Self> {
        Ok(Self {
            size: check_tournament_size("tournament_size", size)?,
            bucket_ratio: None,
        })
    }

    /// Ratio bucketing (Luke and Panait 2002): the population, sorted by fitness, goes into
    /// buckets, and tournaments compare the buckets instead of the fitness values. The worst
    /// `ratio` of the population (`1 / r` in the paper, rounded up to a whole individual) goes
    /// into the lowest bucket, with every other individual of the same fitness as the best in
    /// it; the worst `ratio` of the rest into the next; and so on, so buckets are finer toward
    /// the top. Luke and Panait (2006) recommend a ratio of 1/2.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] (setting `bucket_ratio`) for a ratio outside (0, 1].
    pub fn ratio_buckets(mut self, ratio: f64) -> Result<Self> {
        self.bucket_ratio = Some(check_rate("bucket_ratio", ratio)?);
        Ok(self)
    }

    /// The tournament size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The ratio of the buckets, if fitness values are bucketed.
    pub fn bucket_ratio(&self) -> Option<f64> {
        self.bucket_ratio
    }
}

// the bucket of each individual, from its sort key (the larger, the better): the worst `ratio`
// of the individuals, rounded up, and the others equal to the best of them, in bucket 0; of the
// rest, the worst `ratio` in bucket 1; and so on
fn ratio_buckets(keys: &[u128], ratio: f64) -> Vec<u32> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by_key(|&index| keys[index]);
    let mut buckets = vec![0; keys.len()];
    let (mut start, mut bucket) = (0, 0);
    while start < order.len() {
        let left = order.len() - start;
        let take = ((left as f64 * ratio).ceil() as usize).clamp(1, left);
        let mut end = start + take;
        while end < order.len() && keys[order[end]] == keys[order[end - 1]] {
            end += 1;
        }
        for &index in &order[start..end] {
            buckets[index] = bucket;
        }
        (start, bucket) = (end, bucket + 1);
    }
    buckets
}

impl Select for LexicographicTournament {
    fn select<G: Genome>(
        &self,
        population: &Population<G>,
        objective: Objective,
        count: usize,
        rng: &mut StreamRng,
    ) -> Vec<usize> {
        let len = population.len();
        if let Some(ratio) = self.bucket_ratio {
            let buckets = ratio_buckets(&sort_keys(population, objective), ratio);
            let size = |index: usize| std::cmp::Reverse(population[index].genome().len());
            let keys: Vec<_> = (0..len)
                .map(|index| (buckets[index], size(index)))
                .collect();
            return tournaments(len, count, self.size, rng, |a, b| keys[a] > keys[b]);
        }
        let key = |index: usize| {
            let individual = &population[index];
            (
                sort_key(individual.fitness(), objective),
                std::cmp::Reverse(individual.genome().len()),
            )
        };
        if count.saturating_mul(self.size - 1) < len {
            // fewer comparisons than individuals: keys computed as needed
            return tournaments(len, count, self.size, rng, |a, b| key(a) > key(b));
        }
        let keys: Vec<_> = (0..len).map(key).collect();
        tournaments(len, count, self.size, rng, |a, b| keys[a] > keys[b])
    }
}

/// Double tournament (Luke and Panait 2002): tournaments of fitness whose contestants are the
/// winners of tournaments of size, or the other way around.
///
/// - **A size tournament** takes two individuals and returns the smaller with probability
///   `parsimony / 2`, else the larger: `parsimony`, D in [1, 2], sets the pressure, from none
///   (1) to always the smaller (2). Of two of the same size, the first wins, which is a random
///   one, as in the paper.
/// - **Fitness first** (the default): a size tournament between the winners of two fitness
///   tournaments of `fitness_size` individuals.
/// - **Size first** ([`size_first`](DoubleTournament::size_first)): a fitness tournament of
///   `fitness_size` contestants, each the winner of a size tournament of two random individuals.
///
/// Size is [`Genome::len`]: a tree's number of nodes ([`gp::Tree`](crate::gp::Tree)). Unlike a
/// size penalty, the pressure doesn't depend on the scale of the fitness values. With Koza's
/// depth limit, Luke and Panait (2002) found D from 1.2 to 1.6 as fit as no parsimony pressure,
/// with trees often half the size, and the order of the tournaments of no significant effect;
/// their comparison of bloat control methods (2006) found fitness tournaments of 7 and D = 1.4
/// consistently the best double tournament, and double tournament among the best methods.
///
/// ```
/// use genoxide::prelude::*;
///
/// let select = DoubleTournament::new(7, 1.4)?;
/// assert_eq!((select.fitness_size(), select.parsimony()), (7, 1.4));
/// assert!(!select.is_size_first());
/// assert!(DoubleTournament::new(7, 2.5).is_err());
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DoubleTournament {
    fitness_size: usize,
    parsimony: f64,
    size_first: bool,
}

impl DoubleTournament {
    /// Fitness tournaments of `fitness_size` individuals (at least 1 and at most 2^24) and size
    /// tournaments in which the smaller of two wins with probability `parsimony / 2`,
    /// `parsimony` in [1, 2]; fitness first.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a fitness size of 0 or above 2^24 (setting
    /// `tournament_size`), or a parsimony outside [1, 2] (setting `parsimony`).
    pub fn new(fitness_size: usize, parsimony: f64) -> Result<Self> {
        let fitness_size = check_tournament_size("tournament_size", fitness_size)?;
        if !(1.0..=2.0).contains(&parsimony) {
            return Err(Error::InvalidSetting {
                setting: "parsimony",
                reason: format!("must be between 1 and 2, got {parsimony}"),
            });
        }
        Ok(Self {
            fitness_size,
            parsimony,
            size_first: false,
        })
    }

    /// Size tournaments first: their winners are the contestants of the fitness tournaments.
    pub fn size_first(mut self) -> Self {
        self.size_first = true;
        self
    }

    /// The size of the fitness tournaments.
    pub fn fitness_size(&self) -> usize {
        self.fitness_size
    }

    /// D: the smaller of two wins a size tournament with probability D / 2.
    pub fn parsimony(&self) -> f64 {
        self.parsimony
    }

    /// Whether the size tournaments come first.
    pub fn is_size_first(&self) -> bool {
        self.size_first
    }
}

impl Select for DoubleTournament {
    fn select<G: Genome>(
        &self,
        population: &Population<G>,
        objective: Objective,
        count: usize,
        rng: &mut StreamRng,
    ) -> Vec<usize> {
        let len = population.len();
        if len == 0 {
            return Vec::new();
        }
        let keys = sort_keys(population, objective);
        let sizes: Vec<usize> = population.iter().map(|i| i.genome().len()).collect();
        let smaller = Chance::new(self.parsimony / 2.0);
        let size_tournament = |a: usize, b: usize, rng: &mut StreamRng| {
            if sizes[a] == sizes[b] {
                return a;
            }
            let (small, large) = if sizes[a] < sizes[b] { (a, b) } else { (b, a) };
            if rng.chance(smaller) { small } else { large }
        };
        let fitness_tournament = |rng: &mut StreamRng, draw: &dyn Fn(&mut StreamRng) -> usize| {
            let mut winner = draw(rng);
            for _ in 1..self.fitness_size {
                let contender = draw(rng);
                if keys[contender] > keys[winner] {
                    winner = contender;
                }
            }
            winner
        };
        let random = |rng: &mut StreamRng| rng.below(len);
        let contestant = |rng: &mut StreamRng| {
            let a = rng.below(len);
            let b = rng.below(len);
            size_tournament(a, b, rng)
        };
        (0..count)
            .map(|_| {
                if self.size_first {
                    fitness_tournament(rng, &contestant)
                } else {
                    let a = fitness_tournament(rng, &random);
                    let b = fitness_tournament(rng, &random);
                    size_tournament(a, b, rng)
                }
            })
            .collect()
    }
}

/// Tarpeian bloat control (Poli 2003): a selection in which some of the genomes larger than the
/// population's mean size count as invalid.
///
/// Once per call, each genome larger than the mean [`Genome::len`] (a tree's number of nodes)
/// is marked with probability `rate`, independently, and `select` then chooses as if the marked
/// individuals were invalid: worse than every other. The more individuals above the mean, the
/// more are held back, whatever the scale of the fitness values.
///
/// Poli marks new individuals before they're evaluated and skips their evaluation. As a
/// selection, this keeps the effect on selection, not the saved evaluations, which would need
/// the algorithm's cooperation.
///
/// ```
/// use genoxide::prelude::*;
///
/// let select = Tarpeian::new(Tournament::new(7)?, 0.3)?;
/// assert_eq!(select.rate(), 0.3);
/// assert!(Tarpeian::new(Tournament::new(7)?, 0.0).is_err());
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tarpeian<S> {
    select: S,
    rate: f64,
}

impl<S: Select> Tarpeian<S> {
    /// `select`, with each genome larger than the mean size counted as invalid with probability
    /// `rate`, greater than 0 and at most 1.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] (setting `tarpeian_rate`) for a rate outside (0, 1].
    pub fn new(select: S, rate: f64) -> Result<Self> {
        Ok(Self {
            select,
            rate: check_rate("tarpeian_rate", rate)?,
        })
    }

    /// The selection it wraps.
    pub fn inner(&self) -> &S {
        &self.select
    }

    /// The probability that a genome larger than the mean counts as invalid.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    // the genomes larger than the mean, each with probability `rate`, in ascending order
    fn marks<G: Genome>(&self, population: &Population<G>, rng: &mut StreamRng) -> Vec<usize> {
        let len = population.len() as u128;
        let total: u128 = population.iter().map(|i| i.genome().len() as u128).sum();
        // larger than the mean: size * len > total, exactly
        let above: Vec<usize> = (0..population.len())
            .filter(|&index| population[index].genome().len() as u128 * len > total)
            .collect();
        let mut marked = Vec::new();
        rng.chosen(Chance::new(self.rate), above.len(), |_, k| {
            marked.push(above[k]);
        });
        marked
    }
}

impl<S: Select> Select for Tarpeian<S> {
    fn select<G: Genome>(
        &self,
        population: &Population<G>,
        objective: Objective,
        count: usize,
        rng: &mut StreamRng,
    ) -> Vec<usize> {
        let marked = self.marks(population, rng);
        // a view of the population, sharing its genomes
        let mut view: Population<&G> = population.iter().map(|i| i.borrowed()).collect();
        for index in marked {
            view[index].set_fitness(Fitness::invalid());
        }
        self.select.select(&view, objective, count, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_infinitely_bad_score_gets_no_weight_and_the_others_equal_ones() {
        // the limit of a finite worst score going to minus infinity
        let draws = 3_000;
        for (scores, objective) in [
            (
                vec![Some(f64::NEG_INFINITY), Some(1.0), Some(2.0), Some(3.0)],
                Objective::Maximize,
            ),
            (
                vec![Some(f64::INFINITY), Some(1.0), Some(2.0), Some(3.0)],
                Objective::Minimize,
            ),
        ] {
            for picked in [
                counts(&Roulette, &scores, objective, draws),
                counts(&StochasticUniversalSampling, &scores, objective, draws),
            ] {
                assert_eq!(picked[0], 0, "{picked:?}");
                for &count in &picked[1..] {
                    assert!((800..1_200).contains(&count), "{picked:?}");
                }
            }
        }
        // an infinitely good score still takes every pick
        let scores = [Some(1.0), Some(f64::INFINITY), Some(2.0)];
        let picked = counts(&Roulette, &scores, Objective::Maximize, draws);
        assert_eq!(picked, [0, draws, 0]);
    }

    #[test]
    fn sus_picks_come_in_a_random_order() {
        let scores: Vec<Option<f64>> = (0..10).map(|score| Some(f64::from(score))).collect();
        let population = population(&scores);
        let mut rng = StreamRng::seed_from_u64(3);
        let (mut sorted, mut self_pairs, mut pairs) = (0, 0, 0);
        for _ in 0..1_000 {
            let picks =
                StochasticUniversalSampling.select(&population, Objective::Maximize, 10, &mut rng);
            sorted += usize::from(picks.is_sorted());
            for [a, b] in picks.as_chunks::<2>().0 {
                self_pairs += usize::from(a == b);
                pairs += 1;
            }
        }
        // before, every selection was sorted, and 44% of the pairs were one individual twice
        assert!(sorted < 10, "{sorted}");
        let fraction = self_pairs as f64 / pairs as f64;
        assert!(fraction < 0.2, "{fraction}");
    }
    use crate::Individual;
    use crate::genome::Bits;
    use proptest::prelude::*;

    // individual i has genome i (distinct) and the given score (None = invalid)
    fn population(scores: &[Option<f64>]) -> Population<Bits> {
        scores
            .iter()
            .enumerate()
            .map(|(index, score)| {
                let mut individual =
                    Individual::new((0..8).map(|bit| index >> bit & 1 == 1).collect());
                individual.set_fitness(score.map_or(Fitness::invalid(), Fitness::new));
                individual
            })
            .collect()
    }

    fn counts<S: Select>(
        select: &S,
        scores: &[Option<f64>],
        objective: Objective,
        draws: usize,
    ) -> Vec<usize> {
        let population = population(scores);
        let mut rng = StreamRng::seed_from_u64(1);
        let mut counts = vec![0; scores.len()];
        for index in select.select(&population, objective, draws, &mut rng) {
            counts[index] += 1;
        }
        counts
    }

    #[test]
    fn ratio_buckets_are_finer_toward_the_top_and_keep_equal_fitness_together() {
        // keys 1 to 8: halves of halves, the worst 4, then 2, 1, 1
        let keys: Vec<u128> = (1..=8).collect();
        assert_eq!(ratio_buckets(&keys, 0.5), [0, 0, 0, 0, 1, 1, 2, 3]);
        // equal to the best of a bucket: in it
        let keys = [1, 2, 3, 4, 4, 4, 5, 6];
        assert_eq!(ratio_buckets(&keys, 0.5), [0, 0, 0, 0, 0, 0, 1, 2]);
        // rounded up to whole individuals, in any order
        let keys = [30, 10, 20];
        assert_eq!(ratio_buckets(&keys, 0.5), [1, 0, 0]);
        assert_eq!(ratio_buckets(&keys, 1.0), [0, 0, 0]);
        assert_eq!(ratio_buckets(&keys, 0.1), [2, 0, 1]);
    }

    #[test]
    fn validation() {
        assert!(Tournament::new(0).is_err());
        assert!(Rank::new(0.9).is_err());
        assert!(Rank::new(2.1).is_err());
        assert!(Truncation::new(0.0).is_err());
        assert!(Truncation::new(1.1).is_err());
    }

    #[test]
    fn tournament_pressure() {
        let scores = [Some(1.0), Some(2.0), Some(3.0), Some(4.0)];
        let size_1 = counts(
            &Tournament::new(1).unwrap(),
            &scores,
            Objective::Maximize,
            40_000,
        );
        assert!(
            size_1.iter().all(|&c| (9_500..10_500).contains(&c)),
            "{size_1:?}"
        );
        let size_3 = counts(
            &Tournament::new(3).unwrap(),
            &scores,
            Objective::Maximize,
            40_000,
        );
        assert!(
            size_3.windows(2).all(|pair| pair[0] < pair[1]),
            "{size_3:?}"
        );
        let minimize = counts(
            &Tournament::new(3).unwrap(),
            &scores,
            Objective::Minimize,
            40_000,
        );
        assert!(
            minimize.windows(2).all(|pair| pair[0] > pair[1]),
            "{minimize:?}"
        );
    }

    #[test]
    fn roulette_is_proportional_to_improvement_over_worst() {
        // weights 0, 1, 3 (improvement over the worst score 10), invalid 0
        let scores = [Some(10.0), Some(11.0), None, Some(13.0)];
        let counts = counts(&Roulette, &scores, Objective::Maximize, 40_000);
        assert_eq!(counts[0], 0);
        assert_eq!(counts[2], 0);
        assert!((9_500..10_500).contains(&counts[1]), "{counts:?}");
        assert!((29_500..30_500).contains(&counts[3]), "{counts:?}");
    }

    #[test]
    fn sus_is_evenly_spread() {
        // expected selections 1.2 and 2.8 out of 4: each count is the floor or the ceiling
        let scores = [Some(0.0), Some(3.0), Some(7.0)];
        for seed in 0..50 {
            let population = population(&scores);
            let mut rng = StreamRng::seed_from_u64(seed);
            let mut counts = [0; 3];
            for index in
                StochasticUniversalSampling.select(&population, Objective::Maximize, 4, &mut rng)
            {
                counts[index] += 1;
            }
            assert_eq!(counts[0], 0);
            assert!(
                (1..=2).contains(&counts[1]) && (2..=3).contains(&counts[2]),
                "{counts:?}"
            );
        }
    }

    #[test]
    fn proportional_falls_back_to_uniform() {
        for select_counts in [
            counts(
                &Roulette,
                &[Some(5.0), Some(5.0), None],
                Objective::Maximize,
                30_000,
            ),
            counts(
                &StochasticUniversalSampling,
                &[None, None, None],
                Objective::Minimize,
                30_000,
            ),
        ] {
            assert!(
                select_counts.iter().all(|&c| (9_500..10_500).contains(&c)),
                "{select_counts:?}"
            );
        }
    }

    #[test]
    fn infinite_scores_select_the_best() {
        let counts = counts(
            &Roulette,
            &[Some(1.0), Some(f64::INFINITY), Some(f64::NEG_INFINITY)],
            Objective::Maximize,
            1_000,
        );
        assert_eq!(counts, vec![0, 1_000, 0]);
    }

    #[test]
    fn proportional_ignores_infeasible_individuals() {
        let mut population = population(&[Some(1.0), Some(2.0), Some(3.0)]);
        // the best score, but infeasible
        population[2].set_fitness(Fitness::constrained(100.0, 1.0));
        let mut rng = StreamRng::seed_from_u64(0);
        let mut counts = [0; 3];
        for index in Roulette.select(&population, Objective::Maximize, 3_000, &mut rng) {
            counts[index] += 1;
        }
        // weights 0 (the worst feasible), 1 and 0 (infeasible)
        assert_eq!(counts, [0, 3_000, 0]);
    }

    #[test]
    fn huge_finite_scores_keep_their_proportions() {
        // weights 2 * MAX and MAX would overflow without scaling
        let scores = [Some(f64::MAX), Some(-f64::MAX), Some(0.0)];
        for select in [
            counts(&Roulette, &scores, Objective::Maximize, 30_000),
            counts(
                &StochasticUniversalSampling,
                &scores,
                Objective::Maximize,
                30_000,
            ),
        ] {
            assert_eq!(select[1], 0);
            assert!((19_500..20_500).contains(&select[0]), "{select:?}");
            assert!((9_500..10_500).contains(&select[2]), "{select:?}");
        }
    }

    #[test]
    fn rank_shares_ranks_on_ties() {
        let scores = [Some(1.0), Some(2.0), Some(2.0), Some(3.0)];
        let counts = counts(
            &Rank::new(2.0).unwrap(),
            &scores,
            Objective::Maximize,
            60_000,
        );
        // pressure 2: weights 0, 1, 1, 2 (ranks 0, 1.5, 1.5, 3 over 3)
        assert_eq!(counts[0], 0);
        assert!(
            (counts[1] as i64 - counts[2] as i64).abs() < 800,
            "{counts:?}"
        );
        assert!((29_000..31_000).contains(&counts[3]), "{counts:?}");
    }

    #[test]
    fn truncation_selects_from_the_top() {
        let scores = [Some(1.0), Some(4.0), Some(2.0), Some(3.0)];
        let counts = counts(
            &Truncation::new(0.5).unwrap(),
            &scores,
            Objective::Maximize,
            20_000,
        );
        assert_eq!((counts[0], counts[2]), (0, 0));
        assert!((9_500..10_500).contains(&counts[1]), "{counts:?}");
    }

    #[test]
    fn truncation_size_is_the_ceiling_without_floating_point_error() {
        // fraction × size one ulp above a whole number, e.g. 0.07 * 100.0 == 7.000000000000001
        for (fraction, n, size) in [
            (0.07, 100, 7),
            (0.14, 50, 7),
            (0.28, 25, 7),
            (0.55, 100, 55),
            (0.17, 100, 17),
            (0.81, 100, 81),
            (0.1, 30, 3),
            (1.0, 7, 7),
        ] {
            assert_eq!(truncation_size(fraction, n), size, "{fraction} of {n}");
        }
        // a fraction that isn't a whole number of individuals still rounds up, to at least 1
        assert_eq!(truncation_size(0.5, 5), 3);
        assert_eq!(truncation_size(0.3, 10), 3);
        assert_eq!(truncation_size(0.31, 10), 4);
        assert_eq!(truncation_size(0.001, 10), 1);
        // no fraction from 0.01 to 1 in steps of 0.01 selects more than its share of 100
        for percent in 1..=100 {
            let fraction = percent as f64 / 100.0;
            assert_eq!(truncation_size(fraction, 100), percent, "{fraction}");
        }
    }

    fn any_scores() -> impl Strategy<Value = Vec<Option<f64>>> {
        prop::collection::vec(
            prop_oneof![Just(None), (-5i32..5).prop_map(|v| Some(v as f64))],
            0..20,
        )
    }

    fn check<S: Select>(
        select: &S,
        scores: &[Option<f64>],
        objective: Objective,
        count: usize,
        seed: u64,
    ) -> std::result::Result<(), TestCaseError> {
        let population = population(scores);
        let selected = select.select(
            &population,
            objective,
            count,
            &mut StreamRng::seed_from_u64(seed),
        );
        prop_assert_eq!(selected.len(), if scores.is_empty() { 0 } else { count });
        prop_assert!(selected.iter().all(|&index| index < scores.len()));
        let again = select.select(
            &population,
            objective,
            count,
            &mut StreamRng::seed_from_u64(seed),
        );
        prop_assert_eq!(&selected, &again, "not deterministic");
        // proportional and truncation selection never pick an invalid individual when a better one exists
        Ok(())
    }

    fn any_fitness() -> impl Strategy<Value = Option<Fitness>> {
        let score = prop_oneof![
            any::<f64>(),
            Just(0.0),
            Just(-0.0),
            Just(f64::INFINITY),
            Just(f64::NEG_INFINITY),
            -3.0..3.0f64,
        ];
        let violation = prop_oneof![Just(0.0), Just(-0.0), Just(f64::INFINITY), 0.0..2.0f64];
        prop_oneof![
            Just(None),
            (score, violation)
                .prop_map(|(score, violation)| { Some(Fitness::constrained(score, violation)) }),
        ]
    }

    proptest! {
        #[test]
        fn sort_keys_are_in_the_order_of_compare(a in any_fitness(), b in any_fitness(), maximize: bool) {
            let objective = if maximize { Objective::Maximize } else { Objective::Minimize };
            let invalid = Fitness::invalid();
            prop_assert_eq!(
                sort_key(a, objective).cmp(&sort_key(b, objective)),
                objective.compare(a.unwrap_or(invalid), b.unwrap_or(invalid))
            );
        }

        #[test]
        fn selections_are_valid_and_deterministic(scores in any_scores(), count in 0usize..30, seed: u64, maximize: bool) {
            let objective = if maximize { Objective::Maximize } else { Objective::Minimize };
            check(&Tournament::new(3).unwrap(), &scores, objective, count, seed)?;
            check(&Roulette, &scores, objective, count, seed)?;
            check(&StochasticUniversalSampling, &scores, objective, count, seed)?;
            check(&Rank::default(), &scores, objective, count, seed)?;
            check(&Truncation::new(0.3).unwrap(), &scores, objective, count, seed)?;
            check(&RandomSelection, &scores, objective, count, seed)?;
        }

        #[test]
        fn proportional_never_selects_worse_than_all_valid_when_improvement_exists(scores in any_scores(), seed: u64) {
            let valid: Vec<f64> = scores.iter().flatten().copied().collect();
            let has_improvement = valid.iter().any(|&v| v != valid[0]);
            prop_assume!(has_improvement);
            let worst = valid.iter().copied().fold(f64::INFINITY, f64::min);
            let population = population(&scores);
            for index in Roulette.select(&population, Objective::Maximize, 50, &mut StreamRng::seed_from_u64(seed)) {
                prop_assert!(scores[index].is_some_and(|score| score > worst));
            }
        }
    }
}
