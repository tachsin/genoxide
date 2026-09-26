//! Parent selection.

use super::{MAX_SIZE, Select, check_rate};
use crate::genome::Genome;
use crate::{Error, Fitness, Objective, Population, Result, StreamRng};
use std::cmp::Ordering;

fn fitness_of<G: Genome>(population: &Population<G>, index: usize) -> Fitness {
    population[index].fitness().unwrap_or(Fitness::invalid())
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
        if population.is_empty() {
            return Vec::new();
        }
        (0..count)
            .map(|_| {
                let mut winner = rng.below(population.len());
                for _ in 1..self.size {
                    let contender = rng.below(population.len());
                    if objective.is_better(
                        fitness_of(population, contender),
                        fitness_of(population, winner),
                    ) {
                        winner = contender;
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
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            objective.compare(fitness_of(population, a), fitness_of(population, b))
        });
        // average rank (0 = worst) for ties
        let mut ranks = vec![0.0; n];
        let mut start = 0;
        while start < n {
            let fitness = fitness_of(population, order[start]);
            let end = (start..n)
                .find(|&i| {
                    objective.compare(fitness_of(population, order[i]), fitness) != Ordering::Equal
                })
                .unwrap_or(n);
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

/// Truncation selection: uniformly random from the best `fraction` of the population.
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
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            objective.compare(fitness_of(population, b), fitness_of(population, a))
        });
        let top = ((self.fraction * n as f64).ceil() as usize).clamp(1, n);
        (0..count).map(|_| order[rng.below(top)]).collect()
    }
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
            for pair in picks.chunks_exact(2) {
                self_pairs += usize::from(pair[0] == pair[1]);
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

    proptest! {
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
