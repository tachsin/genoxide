//! NSGA-III: non-dominated sorting with reference points, for many objectives.

use super::breed::{Variation, scores_of};
use super::pareto::gains;
use super::{MultiObjectiveAlgorithm, Scores, non_dominated_sort};
use crate::algorithm::{Candidates, Unset};
use crate::genome::Representation;
use crate::operator::{Crossover, Mutate, check_probability};
use crate::rng::Chance;
use crate::{Error, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;

/// NSGA-III (Deb and Jain, 2014): NSGA-II for many objectives, as an ask / tell
/// [`MultiObjectiveAlgorithm`].
///
/// Crowding distance spreads a front poorly beyond 2 or 3 objectives, so NSGA-III spreads it
/// along reference directions instead, usually [Das-Dennis points](super::das_dennis):
///
/// 1. Parents are chosen at random, except that between two parents with constraint violations
///    the smaller violation wins. Pairs are recombined and mutated as in [`Nsga2`](super::Nsga2).
/// 2. Parents and children compete: the next population takes whole fronts, best first. The
///    last front that fits partly is chosen by niching: the objectives are normalized by the
///    ideal point and the intercepts of the hyperplane through the extreme points, each solution
///    joins the reference direction nearest to it, and the directions with the fewest members
///    so far each take one more (the nearest one if the direction is still empty, a random one
///    otherwise).
///
/// Only feasible solutions take part in the niching; infeasible ones fill the rest of the
/// population by constraint violation. The normalization follows pymoo's, and keeps the ideal
/// point, the worst point and the extreme points over the whole run.
///
/// The population size is the number of reference directions by default.
///
/// ```
/// use genoxide::Objective::Minimize;
/// use genoxide::multi::problems::{Dtlz2, TestProblem};
/// use genoxide::multi::{Nsga3, das_dennis};
/// use genoxide::prelude::*;
///
/// let problem = Dtlz2::<3>::default();
/// let nsga3 = Nsga3::builder(problem.real(), [Minimize; 3], das_dennis::<3>(12))
///     .crossover(SimulatedBinaryCrossover::new(30.0)?)
///     .mutate(PolynomialMutation::per_gene(1.0 / 12.0, 20.0)?)
///     .seed(1)
///     .build()?;
/// let outcome = MultiEngine::new(nsga3, problem).stop_when(Stop::generations(100)).run()?;
/// // on the unit sphere
/// for values in outcome.front_values() {
///     let radius: f64 = values.iter().map(|v| v * v).sum::<f64>().sqrt();
///     assert!((radius - 1.0).abs() < 0.05);
/// }
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "R: serde::Serialize, C: serde::Serialize, X: serde::Serialize, R::Genome: serde::Serialize",
        deserialize = "R: serde::Deserialize<'de>, C: serde::Deserialize<'de>, X: serde::Deserialize<'de>, R::Genome: serde::Deserialize<'de>"
    ))
)]
pub struct Nsga3<R: Representation, C, X, const M: usize> {
    variation: Variation<R, C, X>,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays::array"))]
    objectives: [Objective; M],
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays::vec_of_arrays"))]
    reference: Vec<[f64; M]>,
    population_size: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    seed: u64,
    rng: StreamRng,
    population: Population<R::Genome, Scores<M>>,
    offspring: Vec<Individual<R::Genome, Scores<M>>>,
    pending: Vec<usize>,
    front: Vec<Individual<R::Genome, Scores<M>>>,
    discarded: Vec<Individual<R::Genome, Scores<M>>>,
    // the normalization, with the objectives minimized: kept over the whole run
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays::array"))]
    ideal: [f64; M],
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays::array"))]
    worst: [f64; M],
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays::option_matrix"))]
    extremes: Option<[[f64; M]; M]>,
    started: bool,
    asked: bool,
    generation: u64,
    evaluations: u64,
    front_generation: u64,
}

impl<R: Representation, const M: usize> Nsga3<R, Unset, Unset, M> {
    /// A builder for NSGA-III on `representation`, with the direction of each objective and the
    /// reference directions, e.g. [`das_dennis`](super::das_dennis): non-negative, not all 0.
    pub fn builder(
        representation: R,
        objectives: [Objective; M],
        reference_directions: Vec<[f64; M]>,
    ) -> Nsga3Builder<R, M> {
        Nsga3Builder {
            representation,
            objectives,
            reference: reference_directions,
            crossover: Unset,
            mutate: Unset,
            population_size: None,
            crossover_rate: 1.0,
            mutation_rate: 1.0,
            seed: None,
            eliminate_duplicates: true,
            initial_genomes: Vec::new(),
        }
    }
}

// the values with every objective turned into one to minimize
fn minimized<const M: usize>(scores: &Scores<M>, objectives: &[Objective; M]) -> [f64; M] {
    let values = scores.raw();
    std::array::from_fn(|j| match objectives[j] {
        Objective::Minimize => values[j],
        Objective::Maximize => -values[j],
    })
}

fn is_ok<const M: usize>(scores: &Scores<M>) -> bool {
    scores.is_feasible()
}

impl<R, C, X, const M: usize> Nsga3<R, C, X, M>
where
    R: Representation,
    C: Crossover<R>,
    X: Mutate<R>,
{
    /// The representation.
    pub fn representation(&self) -> &R {
        &self.variation.representation
    }

    /// The reference directions.
    pub fn reference_directions(&self) -> &[[f64; M]] {
        &self.reference
    }

    /// The population size.
    pub fn population_size(&self) -> usize {
        self.population_size
    }

    /// The probability that a pair of parents is recombined.
    pub fn crossover_rate(&self) -> f64 {
        self.crossover_rate
    }

    /// The probability that a child is mutated.
    pub fn mutation_rate(&self) -> f64 {
        self.mutation_rate
    }

    /// The ideal point so far: the best value of each objective among the feasible solutions,
    /// `None` before the first survival, or while an objective's best value is infinite.
    pub fn ideal_point(&self) -> Option<[f64; M]> {
        let ideal = self.ideal;
        ideal.iter().all(|v| v.is_finite()).then(|| {
            std::array::from_fn(|j| match self.objectives[j] {
                Objective::Minimize => ideal[j],
                Objective::Maximize => -ideal[j],
            })
        })
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn breed(&mut self) {
        let population = &self.population;
        let violation = |index: usize| {
            let scores = population[index].fitness().unwrap_or(Scores::invalid());
            if scores.is_valid() {
                scores.violation()
            } else {
                f64::INFINITY
            }
        };
        self.variation.breed(
            population,
            self.population_size,
            &mut self.rng,
            |rng| {
                // a random pair: the smaller violation wins, and a coin flip otherwise
                let size = population.len();
                let a = rng.below(size);
                let mut b = rng.below(size - 1);
                if b >= a {
                    b += 1;
                }
                let (va, vb) = (violation(a), violation(b));
                if va < vb {
                    a
                } else if vb < va || rng.below(2) != 0 {
                    b
                } else {
                    a
                }
            },
            &mut self.offspring,
        );
    }

    // the next population from the parents and the offspring
    fn survive(&mut self) {
        let parents = std::mem::take(&mut self.population).into_vec();
        let parent_count = parents.len();
        let mut pool = parents;
        pool.append(&mut self.offspring);
        let scores = scores_of(&pool);
        let size = self.population_size;
        let feasible: Vec<usize> = (0..pool.len()).filter(|&i| is_ok(&scores[i])).collect();
        let mut chosen: Vec<usize> = Vec::with_capacity(size);
        if feasible.len() <= size {
            // every feasible solution, then the least infeasible ones
            chosen.extend(&feasible);
            let mut others: Vec<usize> = (0..pool.len()).filter(|&i| !is_ok(&scores[i])).collect();
            let violation = |i: usize| {
                if scores[i].is_valid() {
                    scores[i].violation()
                } else {
                    f64::INFINITY
                }
            };
            let keys: Vec<u64> = others.iter().map(|_| self.rng.next_u64()).collect();
            let mut order: Vec<usize> = (0..others.len()).collect();
            order.sort_by(|&a, &b| {
                violation(others[a])
                    .total_cmp(&violation(others[b]))
                    .then(keys[a].cmp(&keys[b]))
            });
            others = order.into_iter().map(|position| others[position]).collect();
            chosen.extend(others.into_iter().take(size - chosen.len()));
            if !feasible.is_empty() {
                let points: Vec<[f64; M]> = feasible
                    .iter()
                    .map(|&i| minimized(&scores[i], &self.objectives))
                    .collect();
                let fronts = non_dominated_sort(
                    &feasible.iter().map(|&i| scores[i]).collect::<Vec<_>>(),
                    &self.objectives,
                );
                self.normalize(&points, &fronts[0]);
            }
        } else {
            self.select_feasible(&scores, &feasible, &mut chosen);
        }
        let mut selected = vec![false; pool.len()];
        for &index in &chosen {
            selected[index] = true;
        }
        let mut slots: Vec<Option<Individual<R::Genome, Scores<M>>>> =
            pool.into_iter().map(Some).collect();
        self.discarded.clear();
        for (index, slot) in slots.iter_mut().enumerate() {
            if !selected[index] && index >= parent_count {
                self.discarded.push(slot.take().expect("not taken yet"));
            }
        }
        let mut population = Vec::with_capacity(chosen.len());
        for index in chosen {
            let mut individual = slots[index].take().expect("chosen once");
            if index < parent_count {
                individual.increment_age();
            }
            population.push(individual);
        }
        self.population = Population::new(population);
    }

    // chooses `population_size` of the more numerous feasible solutions: whole fronts, then
    // niching on the last one
    fn select_feasible(
        &mut self,
        scores: &[Scores<M>],
        feasible: &[usize],
        chosen: &mut Vec<usize>,
    ) {
        let size = self.population_size;
        let feasible_scores: Vec<Scores<M>> = feasible.iter().map(|&i| scores[i]).collect();
        let fronts = non_dominated_sort(&feasible_scores, &self.objectives);
        let points: Vec<[f64; M]> = feasible_scores
            .iter()
            .map(|s| minimized(s, &self.objectives))
            .collect();
        // the fronts that fill at least the population, by position in `feasible`
        let mut members: Vec<usize> = Vec::new();
        let mut last = Vec::new();
        for front in &fronts {
            if members.len() + front.len() > size {
                last = front.clone();
                break;
            }
            members.extend(front);
            if members.len() == size {
                break;
            }
        }
        // the normalization uses every feasible solution and the first front
        self.normalize(&points, &fronts[0]);
        if !last.is_empty() {
            let (ideal, nadir) = (self.ideal, self.nadir(&points, &fronts[0]));
            let associate = |position: usize| self.associate(&points[position], &ideal, &nadir);
            let mut counts = vec![0usize; self.reference.len()];
            for &position in &members {
                counts[associate(position).0] += 1;
            }
            let candidates: Vec<(usize, usize, f64)> = last
                .iter()
                .map(|&position| {
                    let (niche, distance) = associate(position);
                    (position, niche, distance)
                })
                .collect();
            let picked = self.niching(&candidates, &mut counts, size - members.len());
            members.extend(picked);
        }
        chosen.extend(members.into_iter().map(|position| feasible[position]));
    }

    // updates the ideal point, the worst point and the extreme points
    fn normalize(&mut self, points: &[[f64; M]], first_front: &[usize]) {
        for point in points {
            for ((ideal, worst), &value) in self.ideal.iter_mut().zip(&mut self.worst).zip(point) {
                *ideal = ideal.min(value);
                *worst = worst.max(value);
            }
        }
        // the extreme point of each axis minimizes the achievement scalarizing function with that
        // axis's weight 1 and the others' 1e6, among the first front and the previous extremes
        let mut candidates: Vec<[f64; M]> = self.extremes.map(|e| e.to_vec()).unwrap_or_default();
        candidates.extend(first_front.iter().map(|&position| points[position]));
        let ideal = self.ideal;
        let asf = |point: &[f64; M], axis: usize| {
            (0..M)
                .map(|j| {
                    let mut value = point[j] - ideal[j];
                    if value < 1e-3 {
                        value = 0.0;
                    }
                    value * if j == axis { 1.0 } else { 1e6 }
                })
                .fold(f64::NEG_INFINITY, f64::max)
        };
        self.extremes = Some(std::array::from_fn(|axis| {
            let mut best = 0;
            for (index, candidate) in candidates.iter().enumerate() {
                if asf(candidate, axis) < asf(&candidates[best], axis) {
                    best = index;
                }
            }
            candidates[best]
        }));
    }

    // the nadir point: the intercepts of the hyperplane through the extreme points, with pymoo's
    // fallbacks to the worst point of the first front and of the population
    fn nadir(&self, points: &[[f64; M]], first_front: &[usize]) -> [f64; M] {
        let ideal = self.ideal;
        let worst_of = |positions: &mut dyn Iterator<Item = usize>| {
            let mut worst = [f64::NEG_INFINITY; M];
            for position in positions {
                for j in 0..M {
                    worst[j] = worst[j].max(points[position][j]);
                }
            }
            worst
        };
        let worst_of_front = worst_of(&mut first_front.iter().copied());
        let worst_of_population = worst_of(&mut (0..points.len()));
        let extremes = self.extremes.expect("normalized before");
        let shifted: Vec<Vec<f64>> = extremes
            .iter()
            .map(|e| (0..M).map(|j| e[j] - ideal[j]).collect())
            .collect();
        let mut nadir = match solve(&shifted) {
            Some(plane) if plane.iter().all(|&b| 1.0 / b > 1e-6) => {
                // no further than the worst point
                std::array::from_fn(|j| (ideal[j] + 1.0 / plane[j]).min(self.worst[j]))
            }
            _ => worst_of_front,
        };
        for j in 0..M {
            if nadir[j] - ideal[j] <= 1e-6 {
                nadir[j] = worst_of_population[j];
            }
        }
        nadir
    }

    // the reference direction nearest to a point in normalized objective space, and the
    // perpendicular distance to it; the first one on ties
    fn associate(&self, point: &[f64; M], ideal: &[f64; M], nadir: &[f64; M]) -> (usize, f64) {
        let normalized: [f64; M] = std::array::from_fn(|j| {
            let mut range = nadir[j] - ideal[j];
            if range == 0.0 {
                range = 1e-12;
            }
            (point[j] - ideal[j]) / range
        });
        let mut best = (0, f64::INFINITY);
        for (index, direction) in self.reference.iter().enumerate() {
            let length: f64 = direction.iter().map(|d| d * d).sum();
            let projection: f64 = normalized
                .iter()
                .zip(direction)
                .map(|(n, d)| n * d)
                .sum::<f64>()
                / length;
            let distance = normalized
                .iter()
                .zip(direction)
                .map(|(n, d)| (n - projection * d) * (n - projection * d))
                .sum::<f64>()
                .sqrt();
            if distance < best.1 {
                best = (index, distance);
            }
        }
        best
    }

    // chooses `remaining` of the candidates (position, niche, distance): the niches with the
    // fewest members first, in random order
    fn niching(
        &mut self,
        candidates: &[(usize, usize, f64)],
        counts: &mut [usize],
        remaining: usize,
    ) -> Vec<usize> {
        let mut picked = Vec::with_capacity(remaining);
        let mut available = vec![true; candidates.len()];
        while picked.len() < remaining {
            // the niches that still have candidates, with the smallest count
            let mut niches: Vec<usize> = candidates
                .iter()
                .zip(&available)
                .filter(|(_, available)| **available)
                .map(|(candidate, _)| candidate.1)
                .collect();
            niches.sort_unstable();
            niches.dedup();
            let fewest = niches
                .iter()
                .map(|&n| counts[n])
                .min()
                .expect("candidates left");
            niches.retain(|&n| counts[n] == fewest);
            shuffle(&mut niches, &mut self.rng);
            niches.truncate(remaining - picked.len());
            for niche in niches {
                let mut members: Vec<usize> = (0..candidates.len())
                    .filter(|&c| available[c] && candidates[c].1 == niche)
                    .collect();
                shuffle(&mut members, &mut self.rng);
                let member = if counts[niche] == 0 {
                    // the nearest, the first after shuffling on ties
                    let mut nearest = members[0];
                    for &c in &members {
                        if candidates[c].2 < candidates[nearest].2 {
                            nearest = c;
                        }
                    }
                    nearest
                } else {
                    members[0]
                };
                available[member] = false;
                picked.push(candidates[member].0);
                counts[niche] += 1;
            }
        }
        picked
    }

    // the new front, and whether it improved on the previous one
    fn update_front(&mut self) {
        let scores = scores_of(self.population.as_slice());
        let fronts = non_dominated_sort(&scores, &self.objectives);
        let front: Vec<Individual<R::Genome, Scores<M>>> = fronts
            .first()
            .map(|first| first.iter().map(|&i| self.population[i].clone()).collect())
            .unwrap_or_default();
        if gains(
            &scores_of(&front),
            &scores_of(&self.front),
            &self.objectives,
        ) {
            self.front_generation = self.generation;
        }
        self.front = front;
    }
}

// a Fisher-Yates shuffle
fn shuffle<T>(items: &mut [T], rng: &mut StreamRng) {
    for i in (1..items.len()).rev() {
        items.swap(i, rng.below(i + 1));
    }
}

// the solution x of A x = 1 by Gaussian elimination with partial pivoting, or `None` if A is
// singular or the solution isn't accurate
fn solve(matrix: &[Vec<f64>]) -> Option<Vec<f64>> {
    let n = matrix.len();
    let mut a: Vec<Vec<f64>> = matrix
        .iter()
        .map(|row| {
            let mut row = row.clone();
            row.push(1.0);
            row
        })
        .collect();
    for column in 0..n {
        let pivot =
            (column..n).max_by(|&i, &j| a[i][column].abs().total_cmp(&a[j][column].abs()))?;
        if a[pivot][column] == 0.0 || !a[pivot][column].is_finite() {
            return None;
        }
        a.swap(column, pivot);
        let (upper, lower) = a.split_at_mut(column + 1);
        let pivot_row = &upper[column];
        for row in lower {
            let factor = row[column] / pivot_row[column];
            for (value, pivot_value) in row[column..].iter_mut().zip(&pivot_row[column..]) {
                *value -= factor * pivot_value;
            }
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let sum: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (a[row][n] - sum) / a[row][row];
    }
    // like numpy's allclose: every residual within 1e-8 + 1e-5
    let accurate = matrix.iter().all(|row| {
        let value: f64 = row.iter().zip(&x).map(|(a, x)| a * x).sum();
        (value - 1.0).abs() <= 1e-8 + 1e-5
    });
    (accurate && x.iter().all(|v| v.is_finite())).then_some(x)
}

impl<R, C, X, const M: usize> MultiObjectiveAlgorithm<M> for Nsga3<R, C, X, M>
where
    R: Representation,
    C: Crossover<R>,
    X: Mutate<R>,
{
    type Genome = R::Genome;

    fn objectives(&self) -> [Objective; M] {
        self.objectives
    }

    fn ask(&mut self) -> Candidates<'_, R::Genome, Scores<M>> {
        if !self.asked {
            self.pending.clear();
            if self.started {
                self.breed();
                self.pending.extend(
                    (0..self.offspring.len()).filter(|&i| !self.offspring[i].is_evaluated()),
                );
            } else {
                self.pending.extend(0..self.population.len());
            }
            self.asked = true;
        }
        let individuals = if self.started {
            &self.offspring
        } else {
            self.population.as_slice()
        };
        Candidates::new(individuals, &self.pending)
    }

    fn tell(&mut self, scores: &[Scores<M>]) -> Result<()> {
        if !self.asked {
            return Err(Error::TellWithoutAsk);
        }
        if scores.len() != self.pending.len() {
            return Err(Error::FitnessCount {
                expected: self.pending.len(),
                got: scores.len(),
            });
        }
        self.asked = false;
        self.evaluations += scores.len() as u64;
        if self.started {
            for (&index, &score) in self.pending.iter().zip(scores) {
                self.offspring[index].set_fitness(score);
            }
            self.generation += 1;
            self.survive();
        } else {
            for (individual, &score) in self.population.iter_mut().zip(scores) {
                individual.set_fitness(score);
            }
        }
        self.update_front();
        self.started = true;
        Ok(())
    }

    fn population(&self) -> &Population<R::Genome, Scores<M>> {
        &self.population
    }

    fn front(&self) -> &[Individual<R::Genome, Scores<M>>] {
        &self.front
    }

    fn discarded(&self) -> &[Individual<R::Genome, Scores<M>>] {
        &self.discarded
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn evaluations(&self) -> u64 {
        self.evaluations
    }

    fn front_generation(&self) -> u64 {
        self.front_generation
    }
}

/// A builder for [`Nsga3`], from [`Nsga3::builder`].
///
/// The crossover and the mutation are required. Defaults: a population size equal to the number
/// of reference directions, `crossover_rate` 1.0 and `mutation_rate` 1.0 (as in Deb and Jain,
/// and pymoo), a random initial population and a random seed.
#[derive(Clone, Debug)]
pub struct Nsga3Builder<R: Representation, const M: usize, C = Unset, X = Unset> {
    representation: R,
    objectives: [Objective; M],
    reference: Vec<[f64; M]>,
    crossover: C,
    mutate: X,
    population_size: Option<usize>,
    crossover_rate: f64,
    mutation_rate: f64,
    seed: Option<u64>,
    eliminate_duplicates: bool,
    initial_genomes: Vec<R::Genome>,
}

impl<R: Representation, const M: usize, C, X> Nsga3Builder<R, M, C, X> {
    /// The crossover operator. Required; SBX with η 30 is the usual choice for real genomes.
    pub fn crossover<T>(self, crossover: T) -> Nsga3Builder<R, M, T, X> {
        Nsga3Builder {
            representation: self.representation,
            objectives: self.objectives,
            reference: self.reference,
            crossover,
            mutate: self.mutate,
            population_size: self.population_size,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            seed: self.seed,
            eliminate_duplicates: self.eliminate_duplicates,
            initial_genomes: self.initial_genomes,
        }
    }

    /// The mutation operator. Required; polynomial mutation with η 20 and a rate of 1 / the
    /// number of genes is the usual choice for real genomes.
    pub fn mutate<T>(self, mutate: T) -> Nsga3Builder<R, M, C, T> {
        Nsga3Builder {
            representation: self.representation,
            objectives: self.objectives,
            reference: self.reference,
            crossover: self.crossover,
            mutate,
            population_size: self.population_size,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            seed: self.seed,
            eliminate_duplicates: self.eliminate_duplicates,
            initial_genomes: self.initial_genomes,
        }
    }

    /// The population size, at least 2; also the number of children per generation. The number
    /// of reference directions by default; smaller populations can't fill every direction.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// The probability that a pair of parents is recombined, between 0 and 1. 1 by default.
    pub fn crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// The probability that a child is mutated, between 0 and 1. 1 by default.
    pub fn mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// The seed of the random numbers, for a reproducible run. Random by default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Whether a child that equals a member of the population, or an earlier child of the same
    /// generation, is dropped and another bred instead, which keeps the population and its front
    /// free of copies. On by default, as in pymoo. When copies are all a population can breed,
    /// after 100 dropped children per child needed, copies are accepted.
    pub fn eliminate_duplicates(mut self, eliminate: bool) -> Self {
        self.eliminate_duplicates = eliminate;
        self
    }

    /// Genomes for the initial population, at most the population size, each valid for the
    /// representation. The rest is random.
    pub fn initial_genomes<I: IntoIterator<Item = R::Genome>>(mut self, genomes: I) -> Self {
        self.initial_genomes = genomes.into_iter().collect();
        self
    }

    /// Validates the settings and creates the algorithm, with its initial population.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidSetting`] for no objectives, no reference directions or one with a
    ///   negative, non-finite or all-zero component, a population size below 2, rates out of
    ///   range or both 0, or more initial genomes than the population size.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<Nsga3<R, C, X, M>>
    where
        C: Crossover<R>,
        X: Mutate<R>,
    {
        let invalid = |setting, reason: String| Err(Error::InvalidSetting { setting, reason });
        if M == 0 {
            return invalid("objectives", "at least 1 objective is needed".to_string());
        }
        if self.reference.is_empty() {
            return invalid(
                "reference_directions",
                "at least 1 reference direction is needed".to_string(),
            );
        }
        for direction in &self.reference {
            let valid = direction.iter().all(|v| *v >= 0.0 && v.is_finite())
                && direction.iter().any(|v| *v > 0.0);
            if !valid {
                return invalid(
                    "reference_directions",
                    format!("must be non-negative, finite and not all 0, got {direction:?}"),
                );
            }
        }
        let size = self.population_size.unwrap_or(self.reference.len());
        if size < 2 {
            return invalid(
                "population_size",
                format!("NSGA-III needs at least 2 individuals, got {size}"),
            );
        }
        let crossover_rate = check_probability("crossover_rate", self.crossover_rate)?;
        let mutation_rate = check_probability("mutation_rate", self.mutation_rate)?;
        if crossover_rate == 0.0 && mutation_rate == 0.0 {
            return invalid(
                "mutation_rate",
                "crossover_rate and mutation_rate are both 0, so every child would be a copy of a parent".to_string(),
            );
        }
        if self.initial_genomes.len() > size {
            return invalid(
                "initial_genomes",
                format!(
                    "at most the population size {size}, got {}",
                    self.initial_genomes.len()
                ),
            );
        }
        for genome in &self.initial_genomes {
            self.representation.validate(genome)?;
        }
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let mut rng = StreamRng::seed_from_u64(seed);
        let random = size - self.initial_genomes.len();
        let mut genomes = self.initial_genomes;
        genomes.extend((0..random).map(|_| self.representation.random_genome(&mut rng)));
        Ok(Nsga3 {
            variation: Variation {
                representation: self.representation,
                crossover: self.crossover,
                mutate: self.mutate,
                crossover_chance: Chance::new(crossover_rate),
                mutation_chance: Chance::new(mutation_rate),
                eliminate_duplicates: self.eliminate_duplicates,
            },
            objectives: self.objectives,
            reference: self.reference,
            population_size: size,
            crossover_rate,
            mutation_rate,
            seed,
            rng,
            population: genomes.into_iter().map(Individual::unevaluated).collect(),
            offspring: Vec::new(),
            pending: Vec::new(),
            front: Vec::new(),
            discarded: Vec::new(),
            ideal: [f64::INFINITY; M],
            worst: [f64::NEG_INFINITY; M],
            extremes: None,
            started: false,
            asked: false,
            generation: 0,
            evaluations: 0,
            front_generation: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective::{Maximize, Minimize};
    use crate::genome::{Real, Reals};
    use crate::multi::{das_dennis, dominates};
    use crate::operator::{PolynomialMutation, SimulatedBinaryCrossover};
    use proptest::prelude::*;

    fn builder<const M: usize>(
        objectives: [Objective; M],
        divisions: usize,
        seed: u64,
    ) -> Nsga3Builder<Real, M, SimulatedBinaryCrossover, PolynomialMutation> {
        Nsga3::builder(
            Real::uniform(M + 2, 0.0..=1.0).unwrap(),
            objectives,
            das_dennis::<M>(divisions),
        )
        .crossover(SimulatedBinaryCrossover::new(30.0).unwrap())
        .mutate(PolynomialMutation::per_gene(0.2, 20.0).unwrap())
        .seed(seed)
    }

    fn setting<T: std::fmt::Debug>(result: Result<T>) -> &'static str {
        match result {
            Err(Error::InvalidSetting { setting, .. } | Error::MissingSetting { setting }) => {
                setting
            }
            other => panic!("expected a setting error, got {other:?}"),
        }
    }

    #[test]
    fn validation() {
        let with = |directions: Vec<[f64; 2]>| {
            Nsga3::builder(
                Real::uniform(2, 0.0..=1.0).unwrap(),
                [Minimize; 2],
                directions,
            )
            .crossover(SimulatedBinaryCrossover::new(30.0).unwrap())
            .mutate(PolynomialMutation::per_gene(0.5, 20.0).unwrap())
            .build()
        };
        assert_eq!(setting(with(Vec::new())), "reference_directions");
        assert_eq!(setting(with(vec![[0.0, 0.0]])), "reference_directions");
        assert_eq!(setting(with(vec![[-1.0, 2.0]])), "reference_directions");
        assert_eq!(setting(with(vec![[f64::NAN, 1.0]])), "reference_directions");
        // one direction: a population of 1 is too small
        assert_eq!(setting(with(vec![[1.0, 1.0]])), "population_size");
        let nsga3 = with(das_dennis::<2>(9)).unwrap();
        assert_eq!(nsga3.population_size(), 10);
        assert_eq!(nsga3.reference_directions().len(), 10);
        assert_eq!(
            setting(builder([Minimize; 2], 4, 0).crossover_rate(2.0).build()),
            "crossover_rate"
        );
        assert_eq!(nsga3.ideal_point(), None);
    }

    #[test]
    fn linear_systems() {
        let x = solve(&[vec![2.0, 0.0], vec![0.0, 4.0]]).unwrap();
        assert_eq!(x, [0.5, 0.25]);
        let x = solve(&[vec![0.0, 1.0], vec![1.0, 1.0]]).unwrap();
        assert_eq!(x, [0.0, 1.0]);
        assert_eq!(solve(&[vec![1.0, 2.0], vec![2.0, 4.0]]), None);
    }

    #[test]
    fn association_to_reference_directions() {
        let nsga3 = builder([Minimize; 2], 2, 0).build().unwrap();
        // the directions (0, 1), (0.5, 0.5), (1, 0)
        let (ideal, nadir) = ([0.0, 0.0], [2.0, 2.0]);
        assert_eq!(nsga3.associate(&[1.0, 1.0], &ideal, &nadir), (1, 0.0));
        assert_eq!(nsga3.associate(&[0.0, 2.0], &ideal, &nadir), (0, 0.0));
        let (niche, distance) = nsga3.associate(&[2.0, 0.2], &ideal, &nadir);
        assert_eq!(niche, 2);
        assert!((distance - 0.1).abs() < 1e-12);
    }

    #[test]
    fn ask_tell_protocol() {
        let mut nsga3 = builder([Minimize; 2], 7, 0).build().unwrap();
        assert_eq!(nsga3.tell(&[]), Err(Error::TellWithoutAsk));
        assert_eq!(nsga3.ask().len(), 8);
        let f = |x: &Reals| Scores::new([x[0], 1.0 - x[0] + x[1]]);
        let mut asked = 0;
        for generation in 0..5 {
            let told: Vec<Scores<2>> = nsga3.ask().iter().map(f).collect();
            asked += told.len() as u64;
            nsga3.tell(&told).unwrap();
            assert_eq!(nsga3.generation(), generation);
            assert_eq!(nsga3.population().len(), 8);
        }
        // children that are copies of a parent inherit its scores
        assert_eq!(nsga3.evaluations(), asked);
        assert!(asked <= 40);
        assert!(nsga3.ideal_point().is_some());
        let surviving_parents = nsga3.population().iter().filter(|x| x.age() > 0).count();
        assert_eq!(nsga3.discarded().len(), surviving_parents);
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut nsga3 = builder([Minimize, Maximize, Minimize], 4, seed)
                .build()
                .unwrap();
            let f = |x: &Reals| Scores::new([x[0], x[1], x[2] + x[3]]);
            for _ in 0..10 {
                let told: Vec<Scores<3>> = nsga3.ask().iter().map(f).collect();
                nsga3.tell(&told).unwrap();
            }
            nsga3.population().clone()
        };
        assert_eq!(run(4), run(4));
        assert_ne!(run(4), run(5));
    }

    proptest! {
        #[test]
        fn the_new_front_is_never_dominated_by_the_old_population(
            seed: u64,
            divisions in 1usize..6,
            maximize: bool,
        ) {
            let objectives = if maximize { [Maximize, Minimize, Minimize] } else { [Minimize; 3] };
            let mut nsga3 = builder(objectives, divisions, seed).population_size(divisions + 4).build().unwrap();
            let mut rng = StreamRng::seed_from_u64(seed);
            // random scores, some infeasible or invalid
            let mut random = |_: &Reals| match rng.below(10) {
                0 => Scores::invalid(),
                1 => Scores::constrained([0.0; 3], rng.below(3) as f64 + 1.0),
                _ => Scores::new([rng.below(4) as f64, rng.below(4) as f64, rng.below(4) as f64]),
            };
            let told: Vec<Scores<3>> = nsga3.ask().iter().map(&mut random).collect();
            nsga3.tell(&told).unwrap();
            for _ in 0..8 {
                let old: Vec<Scores<3>> =
                    nsga3.population().iter().map(|x| x.fitness().unwrap()).collect();
                let told: Vec<Scores<3>> = nsga3.ask().iter().map(&mut random).collect();
                nsga3.tell(&told).unwrap();
                prop_assert_eq!(nsga3.population().len(), divisions + 4);
                for member in nsga3.front() {
                    let scores = member.fitness().unwrap();
                    prop_assert!(!old.iter().any(|o| dominates(o, &scores, &objectives)));
                }
            }
        }
    }
}
