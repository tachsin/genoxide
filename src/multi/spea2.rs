//! SPEA2: the strength Pareto evolutionary algorithm 2.

use super::breed::{Spares, Variation, distinct_into, scores_of};
use super::pareto::{dominance, gains};
use super::{MultiObjectiveAlgorithm, Scores, non_dominated_sort};
use crate::algorithm::{Candidates, Unset};
use crate::genome::Representation;
use crate::operator::{Crossover, Mutate, check_rates, check_size};
use crate::rng::Chance;
use crate::{Error, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;
use std::cmp::Ordering;
use std::fmt;

/// SPEA2 (Zitzler, Laumanns and Thiele, 2001): a multi-objective genetic algorithm with an
/// archive of the best solutions, as an ask / tell [`MultiObjectiveAlgorithm`].
///
/// The population is the archive. Every generation:
///
/// 1. Parents are chosen from the archive by binary tournament on the SPEA2 fitness (smaller is
///    better; the smaller constraint violation wins first), and recombined and mutated as in
///    [`Nsga2`](super::Nsga2): as many children as the archive size.
/// 2. Archive and children compete. Each solution's fitness is its raw fitness, the summed
///    strength (number of dominated solutions) of the solutions that dominate it, plus a density
///    `1 / (σₖ + 2)` from the distance `σₖ` to its k-th nearest neighbor, `k = √(pool size)`.
/// 3. The next archive takes every non-dominated solution. If they're too few, the best of the
///    others by fitness fill it; if they're too many, the solution with the smallest distance to
///    its nearest neighbor (then second nearest, and so on) is removed until they fit, which
///    keeps the extremes and spreads the front evenly. Truncation costs O(N²) per removal in the
///    worst case, so a generation that removes N solutions costs O(N³): fine for populations of
///    a few hundred.
///
/// Dominance is constrained dominance ([`dominates`](super::dominates)), and distances are
/// measured with each objective scaled to its range among the valid solutions, so objectives with
/// different scales count equally.
///
/// Built with [`Spea2::builder`], run with a [`MultiEngine`](super::MultiEngine).
///
/// ```
/// use genoxide::Objective::Minimize;
/// use genoxide::multi::Spea2;
/// use genoxide::multi::problems::{MultiProblem, Zdt1};
/// use genoxide::prelude::*;
///
/// let problem = Zdt1::new(30);
/// let spea2 = Spea2::builder(problem.representation(), [Minimize; 2])
///     .population_size(100)
///     .crossover(SimulatedBinaryCrossover::new(15.0)?)
///     .mutate(PolynomialMutation::per_gene(1.0 / 30.0, 20.0)?)
///     .seed(1)
///     .build()?;
/// let outcome = MultiEngine::new(spea2, problem).stop_when(Stop::generations(150)).run()?;
/// assert_eq!(outcome.front().len(), 100);
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
pub struct Spea2<R: Representation, C, X, const M: usize> {
    variation: Variation<R, C, X>,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays::array"))]
    objectives: [Objective; M],
    population_size: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    seed: u64,
    rng: StreamRng,
    population: Population<R::Genome, Scores<M>>,
    // the SPEA2 fitness of each member of the archive
    fitness: Vec<f64>,
    offspring: Vec<Individual<R::Genome, Scores<M>>>,
    pending: Vec<usize>,
    front: Vec<Individual<R::Genome, Scores<M>>>,
    discarded: Vec<Individual<R::Genome, Scores<M>>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    spares: Spares<R::Genome>,
    #[cfg_attr(feature = "serde", serde(skip))]
    scratch: Scratch,
    started: bool,
    asked: bool,
    generation: u64,
    evaluations: u64,
    front_generation: u64,
}

impl<R: Representation, const M: usize> Spea2<R, Unset, Unset, M> {
    /// A builder for SPEA2 on `representation`, with the direction of each objective.
    pub fn builder(representation: R, objectives: [Objective; M]) -> Spea2Builder<R, M> {
        Spea2Builder {
            representation,
            objectives,
            crossover: Unset,
            mutate: Unset,
            population_size: None,
            crossover_rate: 0.9,
            mutation_rate: 1.0,
            seed: None,
            eliminate_duplicates: true,
            initial_genomes: Vec::new(),
        }
    }
}

// Space for the SPEA2 fitness and the truncation, reused from generation to generation. A clone
// or a checkpoint starts without it.
#[derive(Default)]
struct Scratch {
    strength: Vec<usize>,
    // every (dominating, dominated) pair
    dominations: Vec<(usize, usize)>,
    raw: Vec<usize>,
    // the SPEA2 fitness of each solution
    fitness: Vec<f64>,
    // the distance from `a` to `b` of `n` solutions at `a * n + b`
    distances: Vec<f64>,
    nearest: Vec<f64>,
    // for the truncation: the sorted distances of each survivor, a row each, and the row of each
    sorted: Vec<f64>,
    rows: Vec<usize>,
}

impl Clone for Scratch {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl fmt::Debug for Scratch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Scratch")
    }
}

// the violation of a solution, infinite if it's invalid
fn violation<const M: usize>(scores: &Scores<M>) -> f64 {
    if scores.is_valid() {
        scores.violation()
    } else {
        f64::INFINITY
    }
}

// the SPEA2 fitness of every solution, and the distances between them (infinite to itself and
// between invalid solutions), with each objective scaled to its range among the valid ones, into
// `scratch.fitness` and `scratch.distances`
fn strength_fitness<const M: usize>(
    scores: &[Scores<M>],
    objectives: &[Objective; M],
    scratch: &mut Scratch,
) {
    let Scratch {
        strength,
        dominations,
        raw,
        fitness,
        distances,
        nearest,
        ..
    } = scratch;
    let n = scores.len();
    strength.clear();
    strength.resize(n, 0);
    dominations.clear();
    for a in 0..n {
        for b in a + 1..n {
            match dominance(&scores[a], &scores[b], objectives) {
                (true, _) => {
                    strength[a] += 1;
                    dominations.push((a, b));
                }
                (_, true) => {
                    strength[b] += 1;
                    dominations.push((b, a));
                }
                _ => {}
            }
        }
    }
    // the raw fitness: the sum of the strengths of the dominating solutions
    raw.clear();
    raw.resize(n, 0);
    for &(dominating, dominated) in dominations.iter() {
        raw[dominated] += strength[dominating];
    }
    let (mut low, mut high) = ([f64::INFINITY; M], [f64::NEG_INFINITY; M]);
    // over the finite values: one infinite value doesn't stop an objective being scaled
    for score in scores.iter().filter(|s| s.is_valid()) {
        for j in 0..M {
            let value = score.raw()[j];
            if value.is_finite() {
                low[j] = low[j].min(value);
                high[j] = high[j].max(value);
            }
        }
    }
    let scale: [f64; M] = std::array::from_fn(|j| {
        let range = high[j] - low[j];
        if range > 0.0 && range.is_finite() {
            range
        } else {
            1.0
        }
    });
    distances.clear();
    distances.resize(n * n, f64::INFINITY);
    for a in 0..n {
        for b in a + 1..n {
            if scores[a].is_valid() && scores[b].is_valid() {
                let distance = (0..M)
                    .map(|j| {
                        let (x, y) = (scores[a].raw()[j], scores[b].raw()[j]);
                        // equal infinities are 0 apart: `inf - inf` would be a NaN, whose sign
                        // (and so its order) differs between processors
                        if x == y {
                            return 0.0;
                        }
                        let d = (x - y) / scale[j];
                        d * d
                    })
                    .sum::<f64>()
                    .sqrt();
                distances[a * n + b] = distance;
                distances[b * n + a] = distance;
            }
        }
    }
    let k = n.isqrt().clamp(1, n.saturating_sub(1).max(1));
    fitness.clear();
    fitness.extend((0..n).map(|i| {
        // the k-th smallest distance: the value a sort would put there, as the order is total
        nearest.clear();
        nearest.extend_from_slice(&distances[i * n..(i + 1) * n]);
        let sigma = if k - 1 < nearest.len() {
            *nearest.select_nth_unstable_by(k - 1, f64::total_cmp).1
        } else {
            f64::INFINITY
        };
        let sigma = if sigma.is_nan() { f64::INFINITY } else { sigma };
        raw[i] as f64 + 1.0 / (sigma + 2.0)
    }));
}

// removes the most crowded survivor until `size` remain: the one whose sorted distances to the
// other survivors are lexicographically smallest (the nearest neighbor first, then the second
// nearest, ...), the first on ties; the sorted distances are kept up to date between removals.
// `distances` are those of `n` solutions, `a * n + b`; `sorted` and `rows` are scratch space.
fn truncate(
    survivors: &mut Vec<usize>,
    distances: &[f64],
    n: usize,
    size: usize,
    sorted: &mut Vec<f64>,
    rows: &mut Vec<usize>,
) {
    // a row of `stride` distances per survivor, of which the first `len` are in use
    let stride = survivors.len().saturating_sub(1);
    sorted.clear();
    for &a in survivors.iter() {
        let start = sorted.len();
        sorted.extend(
            survivors
                .iter()
                .filter(|&&b| b != a)
                .map(|&b| distances[a * n + b]),
        );
        // equal under total_cmp means the same bits: an unstable sort gives the same row
        sorted[start..].sort_unstable_by(f64::total_cmp);
    }
    rows.clear();
    rows.extend(0..survivors.len());
    while survivors.len() > size {
        let len = survivors.len() - 1;
        let row = |position: usize| &sorted[rows[position] * stride..][..len];
        let mut most_crowded = 0;
        for position in 1..survivors.len() {
            let ordering = row(position)
                .iter()
                .zip(row(most_crowded))
                .map(|(a, b)| a.total_cmp(b))
                .find(|ordering| ordering.is_ne())
                .unwrap_or(Ordering::Equal);
            if ordering == Ordering::Less {
                most_crowded = position;
            }
        }
        let removed = survivors.remove(most_crowded);
        rows.remove(most_crowded);
        for (&row, &survivor) in rows.iter().zip(survivors.iter()) {
            let distance = distances[survivor * n + removed];
            let row = &mut sorted[row * stride..][..len];
            let position = row.partition_point(|d| d.total_cmp(&distance) == Ordering::Less);
            row.copy_within(position + 1.., position);
        }
    }
}

impl<R, C, X, const M: usize> Spea2<R, C, X, M>
where
    R: Representation,
    C: Crossover<R>,
    X: Mutate<R>,
{
    /// The representation.
    pub fn representation(&self) -> &R {
        &self.variation.representation
    }

    /// The archive size.
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

    /// The SPEA2 fitness of each member of the archive, in its order: below 1 for the
    /// non-dominated ones. Empty before the first [`tell`](MultiObjectiveAlgorithm::tell).
    pub fn strength_fitness(&self) -> &[f64] {
        &self.fitness
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn breed(&mut self) {
        let population = &self.population;
        let fitness = &self.fitness;
        self.variation.breed(
            population,
            self.population_size,
            &mut self.rng,
            |rng| {
                let size = population.len();
                let a = rng.below(size);
                let mut b = rng.below(size - 1);
                if b >= a {
                    b += 1;
                }
                let score = |i: usize| population[i].fitness().unwrap_or(Scores::invalid());
                let (va, vb) = (violation(&score(a)), violation(&score(b)));
                let (ka, kb) = if va > 0.0 || vb > 0.0 {
                    (va, vb)
                } else {
                    (fitness[a], fitness[b])
                };
                if ka < kb {
                    a
                } else if kb < ka || rng.below(2) != 0 {
                    b
                } else {
                    a
                }
            },
            &mut self.offspring,
            &mut self.spares,
        );
    }

    // the next archive from the archive and the offspring
    fn survive(&mut self) {
        let parents = std::mem::take(&mut self.population).into_vec();
        let parent_count = parents.len();
        let mut pool = parents;
        pool.append(&mut self.offspring);
        let scores = scores_of(&pool);
        strength_fitness(&scores, &self.objectives, &mut self.scratch);
        let Scratch {
            fitness,
            distances,
            sorted,
            rows,
            ..
        } = &mut self.scratch;
        let mut survivors: Vec<usize> = (0..pool.len()).filter(|&i| fitness[i] < 1.0).collect();
        if survivors.len() < self.population_size {
            let mut others: Vec<usize> = (0..pool.len()).filter(|&i| fitness[i] >= 1.0).collect();
            others.sort_by(|&a, &b| fitness[a].total_cmp(&fitness[b]));
            survivors.extend(&others[..self.population_size - survivors.len()]);
        }
        if survivors.len() > self.population_size {
            truncate(
                &mut survivors,
                distances,
                pool.len(),
                self.population_size,
                sorted,
                rows,
            );
        }
        let mut selected = vec![false; pool.len()];
        for &index in &survivors {
            selected[index] = true;
        }
        let mut slots: Vec<Option<Individual<R::Genome, Scores<M>>>> =
            pool.into_iter().map(Some).collect();
        self.spares.keep_all(self.discarded.drain(..));
        for (index, slot) in slots.iter_mut().enumerate() {
            if !selected[index] && index >= parent_count {
                self.discarded.push(slot.take().expect("not taken yet"));
            }
        }
        self.fitness.clear();
        let mut population = Vec::with_capacity(survivors.len());
        for index in survivors {
            let mut individual = slots[index].take().expect("chosen once");
            if index < parent_count {
                individual.increment_age();
            }
            population.push(individual);
            self.fitness.push(fitness[index]);
        }
        // the parents that didn't survive
        self.spares.keep_all(slots.into_iter().flatten());
        self.population = Population::new(population);
    }

    // the new front, and whether it improved on the previous one
    fn update_front(&mut self) {
        let scores = scores_of(self.population.as_slice());
        let fronts = non_dominated_sort(&scores, &self.objectives);
        let first = fronts.first().map(Vec::as_slice).unwrap_or_default();
        let previous = scores_of(&self.front);
        distinct_into(&mut self.front, &self.population, first.iter().copied());
        if gains(&scores_of(&self.front), &previous, &self.objectives) {
            self.front_generation = self.generation;
        }
    }
}

impl<R, C, X, const M: usize> MultiObjectiveAlgorithm<M> for Spea2<R, C, X, M>
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
            let scores = scores_of(self.population.as_slice());
            strength_fitness(&scores, &self.objectives, &mut self.scratch);
            self.fitness.clone_from(&self.scratch.fitness);
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

/// A builder for [`Spea2`], from [`Spea2::builder`].
///
/// The population (archive) size, the crossover and the mutation are required. Defaults:
/// `crossover_rate` 0.9, `mutation_rate` 1.0, `eliminate_duplicates` true, a random initial
/// population and a random seed.
#[derive(Clone, Debug)]
pub struct Spea2Builder<R: Representation, const M: usize, C = Unset, X = Unset> {
    representation: R,
    objectives: [Objective; M],
    crossover: C,
    mutate: X,
    population_size: Option<usize>,
    crossover_rate: f64,
    mutation_rate: f64,
    seed: Option<u64>,
    eliminate_duplicates: bool,
    initial_genomes: Vec<R::Genome>,
}

impl<R: Representation, const M: usize, C, X> Spea2Builder<R, M, C, X> {
    /// The crossover operator. Required.
    pub fn crossover<T>(self, crossover: T) -> Spea2Builder<R, M, T, X> {
        Spea2Builder {
            representation: self.representation,
            objectives: self.objectives,
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

    /// The mutation operator. Required.
    pub fn mutate<T>(self, mutate: T) -> Spea2Builder<R, M, C, T> {
        Spea2Builder {
            representation: self.representation,
            objectives: self.objectives,
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

    /// The archive size, at least 2 and at most 2^24; also the number of children per
    /// generation. Required.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// The probability that a pair of parents is recombined, between 0 and 1. 0.9 by default.
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
    /// free of copies. On by default. When copies are all a population can breed,
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
    /// - [`Error::MissingSetting`] without a population size.
    /// - [`Error::InvalidSetting`] for a population size below 2 or above 2^24, no objectives,
    ///   rates out of range, a mutation rate of 0 with a crossover rate of 0 or
    ///   [`NoCrossover`](crate::operator::NoCrossover), or more initial genomes than the
    ///   population size.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<Spea2<R, C, X, M>>
    where
        C: Crossover<R>,
        X: Mutate<R>,
    {
        let size = self.population_size.ok_or(Error::MissingSetting {
            setting: "population_size",
        })?;
        let invalid = |setting, reason: String| Err(Error::InvalidSetting { setting, reason });
        if size < 2 {
            return invalid(
                "population_size",
                format!("SPEA2 needs at least 2 individuals, got {size}"),
            );
        }
        check_size("population_size", size)?;
        if M == 0 {
            return invalid("objectives", "at least 1 objective is needed".to_string());
        }
        let (crossover_rate, mutation_rate) = check_rates(
            self.crossover_rate,
            self.mutation_rate,
            self.crossover.recombines(),
        )?;
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
        Ok(Spea2 {
            variation: Variation {
                representation: self.representation,
                crossover: self.crossover,
                mutate: self.mutate,
                crossover_chance: Chance::new(crossover_rate),
                mutation_chance: Chance::new(mutation_rate),
                eliminate_duplicates: self.eliminate_duplicates,
            },
            objectives: self.objectives,
            population_size: size,
            crossover_rate,
            mutation_rate,
            seed,
            rng,
            population: genomes.into_iter().map(Individual::unevaluated).collect(),
            fitness: Vec::new(),
            offspring: Vec::new(),
            pending: Vec::new(),
            front: Vec::new(),
            discarded: Vec::new(),
            started: false,
            asked: false,
            generation: 0,
            evaluations: 0,
            front_generation: 0,
            spares: Spares::default(),
            scratch: Scratch::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective::{Maximize, Minimize};
    use crate::genome::{Real, Reals};
    use crate::multi::dominates;
    use crate::operator::{PolynomialMutation, SimulatedBinaryCrossover};
    use proptest::prelude::*;

    #[test]
    fn equal_infinities_are_no_nan_apart() {
        // inf - inf is a NaN whose sign, and so its order, depends on the processor
        let scores = [
            Scores::new([f64::INFINITY, 0.0]),
            Scores::new([f64::INFINITY, 1.0]),
            Scores::new([0.0, 0.5]),
            Scores::new([4.0, 0.2]),
        ];
        let mut scratch = Scratch::default();
        strength_fitness(&scores, &[Minimize, Minimize], &mut scratch);
        assert!(scratch.fitness.iter().all(|value| !value.is_nan()));
        assert!(scratch.distances.iter().all(|distance| !distance.is_nan()));
        // the objective with infinite values is still scaled, by its finite range 0 to 4
        let expected = (1.0f64 + (0.5f64 - 0.2) * (0.5 - 0.2)).sqrt();
        assert_eq!(scratch.distances[2 * 4 + 3], expected);
    }

    fn builder(
        size: usize,
        seed: u64,
    ) -> Spea2Builder<Real, 2, SimulatedBinaryCrossover, PolynomialMutation> {
        Spea2::builder(Real::uniform(3, 0.0..=1.0).unwrap(), [Minimize, Minimize])
            .population_size(size)
            .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
            .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0).unwrap())
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
    fn fitness_of_a_small_set() {
        // a, b, c are non-dominated; d is dominated by b only; e by all four
        let scores = [
            Scores::new([0.0, 4.0]),
            Scores::new([2.0, 2.0]),
            Scores::new([4.0, 0.0]),
            Scores::new([3.0, 3.0]),
            Scores::new([5.0, 5.0]),
        ];
        let mut scratch = Scratch::default();
        strength_fitness(&scores, &[Minimize, Minimize], &mut scratch);
        let (fitness, distances) = (&scratch.fitness, &scratch.distances);
        // strengths: a 1 (e), b 2 (d, e), c 1 (e), d 1 (e), e 0; raw fitness: d 2, e 1 + 2 + 1 + 1
        let raw: Vec<f64> = fitness.iter().map(|f| f.floor()).collect();
        assert_eq!(raw, [0.0, 0.0, 0.0, 2.0, 5.0]);
        assert!(
            fitness
                .iter()
                .all(|f| f - f.floor() > 0.0 && f - f.floor() <= 0.5)
        );
        // distances with both objectives scaled to 0..5
        assert!((distances[2] - (0.8f64 * 0.8 * 2.0).sqrt()).abs() < 1e-12);
        assert_eq!(distances[5 + 1], f64::INFINITY);
        // the maximized mirror image gives the same fitness
        let mirrored: Vec<Scores<2>> = scores
            .iter()
            .map(|s| Scores::new(s.values().unwrap().map(|v| -v)))
            .collect();
        let mut mirrored_scratch = Scratch::default();
        strength_fitness(&mirrored, &[Maximize, Maximize], &mut mirrored_scratch);
        assert_eq!(&mirrored_scratch.fitness, fitness);
    }

    #[test]
    fn truncation_removes_the_most_crowded() {
        // five points on a line: 1 and 2 are the closest pair, and 2 is closer to 3
        let xs: [f64; 5] = [0.0, 1.0, 1.2, 2.0, 4.0];
        let distances: Vec<f64> = xs
            .iter()
            .flat_map(|a| {
                xs.map(|b| {
                    if *a == b {
                        f64::INFINITY
                    } else {
                        (a - b).abs()
                    }
                })
            })
            .collect();
        let (mut sorted, mut rows) = (Vec::new(), Vec::new());
        let mut survivors = vec![0, 1, 2, 3, 4];
        truncate(&mut survivors, &distances, 5, 3, &mut sorted, &mut rows);
        // 1 and 2 are nearest (0.2); 1's second nearest is 1.0 (to 0), 2's is 0.8 (to 3), so 2
        // goes first; then 1 and 3 are nearest (1.0 each), and 1's second nearest, 1.0 to 0, is
        // smaller than 3's, 2.0 to 4
        assert_eq!(survivors, [0, 3, 4]);
        // the extremes stay
        let mut survivors = vec![0, 1, 2, 3, 4];
        truncate(&mut survivors, &distances, 5, 2, &mut sorted, &mut rows);
        assert_eq!(survivors, [0, 4]);
    }

    #[test]
    fn validation_and_protocol() {
        assert_eq!(setting(builder(1, 0).build()), "population_size");
        assert_eq!(
            setting(builder(4, 0).mutation_rate(2.0).build()),
            "mutation_rate"
        );
        let mut spea2 = builder(10, 0).build().unwrap();
        assert_eq!(spea2.tell(&[]), Err(Error::TellWithoutAsk));
        let f = |x: &Reals| Scores::new([x[0], 1.0 - x[0] + x[1]]);
        for generation in 0..5 {
            let told: Vec<Scores<2>> = spea2.ask().iter().map(f).collect();
            spea2.tell(&told).unwrap();
            assert_eq!(spea2.generation(), generation);
            assert_eq!(spea2.population().len(), 10);
            assert_eq!(spea2.strength_fitness().len(), 10);
        }
        // the front is the non-dominated members of the archive: fitness below 1
        let non_dominated = spea2
            .strength_fitness()
            .iter()
            .filter(|&&f| f < 1.0)
            .count();
        assert_eq!(spea2.front().len(), non_dominated);
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut spea2 = builder(12, seed).build().unwrap();
            let f = |x: &Reals| Scores::new([x[0], x[1] + x[2]]);
            for _ in 0..10 {
                let told: Vec<Scores<2>> = spea2.ask().iter().map(f).collect();
                spea2.tell(&told).unwrap();
            }
            spea2.population().clone()
        };
        assert_eq!(run(2), run(2));
        assert_ne!(run(2), run(3));
    }

    proptest! {
        #[test]
        fn the_new_front_is_never_dominated_by_the_old_population(
            seed: u64,
            size in 2usize..14,
            maximize: bool,
        ) {
            let objectives = if maximize { [Maximize, Minimize] } else { [Minimize, Minimize] };
            let mut spea2 = Spea2::builder(Real::uniform(3, 0.0..=1.0).unwrap(), objectives)
                .population_size(size)
                .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
                .mutate(PolynomialMutation::per_gene(0.5, 20.0).unwrap())
                .seed(seed)
                .build()
                .unwrap();
            let mut rng = StreamRng::seed_from_u64(seed);
            let mut random = |_: &Reals| match rng.below(10) {
                0 => Scores::invalid(),
                1 => Scores::constrained([0.0, 0.0], rng.below(3) as f64 + 1.0),
                _ => Scores::new([rng.below(5) as f64, rng.below(5) as f64]),
            };
            let told: Vec<Scores<2>> = spea2.ask().iter().map(&mut random).collect();
            spea2.tell(&told).unwrap();
            for _ in 0..8 {
                let old: Vec<Scores<2>> =
                    spea2.population().iter().map(|x| x.fitness().unwrap()).collect();
                let told: Vec<Scores<2>> = spea2.ask().iter().map(&mut random).collect();
                spea2.tell(&told).unwrap();
                prop_assert_eq!(spea2.population().len(), size);
                for member in spea2.front() {
                    let scores = member.fitness().unwrap();
                    prop_assert!(!old.iter().any(|o| dominates(o, &scores, &objectives)));
                }
            }
        }
    }
}
