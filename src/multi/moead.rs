//! MOEA/D: multi-objective optimization by decomposition into single-objective subproblems.

use super::breed::{Variation, scores_of};
use super::pareto::gains;
use super::{MultiObjectiveAlgorithm, Scores, non_dominated_sort};
use crate::algorithm::{Candidates, Unset};
use crate::genome::Representation;
use crate::operator::{Crossover, Mutate, check_probability};
use crate::rng::Chance;
use crate::{Error, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;

/// How a [`Moead`] turns the objectives into one value per subproblem, around the ideal point
/// `z` (the best value of each objective so far), for a weight vector `w`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Decomposition {
    /// The weighted Tchebycheff distance `maxⱼ wⱼ |fⱼ − zⱼ|`: the default, for any front shape.
    Tchebycheff,
    /// Penalty-based boundary intersection: `d₁ + θ d₂`, the distance along the weight vector
    /// plus `theta` times the distance from it. Spreads many-objective fronts evenly; `theta` 5
    /// is common.
    Pbi {
        /// The penalty for the distance from the weight vector, 0 or more.
        theta: f64,
    },
}

impl Decomposition {
    // the value of minimized objective values for a weight vector and the ideal point
    fn value<const M: usize>(
        &self,
        values: &[f64; M],
        weights: &[f64; M],
        ideal: &[f64; M],
    ) -> f64 {
        match *self {
            Decomposition::Tchebycheff => (0..M)
                .map(|j| weights[j] * (values[j] - ideal[j]).abs())
                .fold(f64::NEG_INFINITY, f64::max),
            Decomposition::Pbi { theta } => {
                let norm = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
                let shifted: [f64; M] = std::array::from_fn(|j| values[j] - ideal[j]);
                let along = (0..M).map(|j| shifted[j] * weights[j]).sum::<f64>() / norm;
                let across = (0..M)
                    .map(|j| {
                        let d = shifted[j] - along * weights[j] / norm;
                        d * d
                    })
                    .sum::<f64>()
                    .sqrt();
                along + theta * across
            }
        }
    }
}

/// MOEA/D (Zhang and Li, 2007): multi-objective optimization by decomposition, as an ask / tell
/// [`MultiObjectiveAlgorithm`].
///
/// Every weight vector (usually [Das-Dennis points](super::das_dennis)) defines a
/// single-objective subproblem by [`Decomposition`], and the population holds one solution per
/// subproblem. Neighboring weight vectors (the `neighbors` nearest) define neighboring
/// subproblems, which share good solutions. Every generation:
///
/// 1. Each subproblem gets a child: two parents from its neighborhood (with probability
///    `neighbor_mating`, 0.9) or from the whole population, recombined with the crossover (one
///    of its two children, at random) and mutated.
/// 2. The children are evaluated together, so a generation can be evaluated in parallel (like
///    pymoo's `ParallelMOEAD`). The ideal point moves to the best feasible values seen.
/// 3. In a random order, each child replaces the solutions of its neighborhood that it improves
///    on, for their subproblems: at most `max_replacements` (2, as in MOEA/D-DE, Li and Zhang,
///    2009), visiting the neighbors in a random order. The limit keeps one good child from taking
///    over a whole neighborhood, which matters more when a generation's children are applied
///    together; pymoo's MOEA/D has no limit.
///
/// Between solutions with different constraint violations, the smaller violation is better, so
/// constraints are handled too (unlike pymoo's MOEA/D).
///
/// ```
/// use genoxide::Objective::Minimize;
/// use genoxide::multi::problems::{TestProblem, Zdt1};
/// use genoxide::multi::{Moead, das_dennis};
/// use genoxide::prelude::*;
///
/// let problem = Zdt1::new(30);
/// let moead = Moead::builder(problem.real(), [Minimize; 2], das_dennis::<2>(99))
///     .crossover(SimulatedBinaryCrossover::new(20.0)?)
///     .mutate(PolynomialMutation::per_gene(1.0 / 30.0, 20.0)?)
///     .seed(1)
///     .build()?;
/// let outcome = MultiEngine::new(moead, problem).stop_when(Stop::generations(150)).run()?;
/// assert!(outcome.front().len() > 50);
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
pub struct Moead<R: Representation, C, X, const M: usize> {
    variation: Variation<R, C, X>,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays::array"))]
    objectives: [Objective; M],
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays::vec_of_arrays"))]
    weights: Vec<[f64; M]>,
    // the nearest weight vectors of each, itself first
    neighborhoods: Vec<Vec<usize>>,
    neighbor_mating: f64,
    neighbor_chance: Chance,
    max_replacements: usize,
    decomposition: Decomposition,
    seed: u64,
    rng: StreamRng,
    population: Population<R::Genome, Scores<M>>,
    offspring: Vec<Individual<R::Genome, Scores<M>>>,
    pending: Vec<usize>,
    front: Vec<Individual<R::Genome, Scores<M>>>,
    discarded: Vec<Individual<R::Genome, Scores<M>>>,
    // the best feasible value of each objective so far, minimized
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays::array"))]
    ideal: [f64; M],
    started: bool,
    asked: bool,
    generation: u64,
    evaluations: u64,
    front_generation: u64,
}

impl<R: Representation, const M: usize> Moead<R, Unset, Unset, M> {
    /// A builder for MOEA/D on `representation`, with the direction of each objective and the
    /// weight vectors, e.g. [`das_dennis`](super::das_dennis): non-negative, not all 0. The
    /// population size is their number.
    pub fn builder(
        representation: R,
        objectives: [Objective; M],
        weights: Vec<[f64; M]>,
    ) -> MoeadBuilder<R, M> {
        MoeadBuilder {
            representation,
            objectives,
            weights,
            crossover: Unset,
            mutate: Unset,
            neighbors: 20,
            neighbor_mating: 0.9,
            max_replacements: 2,
            decomposition: Decomposition::Tchebycheff,
            crossover_rate: 1.0,
            mutation_rate: 1.0,
            seed: None,
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

impl<R, C, X, const M: usize> Moead<R, C, X, M>
where
    R: Representation,
    C: Crossover<R>,
    X: Mutate<R>,
{
    /// The representation.
    pub fn representation(&self) -> &R {
        &self.variation.representation
    }

    /// The weight vectors, one per subproblem and member of the population, in its order.
    pub fn weights(&self) -> &[[f64; M]] {
        &self.weights
    }

    /// The neighborhood of each subproblem: the nearest weight vectors, itself first.
    pub fn neighborhoods(&self) -> &[Vec<usize>] {
        &self.neighborhoods
    }

    /// The decomposition.
    pub fn decomposition(&self) -> Decomposition {
        self.decomposition
    }

    /// The probability that a subproblem's parents come from its neighborhood.
    pub fn neighbor_mating(&self) -> f64 {
        self.neighbor_mating
    }

    /// The most solutions a child replaces.
    pub fn max_replacements(&self) -> usize {
        self.max_replacements
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    // one child per subproblem
    fn breed(&mut self) {
        let size = self.population.len();
        self.offspring.clear();
        for subproblem in 0..size {
            let neighborhood = &self.neighborhoods[subproblem];
            let from_neighborhood =
                neighborhood.len() >= 2 && self.rng.chance(self.neighbor_chance);
            let (a, b) = if from_neighborhood {
                let picked = self.rng.sample_distinct(2, neighborhood.len());
                (neighborhood[picked[0]], neighborhood[picked[1]])
            } else {
                let picked = self.rng.sample_distinct(2, size);
                (picked[0], picked[1])
            };
            // sample_distinct is in ascending order: a random order for the crossover
            let (a, b) = if self.rng.below(2) == 0 {
                (a, b)
            } else {
                (b, a)
            };
            let variation = &self.variation;
            let mut first = self.population[a].genome().clone();
            let mut second = self.population[b].genome().clone();
            if self.rng.chance(variation.crossover_chance) {
                variation.crossover.crossover(
                    &variation.representation,
                    &mut first,
                    &mut second,
                    &mut self.rng,
                );
            }
            let mut genome = if self.rng.below(2) == 0 {
                first
            } else {
                second
            };
            if self.rng.chance(variation.mutation_chance) {
                variation
                    .mutate
                    .mutate(&variation.representation, &mut genome, &mut self.rng);
            }
            let inherited = [a, b]
                .iter()
                .map(|&parent| &self.population[parent])
                .find(|parent| parent.genome() == &genome)
                .and_then(Individual::fitness);
            let mut child = Individual::unevaluated(genome);
            if let Some(scores) = inherited {
                child.set_fitness(scores);
            }
            self.offspring.push(child);
        }
    }

    // moves the ideal point to the best feasible values of `scores`
    fn update_ideal(&mut self, scores: &[Scores<M>]) {
        for score in scores.iter().filter(|s| s.is_feasible()) {
            let values = minimized(score, &self.objectives);
            for (ideal, value) in self.ideal.iter_mut().zip(values) {
                *ideal = ideal.min(value);
            }
        }
    }

    // whether `a` is better than `b` for a subproblem: the smaller violation, and then the
    // smaller decomposition value; invalid is the worst
    fn improves(&self, a: &Scores<M>, b: &Scores<M>, subproblem: usize) -> bool {
        match (a.is_valid(), b.is_valid()) {
            (false, _) => return false,
            (true, false) => return true,
            (true, true) => {}
        }
        if a.violation() != b.violation() {
            return a.violation() < b.violation();
        }
        let weights = &self.weights[subproblem];
        let value = |s: &Scores<M>| {
            self.decomposition
                .value(&minimized(s, &self.objectives), weights, &self.ideal)
        };
        value(a) < value(b)
    }

    // the children replace the neighbors they improve on, in a random order
    fn replace(&mut self) {
        let children = std::mem::take(&mut self.offspring);
        let size = self.population.len();
        let mut order: Vec<usize> = (0..size).collect();
        for i in (1..size).rev() {
            order.swap(i, self.rng.below(i + 1));
        }
        for individual in self.population.iter_mut() {
            individual.increment_age();
        }
        // the child in each slot, if a child replaced its solution
        let mut holders: Vec<Option<usize>> = vec![None; size];
        for subproblem in order {
            let child = &children[subproblem];
            let scores = child.fitness().unwrap_or(Scores::invalid());
            let mut neighbors = self.neighborhoods[subproblem].clone();
            for i in (1..neighbors.len()).rev() {
                neighbors.swap(i, self.rng.below(i + 1));
            }
            let mut replaced = 0;
            for neighbor in neighbors {
                if replaced == self.max_replacements {
                    break;
                }
                let current = self.population[neighbor]
                    .fitness()
                    .unwrap_or(Scores::invalid());
                if self.improves(&scores, &current, neighbor) {
                    self.population[neighbor] = child.clone();
                    holders[neighbor] = Some(subproblem);
                    replaced += 1;
                }
            }
        }
        // the children that aren't in the population: never placed, or replaced by later children
        let mut kept = vec![false; size];
        for holder in holders.into_iter().flatten() {
            kept[holder] = true;
        }
        self.discarded = children
            .into_iter()
            .zip(kept)
            .filter(|(_, kept)| !kept)
            .map(|(child, _)| child)
            .collect();
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

impl<R, C, X, const M: usize> MultiObjectiveAlgorithm<M> for Moead<R, C, X, M>
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
            let children = scores_of(&self.offspring);
            self.update_ideal(&children);
            self.replace();
        } else {
            for (individual, &score) in self.population.iter_mut().zip(scores) {
                individual.set_fitness(score);
            }
            self.update_ideal(scores);
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

/// A builder for [`Moead`], from [`Moead::builder`].
///
/// The crossover and the mutation are required. Defaults: 20 neighbors, parents from the
/// neighborhood with probability 0.9, at most 2 replacements per child, Tchebycheff decomposition, `crossover_rate` and
/// `mutation_rate` 1.0 (as in pymoo), a random initial population and a random seed.
#[derive(Clone, Debug)]
pub struct MoeadBuilder<R: Representation, const M: usize, C = Unset, X = Unset> {
    representation: R,
    objectives: [Objective; M],
    weights: Vec<[f64; M]>,
    crossover: C,
    mutate: X,
    neighbors: usize,
    neighbor_mating: f64,
    max_replacements: usize,
    decomposition: Decomposition,
    crossover_rate: f64,
    mutation_rate: f64,
    seed: Option<u64>,
    initial_genomes: Vec<R::Genome>,
}

impl<R: Representation, const M: usize, C, X> MoeadBuilder<R, M, C, X> {
    /// The crossover operator. Required; SBX with η 20 is the usual choice for real genomes.
    pub fn crossover<T>(self, crossover: T) -> MoeadBuilder<R, M, T, X> {
        MoeadBuilder {
            representation: self.representation,
            objectives: self.objectives,
            weights: self.weights,
            crossover,
            mutate: self.mutate,
            neighbors: self.neighbors,
            neighbor_mating: self.neighbor_mating,
            max_replacements: self.max_replacements,
            decomposition: self.decomposition,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            seed: self.seed,
            initial_genomes: self.initial_genomes,
        }
    }

    /// The mutation operator. Required; polynomial mutation with η 20 and a rate of 1 / the
    /// number of genes is the usual choice for real genomes.
    pub fn mutate<T>(self, mutate: T) -> MoeadBuilder<R, M, C, T> {
        MoeadBuilder {
            representation: self.representation,
            objectives: self.objectives,
            weights: self.weights,
            crossover: self.crossover,
            mutate,
            neighbors: self.neighbors,
            neighbor_mating: self.neighbor_mating,
            max_replacements: self.max_replacements,
            decomposition: self.decomposition,
            crossover_rate: self.crossover_rate,
            mutation_rate: self.mutation_rate,
            seed: self.seed,
            initial_genomes: self.initial_genomes,
        }
    }

    /// The size of each neighborhood, at least 2 (itself included); at most the number of
    /// weight vectors, fewer if there are fewer. 20 by default.
    pub fn neighbors(mut self, neighbors: usize) -> Self {
        self.neighbors = neighbors;
        self
    }

    /// The probability that a subproblem's parents come from its neighborhood rather than the
    /// whole population, between 0 and 1. 0.9 by default.
    pub fn neighbor_mating(mut self, probability: f64) -> Self {
        self.neighbor_mating = probability;
        self
    }

    /// The most solutions a child replaces, at least 1. 2 by default (MOEA/D-DE); the
    /// neighborhood size (or more) removes the limit, as in pymoo.
    pub fn max_replacements(mut self, count: usize) -> Self {
        self.max_replacements = count;
        self
    }

    /// The decomposition. [`Decomposition::Tchebycheff`] by default;
    /// [`Decomposition::Pbi`] with `theta` 5 spreads fronts of 3 or more objectives well.
    pub fn decomposition(mut self, decomposition: Decomposition) -> Self {
        self.decomposition = decomposition;
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

    /// Genomes for the initial population, at most the number of weight vectors, each valid
    /// for the representation; the first goes to the first subproblem, and so on. The rest is
    /// random.
    pub fn initial_genomes<I: IntoIterator<Item = R::Genome>>(mut self, genomes: I) -> Self {
        self.initial_genomes = genomes.into_iter().collect();
        self
    }

    /// Validates the settings and creates the algorithm, with its initial population.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidSetting`] for no objectives, fewer than 2 weight vectors or one with a
    ///   negative, non-finite or all-zero component, fewer than 2 neighbors, no replacements, a
    ///   probability or
    ///   rate out of range, both rates 0, a negative or non-finite `theta`, or more initial
    ///   genomes than weight vectors.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<Moead<R, C, X, M>>
    where
        C: Crossover<R>,
        X: Mutate<R>,
    {
        let invalid = |setting, reason: String| Err(Error::InvalidSetting { setting, reason });
        if M == 0 {
            return invalid("objectives", "at least 1 objective is needed".to_string());
        }
        let size = self.weights.len();
        if size < 2 {
            return invalid(
                "weights",
                format!("MOEA/D needs at least 2 weight vectors, got {size}"),
            );
        }
        for weights in &self.weights {
            let valid = weights.iter().all(|w| *w >= 0.0 && w.is_finite())
                && weights.iter().any(|w| *w > 0.0);
            if !valid {
                return invalid(
                    "weights",
                    format!("must be non-negative, finite and not all 0, got {weights:?}"),
                );
            }
        }
        if self.neighbors < 2 {
            return invalid("neighbors", format!("at least 2, got {}", self.neighbors));
        }
        let neighbor_mating = check_probability("neighbor_mating", self.neighbor_mating)?;
        if self.max_replacements == 0 {
            return invalid("max_replacements", "must be at least 1".to_string());
        }
        if let Decomposition::Pbi { theta } = self.decomposition {
            if !(theta >= 0.0 && theta.is_finite()) {
                return invalid(
                    "theta",
                    format!("must be 0 or more and finite, got {theta}"),
                );
            }
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
                    "at most the {size} weight vectors, got {}",
                    self.initial_genomes.len()
                ),
            );
        }
        for genome in &self.initial_genomes {
            self.representation.validate(genome)?;
        }
        // the nearest weight vectors, the lower index on ties
        let count = self.neighbors.min(size);
        let neighborhoods = self
            .weights
            .iter()
            .map(|a| {
                let distance =
                    |b: &[f64; M]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>();
                let mut order: Vec<usize> = (0..size).collect();
                order.sort_by(|&i, &j| {
                    distance(&self.weights[i])
                        .total_cmp(&distance(&self.weights[j]))
                        .then(i.cmp(&j))
                });
                order.truncate(count);
                order
            })
            .collect();
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let mut rng = StreamRng::seed_from_u64(seed);
        let random = size - self.initial_genomes.len();
        let mut genomes = self.initial_genomes;
        genomes.extend((0..random).map(|_| self.representation.random_genome(&mut rng)));
        Ok(Moead {
            variation: Variation {
                representation: self.representation,
                crossover: self.crossover,
                mutate: self.mutate,
                crossover_chance: Chance::new(crossover_rate),
                mutation_chance: Chance::new(mutation_rate),
            },
            objectives: self.objectives,
            weights: self.weights,
            neighborhoods,
            neighbor_mating,
            neighbor_chance: Chance::new(neighbor_mating),
            max_replacements: self.max_replacements,
            decomposition: self.decomposition,
            seed,
            rng,
            population: genomes.into_iter().map(Individual::unevaluated).collect(),
            offspring: Vec::new(),
            pending: Vec::new(),
            front: Vec::new(),
            discarded: Vec::new(),
            ideal: [f64::INFINITY; M],
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
    use crate::multi::das_dennis;
    use crate::operator::{PolynomialMutation, SimulatedBinaryCrossover};
    use proptest::prelude::*;

    type Real2 = Moead<Real, SimulatedBinaryCrossover, PolynomialMutation, 2>;

    fn builder(
        divisions: usize,
        seed: u64,
    ) -> MoeadBuilder<Real, 2, SimulatedBinaryCrossover, PolynomialMutation> {
        Moead::builder(
            Real::uniform(3, 0.0..=1.0).unwrap(),
            [Minimize, Minimize],
            das_dennis::<2>(divisions),
        )
        .crossover(SimulatedBinaryCrossover::new(20.0).unwrap())
        .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0).unwrap())
        .seed(seed)
    }

    fn step(moead: &mut Real2, f: impl Fn(&Reals) -> Scores<2>) {
        let told: Vec<Scores<2>> = moead.ask().iter().map(f).collect();
        moead.tell(&told).unwrap();
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
        let with = |weights: Vec<[f64; 2]>| {
            Moead::builder(Real::uniform(2, 0.0..=1.0).unwrap(), [Minimize; 2], weights)
                .crossover(SimulatedBinaryCrossover::new(20.0).unwrap())
                .mutate(PolynomialMutation::per_gene(0.5, 20.0).unwrap())
                .build()
        };
        assert_eq!(setting(with(vec![[1.0, 0.0]])), "weights");
        assert_eq!(setting(with(vec![[1.0, 0.0], [0.0, 0.0]])), "weights");
        assert_eq!(setting(with(vec![[1.0, 0.0], [-1.0, 1.0]])), "weights");
        assert_eq!(setting(builder(4, 0).neighbors(1).build()), "neighbors");
        assert_eq!(
            setting(builder(4, 0).max_replacements(0).build()),
            "max_replacements"
        );
        assert_eq!(
            setting(builder(4, 0).neighbor_mating(1.5).build()),
            "neighbor_mating"
        );
        let pbi = |theta| {
            builder(4, 0)
                .decomposition(Decomposition::Pbi { theta })
                .build()
        };
        assert_eq!(setting(pbi(-1.0)), "theta");
        assert_eq!(setting(pbi(f64::NAN)), "theta");
        assert!(pbi(0.0).is_ok());
        let moead = builder(4, 0).neighbors(100).build().unwrap();
        assert_eq!(moead.population().len(), 5);
        assert_eq!(moead.neighborhoods()[0].len(), 5);
    }

    #[test]
    fn neighborhoods_are_the_nearest_weights() {
        // weights (0, 1), (0.25, 0.75), ..., (1, 0)
        let moead = builder(4, 0).neighbors(3).build().unwrap();
        assert_eq!(moead.neighborhoods()[0], [0, 1, 2]);
        assert_eq!(moead.neighborhoods()[2], [2, 1, 3]);
        assert_eq!(moead.neighborhoods()[4], [4, 3, 2]);
    }

    #[test]
    fn decomposition_values() {
        let ideal = [0.0, 0.0];
        let tchebycheff = Decomposition::Tchebycheff;
        assert_eq!(tchebycheff.value(&[2.0, 1.0], &[0.5, 0.5], &ideal), 1.0);
        assert_eq!(tchebycheff.value(&[2.0, 1.0], &[0.0, 1.0], &ideal), 1.0);
        // PBI: along (1, 1) / √2, the point (2, 2) is 2√2 away and on the line
        let pbi = Decomposition::Pbi { theta: 5.0 };
        let value = pbi.value(&[2.0, 2.0], &[0.5, 0.5], &ideal);
        assert!((value - 8f64.sqrt()).abs() < 1e-12);
        // (2, 0): √2 along and √2 across
        let value = pbi.value(&[2.0, 0.0], &[0.5, 0.5], &ideal);
        assert!((value - 6.0 * 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn constraints_come_first() {
        let moead = builder(4, 0).build().unwrap();
        let feasible = Scores::new([9.0, 9.0]);
        let slightly = Scores::constrained([0.0, 0.0], 0.1);
        let far = Scores::constrained([0.0, 0.0], 2.0);
        assert!(moead.improves(&feasible, &slightly, 0));
        assert!(moead.improves(&slightly, &far, 0));
        assert!(!moead.improves(&far, &slightly, 0));
        assert!(moead.improves(&far, &Scores::invalid(), 0));
        assert!(!moead.improves(&Scores::invalid(), &far, 0));
    }

    #[test]
    fn ask_tell_protocol_and_replacements() {
        let mut moead = builder(9, 0).max_replacements(1).build().unwrap();
        assert_eq!(moead.tell(&[]), Err(Error::TellWithoutAsk));
        assert_eq!(moead.ask().len(), 10);
        let f = |x: &Reals| Scores::new([x[0], 1.0 - x[0] + x[1] + x[2]]);
        step(&mut moead, f);
        assert_eq!(moead.generation(), 0);
        for _ in 0..5 {
            let before: Vec<Reals> = moead
                .population()
                .iter()
                .map(|x| x.genome().clone())
                .collect();
            step(&mut moead, f);
            // with one replacement per child, each child is in the population at most once
            let after: Vec<&Reals> = moead.population().iter().map(|x| x.genome()).collect();
            for (index, genome) in after.iter().enumerate() {
                if **genome != before[index] {
                    let copies = after.iter().filter(|other| **other == *genome).count();
                    let old = before.iter().filter(|old| *old == *genome).count();
                    assert!(copies <= old + 1);
                }
            }
            assert_eq!(moead.population().len(), 10);
        }
        assert_eq!(moead.generation(), 5);
    }

    #[test]
    fn every_child_is_kept_or_discarded() {
        // without a replacement limit, later children often overwrite earlier ones
        let mut moead = builder(19, 1).max_replacements(20).build().unwrap();
        let f = |x: &Reals| Scores::new([x[0], 1.0 - x[0] + x[1] + x[2]]);
        step(&mut moead, f);
        for _ in 0..10 {
            let children: Vec<Reals> = moead.ask().iter().cloned().collect();
            step(&mut moead, f);
            let present = |genome: &Reals| {
                moead.population().iter().any(|x| x.genome() == genome)
                    || moead.discarded().iter().any(|x| x.genome() == genome)
            };
            assert!(children.iter().all(present));
        }
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut moead = builder(9, seed)
                .decomposition(Decomposition::Pbi { theta: 5.0 })
                .build()
                .unwrap();
            for _ in 0..10 {
                step(&mut moead, |x| Scores::new([x[0], x[1] + x[2]]));
            }
            moead.population().clone()
        };
        assert_eq!(run(6), run(6));
        assert_ne!(run(6), run(7));
    }

    proptest! {
        #[test]
        fn the_ideal_point_bounds_every_feasible_member(
            seed: u64,
            divisions in 1usize..12,
            neighbors in 2usize..8,
            maximize: bool,
        ) {
            let objectives = if maximize { [Maximize, Minimize] } else { [Minimize, Minimize] };
            let mut moead = Moead::builder(Real::uniform(3, 0.0..=1.0).unwrap(), objectives, das_dennis::<2>(divisions))
                .neighbors(neighbors)
                .crossover(SimulatedBinaryCrossover::new(20.0).unwrap())
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
            for _ in 0..8 {
                let told: Vec<Scores<2>> = moead.ask().iter().map(&mut random).collect();
                moead.tell(&told).unwrap();
                prop_assert_eq!(moead.population().len(), divisions + 1);
                for member in moead.population().iter() {
                    let scores = member.fitness().unwrap();
                    if scores.is_feasible() {
                        let values = minimized(&scores, &objectives);
                        prop_assert!(values.iter().zip(&moead.ideal).all(|(v, z)| v >= z));
                    }
                }
            }
        }
    }
}
