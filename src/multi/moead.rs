//! MOEA/D: multi-objective optimization by decomposition into single-objective subproblems.

use super::breed::{Spares, Variation, distinct_into, scores_of};
use super::pareto::gains;
use super::{MultiObjectiveAlgorithm, Scores, non_dominated_sort};
use crate::algorithm::{Candidates, Unset};
use crate::genome::{Real, Reals, Representation};
use crate::operator::{Crossover, Mutate, check_probability, check_rates, check_size};
use crate::rng::Chance;
use crate::{Error, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;
use sealed::Sealed;
use std::fmt::Debug;

/// The differential evolution operator of MOEA/D-DE (Li and Zhang, 2009), for a [`Moead`] on
/// [`Real`] genomes, in place of a crossover.
///
/// A subproblem's child starts from its own solution `x` and moves by a scaled difference of the
/// two parents `a` and `b` (chosen as for a crossover: from the neighborhood with probability
/// `neighbor_mating`, else from the whole population, in a random order): each gene `k` becomes
/// `x_k + F (a_k − b_k)` with probability `CR`, and stays `x_k` otherwise (the paper's eq. 6,
/// with no gene forced to move). A gene that leaves its bounds is brought back by [`Repair`].
/// Then the mutation applies; Li and Zhang use polynomial mutation with η 20 at a rate of 1 / the
/// number of genes.
///
/// With this operator, a child whose parents came from the whole population may replace the
/// solutions of any subproblems, not only of its neighborhood: the paper's update range is its
/// mating range (step 2.1). The steps are differences between solutions of the population, so
/// once it lies near the front they point along it, in all the genes together: suited to Pareto
/// sets whose genes are linked (complicated Pareto sets), where SBX and polynomial mutation, which
/// change each gene on its own, stall. With `CR` 1 it is invariant under rotations of the genes.
/// On the paper's F2, with its settings, genoxide's MOEA/D-DE reaches an IGD of 0.0026 to 0.0039
/// over 10 seeds (the paper's table II: 0.0028 on average, 0.0023 at best), and MOEA/D with SBX
/// 0.06 to 0.13. On fronts that SBX reaches easily (ZDT, DTLZ), it converges more slowly; on
/// DTLZ2 a `CR` below 1, such as 0.5, converges far better than 1.
///
/// Li and Zhang's settings (section IV-A): `F` 0.5, `CR` 1, 20 neighbors and `neighbor_mating`
/// 0.9 (the defaults), at most 2 replacements (the default), polynomial mutation with η 20 at a
/// rate of 1 / n, 300 weight vectors for 2 objectives and 595 for 3, 500 generations. genoxide's
/// [`Moead`] differs from their algorithm where it does for any crossover: the children of a
/// generation are bred from the same population and evaluated together, then replace solutions
/// in a random order; a child replaces a solution only if strictly better (the paper's step 2.5
/// also replaces an equal one); the ideal point moves to feasible values only. Their polynomial
/// mutation can leave the bounds and is repaired after; genoxide's
/// [`PolynomialMutation`](crate::operator::PolynomialMutation) stays within them, so only the
/// difference step is repaired.
///
/// On constrained problems, where a [`Moead`] compares solutions by violation first, a lower
/// `neighbor_mating`, such as 0.2, keeps the population from collapsing onto the part of the
/// front it first finds feasible: with it, 300 weight vectors and 1,000 generations reach the
/// fronts of DAS-CMOP1, 2 and 3 (Fan et al., 2020) in every one of 20 seeds.
///
/// Li, H. and Zhang, Q. (2009). Multiobjective optimization problems with complicated Pareto
/// sets, MOEA/D and NSGA-II. *IEEE Transactions on Evolutionary Computation* 13(2): 284-302.
/// doi:10.1109/TEVC.2008.925798. Section III-A.
///
/// ```
/// use genoxide::Objective::Minimize;
/// use genoxide::multi::problems::{MultiProblem, Zdt1};
/// use genoxide::multi::{DifferentialEvolutionCrossover, Moead, das_dennis};
/// use genoxide::prelude::*;
///
/// let problem = Zdt1::new(30);
/// let moead = Moead::builder(problem.representation(), [Minimize; 2], das_dennis::<2>(99))
///     .crossover(DifferentialEvolutionCrossover::new(0.5, 1.0)?)
///     .mutate(PolynomialMutation::per_gene(1.0 / 30.0, 20.0)?)
///     .seed(1)
///     .build()?;
/// let outcome = MultiEngine::new(moead, problem).stop_when(Stop::generations(150)).run()?;
/// assert!(outcome.front().len() > 50);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DifferentialEvolutionCrossover {
    f: f64,
    cr: f64,
    chance: Chance,
    repair: Repair,
}

/// How a [`DifferentialEvolutionCrossover`] brings back a gene that `x + F (a − b)` takes out of
/// its bounds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Repair {
    /// A uniformly random value between the subproblem's gene `x` and the bound the gene
    /// crossed: the gene moves toward that bound, never away from it. The default: it reaches
    /// genes on their bounds, where many fronts' solutions lie (ZDT's distance genes at 0, the
    /// ends of DAS-CMOP's fronts), and with it genoxide matches the paper's results on its F2.
    #[default]
    Bounce,
    /// A uniformly random value anywhere within the bounds: the repair the paper's text
    /// describes (step 2.3). Genes on a bound are then hard to reach: on the paper's F2, the
    /// IGD is 0.007 to 0.031 over 10 seeds, against the paper's 0.0028 on average.
    Random,
}

impl DifferentialEvolutionCrossover {
    /// The operator with the scale factor `f` of the difference, greater than 0 and at most 2,
    /// and the probability `cr` that a gene moves, between 0 and 1. Li and Zhang use 0.5 and 1.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for an `f` or a `cr` out of range.
    pub fn new(f: f64, cr: f64) -> Result<Self> {
        if !(f > 0.0 && f <= 2.0) {
            return Err(Error::InvalidSetting {
                setting: "f",
                reason: format!("must be greater than 0 and at most 2, got {f}"),
            });
        }
        let cr = check_probability("cr", cr)?;
        Ok(Self {
            f,
            cr,
            chance: Chance::new(cr),
            repair: Repair::Bounce,
        })
    }

    /// The repair of a gene out of its bounds: [`Repair::Bounce`] by default.
    pub fn with_repair(mut self, repair: Repair) -> Self {
        self.repair = repair;
        self
    }

    /// The repair of a gene out of its bounds.
    pub fn repair(&self) -> Repair {
        self.repair
    }

    /// The scale factor of the difference.
    pub fn f(&self) -> f64 {
        self.f
    }

    /// The probability that a gene moves.
    pub fn cr(&self) -> f64 {
        self.cr
    }
}

/// The operator that makes a [`Moead`]'s children from parents: any [`Crossover`], or a
/// [`DifferentialEvolutionCrossover`] on [`Real`] genomes. It is implemented for those only.
pub trait MoeadCrossover<R: Representation>: Sealed<R> + Clone + Debug + Send + Sync {}

impl<R: Representation, T: Sealed<R> + Clone + Debug + Send + Sync> MoeadCrossover<R> for T {}

mod sealed {
    use super::{
        Crossover, DifferentialEvolutionCrossover, Real, Reals, Repair, Representation, Spares,
    };
    use crate::StreamRng;
    use crate::genome::real::random_in;

    // how a child is made: the methods of `MoeadCrossover`, which no other crate implements
    pub trait Sealed<R: Representation> {
        // whether it can change the genomes, for the rates' check
        fn can_recombine(&self) -> bool;

        // whether the child is the subproblem's solution moved by the difference of the parents
        // (MOEA/D-DE), which may then replace solutions anywhere in its mating range
        fn differential(&self) -> bool;

        // the child of the subproblem whose solution is `parents[0]`, from the parents
        // `parents[1]` and `parents[2]` (in a random order), recombined if `recombine`, in a
        // copy from `spares`
        fn child(
            &self,
            representation: &R,
            parents: [&R::Genome; 3],
            recombine: bool,
            spares: &mut Spares<R::Genome>,
            rng: &mut StreamRng,
        ) -> R::Genome;
    }

    impl<R: Representation, C: Crossover<R>> Sealed<R> for C {
        fn can_recombine(&self) -> bool {
            self.recombines()
        }

        fn differential(&self) -> bool {
            false
        }

        // one of the crossover's two children, at random, or a copy of a random parent
        fn child(
            &self,
            representation: &R,
            [_, first, second]: [&R::Genome; 3],
            recombine: bool,
            spares: &mut Spares<R::Genome>,
            rng: &mut StreamRng,
        ) -> R::Genome {
            if recombine {
                let mut a = spares.copy(first);
                let mut b = spares.copy(second);
                self.crossover(representation, &mut a, &mut b, rng);
                let (child, other) = if rng.below(2) == 0 { (a, b) } else { (b, a) };
                spares.keep(other);
                child
            } else {
                // a copy of one of the parents: only that one is copied
                let parent = if rng.below(2) == 0 { first } else { second };
                spares.copy(parent)
            }
        }
    }

    impl Sealed<Real> for DifferentialEvolutionCrossover {
        fn can_recombine(&self) -> bool {
            true
        }

        fn differential(&self) -> bool {
            true
        }

        // `x + F (a − b)` in the genes chosen with probability CR, a gene out of its bounds
        // repaired; the subproblem's solution if not recombined
        fn child(
            &self,
            real: &Real,
            [current, first, second]: [&Reals; 3],
            recombine: bool,
            spares: &mut Spares<Reals>,
            rng: &mut StreamRng,
        ) -> Reals {
            let mut child = spares.copy(current);
            if recombine {
                let bounds = real.bounds();
                rng.chosen(self.chance, child.len(), |rng, gene| {
                    let range = &bounds[gene];
                    let moved = current[gene] + self.f * (first[gene] - second[gene]);
                    child[gene] = if range.contains(&moved) {
                        moved
                    } else {
                        match self.repair {
                            Repair::Random => random_in(range, rng),
                            Repair::Bounce => {
                                let (start, end) = (*range.start(), *range.end());
                                let x = current[gene];
                                let u = rng.unit_f64();
                                // above the end, or below the start (or NaN)
                                if moved > end {
                                    (end - u * (end - x)).max(start)
                                } else {
                                    (start + u * (x - start)).min(end)
                                }
                            }
                        }
                    };
                });
            }
            child
        }
    }
}

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

// whether `a` is better than `b` for a subproblem: the smaller violation, and then the smaller
// decomposition value, from `a_value` and `b_value` only if the violations are equal; invalid is
// the worst
fn improves<const M: usize>(
    a: &Scores<M>,
    b: &Scores<M>,
    a_value: impl FnOnce() -> f64,
    b_value: impl FnOnce() -> f64,
) -> bool {
    match (a.is_valid(), b.is_valid()) {
        (false, _) => return false,
        (true, false) => return true,
        (true, true) => {}
    }
    if a.violation() != b.violation() {
        return a.violation() < b.violation();
    }
    a_value() < b_value()
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
///    of its two children, at random) and mutated. With a [`DifferentialEvolutionCrossover`]
///    (MOEA/D-DE), the child is instead the subproblem's own solution moved by the scaled
///    difference of the two parents.
/// 2. The children are evaluated together, so a generation can be evaluated in parallel. The
///    ideal point moves to the best feasible values seen.
/// 3. In a random order, each child replaces the solutions of its neighborhood that it improves
///    on, for their subproblems: at most `max_replacements` (2, as in MOEA/D-DE, Li and Zhang,
///    2009), visiting the neighbors in a random order. The limit keeps one good child from taking
///    over a whole neighborhood, which matters more when a generation's children are applied
///    together. In MOEA/D-DE, a child whose parents came from the whole population visits the
///    whole population instead.
///
/// Constraints are handled by Deb's feasibility rules in the comparison of a child with a
/// solution (MOEA/D-CDP): a feasible solution beats an infeasible one, of two infeasible ones the
/// smaller violation wins, and of two with the same violation, the subproblem's value decides.
///
/// ```
/// use genoxide::Objective::Minimize;
/// use genoxide::multi::problems::{MultiProblem, Zdt1};
/// use genoxide::multi::{Moead, das_dennis};
/// use genoxide::prelude::*;
///
/// let problem = Zdt1::new(30);
/// let moead = Moead::builder(problem.representation(), [Minimize; 2], das_dennis::<2>(99))
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
    #[cfg_attr(feature = "serde", serde(skip))]
    spares: Spares<R::Genome>,
    // with a differential crossover, whether each child's parents came from the whole population,
    // which is then where it may replace solutions
    #[cfg_attr(feature = "serde", serde(default))]
    from_population: Vec<bool>,
    // the whole population in the order its solutions are visited by such a child
    #[cfg_attr(feature = "serde", serde(skip))]
    visit: Vec<usize>,
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
    C: MoeadCrossover<R>,
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
        let differential = self.variation.crossover.differential();
        self.offspring.clear();
        self.from_population.clear();
        for subproblem in 0..size {
            let neighborhood = &self.neighborhoods[subproblem];
            let from_neighborhood =
                neighborhood.len() >= 2 && self.rng.chance(self.neighbor_chance);
            // sample_distinct(2, n) without an allocation
            let (a, b) = if from_neighborhood {
                let (first, second) = self.rng.sample_pair(neighborhood.len());
                (neighborhood[first], neighborhood[second])
            } else {
                self.rng.sample_pair(size)
            };
            // the pair is in ascending order: a random order for the crossover
            let (a, b) = if self.rng.below(2) == 0 {
                (a, b)
            } else {
                (b, a)
            };
            let variation = &self.variation;
            let recombine = self.rng.chance(variation.crossover_chance);
            let parents = [subproblem, a, b].map(|parent| self.population[parent].genome());
            let mut genome = variation.crossover.child(
                &variation.representation,
                parents,
                recombine,
                &mut self.spares,
                &mut self.rng,
            );
            if self.rng.chance(variation.mutation_chance) {
                variation
                    .mutate
                    .mutate(&variation.representation, &mut genome, &mut self.rng);
            }
            // a differential child starts from the subproblem's solution, which it may equal
            let others: &[usize] = if differential {
                &[subproblem, a, b]
            } else {
                &[a, b]
            };
            let inherited = others
                .iter()
                .map(|&parent| &self.population[parent])
                .find(|parent| parent.genome() == &genome)
                .and_then(Individual::fitness);
            if differential {
                self.from_population.push(!from_neighborhood);
            }
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
    #[cfg(test)]
    fn improves(&self, a: &Scores<M>, b: &Scores<M>, subproblem: usize) -> bool {
        improves(
            a,
            b,
            || self.value(a, subproblem),
            || self.value(b, subproblem),
        )
    }

    // the decomposition value of scores for a subproblem
    fn value(&self, scores: &Scores<M>, subproblem: usize) -> f64 {
        let values = minimized(scores, &self.objectives);
        (self.decomposition).value(&values, &self.weights[subproblem], &self.ideal)
    }

    // the children replace the neighbors they improve on, in a random order
    fn replace(&mut self) {
        let mut children = std::mem::take(&mut self.offspring);
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
        // the decomposition value of each slot's solution for its subproblem, once needed: the
        // ideal point doesn't move while children replace solutions
        let mut values: Vec<Option<f64>> = vec![None; size];
        let mut neighbors = Vec::new();
        for subproblem in order {
            let child = &children[subproblem];
            let scores = child.fitness().unwrap_or(Scores::invalid());
            let anywhere = self.from_population.get(subproblem) == Some(&true);
            if anywhere {
                // the whole population, in a random order drawn as it's visited: a child seldom
                // visits all of it
                self.visit.clear();
                self.visit.extend(0..size);
            } else {
                neighbors.clone_from(&self.neighborhoods[subproblem]);
                for i in (1..neighbors.len()).rev() {
                    neighbors.swap(i, self.rng.below(i + 1));
                }
            }
            let candidates = if anywhere { size } else { neighbors.len() };
            let mut replaced = 0;
            // by index: the whole population's order is drawn at each index as it's visited
            #[expect(clippy::needless_range_loop)]
            for visited in 0..candidates {
                if replaced == self.max_replacements {
                    break;
                }
                let neighbor = if anywhere {
                    let next = visited + self.rng.below(size - visited);
                    self.visit.swap(visited, next);
                    self.visit[visited]
                } else {
                    neighbors[visited]
                };
                let current = self.population[neighbor]
                    .fitness()
                    .unwrap_or(Scores::invalid());
                let mut child_value = None;
                let better = improves(
                    &scores,
                    &current,
                    || *child_value.insert(self.value(&scores, neighbor)),
                    || *values[neighbor].get_or_insert_with(|| self.value(&current, neighbor)),
                );
                if better {
                    self.population[neighbor].clone_from(child);
                    holders[neighbor] = Some(subproblem);
                    values[neighbor] = child_value;
                    replaced += 1;
                }
            }
        }
        // the children that aren't in the population: never placed, or replaced by later children
        let mut kept = vec![false; size];
        for holder in holders.into_iter().flatten() {
            kept[holder] = true;
        }
        self.spares.keep_all(self.discarded.drain(..));
        for (child, kept) in children.drain(..).zip(kept) {
            if kept {
                // the population has a copy
                self.spares.keep(child.into_genome());
            } else {
                self.discarded.push(child);
            }
        }
        self.offspring = children;
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

impl<R, C, X, const M: usize> MultiObjectiveAlgorithm<M> for Moead<R, C, X, M>
where
    R: Representation,
    C: MoeadCrossover<R>,
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
/// neighborhood with probability 0.9, at most 2 replacements per child, Tchebycheff
/// decomposition, `crossover_rate` and `mutation_rate` 1.0 (every child is recombined and
/// mutated), a random initial population and a random seed.
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
    /// The crossover operator: a [`Crossover`], or a [`DifferentialEvolutionCrossover`] for
    /// MOEA/D-DE on real genomes ([`MoeadCrossover`]). Required; SBX with η 20 is the usual
    /// choice for real genomes, differential evolution for Pareto sets whose genes are linked.
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
    /// neighborhood size (or more) removes the limit.
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
    ///   probability or rate out of range, a mutation rate of 0 with a crossover rate of 0 or
    ///   [`NoCrossover`](crate::operator::NoCrossover), a negative or non-finite `theta`, or more
    ///   initial genomes than weight vectors.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<Moead<R, C, X, M>>
    where
        C: MoeadCrossover<R>,
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
        check_size("weights", size)?;
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
        if let Decomposition::Pbi { theta } = self.decomposition
            && !(theta >= 0.0 && theta.is_finite())
        {
            return invalid(
                "theta",
                format!("must be 0 or more and finite, got {theta}"),
            );
        }
        let (crossover_rate, mutation_rate) = check_rates(
            self.crossover_rate,
            self.mutation_rate,
            self.crossover.can_recombine(),
        )?;
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
        // the nearest weight vectors: itself first, then the lower index on ties
        let count = self.neighbors.min(size);
        let neighborhoods = self
            .weights
            .iter()
            .enumerate()
            .map(|(own, a)| {
                let distance =
                    |b: &[f64; M]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>();
                let nearer = |&i: &usize, &j: &usize| {
                    distance(&self.weights[i])
                        .total_cmp(&distance(&self.weights[j]))
                        .then((i != own).cmp(&(j != own)))
                        .then(i.cmp(&j))
                };
                // the `count` nearest, then only they sorted: O(size) per weight vector, not
                // O(size log size), and the same order, as `nearer` is a total order
                let mut order: Vec<usize> = (0..size).collect();
                if count < size {
                    order.select_nth_unstable_by(count, nearer);
                    order.truncate(count);
                }
                order.sort_by(nearer);
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
                // MOEA/D replaces neighbors one child at a time, and doesn't eliminate duplicates
                eliminate_duplicates: false,
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
            spares: Spares::default(),
            from_population: Vec::new(),
            visit: Vec::new(),
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
        // with duplicate weights, each subproblem still comes first in its own neighborhood
        let duplicates = Moead::builder(
            Real::uniform(3, 0.0..=1.0).unwrap(),
            [Minimize, Minimize],
            vec![[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [1.0, 0.0]],
        )
        .neighbors(2)
        .crossover(SimulatedBinaryCrossover::new(20.0).unwrap())
        .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0).unwrap())
        .seed(0)
        .build()
        .unwrap();
        for (index, neighborhood) in duplicates.neighborhoods().iter().enumerate() {
            assert_eq!(neighborhood[0], index);
        }
        // weights (0, 1), (0.25, 0.75), ..., (1, 0)
        let moead = builder(4, 0).neighbors(3).build().unwrap();
        assert_eq!(moead.neighborhoods()[0], [0, 1, 2]);
        assert_eq!(moead.neighborhoods()[2], [2, 1, 3]);
        assert_eq!(moead.neighborhoods()[4], [4, 3, 2]);
        // the same as sorting every weight vector by distance, with its ties, for every size
        let weights = crate::multi::das_dennis::<3>(6);
        for neighbors in [2, 5, 7, weights.len(), weights.len() + 3] {
            let moead = Moead::builder(
                Real::uniform(3, 0.0..=1.0).unwrap(),
                [Minimize; 3],
                weights.clone(),
            )
            .neighbors(neighbors)
            .crossover(SimulatedBinaryCrossover::new(20.0).unwrap())
            .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0).unwrap())
            .seed(0)
            .build()
            .unwrap();
            for (own, neighborhood) in moead.neighborhoods().iter().enumerate() {
                let distance = |i: usize| -> f64 {
                    (0..3)
                        .map(|k| (weights[i][k] - weights[own][k]).powi(2))
                        .sum()
                };
                let mut sorted: Vec<usize> = (0..weights.len()).collect();
                sorted.sort_by(|&i, &j| {
                    distance(i)
                        .total_cmp(&distance(j))
                        .then((i != own).cmp(&(j != own)))
                        .then(i.cmp(&j))
                });
                sorted.truncate(neighbors);
                assert_eq!(*neighborhood, sorted);
            }
        }
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

    fn differential(
        crossover: &DifferentialEvolutionCrossover,
        parents: [&[f64]; 3],
        recombine: bool,
        seed: u64,
    ) -> Vec<f64> {
        let real = Real::uniform(3, 0.0..=1.0).unwrap();
        let parents = parents.map(|genes| Reals::from(genes.to_vec()));
        let child = crossover.child(
            &real,
            [&parents[0], &parents[1], &parents[2]],
            recombine,
            &mut Spares::default(),
            &mut StreamRng::seed_from_u64(seed),
        );
        child.to_vec()
    }

    #[test]
    fn differential_evolution_by_hand() {
        let x: &[f64] = &[0.5, 0.5, 0.5];
        let a: &[f64] = &[0.8, 0.2, 0.75];
        let b: &[f64] = &[0.6, 0.4, 0.25];
        // x + 0.5 (a − b): every gene with CR 1, and these are exact in binary
        let de = DifferentialEvolutionCrossover::new(0.5, 1.0).unwrap();
        let child = differential(&de, [x, a, b], true, 1);
        let expected = [0.5 + 0.5 * (0.8 - 0.6), 0.5 + 0.5 * (0.2 - 0.4), 0.75];
        assert_eq!(child, expected);
        // not recombined, or with CR 0: the subproblem's solution
        assert_eq!(differential(&de, [x, a, b], false, 1), x);
        let none = DifferentialEvolutionCrossover::new(0.5, 0.0).unwrap();
        assert_eq!(differential(&none, [x, a, b], true, 1), x);
        // with CR 0.5, each gene is x's or the moved one, and both happen
        let half = DifferentialEvolutionCrossover::new(0.5, 0.5).unwrap();
        let (mut kept, mut moved) = (0, 0);
        for seed in 0..50 {
            for (k, gene) in differential(&half, [x, a, b], true, seed)
                .into_iter()
                .enumerate()
            {
                if gene == x[k] {
                    kept += 1;
                } else {
                    assert_eq!(gene, expected[k]);
                    moved += 1;
                }
            }
        }
        assert!(kept > 50 && moved > 50, "{kept} {moved}");
        // F 2: the third gene, 0.5 + 2 × 0.5 = 1.5, leaves the bounds and is repaired: anywhere
        // in them at random, or between x and the upper bound with a bounce
        let bounce = DifferentialEvolutionCrossover::new(2.0, 1.0).unwrap();
        let far = bounce.with_repair(Repair::Random);
        let (mut below, mut above) = (false, false);
        for seed in 0..50 {
            let random = differential(&far, [x, a, b], true, seed);
            let moved = [0.5 + 2.0 * (0.8 - 0.6), 0.5 + 2.0 * (0.2 - 0.4)];
            assert_eq!(random[..2], moved);
            assert!((0.0..=1.0).contains(&random[2]));
            below |= random[2] < 0.5;
            above |= random[2] > 0.5;
            let bounced = differential(&bounce, [x, a, b], true, seed);
            assert_eq!(bounced[..2], moved);
            assert!((0.5..=1.0).contains(&bounced[2]), "{bounced:?}");
        }
        assert!(below && above);
        // below the lower bound: between it and x
        let low = differential(&bounce, [x, b, a], true, 3);
        assert!((0.0..=0.5).contains(&low[2]), "{low:?}");
    }

    #[test]
    fn differential_evolution_settings() {
        for (f, cr, name) in [
            (0.0, 1.0, "f"),
            (2.5, 1.0, "f"),
            (f64::NAN, 1.0, "f"),
            (0.5, -0.1, "cr"),
            (0.5, 1.5, "cr"),
            (0.5, f64::NAN, "cr"),
        ] {
            assert_eq!(setting(DifferentialEvolutionCrossover::new(f, cr)), name);
        }
        let de = DifferentialEvolutionCrossover::new(2.0, 0.0).unwrap();
        assert_eq!((de.f(), de.cr(), de.repair()), (2.0, 0.0, Repair::Bounce));
        assert_eq!(de.with_repair(Repair::Random).repair(), Repair::Random);
        // a mutation rate of 0 is allowed: the difference changes the genomes
        let moead = Moead::builder(
            Real::uniform(3, 0.0..=1.0).unwrap(),
            [Minimize; 2],
            das_dennis::<2>(4),
        )
        .crossover(de)
        .mutate(PolynomialMutation::per_gene(0.5, 20.0).unwrap())
        .mutation_rate(0.0)
        .build();
        assert!(moead.is_ok());
    }

    type RealDe = Moead<Real, DifferentialEvolutionCrossover, PolynomialMutation, 2>;

    fn de_builder(
        divisions: usize,
        seed: u64,
    ) -> MoeadBuilder<Real, 2, DifferentialEvolutionCrossover, PolynomialMutation> {
        Moead::builder(
            Real::uniform(3, 0.0..=1.0).unwrap(),
            [Minimize, Minimize],
            das_dennis::<2>(divisions),
        )
        .crossover(DifferentialEvolutionCrossover::new(0.5, 1.0).unwrap())
        .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0).unwrap())
        .seed(seed)
    }

    // the slots outside the child's neighborhood that hold a copy of a child after a generation,
    // over 10 generations of a problem whose violations differ a lot
    fn replaced_outside<C: MoeadCrossover<Real>>(
        mut moead: Moead<Real, C, PolynomialMutation, 2>,
    ) -> usize {
        let f = |x: &Reals| Scores::constrained([x[0], 1.0 - x[0]], x[1] + x[2]);
        let told: Vec<Scores<2>> = moead.ask().iter().map(f).collect();
        moead.tell(&told).unwrap();
        let mut outside = 0;
        for _ in 0..10 {
            let told: Vec<Scores<2>> = moead.ask().iter().map(f).collect();
            // every subproblem's child, in their order
            let children: Vec<Reals> = moead.offspring.iter().map(|x| x.genome().clone()).collect();
            let before: Vec<Reals> = moead
                .population()
                .iter()
                .map(|x| x.genome().clone())
                .collect();
            moead.tell(&told).unwrap();
            for (slot, member) in moead.population().iter().enumerate() {
                if *member.genome() == before[slot] {
                    continue;
                }
                // the children it may be a copy of: outside if none has the slot in its neighborhood
                let inside = children
                    .iter()
                    .enumerate()
                    .filter(|(_, child)| *child == member.genome())
                    .any(|(from, _)| moead.neighborhoods()[from].contains(&slot));
                if !inside {
                    outside += 1;
                }
            }
        }
        outside
    }

    #[test]
    fn children_from_the_whole_population_replace_anywhere_with_differential_evolution() {
        // parents from the whole population: MOEA/D-DE's children replace anywhere, others only
        // in the neighborhood
        let de = de_builder(19, 2)
            .neighbors(3)
            .neighbor_mating(0.0)
            .build()
            .unwrap();
        assert!(replaced_outside(de) > 0);
        let sbx = builder(19, 2)
            .neighbors(3)
            .neighbor_mating(0.0)
            .build()
            .unwrap();
        assert_eq!(replaced_outside(sbx), 0);
        let local = de_builder(19, 2)
            .neighbors(3)
            .neighbor_mating(1.0)
            .build()
            .unwrap();
        assert_eq!(replaced_outside(local), 0);
    }

    #[test]
    fn differential_evolution_same_seed_same_run() {
        let run = |seed| {
            let mut moead: RealDe = de_builder(9, seed).neighbor_mating(0.5).build().unwrap();
            for _ in 0..10 {
                let told: Vec<Scores<2>> = moead
                    .ask()
                    .iter()
                    .map(|x| Scores::new([x[0], x[1] + x[2]]))
                    .collect();
                moead.tell(&told).unwrap();
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
