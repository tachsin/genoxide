//! A population of individuals.

use crate::genome::Genome;
use crate::{Fitness, Individual, Objective};
use std::cmp::Ordering;
use std::ops::{Index, IndexMut};

/// A population of individuals.
///
/// "Best" and sorting are deterministic: ties are broken by position, the earlier individual
/// first. Individuals that aren't evaluated yet are never the best, and sort last.
///
/// ```
/// use genoxide::{Fitness, Objective, Population};
/// use genoxide::genome::Bits;
///
/// let mut population = Population::from_genomes([Bits::zeros(2), Bits::ones(2)]);
/// for individual in population.iter_mut() {
///     let ones = individual.genome().count_ones();
///     individual.set_fitness(Fitness::new(ones as f64));
/// }
/// let best = population.best(Objective::Maximize).unwrap();
/// assert_eq!(best.genome().to_string(), "11");
/// ```
///
/// The fitness type `F` is [`Fitness`] by default, and [`Scores`](crate::multi::Scores) in
/// multi-objective optimization.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Population<G: Genome, F = Fitness> {
    individuals: Vec<Individual<G, F>>,
}

impl<G: Genome, F> Population<G, F> {
    /// A population of these individuals.
    pub fn new(individuals: Vec<Individual<G, F>>) -> Self {
        Self { individuals }
    }

    /// The number of individuals.
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Whether the population is empty.
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    /// The individuals.
    pub fn iter(&self) -> std::slice::Iter<'_, Individual<G, F>> {
        self.individuals.iter()
    }

    /// The individuals, mutable.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Individual<G, F>> {
        self.individuals.iter_mut()
    }

    /// Adds an individual.
    pub fn push(&mut self, individual: Individual<G, F>) {
        self.individuals.push(individual);
    }

    /// Keeps the first `len` individuals and drops the rest. No effect if there are fewer.
    pub fn truncate(&mut self, len: usize) {
        self.individuals.truncate(len);
    }

    /// The individuals as a slice.
    pub fn as_slice(&self) -> &[Individual<G, F>] {
        &self.individuals
    }

    /// The individuals, consuming the population.
    pub fn into_vec(self) -> Vec<Individual<G, F>> {
        self.individuals
    }
}

impl<G: Genome> Population<G> {
    /// A population of new, not yet evaluated individuals with these genomes.
    pub fn from_genomes<I: IntoIterator<Item = G>>(genomes: I) -> Self {
        genomes.into_iter().map(Individual::new).collect()
    }

    /// The position of the best evaluated individual (the first one on ties), or `None` if no
    /// individual is evaluated.
    pub fn best_index(&self, objective: Objective) -> Option<usize> {
        let mut best: Option<(usize, Fitness)> = None;
        for (index, individual) in self.individuals.iter().enumerate() {
            let Some(fitness) = individual.fitness() else {
                continue;
            };
            let is_better = match best {
                Some((_, best_fitness)) => objective.is_better(fitness, best_fitness),
                None => true,
            };
            if is_better {
                best = Some((index, fitness));
            }
        }
        best.map(|(index, _)| index)
    }

    /// The best evaluated individual (the first one on ties), or `None` if no individual is
    /// evaluated. Invalid individuals are only the best if every evaluated individual is invalid.
    pub fn best(&self, objective: Objective) -> Option<&Individual<G>> {
        self.best_index(objective)
            .map(|index| &self.individuals[index])
    }

    /// Sorts best first. The sort is stable: ties keep their order. Individuals that aren't
    /// evaluated yet go last.
    pub fn sort_best_first(&mut self, objective: Objective) {
        self.individuals
            .sort_by(|a, b| match (a.fitness(), b.fitness()) {
                (Some(a), Some(b)) => objective.compare(b, a),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            });
    }
}

impl<G: Genome, F> Default for Population<G, F> {
    /// An empty population.
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl<G: Genome, F> Extend<Individual<G, F>> for Population<G, F> {
    fn extend<I: IntoIterator<Item = Individual<G, F>>>(&mut self, iter: I) {
        self.individuals.extend(iter);
    }
}

impl<G: Genome, F> FromIterator<Individual<G, F>> for Population<G, F> {
    fn from_iter<I: IntoIterator<Item = Individual<G, F>>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

impl<G: Genome, F> IntoIterator for Population<G, F> {
    type Item = Individual<G, F>;
    type IntoIter = std::vec::IntoIter<Individual<G, F>>;

    fn into_iter(self) -> Self::IntoIter {
        self.individuals.into_iter()
    }
}

impl<'a, G: Genome, F> IntoIterator for &'a Population<G, F> {
    type Item = &'a Individual<G, F>;
    type IntoIter = std::slice::Iter<'a, Individual<G, F>>;

    fn into_iter(self) -> Self::IntoIter {
        self.individuals.iter()
    }
}

impl<G: Genome, F> Index<usize> for Population<G, F> {
    type Output = Individual<G, F>;

    fn index(&self, index: usize) -> &Individual<G, F> {
        &self.individuals[index]
    }
}

impl<G: Genome, F> IndexMut<usize> for Population<G, F> {
    fn index_mut(&mut self, index: usize) -> &mut Individual<G, F> {
        &mut self.individuals[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Fitness;
    use crate::genome::Bits;
    use proptest::prelude::*;

    // None: not evaluated, Some(None): invalid, Some(Some(score))
    fn population(fitness: &[Option<Option<f64>>]) -> Population<Bits> {
        fitness
            .iter()
            .enumerate()
            .map(|(index, fitness)| {
                // distinct genomes, to follow individuals through a sort
                let mut individual =
                    Individual::new((0..16).map(|bit| index >> bit & 1 == 1).collect());
                if let Some(fitness) = fitness {
                    individual.set_fitness(fitness.map_or(Fitness::invalid(), Fitness::new));
                }
                individual
            })
            .collect()
    }

    #[test]
    fn best_skips_unevaluated_and_prefers_first_on_ties() {
        let population = population(&[
            None,
            Some(Some(1.0)),
            Some(None),
            Some(Some(3.0)),
            Some(Some(3.0)),
        ]);
        assert_eq!(population.best_index(Objective::Maximize), Some(3));
        assert_eq!(population.best_index(Objective::Minimize), Some(1));
    }

    #[test]
    fn best_of_only_invalid_or_unevaluated() {
        assert_eq!(
            population(&[None, None]).best_index(Objective::Maximize),
            None
        );
        assert_eq!(
            population(&[None, Some(None), Some(None)]).best_index(Objective::Maximize),
            Some(1)
        );
        assert_eq!(population(&[]).best_index(Objective::Minimize), None);
    }

    fn any_fitness() -> impl Strategy<Value = Option<Option<f64>>> {
        prop_oneof![
            Just(None),
            Just(Some(None)),
            (-3i32..3).prop_map(|score| Some(Some(score as f64))),
        ]
    }

    fn any_objective() -> impl Strategy<Value = Objective> {
        prop_oneof![Just(Objective::Maximize), Just(Objective::Minimize)]
    }

    proptest! {
        #[test]
        fn best_is_first_maximum(fitness in prop::collection::vec(any_fitness(), 0..20), objective in any_objective()) {
            let population = population(&fitness);
            let expected = population.iter().enumerate()
                .filter_map(|(index, individual)| individual.fitness().map(|fitness| (index, fitness)))
                .fold(None, |best: Option<(usize, Fitness)>, (index, fitness)| match best {
                    Some((_, best_fitness)) if !objective.is_better(fitness, best_fitness) => best,
                    _ => Some((index, fitness)),
                })
                .map(|(index, _)| index);
            prop_assert_eq!(population.best_index(objective), expected);
        }

        #[test]
        fn sort_best_first_is_ordered_and_stable(fitness in prop::collection::vec(any_fitness(), 0..20), objective in any_objective()) {
            let original = population(&fitness);
            let mut sorted = original.clone();
            sorted.sort_best_first(objective);
            prop_assert_eq!(sorted.len(), original.len());
            for pair in sorted.as_slice().windows(2) {
                match (pair[0].fitness(), pair[1].fitness()) {
                    (Some(a), Some(b)) => {
                        prop_assert!(!objective.is_better(b, a));
                        if a == b {
                            // stable: equal fitness keeps the original order
                            let position = |individual: &Individual<Bits>| original.iter().position(|o| o == individual);
                            prop_assert!(position(&pair[0]) < position(&pair[1]));
                        }
                    }
                    (None, Some(_)) => prop_assert!(false, "unevaluated before evaluated"),
                    _ => {}
                }
            }
            if let Some(best) = original.best(objective) {
                prop_assert_eq!(&sorted[0], best);
            }
        }
    }
}
