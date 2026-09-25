//! Breeding shared by the multi-objective genetic algorithms.

use super::Scores;
use crate::genome::Representation;
use crate::operator::{Crossover, Mutate};
use crate::rng::Chance;
use crate::{Individual, Population, StreamRng};
use std::collections::HashSet;

// with duplicate elimination, the children rejected as copies, per child needed, before copies
// are accepted: a population of few distinct genomes still gets its children
const REJECTIONS_PER_CHILD: usize = 100;

// the operators and rates of a genetic algorithm
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Variation<R, C, X> {
    pub(crate) representation: R,
    pub(crate) crossover: C,
    pub(crate) mutate: X,
    pub(crate) crossover_chance: Chance,
    pub(crate) mutation_chance: Chance,
    // whether a child that equals a member of the population or an earlier child is bred again;
    // a checkpoint from before this setting resumes without it, as it was saved
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) eliminate_duplicates: bool,
}

impl<R, C, X> Variation<R, C, X>
where
    R: Representation,
    C: Crossover<R>,
    X: Mutate<R>,
{
    // `count` children of pairs of parents chosen by `select`: recombined with the crossover
    // chance, each mutated with the mutation chance. With duplicate elimination, a child equal to
    // a member of the population or to an earlier child is dropped. A child equal to a parent
    // (only when copies are accepted) inherits its scores.
    pub(crate) fn breed<const M: usize>(
        &self,
        population: &Population<R::Genome, Scores<M>>,
        count: usize,
        rng: &mut StreamRng,
        mut select: impl FnMut(&mut StreamRng) -> usize,
        offspring: &mut Vec<Individual<R::Genome, Scores<M>>>,
    ) {
        offspring.clear();
        let mut population_genomes: HashSet<&R::Genome> = HashSet::new();
        let mut children: HashSet<R::Genome> = HashSet::new();
        if self.eliminate_duplicates {
            population_genomes.extend(population.iter().map(Individual::genome));
        }
        let mut rejections = count.saturating_mul(REJECTIONS_PER_CHILD);
        while offspring.len() < count {
            let parents = [select(rng), select(rng)];
            let mut a = population[parents[0]].genome().clone();
            let mut b = population[parents[1]].genome().clone();
            if rng.chance(self.crossover_chance) {
                self.crossover
                    .crossover(&self.representation, &mut a, &mut b, rng);
            }
            for mut genome in [a, b] {
                if offspring.len() == count {
                    break;
                }
                if rng.chance(self.mutation_chance) {
                    self.mutate.mutate(&self.representation, &mut genome, rng);
                }
                if self.eliminate_duplicates && rejections > 0 {
                    if population_genomes.contains(&genome) || children.contains(&genome) {
                        rejections -= 1;
                        continue;
                    }
                    children.insert(genome.clone());
                }
                let inherited = parents
                    .iter()
                    .map(|&parent| &population[parent])
                    .find(|parent| parent.genome() == &genome)
                    .and_then(Individual::fitness);
                let mut child = Individual::unevaluated(genome);
                if let Some(scores) = inherited {
                    child.set_fitness(scores);
                }
                offspring.push(child);
            }
        }
    }
}

// the scores of individuals, invalid if not evaluated
pub(crate) fn scores_of<G, const M: usize>(
    individuals: &[Individual<G, Scores<M>>],
) -> Vec<Scores<M>>
where
    G: crate::genome::Genome,
{
    individuals
        .iter()
        .map(|individual| individual.fitness().unwrap_or(Scores::invalid()))
        .collect()
}
