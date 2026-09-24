//! Breeding shared by the multi-objective genetic algorithms.

use super::Scores;
use crate::genome::Representation;
use crate::operator::{Crossover, Mutate};
use crate::rng::Chance;
use crate::{Individual, Population, StreamRng};

// the operators and rates of a genetic algorithm
#[derive(Clone, Debug)]
pub(crate) struct Variation<R, C, X> {
    pub(crate) representation: R,
    pub(crate) crossover: C,
    pub(crate) mutate: X,
    pub(crate) crossover_chance: Chance,
    pub(crate) mutation_chance: Chance,
}

impl<R, C, X> Variation<R, C, X>
where
    R: Representation,
    C: Crossover<R>,
    X: Mutate<R>,
{
    // `count` children of pairs of parents chosen by `select`: recombined with the crossover
    // chance, each mutated with the mutation chance; a child equal to a parent inherits its scores
    pub(crate) fn breed<const M: usize>(
        &self,
        population: &Population<R::Genome, Scores<M>>,
        count: usize,
        rng: &mut StreamRng,
        mut select: impl FnMut(&mut StreamRng) -> usize,
        offspring: &mut Vec<Individual<R::Genome, Scores<M>>>,
    ) {
        offspring.clear();
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
