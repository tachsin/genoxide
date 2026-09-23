//! Statistics per generation.

use super::{Observer, Snapshot};
use crate::Fitness;
use crate::genome::Genome;
use std::collections::HashSet;
use std::time::Duration;

/// Records [`GenerationStatistics`] after every generation.
///
/// ```
/// use genoxide::prelude::*;
///
/// let ga = Ga::builder(Binary::new(16)?)
///     .population_size(20)
///     .select(Tournament::new(2)?)
///     .crossover(UniformCrossover::new())
///     .mutate(BitFlip::count(1)?)
///     .seed(3)
///     .build()?;
/// let mut statistics = Statistics::new();
/// Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
///     .stop_when(Stop::generations(10))
///     .observe(&mut statistics)
///     .run()?;
/// for record in statistics.records() {
///     println!("{} {:?} {:?}", record.generation, record.best, record.mean);
/// }
/// assert_eq!(statistics.records().len(), 11);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Statistics {
    records: Vec<GenerationStatistics>,
}

/// Statistics of the population after a generation.
///
/// The mean and standard deviation are over the valid scores. Infinite scores make them infinite
/// or NaN.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct GenerationStatistics {
    /// The generation, 0 for the initial population.
    pub generation: u64,
    /// The number of fitness evaluations so far.
    pub evaluations: u64,
    /// The time since the run started.
    pub elapsed: Duration,
    /// The best fitness found so far.
    pub best_so_far: Option<Fitness>,
    /// The best fitness in the population.
    pub best: Option<Fitness>,
    /// The mean score, `None` without valid scores.
    pub mean: Option<f64>,
    /// The (population) standard deviation of the scores, `None` without valid scores.
    pub std_dev: Option<f64>,
    /// The number of individuals with an invalid fitness.
    pub invalid: usize,
    /// The number of distinct genomes, a measure of diversity.
    pub unique: usize,
    /// The population size.
    pub size: usize,
}

impl Statistics {
    /// Empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// The statistics of every generation so far, in order.
    pub fn records(&self) -> &[GenerationStatistics] {
        &self.records
    }

    /// The statistics of the last generation.
    pub fn last(&self) -> Option<&GenerationStatistics> {
        self.records.last()
    }
}

impl<G: Genome> Observer<G> for Statistics {
    fn observe(&mut self, snapshot: &Snapshot<'_, G>) {
        let population = snapshot.population();
        let progress = snapshot.progress();
        let scores: Vec<f64> = population
            .iter()
            .filter_map(|individual| individual.fitness()?.score())
            .collect();
        let evaluated = population
            .iter()
            .filter(|individual| individual.is_evaluated())
            .count();
        let (mean, std_dev) = if scores.is_empty() {
            (None, None)
        } else {
            let count = scores.len() as f64;
            let mean = scores.iter().sum::<f64>() / count;
            let variance = scores
                .iter()
                .map(|score| (score - mean) * (score - mean))
                .sum::<f64>()
                / count;
            (Some(mean), Some(variance.sqrt()))
        };
        let unique = population
            .iter()
            .map(|individual| individual.genome())
            .collect::<HashSet<_>>()
            .len();
        self.records.push(GenerationStatistics {
            generation: progress.generation(),
            evaluations: progress.evaluations(),
            elapsed: progress.elapsed(),
            best_so_far: progress.best(),
            best: population
                .best(progress.objective())
                .and_then(|individual| individual.fitness()),
            mean,
            std_dev,
            invalid: evaluated - scores.len(),
            unique,
            size: population.len(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Progress;
    use crate::genome::Bits;
    use crate::{Individual, Objective, Population};

    #[test]
    fn statistics_of_a_population() {
        let fitness = [Some(1.0), Some(3.0), None, Some(2.0)];
        let genomes = ["00", "11", "01", "11"];
        let population: Population<Bits> = genomes
            .iter()
            .zip(fitness)
            .map(|(genome, fitness)| {
                let mut individual = Individual::new(genome.chars().map(|c| c == '1').collect());
                individual.set_fitness(fitness.map_or(Fitness::invalid(), Fitness::new));
                individual
            })
            .collect();
        let progress = Progress::for_test(4, Objective::Minimize);
        let mut statistics = Statistics::new();
        statistics.observe(&Snapshot::new(&population, &[], &population[0], &progress));
        let record = statistics.last().unwrap();
        assert_eq!(record.generation, 4);
        assert_eq!(record.best, Some(Fitness::new(1.0)));
        assert_eq!(record.mean, Some(2.0));
        assert_eq!(record.std_dev, Some((2.0f64 / 3.0).sqrt()));
        assert_eq!(record.invalid, 1);
        assert_eq!(record.unique, 3);
        assert_eq!(record.size, 4);
    }
}
