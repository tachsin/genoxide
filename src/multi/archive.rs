//! An archive of non-dominated solutions.

use super::{MultiSnapshot, Scores, crowding_distance, dominates};
use crate::genome::Genome;
use crate::{Error, Individual, Objective, Result};

/// The non-dominated solutions among every solution it's given: the best trade-offs found in a
/// whole run, not only in the final population.
///
/// A population's front can lose non-dominated solutions when it's larger than the population
/// (NSGA-II keeps the least crowded ones), so the archive can be better than the final front.
/// Solutions are compared by constrained [dominance](dominates); the archive keeps one member per
/// objective vector, the first one given. With a capacity, the most crowded member makes room
/// for a new one: the smallest [crowding distance](crowding_distance), the earliest on ties. The
/// extremes of each objective have an infinite distance, so they stay unless every member is an
/// extreme. They aren't infinite for an objective whose range isn't finite, or when any member is
/// invalid (every distance is then 0).
///
/// Record a run with [`update`](ParetoArchive::update) in
/// [`MultiEngine::on_generation`](super::MultiEngine::on_generation):
///
/// ```
/// use genoxide::Objective::Minimize;
/// use genoxide::multi::ParetoArchive;
/// use genoxide::prelude::*;
///
/// let nsga2 = Nsga2::builder(Real::uniform(1, -10.0..=10.0)?, [Minimize, Minimize])
///     .population_size(10)
///     .crossover(SimulatedBinaryCrossover::new(15.0)?)
///     .mutate(PolynomialMutation::per_gene(1.0, 20.0)?)
///     .seed(1)
///     .build()?;
/// let mut archive = ParetoArchive::new([Minimize, Minimize]);
/// let outcome = MultiEngine::new(nsga2, |x: &Reals| [x[0] * x[0], (x[0] - 2.0) * (x[0] - 2.0)])
///     .on_generation(|snapshot| archive.update(snapshot))
///     .stop_when(Stop::generations(30))
///     .run()?;
/// // more trade-offs than the final population holds
/// assert!(archive.len() > outcome.front().len());
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParetoArchive<G: Genome, const M: usize> {
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_arrays::array"))]
    objectives: [Objective; M],
    capacity: Option<usize>,
    members: Vec<Individual<G, Scores<M>>>,
}

impl<G: Genome, const M: usize> ParetoArchive<G, M> {
    /// An empty, unbounded archive for these objective directions.
    pub fn new(objectives: [Objective; M]) -> Self {
        Self {
            objectives,
            capacity: None,
            members: Vec::new(),
        }
    }

    /// An empty archive that holds at most `capacity` members, at least 2.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a capacity below 2.
    pub fn with_capacity(objectives: [Objective; M], capacity: usize) -> Result<Self> {
        if capacity < 2 {
            return Err(Error::InvalidSetting {
                setting: "capacity",
                reason: format!("an archive holds at least 2 members, got {capacity}"),
            });
        }
        Ok(Self {
            capacity: Some(capacity),
            ..Self::new(objectives)
        })
    }

    /// The members, in the order they were added.
    pub fn members(&self) -> &[Individual<G, Scores<M>>] {
        &self.members
    }

    /// The objective values of the valid members.
    pub fn values(&self) -> Vec<[f64; M]> {
        self.members
            .iter()
            .filter_map(|member| member.fitness().and_then(|scores| scores.values()))
            .collect()
    }

    /// The number of members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the archive is empty.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The members, consuming the archive.
    pub fn into_members(self) -> Vec<Individual<G, Scores<M>>> {
        self.members
    }

    /// Adds an evaluated individual unless a member dominates it or has the same scores, and
    /// removes the members it dominates. Returns whether it was added (with a capacity, it can
    /// then be the one that makes room). Individuals that aren't evaluated are ignored.
    pub fn insert(&mut self, individual: &Individual<G, Scores<M>>) -> bool {
        let Some(scores) = individual.fitness() else {
            return false;
        };
        let objectives = &self.objectives;
        // the members are evaluated
        let rejected = self.members.iter().any(|member| {
            let Some(other) = member.fitness() else {
                return false;
            };
            other == scores || dominates(&other, &scores, objectives)
        });
        if rejected {
            return false;
        }
        self.members.retain(|member| {
            let Some(other) = member.fitness() else {
                return true;
            };
            !dominates(&scores, &other, objectives)
        });
        self.members.push(individual.clone());
        if let Some(capacity) = self.capacity {
            while self.members.len() > capacity {
                self.remove_most_crowded();
            }
        }
        true
    }

    /// Inserts every individual evaluated in a generation: the population and the discarded
    /// individuals. Call it from [`MultiEngine::on_generation`](super::MultiEngine::on_generation)
    /// to archive a whole run.
    pub fn update(&mut self, snapshot: &MultiSnapshot<'_, G, M>) {
        for individual in snapshot.population().iter().chain(snapshot.discarded()) {
            self.insert(individual);
        }
    }

    // removes the member with the smallest crowding distance, the earliest on ties
    fn remove_most_crowded(&mut self) {
        let scores: Vec<Scores<M>> = self
            .members
            .iter()
            .map(|member| member.fitness().expect("members are evaluated"))
            .collect();
        let all: Vec<usize> = (0..scores.len()).collect();
        let distances = crowding_distance(&scores, &all);
        let mut most_crowded = 0;
        for (index, distance) in distances.iter().enumerate() {
            if *distance < distances[most_crowded] {
                most_crowded = index;
            }
        }
        self.members.remove(most_crowded);
    }
}

impl<G: Genome, const M: usize> Extend<Individual<G, Scores<M>>> for ParetoArchive<G, M> {
    fn extend<I: IntoIterator<Item = Individual<G, Scores<M>>>>(&mut self, iter: I) {
        for individual in iter {
            self.insert(&individual);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective::{Maximize, Minimize};
    use crate::genome::Integers;
    use proptest::prelude::*;

    fn individual(id: i64, scores: Scores<2>) -> Individual<Integers, Scores<2>> {
        let mut individual = Individual::unevaluated(Integers::from(vec![id]));
        individual.set_fitness(scores);
        individual
    }

    #[test]
    fn keeps_the_non_dominated() {
        let mut archive = ParetoArchive::new([Minimize, Minimize]);
        assert!(archive.insert(&individual(0, Scores::new([2.0, 2.0]))));
        assert!(archive.insert(&individual(1, Scores::new([1.0, 3.0]))));
        // dominated, then equal: rejected
        assert!(!archive.insert(&individual(2, Scores::new([3.0, 3.0]))));
        assert!(!archive.insert(&individual(3, Scores::new([2.0, 2.0]))));
        // dominates the first member
        assert!(archive.insert(&individual(4, Scores::new([1.5, 1.5]))));
        let ids: Vec<i64> = archive.members().iter().map(|m| m.genome()[0]).collect();
        assert_eq!(ids, [1, 4]);
        assert_eq!(archive.values(), [[1.0, 3.0], [1.5, 1.5]]);
        // not evaluated: ignored
        assert!(!archive.insert(&Individual::unevaluated(Integers::from(vec![5]))));
        // feasibility first
        let mut archive = ParetoArchive::new([Maximize, Maximize]);
        archive.insert(&individual(0, Scores::constrained([9.0, 9.0], 1.0)));
        archive.insert(&individual(1, Scores::invalid()));
        assert_eq!(archive.len(), 1);
        archive.insert(&individual(2, Scores::new([0.0, 0.0])));
        assert_eq!(archive.members()[0].genome()[0], 2);
        assert_eq!(archive.len(), 1);
    }

    #[test]
    fn a_full_archive_drops_the_most_crowded() {
        assert!(ParetoArchive::<Integers, 2>::with_capacity([Minimize; 2], 1).is_err());
        let mut archive = ParetoArchive::with_capacity([Minimize, Minimize], 3).unwrap();
        for (id, x) in [0.0, 1.0, 4.0, 1.5].into_iter().enumerate() {
            archive.insert(&individual(id as i64, Scores::new([x, 4.0 - x])));
        }
        // the ends 0 and 4 stay; 1 has the crowding distance (1.5 − 0) / 4 · 2 = 0.75 and 1.5
        // has (4 − 1) / 4 · 2 = 1.5, so 1 makes room
        let ids: Vec<i64> = archive.members().iter().map(|m| m.genome()[0]).collect();
        assert_eq!(ids, [0, 2, 3]);
        let mut unbounded = ParetoArchive::new([Minimize, Minimize]);
        unbounded.extend((0..100).map(|i| individual(i, Scores::new([i as f64, -(i as f64)]))));
        assert_eq!(unbounded.len(), 100);
    }

    proptest! {
        #[test]
        fn members_are_the_non_dominated_unique_scores(
            values in prop::collection::vec(((0..6), (0..6), (0..3)), 0..60),
        ) {
            let objectives = [Minimize, Maximize];
            let all: Vec<Scores<2>> = values
                .iter()
                .map(|&(a, b, violation)| {
                    Scores::constrained([a as f64, b as f64], if violation == 2 { 1.0 } else { 0.0 })
                })
                .collect();
            let mut archive = ParetoArchive::new(objectives);
            archive.extend(all.iter().enumerate().map(|(i, s)| individual(i as i64, *s)));
            let members: Vec<Scores<2>> =
                archive.members().iter().map(|m| m.fitness().unwrap()).collect();
            for (i, a) in members.iter().enumerate() {
                prop_assert!(!all.iter().any(|b| dominates(b, a, &objectives)));
                prop_assert!(!members[..i].contains(a));
            }
            // every non-dominated score is represented
            for a in &all {
                if !all.iter().any(|b| dominates(b, a, &objectives)) {
                    prop_assert!(members.contains(a));
                }
            }
        }
    }
}
