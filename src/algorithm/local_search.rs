//! Local search: improving a single solution, one step at a time.

use super::ga::Unset;
use super::{Algorithm, Candidates};
use crate::genome::Representation;
use crate::math::exp;
use crate::operator::{Mutate, neighbor};
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;
use std::collections::{HashSet, VecDeque};

/// When a [`LocalSearch`] moves to a neighbor.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Acceptance {
    /// Only to a strictly better neighbor: hill climbing, which stops at the first local optimum.
    Improving,
    /// To a better or equal neighbor (the default): hill climbing that drifts across plateaus.
    NotWorse,
    /// Simulated annealing: to a better or equal neighbor, and to a neighbor that is worse by `Δ`
    /// with probability `exp(-Δ / T)`. The temperature `T` starts at `initial_temperature` and is
    /// multiplied by `cooling` after every step among neighbors; a restart doesn't cool it.
    Annealing {
        /// The starting temperature, positive, in units of fitness: a worse move by this much is
        /// accepted with probability 1/e at the start.
        initial_temperature: f64,
        /// The factor the temperature is multiplied by after every step, greater than 0 and at
        /// most 1, e.g. 0.999.
        cooling: f64,
    },
    /// Tabu search: to the best neighbor that isn't one of the last `tenure` solutions, even if
    /// it's worse, so the search walks out of a local optimum instead of cycling back into it. A
    /// tabu neighbor is still allowed if it's better than the best solution so far (aspiration).
    /// If every neighbor is tabu, the search stays. Use it with several neighbors per step.
    Tabu {
        /// The number of recent solutions that are tabu, at least 1.
        tenure: usize,
    },
}

impl Default for Acceptance {
    /// [`Acceptance::NotWorse`].
    fn default() -> Self {
        Acceptance::NotWorse
    }
}

impl Acceptance {
    fn validate(self) -> Result<()> {
        if let Acceptance::Annealing {
            initial_temperature,
            cooling,
        } = self
        {
            if !(initial_temperature > 0.0 && initial_temperature.is_finite()) {
                return Err(Error::InvalidSetting {
                    setting: "initial_temperature",
                    reason: format!("must be positive and finite, got {initial_temperature}"),
                });
            }
            if !(cooling > 0.0 && cooling <= 1.0) {
                return Err(Error::InvalidSetting {
                    setting: "cooling",
                    reason: format!("must be greater than 0 and at most 1, got {cooling}"),
                });
            }
        }
        if let Acceptance::Tabu { tenure: 0 } = self {
            return Err(Error::InvalidSetting {
                setting: "tenure",
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(())
    }
}

/// Local search, as an ask / tell [`Algorithm`]: hill climbing, simulated annealing, tabu search,
/// and iterated local search on top of any of them.
///
/// It keeps one current solution. Every step, it evaluates `neighbors` random neighbors of it,
/// made by the neighbor operator (any [`Mutate`] of the representation), and moves to the best of
/// them if the [`Acceptance`] allows. With one neighbor per step this is first-improvement hill
/// climbing, with more it's best-of-k. The best solution found is kept whether or not it's the
/// current one.
///
/// With [`restart`](LocalSearchBuilder::restart), it's iterated local search: after `patience`
/// steps without a new best, the search restarts from the best solution changed by `kicks` random
/// neighbor moves. A restart is a step of its own, with a single evaluation.
///
/// Built with [`LocalSearch::builder`]. Run it with an [`Engine`](crate::Engine) like any
/// algorithm; its population is the current solution.
///
/// ```
/// use genoxide::prelude::*;
///
/// // N-Queens: permutations have no conflicts in rows and columns, only on diagonals
/// fn conflicts(order: &Order) -> f64 {
///     let n = order.len();
///     let mut count = 0;
///     for i in 0..n {
///         for j in i + 1..n {
///             count += usize::from(order[i].abs_diff(order[j]) == j - i);
///         }
///     }
///     count as f64
/// }
///
/// let search = LocalSearch::builder(Permutation::new(16)?)
///     .neighbor(SwapMutation::new())
///     .neighbors(4)
///     .minimize()
///     .seed(1)
///     .build()?;
/// let outcome = Engine::new(search, conflicts)
///     .stop_when(Stop::target(0.0).or(Stop::generations(20_000)))
///     .run()?;
/// assert_eq!(outcome.best_fitness(), Fitness::new(0.0));
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "R: serde::Serialize, M: serde::Serialize, R::Genome: serde::Serialize",
        deserialize = "R: serde::Deserialize<'de>, M: serde::Deserialize<'de>, R::Genome: serde::Deserialize<'de>"
    ))
)]
pub struct LocalSearch<R: Representation, M> {
    representation: R,
    neighbor: M,
    neighbors: usize,
    objective: Objective,
    acceptance: Acceptance,
    temperature: f64,
    seed: u64,
    rng: StreamRng,
    // the current solution, as a population of one
    current: Population<R::Genome>,
    candidates: Vec<Individual<R::Genome>>,
    discarded: Vec<Individual<R::Genome>>,
    started: bool,
    asked: bool,
    pending: Vec<usize>,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<R::Genome>>,
    best_generation: u64,
    // the recent solutions of tabu search, in order and for lookups
    tabu: VecDeque<R::Genome>,
    // rebuilt from `tabu` after a checkpoint, so a checkpoint doesn't depend on the hash order
    #[cfg_attr(feature = "serde", serde(skip))]
    tabu_set: HashSet<R::Genome>,
    // iterated local search: (patience, kicks)
    restart: Option<(u64, usize)>,
    restarting: bool,
    last_restart: u64,
    restarts: u64,
}

impl<R: Representation> LocalSearch<R, Unset> {
    /// A builder for a local search on `representation`.
    pub fn builder(representation: R) -> LocalSearchBuilder<R> {
        LocalSearchBuilder {
            representation,
            neighbor: Unset,
            neighbors: 1,
            objective: Objective::default(),
            acceptance: Acceptance::default(),
            seed: None,
            initial_genome: None,
            restart: None,
        }
    }
}

impl<R: Representation, M> LocalSearch<R, M> {
    /// The representation.
    pub fn representation(&self) -> &R {
        &self.representation
    }

    /// The operator that makes neighbors.
    pub fn neighbor(&self) -> &M {
        &self.neighbor
    }

    /// The number of neighbors evaluated per step.
    pub fn neighbors(&self) -> usize {
        self.neighbors
    }

    /// When the search moves to a neighbor.
    pub fn acceptance(&self) -> Acceptance {
        self.acceptance
    }

    /// The current temperature of simulated annealing (0 for the other acceptances).
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The number of restarts of iterated local search so far.
    pub fn restarts(&self) -> u64 {
        self.restarts
    }

    // whether iterated local search restarts now: `patience` steps without a new best, since the
    // last restart
    fn restart_due(&self) -> bool {
        self.restart.is_some_and(|(patience, _)| {
            self.generation - self.best_generation.max(self.last_restart) >= patience
        })
    }

    // records `genome` as the newest tabu solution, forgetting the oldest beyond the tenure
    fn remember(&mut self, genome: &R::Genome) {
        if let Acceptance::Tabu { tenure } = self.acceptance {
            if self.tabu_set.insert(genome.clone()) {
                self.tabu.push_back(genome.clone());
            }
            while self.tabu.len() > tenure {
                if let Some(oldest) = self.tabu.pop_front() {
                    self.tabu_set.remove(&oldest);
                }
            }
        }
    }

    // whether to move from `current` to a neighbor with fitness `candidate`
    fn accepts(&mut self, current: Fitness, candidate: Fitness) -> bool {
        let objective = self.objective;
        match self.acceptance {
            Acceptance::Improving => objective.is_better(candidate, current),
            Acceptance::NotWorse => !objective.is_better(current, candidate),
            // tabu search chooses among the neighbors itself, and always moves
            Acceptance::Tabu { .. } => true,
            Acceptance::Annealing { .. } => {
                if !objective.is_better(current, candidate) {
                    return true;
                }
                let (Some(current_score), Some(candidate_score)) =
                    (current.score(), candidate.score())
                else {
                    // an invalid neighbor is never accepted over a valid solution
                    return false;
                };
                let score_loss = match objective {
                    Objective::Maximize => current_score - candidate_score,
                    Objective::Minimize => candidate_score - current_score,
                };
                let loss = match (current.is_feasible(), candidate.is_feasible()) {
                    (true, true) => score_loss,
                    // an infeasible neighbor is never accepted over a feasible solution
                    (true, false) => return false,
                    // both infeasible: how much the violation grows, or the score on a tie
                    _ if candidate.violation() == current.violation() => score_loss,
                    _ => candidate.violation() - current.violation(),
                };
                self.rng.unit_f64() < exp(-loss / self.temperature)
            }
        }
    }
}

impl<R: Representation, M: Mutate<R>> Algorithm for LocalSearch<R, M> {
    type Genome = R::Genome;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, R::Genome> {
        if !self.asked {
            self.pending.clear();
            if self.started && self.restart_due() {
                // iterated local search: kick the best solution
                let (_, kicks) = self.restart.expect("a restart is due");
                let best = self.best.as_ref().expect("best after the first tell");
                let mut genome = best.genome().clone();
                for _ in 0..kicks {
                    genome = neighbor(&self.neighbor, &self.representation, &genome, &mut self.rng);
                }
                self.candidates.clear();
                self.candidates.push(Individual::new(genome));
                self.pending.push(0);
                self.restarting = true;
            } else if self.started {
                let current = self.current[0].genome();
                self.candidates.clear();
                for _ in 0..self.neighbors {
                    let genome =
                        neighbor(&self.neighbor, &self.representation, current, &mut self.rng);
                    self.candidates.push(Individual::new(genome));
                }
                self.pending.extend(0..self.neighbors);
            } else {
                self.pending.push(0);
            }
            self.asked = true;
        }
        let individuals = if self.started {
            &self.candidates
        } else {
            self.current.as_slice()
        };
        Candidates::new(individuals, &self.pending)
    }

    fn tell(&mut self, fitness: &[Fitness]) -> Result<()> {
        if !self.asked {
            return Err(Error::TellWithoutAsk);
        }
        if fitness.len() != self.pending.len() {
            return Err(Error::FitnessCount {
                expected: self.pending.len(),
                got: fitness.len(),
            });
        }
        self.asked = false;
        self.evaluations += fitness.len() as u64;
        if self.tabu_set.len() != self.tabu.len() {
            self.tabu_set = self.tabu.iter().cloned().collect();
        }
        if !self.started {
            self.current[0].set_fitness(fitness[0]);
            self.best = Some(self.current[0].clone());
            self.started = true;
            let initial = self.current[0].genome().clone();
            self.remember(&initial);
            return Ok(());
        }
        for (candidate, &fitness) in self.candidates.iter_mut().zip(fitness) {
            candidate.set_fitness(fitness);
        }
        self.generation += 1;

        // the best neighbor, the first one on ties
        let objective = self.objective;
        let mut best_neighbor = 0;
        for index in 1..fitness.len() {
            if objective.is_better(fitness[index], fitness[best_neighbor]) {
                best_neighbor = index;
            }
        }
        // both are evaluated: the current solution since the first tell, the best since then too
        let current_fitness = self.current[0].fitness().unwrap_or(Fitness::invalid());
        let best_fitness = self
            .best
            .as_ref()
            .and_then(Individual::fitness)
            .unwrap_or(Fitness::invalid());
        if objective.is_better(fitness[best_neighbor], best_fitness) {
            self.best = Some(self.candidates[best_neighbor].clone());
            self.best_generation = self.generation;
        }

        if self.restarting {
            // the kicked solution is the new current solution, whatever its fitness
            self.restarting = false;
            self.last_restart = self.generation;
            self.restarts += 1;
            let kicked = self.candidates.swap_remove(0);
            self.remember(kicked.genome());
            self.current = Population::new(vec![kicked]);
            self.discarded.clear();
            return Ok(());
        }

        // tabu search moves to the best neighbor that isn't tabu, unless it beats the best so far
        let chosen = if let Acceptance::Tabu { .. } = self.acceptance {
            let allowed = |index: usize| {
                !self.tabu_set.contains(self.candidates[index].genome())
                    || objective.is_better(fitness[index], best_fitness)
            };
            (0..fitness.len())
                .filter(|&index| allowed(index))
                .reduce(|chosen, index| {
                    if objective.is_better(fitness[index], fitness[chosen]) {
                        index
                    } else {
                        chosen
                    }
                })
        } else {
            Some(best_neighbor)
        };
        let accepted = chosen.is_some_and(|chosen| self.accepts(current_fitness, fitness[chosen]));
        let chosen = chosen.unwrap_or(best_neighbor);

        self.discarded.clear();
        let mut candidates = std::mem::take(&mut self.candidates);
        if accepted {
            let next = candidates.swap_remove(chosen);
            self.remember(next.genome());
            self.current = Population::new(vec![next]);
        } else {
            self.current[0].increment_age();
        }
        self.discarded.append(&mut candidates);
        self.candidates = candidates;
        if let Acceptance::Annealing { cooling, .. } = self.acceptance {
            self.temperature *= cooling;
        }
        Ok(())
    }

    fn population(&self) -> &Population<R::Genome> {
        &self.current
    }

    fn best(&self) -> Option<&Individual<R::Genome>> {
        self.best.as_ref()
    }

    fn discarded(&self) -> &[Individual<R::Genome>] {
        &self.discarded
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn evaluations(&self) -> u64 {
        self.evaluations
    }

    fn best_generation(&self) -> u64 {
        self.best_generation
    }
}

/// A builder for a [`LocalSearch`], from [`LocalSearch::builder`].
///
/// The neighbor operator is required: a missing one, or one that doesn't fit the representation,
/// is a compile error. Invalid values are errors from [`build`](LocalSearchBuilder::build).
///
/// Defaults: maximize, 1 neighbor per step, [`Acceptance::NotWorse`], a random initial solution
/// and a random seed.
#[derive(Clone, Debug)]
pub struct LocalSearchBuilder<R: Representation, M = Unset> {
    representation: R,
    neighbor: M,
    neighbors: usize,
    objective: Objective,
    acceptance: Acceptance,
    seed: Option<u64>,
    initial_genome: Option<R::Genome>,
    restart: Option<(u64, usize)>,
}

impl<R: Representation, M> LocalSearchBuilder<R, M> {
    /// The operator that makes a neighbor from the current solution: a mutation of the
    /// representation, e.g. `BitFlip::count(1)` or `InversionMutation`.
    pub fn neighbor<T>(self, neighbor: T) -> LocalSearchBuilder<R, T> {
        LocalSearchBuilder {
            representation: self.representation,
            neighbor,
            neighbors: self.neighbors,
            objective: self.objective,
            acceptance: self.acceptance,
            seed: self.seed,
            initial_genome: self.initial_genome,
            restart: self.restart,
        }
    }

    /// The number of neighbors evaluated per step, at least 1: 1 for first-improvement, more to
    /// move to the best of them. 1 by default.
    pub fn neighbors(mut self, neighbors: usize) -> Self {
        self.neighbors = neighbors;
        self
    }

    /// Whether higher or lower fitness is better. Maximize by default.
    pub fn objective(mut self, objective: Objective) -> Self {
        self.objective = objective;
        self
    }

    /// Higher fitness is better (the default).
    pub fn maximize(self) -> Self {
        self.objective(Objective::Maximize)
    }

    /// Lower fitness is better.
    pub fn minimize(self) -> Self {
        self.objective(Objective::Minimize)
    }

    /// When the search moves to a neighbor. [`Acceptance::NotWorse`] by default.
    pub fn acceptance(mut self, acceptance: Acceptance) -> Self {
        self.acceptance = acceptance;
        self
    }

    /// The seed of the random numbers, for a reproducible run. Random by default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Iterated local search: after `patience` steps (at least 1) without a new best solution, the
    /// search restarts from the best solution, changed by `kicks` (at least 1) random neighbor
    /// moves. Strong enough kicks leave the basin of the local optimum, weak enough ones keep most
    /// of what was found; for a tour with 2-opt, 3 to 10 kicks are common. Off by default.
    pub fn restart(mut self, patience: u64, kicks: usize) -> Self {
        self.restart = Some((patience, kicks));
        self
    }

    /// The solution to start from, e.g. one from a greedy heuristic. Random by default.
    pub fn initial_genome(mut self, genome: R::Genome) -> Self {
        self.initial_genome = Some(genome);
        self
    }

    /// Validates the settings and creates the search, with its initial solution.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidSetting`] for 0 neighbors, invalid annealing settings, a tabu tenure of
    ///   0, or a restart patience or kicks of 0.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<LocalSearch<R, M>>
    where
        M: Mutate<R>,
    {
        if self.neighbors == 0 {
            return Err(Error::InvalidSetting {
                setting: "neighbors",
                reason: "must be at least 1".to_string(),
            });
        }
        self.acceptance.validate()?;
        if let Some((patience, kicks)) = self.restart {
            if patience == 0 || kicks == 0 {
                return Err(Error::InvalidSetting {
                    setting: "restart",
                    reason: format!(
                        "patience and kicks must be at least 1, got {patience} and {kicks}"
                    ),
                });
            }
        }
        if let Some(genome) = &self.initial_genome {
            self.representation.validate(genome)?;
        }
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let mut rng = StreamRng::seed_from_u64(seed);
        let initial = match self.initial_genome {
            Some(genome) => genome,
            None => self.representation.random_genome(&mut rng),
        };
        let temperature = match self.acceptance {
            Acceptance::Annealing {
                initial_temperature,
                ..
            } => initial_temperature,
            _ => 0.0,
        };
        Ok(LocalSearch {
            representation: self.representation,
            neighbor: self.neighbor,
            neighbors: self.neighbors,
            objective: self.objective,
            acceptance: self.acceptance,
            temperature,
            seed,
            rng,
            current: Population::from_genomes([initial]),
            candidates: Vec::new(),
            discarded: Vec::new(),
            started: false,
            asked: false,
            pending: Vec::new(),
            generation: 0,
            evaluations: 0,
            best: None,
            best_generation: 0,
            tabu: VecDeque::new(),
            tabu_set: HashSet::new(),
            restart: self.restart,
            restarting: false,
            last_restart: 0,
            restarts: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{Binary, Bits};
    use crate::operator::BitFlip;
    use crate::{Engine, Stop, StopReason};

    type Search = LocalSearch<Binary, BitFlip>;

    fn search(acceptance: Acceptance, neighbors: usize, seed: u64) -> Search {
        LocalSearch::builder(Binary::new(32).unwrap())
            .neighbor(BitFlip::count(1).unwrap())
            .neighbors(neighbors)
            .acceptance(acceptance)
            .seed(seed)
            .build()
            .unwrap()
    }

    // the current fitness after each of `steps` steps, with `fitness`
    fn trajectory(search: &mut Search, steps: usize, fitness: impl Fn(&Bits) -> f64) -> Vec<f64> {
        let mut scores = Vec::new();
        for _ in 0..=steps {
            let values: Vec<Fitness> = search
                .ask()
                .iter()
                .map(|genome| Fitness::new(fitness(genome)))
                .collect();
            search.tell(&values).unwrap();
            scores.push(search.population()[0].fitness().unwrap().score().unwrap());
        }
        scores
    }

    fn ones(genome: &Bits) -> f64 {
        genome.count_ones() as f64
    }

    #[test]
    fn validation() {
        let builder =
            || LocalSearch::builder(Binary::new(8).unwrap()).neighbor(BitFlip::count(1).unwrap());
        assert!(builder().neighbors(0).build().is_err());
        for (initial_temperature, cooling) in
            [(0.0, 0.9), (f64::INFINITY, 0.9), (1.0, 0.0), (1.0, 1.5)]
        {
            let acceptance = Acceptance::Annealing {
                initial_temperature,
                cooling,
            };
            assert!(
                builder().acceptance(acceptance).build().is_err(),
                "{acceptance:?}"
            );
        }
        assert!(matches!(
            builder().initial_genome(Bits::zeros(7)).build(),
            Err(Error::InvalidGenome { .. })
        ));
        assert!(builder().initial_genome(Bits::zeros(8)).build().is_ok());
    }

    #[test]
    fn ask_tell_protocol() {
        let mut search = search(Acceptance::NotWorse, 3, 0);
        assert_eq!(search.tell(&[]), Err(Error::TellWithoutAsk));
        assert_eq!(search.ask().len(), 1);
        search.tell(&[Fitness::new(0.0)]).unwrap();
        assert_eq!(search.ask().len(), 3);
        assert_eq!(
            search.tell(&[Fitness::new(0.0)]),
            Err(Error::FitnessCount {
                expected: 3,
                got: 1
            })
        );
        search
            .tell(&[Fitness::new(1.0), Fitness::new(2.0), Fitness::new(0.0)])
            .unwrap();
        assert_eq!((search.generation(), search.evaluations()), (1, 4));
        // moved to the best neighbor; the other two are discarded
        assert_eq!(search.population()[0].fitness(), Some(Fitness::new(2.0)));
        assert_eq!(search.discarded().len(), 2);
    }

    #[test]
    fn hill_climbing_solves_one_max() {
        let outcome = Engine::new(search(Acceptance::Improving, 1, 0), ones)
            .stop_when(Stop::target(32.0).or(Stop::generations(10_000)))
            .run()
            .unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Target);
    }

    #[test]
    fn improving_never_moves_to_equal_or_worse() {
        for seed in 0..10 {
            let mut search = search(Acceptance::Improving, 2, seed);
            let scores = trajectory(&mut search, 200, ones);
            assert!(scores.windows(2).all(|pair| pair[1] >= pair[0]));
            // on a plateau, it stays where it is
            let mut search = search_on_plateau(Acceptance::Improving, seed);
            let start = search.population()[0].genome().clone();
            trajectory(&mut search, 20, |_| 1.0);
            assert_eq!(search.population()[0].genome(), &start);
        }
    }

    fn search_on_plateau(acceptance: Acceptance, seed: u64) -> Search {
        let mut search = search(acceptance, 1, seed);
        let _ = search.ask();
        search.tell(&[Fitness::new(1.0)]).unwrap();
        search
    }

    #[test]
    fn not_worse_drifts_across_plateaus() {
        let mut search = search_on_plateau(Acceptance::NotWorse, 0);
        let mut visited = std::collections::HashSet::new();
        trajectory(&mut search, 50, |_| 1.0);
        for _ in 0..50 {
            trajectory(&mut search, 0, |_| 1.0);
            visited.insert(search.population()[0].genome().clone());
        }
        assert!(visited.len() > 10, "{}", visited.len());
    }

    #[test]
    fn annealing_accepts_worse_moves_while_hot() {
        let annealing = |initial_temperature, cooling| Acceptance::Annealing {
            initial_temperature,
            cooling,
        };
        // hot: worse moves are accepted
        let mut hot = search(annealing(100.0, 1.0), 1, 0);
        let scores = trajectory(&mut hot, 200, ones);
        assert!(scores.windows(2).any(|pair| pair[1] < pair[0]));
        // cold: never
        let mut cold = search(annealing(1e-9, 1.0), 1, 0);
        let scores = trajectory(&mut cold, 200, ones);
        assert!(scores.windows(2).all(|pair| pair[1] >= pair[0]));
        // cooling: the temperature after n steps is initial * cooling^n
        let mut cooling = search(annealing(10.0, 0.5), 1, 0);
        trajectory(&mut cooling, 3, ones);
        assert_eq!(cooling.temperature(), 10.0 * 0.5 * 0.5 * 0.5);
    }

    #[test]
    fn annealing_follows_the_feasibility_rules() {
        let hot = Acceptance::Annealing {
            initial_temperature: 1e9,
            cooling: 1.0,
        };
        let mut search = search(hot, 1, 0);
        let (feasible, infeasible) = (Fitness::new(1.0), Fitness::constrained(100.0, 0.1));
        // however hot: never from feasible to infeasible, always the other way
        assert!((0..100).all(|_| !search.accepts(feasible, infeasible)));
        assert!(search.accepts(infeasible, feasible));
        // between infeasible solutions, a larger violation is a loss like a worse score
        let (small, large) = (
            Fitness::constrained(0.0, 0.1),
            Fitness::constrained(0.0, 0.2),
        );
        assert!((0..100).any(|_| search.accepts(small, large)));
        let mut cold = super::tests::search(
            Acceptance::Annealing {
                initial_temperature: 1e-12,
                cooling: 1.0,
            },
            1,
            0,
        );
        assert!((0..100).all(|_| !cold.accepts(small, large)));
    }

    #[test]
    fn best_is_kept_when_the_search_moves_away() {
        let mut search = search(
            Acceptance::Annealing {
                initial_temperature: 1_000.0,
                cooling: 1.0,
            },
            1,
            3,
        );
        let scores = trajectory(&mut search, 300, ones);
        let best = scores.iter().copied().fold(f64::MIN, f64::max);
        assert_eq!(search.best().unwrap().fitness(), Some(Fitness::new(best)));
    }

    #[test]
    fn tabu_and_restart_validation() {
        let builder =
            || LocalSearch::builder(Binary::new(8).unwrap()).neighbor(BitFlip::count(1).unwrap());
        assert!(
            builder()
                .acceptance(Acceptance::Tabu { tenure: 0 })
                .build()
                .is_err()
        );
        assert!(builder().restart(0, 2).build().is_err());
        assert!(builder().restart(5, 0).build().is_err());
        assert!(
            builder()
                .acceptance(Acceptance::Tabu { tenure: 1 })
                .restart(5, 2)
                .build()
                .is_ok()
        );
    }

    #[test]
    fn tabu_never_returns_to_recent_solutions() {
        for seed in 0..5 {
            let tenure = 6;
            let mut search = search(Acceptance::Tabu { tenure }, 4, seed);
            let mut visited: Vec<Bits> = Vec::new();
            let mut best = f64::MIN;
            for step in 0..300 {
                let values: Vec<Fitness> = search
                    .ask()
                    .iter()
                    // a rugged landscape, so the search meets local optima
                    .map(|genome| Fitness::new(((genome.count_ones() * 7) % 11) as f64))
                    .collect();
                search.tell(&values).unwrap();
                let current = search.population()[0].clone();
                let score = current.fitness().unwrap().score().unwrap();
                let recent = &visited[visited.len().saturating_sub(tenure)..];
                if step > 0 && current.genome() != &visited[visited.len() - 1] {
                    // a move: not to a recent solution, unless it beats the best so far
                    assert!(
                        !recent.contains(current.genome()) || score > best,
                        "seed {seed}, step {step}"
                    );
                }
                best = best.max(score);
                visited.push(current.genome().clone());
            }
        }
    }

    #[test]
    fn tabu_moves_on_from_a_local_optimum() {
        // at all ones, every neighbor of OneMax is worse: hill climbing stays, tabu moves on
        let at_optimum = |acceptance| {
            LocalSearch::builder(Binary::new(16).unwrap())
                .neighbor(BitFlip::count(1).unwrap())
                .neighbors(4)
                .acceptance(acceptance)
                .initial_genome(Bits::ones(16))
                .seed(0)
                .build()
                .unwrap()
        };
        let mut hill_climbing = at_optimum(Acceptance::NotWorse);
        let scores = trajectory(&mut hill_climbing, 5, ones);
        assert!(scores.iter().all(|&score| score == 16.0));
        let mut tabu = at_optimum(Acceptance::Tabu { tenure: 3 });
        let scores = trajectory(&mut tabu, 5, ones);
        assert!(scores[1..].iter().any(|&score| score < 16.0), "{scores:?}");
        assert_eq!(tabu.best().unwrap().fitness(), Some(Fitness::new(16.0)));
    }

    #[test]
    fn restarts_kick_the_best_after_patience() {
        let mut search = LocalSearch::builder(Binary::new(32).unwrap())
            .neighbor(BitFlip::count(1).unwrap())
            .acceptance(Acceptance::Improving)
            .restart(5, 3)
            .seed(0)
            .build()
            .unwrap();
        // a flat landscape never improves: a restart every 5 steps, plus the restart step itself
        trajectory(&mut search, 30, |_| 1.0);
        assert_eq!(search.restarts(), 5);
        // the kicked solution is within 3 flips of the best
        let mut search = LocalSearch::builder(Binary::new(32).unwrap())
            .neighbor(BitFlip::count(1).unwrap())
            .acceptance(Acceptance::Improving)
            .restart(1, 3)
            .seed(1)
            .build()
            .unwrap();
        for _ in 0..10 {
            trajectory(&mut search, 0, |_| 1.0);
            let best = search.best().unwrap().genome().clone();
            let current = search.population()[0].genome();
            let distance = best
                .iter()
                .zip(current.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert!(distance <= 3, "{distance}");
        }
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut search = search(Acceptance::NotWorse, 3, seed);
            trajectory(&mut search, 50, ones);
            search.population().clone()
        };
        assert_eq!(run(4), run(4));
        assert_ne!(run(4), run(5));
    }
}
