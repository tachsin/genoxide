//! NEAT, NeuroEvolution of Augmenting Topologies (Stanley and Miikkulainen 2002): networks whose
//! structure evolves with their weights, from minimal networks up.
//!
//! A [`Neat`] run evolves [`Network`]s: node genes and connection genes, each connection with an
//! innovation number that records its history. Crossover aligns genes by these numbers, so
//! networks of different shapes recombine meaningfully. Speciation groups similar networks
//! (by the compatibility distance δ = c₁E/N + c₂D/N + c₃W̄ of excess and disjoint genes and
//! weight differences), and explicit fitness sharing gives each species offspring in proportion to
//! its members' mean fitness, protecting new structure while its weights are tuned. Mutations add
//! connections and nodes; a node splits a connection, keeping the network's function nearly the
//! same.
//!
//! ```
//! use genoxide::neat::{Neat, Network};
//! use genoxide::prelude::*;
//!
//! const CASES: [([f64; 2], f64); 4] =
//!     [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)];
//!
//! // XOR: the paper's fitness, (4 − Σ|error|)², maximized
//! let xor = |network: &Network| {
//!     let mut evaluator = network.feed_forward().expect("feed-forward by construction");
//!     let mut output = [0.0];
//!     let error: f64 = CASES
//!         .iter()
//!         .map(|(input, target)| {
//!             evaluator.activate(input, &mut output);
//!             (output[0] - target).abs()
//!         })
//!         .sum();
//!     (4.0 - error).powi(2)
//! };
//! let neat = Neat::builder(2, 1).population_size(150).seed(1).build()?;
//! let outcome = Engine::new(neat, xor)
//!     .stop_when(Stop::target(15.0).or(Stop::generations(300)))
//!     .run()?;
//! assert_eq!(outcome.stop_reason(), StopReason::Target);
//! assert!(outcome.best_genome().hidden() >= 1); // XOR needs a hidden node
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! The settings default to the paper's (population 150; c₁ = c₂ = 1, c₃ = 0.4, δ_t = 3;
//! weights mutated in 80% of the offspring, 10% of them replaced; new nodes 0.03, new connections
//! 0.05; 25% of the offspring by mutation alone; interspecies mating 0.001; a gene disabled in a
//! parent disabled in the child with probability 0.75; the champion of each species of more than
//! five kept; species without improvement for 15 generations stop reproducing; the steepened
//! sigmoid 1 / (1 + e^(−4.9x))). Where the paper gives no value: new weights normal with
//! deviation 1 and the parents of each species its best 20%, as in neat-python, and weight
//! perturbations normal with deviation 1 (neat-python's 0.5 solved XOR in 97 of 100 runs, 1 in all
//! of them, in genoxide's measurements).
//!
//! genoxide's differences from the paper: fitness sharing works for any objective and invalid
//! fitness ([`Sharing::Normalized`], the default; the paper's form is [`Sharing::Raw`]); the same
//! structural mutation gets the same innovation number for the whole run, not only within a
//! generation (the registry is keyed by structure, as neat-python's); offspring counts are split
//! by the largest-remainder method, so the population size is exact; and the species holding the
//! best network never stagnates. Networks are feed-forward by default
//! ([`NeatBuilder::feed_forward`]).
//!
//! - Stanley, K. O. and Miikkulainen, R. (2002). Evolving neural networks through augmenting
//!   topologies. *Evolutionary Computation* 10(2): 99-127. doi:10.1162/106365602320169811

mod network;

pub use network::{ConnectionGene, FeedForward, Network, NodeGene, NodeKind, Recurrent};

use crate::algorithm::{Algorithm, Candidates, Reevaluate};
use crate::nn::Activation;
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;
use std::collections::BTreeMap;

/// How [`Neat`] shares fitness within species.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Sharing {
    /// Each network's fitness normalized in its generation, (f − worst) / (best − worst) in the
    /// objective's direction, and 0 for an invalid or infeasible one, then divided by its
    /// species' size: for any objective and any values.
    #[default]
    Normalized,
    /// The paper's: the raw score divided by the species' size. It needs a maximized, valid and
    /// non-negative fitness: anything else is [`Error::InvalidFitness`].
    Raw,
}

/// The initial networks of a [`Neat`] run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Initial {
    /// Every input and the bias connected to every output, as in the paper, with random weights.
    #[default]
    FullyConnected,
    /// No connections: structure grows from nothing.
    Unconnected,
}

/// A species of a [`Neat`] generation.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Species {
    id: u64,
    members: Vec<usize>,
    representative: Network,
    best: Option<Fitness>,
    improved: u64,
    created: u64,
}

impl Species {
    /// Its id: species are numbered in the order they appear.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Its members: their positions in [`Neat`]'s population.
    pub fn members(&self) -> &[usize] {
        &self.members
    }

    /// The network new networks are compared with: a random member of the previous generation.
    pub fn representative(&self) -> &Network {
        &self.representative
    }

    /// The best fitness its members ever had.
    pub fn best(&self) -> Option<Fitness> {
        self.best
    }

    /// The generation of its last improvement.
    pub fn improved(&self) -> u64 {
        self.improved
    }

    /// The generation it appeared in.
    pub fn created(&self) -> u64 {
        self.created
    }
}

// the innovation numbers and node ids of a run, keyed by structure
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Registry {
    // (from, to) -> innovation
    #[cfg_attr(feature = "serde", serde(with = "entries"))]
    connections: BTreeMap<(u32, u32), u32>,
    // (split connection's innovation, k) -> node id: the k-th node splitting it in a genome
    #[cfg_attr(feature = "serde", serde(with = "entries"))]
    nodes: BTreeMap<(u32, u32), u32>,
    next_innovation: u32,
    next_node: u32,
}

// a map with pairs as keys, saved as a list of entries: JSON keys are strings
#[cfg(feature = "serde")]
mod entries {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    pub(super) fn serialize<S: Serializer>(
        map: &BTreeMap<(u32, u32), u32>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let entries: Vec<(u32, u32, u32)> = map.iter().map(|(&(a, b), &v)| (a, b, v)).collect();
        entries.serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<(u32, u32), u32>, D::Error> {
        let entries = Vec::<(u32, u32, u32)>::deserialize(deserializer)?;
        Ok(entries.into_iter().map(|(a, b, v)| ((a, b), v)).collect())
    }
}

impl Registry {
    fn connection(&mut self, from: u32, to: u32) -> u32 {
        let next = &mut self.next_innovation;
        *self.connections.entry((from, to)).or_insert_with(|| {
            let innovation = *next;
            *next += 1;
            innovation
        })
    }

    fn node(&mut self, split: u32, k: u32) -> u32 {
        let next = &mut self.next_node;
        *self.nodes.entry((split, k)).or_insert_with(|| {
            let id = *next;
            *next += 1;
            id
        })
    }
}

// the settings of a run
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Settings {
    inputs: u32,
    outputs: u32,
    population_size: usize,
    objective: Objective,
    compatibility: [f64; 3],
    threshold: f64,
    weight_rate: f64,
    replace_rate: f64,
    perturbation: f64,
    initial_deviation: f64,
    add_node: f64,
    add_connection: f64,
    mutation_only: f64,
    interspecies: f64,
    disable: f64,
    elitism_size: usize,
    survival: f64,
    stagnation: u64,
    activation: Activation,
    feed_forward: bool,
    initial: Initial,
    sharing: Sharing,
}

/// NEAT (Stanley and Miikkulainen 2002), from [`Neat::builder`]: see [the module](self).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Neat {
    settings: Settings,
    seed: u64,
    rng: StreamRng,
    registry: Registry,
    population: Population<Network>,
    species: Vec<Species>,
    next_species: u64,
    pending: Vec<usize>,
    started: bool,
    asked: bool,
    // the next ask gives the whole population again, without breeding
    reevaluating: bool,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<Network>>,
    best_generation: u64,
}

impl Neat {
    /// A builder for NEAT on networks of `inputs` inputs (a bias input is added) and `outputs`
    /// outputs, both at least 1.
    pub fn builder(inputs: usize, outputs: usize) -> NeatBuilder {
        NeatBuilder {
            inputs,
            outputs,
            population_size: 150,
            objective: Objective::Maximize,
            compatibility: [1.0, 1.0, 0.4],
            threshold: 3.0,
            weight_rate: 0.8,
            replace_rate: 0.1,
            perturbation: 1.0,
            initial_deviation: 1.0,
            add_node: 0.03,
            add_connection: 0.05,
            mutation_only: 0.25,
            interspecies: 0.001,
            disable: 0.75,
            elitism_size: 5,
            survival: 0.2,
            stagnation: 15,
            activation: Activation::SteepSigmoid,
            feed_forward: true,
            initial: Initial::FullyConnected,
            sharing: Sharing::Normalized,
            seed: None,
        }
    }

    /// The seed of the run.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The species of the current generation, by id: empty before the first
    /// [`tell`](Algorithm::tell).
    pub fn species(&self) -> &[Species] {
        &self.species
    }

    /// The number of innovation numbers given so far: the distinct connections the run has made.
    pub fn innovations(&self) -> u32 {
        self.registry.next_innovation
    }

    /// The compatibility distance δ of two networks: c₁E/N + c₂D/N + c₃W̄, with E and D the
    /// numbers of excess and disjoint connection genes, W̄ the mean weight difference of the
    /// matching ones, and N the number of genes of the larger network, or 1 if both have fewer
    /// than 20.
    pub fn distance(&self, a: &Network, b: &Network) -> f64 {
        distance(a, b, self.settings.compatibility)
    }

    /// Scores the population again at the next [`ask`](Algorithm::ask), for a fitness function
    /// that changed; see [`Reevaluate`].
    ///
    /// # Errors
    ///
    /// [`Error::ReevaluationOutOfTurn`] between an ask and its tell.
    pub fn reevaluate(&mut self) -> Result<()> {
        if self.asked {
            return Err(Error::ReevaluationOutOfTurn);
        }
        if self.started {
            self.reevaluating = true;
        }
        Ok(())
    }

    fn random_weight(&mut self) -> f64 {
        self.settings.initial_deviation * self.rng.normal()
    }

    // the initial networks
    fn initial_population(&mut self) -> Population<Network> {
        let (inputs, outputs) = (self.settings.inputs, self.settings.outputs);
        let template = Network::minimal(inputs, outputs, self.settings.activation);
        self.registry.next_node = inputs + 1 + outputs;
        let mut population = Population::default();
        for _ in 0..self.settings.population_size {
            let mut network = template.clone();
            if self.settings.initial == Initial::FullyConnected {
                for k in 0..outputs {
                    for from in 0..=inputs {
                        let to = inputs + 1 + k;
                        let innovation = self.registry.connection(from, to);
                        let weight = self.random_weight();
                        network.add_connection(ConnectionGene::new(
                            innovation, from, to, weight, true,
                        ));
                    }
                }
            }
            population.push(Individual::unevaluated(network));
        }
        population
    }

    // the network's fitness value for sharing, in [0, 1] normalized, or the raw score
    fn shared_values(&self) -> Result<Vec<f64>> {
        let fitness: Vec<Option<Fitness>> = self.population.iter().map(|i| i.fitness()).collect();
        match self.settings.sharing {
            Sharing::Raw => fitness
                .iter()
                .map(|f| {
                    match f
                        .and_then(|f| f.is_feasible().then_some(f))
                        .and_then(Fitness::score)
                    {
                        Some(score) if score >= 0.0 => Ok(score),
                        _ => Err(Error::InvalidFitness {
                            reason: "raw fitness sharing needs valid, feasible and non-negative \
                                 scores"
                                .to_string(),
                        }),
                    }
                })
                .collect(),
            Sharing::Normalized => {
                let scores: Vec<Option<f64>> = fitness
                    .iter()
                    .map(|f| f.filter(|f| f.is_feasible()).and_then(Fitness::score))
                    .collect();
                let valid = scores.iter().flatten();
                let (low, high) = valid.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &s| {
                    (lo.min(s), hi.max(s))
                });
                let maximize = self.settings.objective == Objective::Maximize;
                Ok(scores
                    .iter()
                    .map(|score| match score {
                        None => 0.0,
                        Some(_) if high <= low => 1.0,
                        Some(s) if maximize => (s - low) / (high - low),
                        Some(s) => (high - s) / (high - low),
                    })
                    .collect())
            }
        }
    }

    // assigns the evaluated generation to species, updates their progress and picks their next
    // representatives
    fn speciate(&mut self) {
        let generation = self.generation;
        let coefficients = self.settings.compatibility;
        let threshold = self.settings.threshold;
        for species in &mut self.species {
            species.members.clear();
        }
        for (index, individual) in self.population.iter().enumerate() {
            let genome = individual.genome();
            let found = self.species.iter_mut().find(|species| {
                distance(genome, &species.representative, coefficients) < threshold
            });
            match found {
                Some(species) => species.members.push(index),
                None => {
                    self.species.push(Species {
                        id: self.next_species,
                        members: vec![index],
                        representative: genome.clone(),
                        best: None,
                        improved: generation,
                        created: generation,
                    });
                    self.next_species += 1;
                }
            }
        }
        self.species.retain(|species| !species.members.is_empty());
        let objective = self.settings.objective;
        for species in &mut self.species {
            let best = species
                .members
                .iter()
                .filter_map(|&i| self.population[i].fitness())
                .reduce(|a, b| if objective.is_better(b, a) { b } else { a });
            if let Some(best) = best
                && species
                    .best
                    .is_none_or(|old| objective.is_better(best, old))
            {
                species.best = Some(best);
                species.improved = generation;
            }
            let k = self.rng.below(species.members.len());
            species.representative = self.population[species.members[k]].genome().clone();
        }
    }

    // the offspring counts of the species, adding up to the population size
    fn offspring_counts(&self, values: &[f64]) -> Vec<usize> {
        let generation = self.generation;
        let objective = self.settings.objective;
        let best_index = self.population.best_index(objective);
        let shares: Vec<f64> = self
            .species
            .iter()
            .map(|species| {
                let holds_best = best_index.is_some_and(|b| species.members.contains(&b));
                let stagnant = generation - species.improved >= self.settings.stagnation;
                if stagnant && !holds_best {
                    return 0.0;
                }
                // the sum of the members' shared fitness: their values divided by the size
                let sum: f64 = species.members.iter().map(|&i| values[i]).sum();
                sum / species.members.len() as f64
            })
            .collect();
        let total: f64 = shares.iter().sum();
        let alive: Vec<bool> = self
            .species
            .iter()
            .map(|species| {
                let holds_best = best_index.is_some_and(|b| species.members.contains(&b));
                holds_best || generation - species.improved < self.settings.stagnation
            })
            .collect();
        // all shares 0: in proportion to the sizes of the species that may reproduce
        let weights: Vec<f64> = if total > 0.0 {
            shares
        } else {
            self.species
                .iter()
                .zip(&alive)
                .map(|(s, &alive)| if alive { s.members.len() as f64 } else { 0.0 })
                .collect()
        };
        largest_remainder(&weights, self.settings.population_size)
    }

    // the next generation: each species' champion (of a species of more than `elitism_size`) and
    // offspring
    fn breed(&mut self) -> Result<()> {
        let values = self.shared_values()?;
        let counts = self.offspring_counts(&values);
        let objective = self.settings.objective;
        // each species' members best first, and its parents: the best `survival` of them
        let mut pools: Vec<Vec<usize>> = Vec::with_capacity(self.species.len());
        for species in &self.species {
            let mut members = species.members.clone();
            members.sort_by(|&a, &b| {
                let (fa, fb) = (self.population[a].fitness(), self.population[b].fitness());
                match (fa, fb) {
                    (Some(fa), Some(fb)) => objective.compare(fb, fa),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                }
                .then(a.cmp(&b))
            });
            pools.push(members);
        }
        let mut next = Population::default();
        for (s, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let members = &pools[s];
            let mut made = 0;
            if members.len() > self.settings.elitism_size {
                next.push(self.population[members[0]].clone());
                made += 1;
            }
            let parents = ((members.len() as f64 * self.settings.survival).ceil() as usize)
                .clamp(1, members.len());
            while made < count {
                let first = members[self.rng.below(parents)];
                let child = if parents == 1 || self.rng.unit_f64() < self.settings.mutation_only {
                    let mut child = self.population[first].genome().clone();
                    self.mutate(&mut child);
                    child
                } else {
                    let second = if self.species.len() > 1
                        && self.rng.unit_f64() < self.settings.interspecies
                    {
                        // another species' parent
                        let mut other = self.rng.below(self.species.len() - 1);
                        if other >= s {
                            other += 1;
                        }
                        let pool = &pools[other];
                        let n = ((pool.len() as f64 * self.settings.survival).ceil() as usize)
                            .clamp(1, pool.len());
                        pool[self.rng.below(n)]
                    } else {
                        let mut k = self.rng.below(parents - 1);
                        let first_k = members.iter().position(|&m| m == first).expect("a parent");
                        if k >= first_k {
                            k += 1;
                        }
                        members[k]
                    };
                    let mut child = self.crossover(first, second);
                    self.mutate(&mut child);
                    child
                };
                next.push(Individual::unevaluated(child));
                made += 1;
            }
        }
        self.population = next;
        Ok(())
    }

    // the child of two parents: genes aligned by innovation, the matching ones from either at
    // random, the disjoint and excess ones from the fitter (the first on ties)
    fn crossover(&mut self, a: usize, b: usize) -> Network {
        let objective = self.settings.objective;
        let (fa, fb) = (self.population[a].fitness(), self.population[b].fitness());
        let b_fitter = match (fa, fb) {
            (Some(fa), Some(fb)) => objective.is_better(fb, fa),
            (None, Some(_)) => true,
            _ => false,
        };
        let (fitter, other) = if b_fitter { (b, a) } else { (a, b) };
        let fitter = self.population[fitter].genome();
        let other = self.population[other].genome();
        let mut connections = Vec::with_capacity(fitter.connections().len());
        let (mut j, others) = (0, other.connections());
        for gene in fitter.connections() {
            while j < others.len() && others[j].innovation() < gene.innovation() {
                j += 1;
            }
            let mut child = *gene;
            if j < others.len() && others[j].innovation() == gene.innovation() {
                let partner = others[j];
                if self.rng.unit_f64() < 0.5 {
                    child = partner;
                }
                let disabled = !gene.is_enabled() || !partner.is_enabled();
                child.set_enabled(!(disabled && self.rng.unit_f64() < self.settings.disable));
            }
            connections.push(child);
        }
        // the fitter parent's nodes: the child's genes are a subset of its structure
        Network::from_genes(
            fitter.inputs() as u32,
            fitter.outputs() as u32,
            fitter.nodes().to_vec(),
            connections,
        )
    }

    // mutates a child: a new node, a new connection, and its weights, each with its probability
    fn mutate(&mut self, network: &mut Network) {
        if self.rng.unit_f64() < self.settings.add_node {
            self.add_node(network);
        }
        if self.rng.unit_f64() < self.settings.add_connection {
            self.add_connection(network);
        }
        if self.rng.unit_f64() < self.settings.weight_rate {
            let (replace, perturbation) = (self.settings.replace_rate, self.settings.perturbation);
            for k in 0..network.connections().len() {
                let weight = if self.rng.unit_f64() < replace {
                    self.random_weight()
                } else {
                    network.connections()[k].weight() + perturbation * self.rng.normal()
                };
                network.connections_mut()[k].set_weight(weight);
            }
        }
    }

    // splits a random enabled connection with a new node: in-weight 1, out-weight the old one
    fn add_node(&mut self, network: &mut Network) {
        let enabled: Vec<usize> = (0..network.connections().len())
            .filter(|&k| network.connections()[k].is_enabled())
            .collect();
        if enabled.is_empty() {
            return;
        }
        let k = enabled[self.rng.below(enabled.len())];
        let old = network.connections()[k];
        network.connections_mut()[k].set_enabled(false);
        // the first node of this split the network doesn't have yet (it may have split this
        // connection before, if the connection was enabled again)
        let mut nth = 0;
        let id = loop {
            let id = self.registry.node(old.innovation(), nth);
            if !network.has_node(id) {
                break id;
            }
            nth += 1;
        };
        network.add_node(NodeGene::new(
            id,
            NodeKind::Hidden,
            self.settings.activation,
        ));
        let into = self.registry.connection(old.from(), id);
        let out = self.registry.connection(id, old.to());
        network.add_connection(ConnectionGene::new(into, old.from(), id, 1.0, true));
        network.add_connection(ConnectionGene::new(out, id, old.to(), old.weight(), true));
    }

    // connects two nodes that aren't connected yet, without closing a cycle in a feed-forward
    // run; up to 20 tries
    fn add_connection(&mut self, network: &mut Network) {
        let nodes = network.nodes().len();
        let first_target = network.inputs() + 1;
        for _ in 0..20 {
            let from = network.nodes()[self.rng.below(nodes)].id();
            let to = network.nodes()[first_target + self.rng.below(nodes - first_target)].id();
            if network.connection_between(from, to).is_some() {
                continue;
            }
            if self.settings.feed_forward && network.reaches(to, from) {
                continue;
            }
            let innovation = self.registry.connection(from, to);
            let weight = self.random_weight();
            network.add_connection(ConnectionGene::new(innovation, from, to, weight, true));
            return;
        }
    }
}

// the compatibility distance of two networks
fn distance(a: &Network, b: &Network, [c1, c2, c3]: [f64; 3]) -> f64 {
    let (a, b) = (a.connections(), b.connections());
    let (mut i, mut j) = (0, 0);
    let (mut disjoint, mut matching, mut difference) = (0usize, 0usize, 0.0);
    while i < a.len() && j < b.len() {
        let (x, y) = (a[i].innovation(), b[j].innovation());
        if x == y {
            matching += 1;
            difference += (a[i].weight() - b[j].weight()).abs();
            i += 1;
            j += 1;
        } else if x < y {
            disjoint += 1;
            i += 1;
        } else {
            disjoint += 1;
            j += 1;
        }
    }
    let excess = (a.len() - i) + (b.len() - j);
    let larger = a.len().max(b.len());
    let n = if larger < 20 { 1.0 } else { larger as f64 };
    let mean = if matching > 0 {
        difference / matching as f64
    } else {
        0.0
    };
    c1 * excess as f64 / n + c2 * disjoint as f64 / n + c3 * mean
}

// `total` split in proportion to `weights` by the largest-remainder method, ties to the earlier
fn largest_remainder(weights: &[f64], total: usize) -> Vec<usize> {
    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 {
        let mut counts = vec![0; weights.len()];
        if let Some(first) = counts.first_mut() {
            *first = total;
        }
        return counts;
    }
    let quotas: Vec<f64> = weights.iter().map(|w| w / sum * total as f64).collect();
    let mut counts: Vec<usize> = quotas.iter().map(|q| q.floor() as usize).collect();
    let assigned: usize = counts.iter().sum();
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|&a, &b| {
        let (ra, rb) = (quotas[a] - quotas[a].floor(), quotas[b] - quotas[b].floor());
        rb.total_cmp(&ra).then(a.cmp(&b))
    });
    for &k in order.iter().cycle().take(total.saturating_sub(assigned)) {
        counts[k] += 1;
    }
    counts
}

impl Reevaluate for Neat {
    /// As [`Neat::reevaluate`]: the next ask gives the population.
    fn reevaluate(&mut self) -> Result<()> {
        Neat::reevaluate(self)
    }
}

impl Algorithm for Neat {
    type Genome = Network;

    fn objective(&self) -> Objective {
        self.settings.objective
    }

    fn ask(&mut self) -> Candidates<'_, Network> {
        if !self.asked {
            self.pending.clear();
            if self.reevaluating {
                self.pending.extend(0..self.population.len());
            } else {
                if self.started {
                    // an error here was reported by the tell that led to it
                    let _ = self.breed();
                }
                let unevaluated = self
                    .population
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| !i.is_evaluated())
                    .map(|(k, _)| k);
                self.pending.extend(unevaluated);
            }
            self.asked = true;
        }
        Candidates::new(self.population.as_slice(), &self.pending)
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
        let reevaluating = self.reevaluating;
        let mut population = self.population.clone();
        for (&k, &f) in self.pending.iter().zip(fitness) {
            population[k].set_fitness(f);
        }
        // raw sharing checks the values before anything changes
        if self.settings.sharing == Sharing::Raw {
            let previous = std::mem::replace(&mut self.population, population);
            if let Err(error) = self.shared_values() {
                self.population = previous;
                return Err(error);
            }
        } else {
            self.population = population;
        }
        self.asked = false;
        self.reevaluating = false;
        self.evaluations += fitness.len() as u64;
        if self.started && !reevaluating {
            self.generation += 1;
        }
        let objective = self.settings.objective;
        if reevaluating {
            // measured by another function: the old best and species' progress don't compare
            self.best = None;
            for species in &mut self.species {
                species.best = None;
                species.improved = self.generation;
            }
        }
        let mut improved = false;
        for individual in self.population.iter() {
            let Some(f) = individual.fitness() else {
                continue;
            };
            let better = self
                .best
                .as_ref()
                .is_none_or(|best| objective.is_better(f, best.fitness().expect("evaluated")));
            if better {
                self.best = Some(individual.clone());
                improved = true;
            }
        }
        if improved {
            self.best_generation = self.generation;
        }
        self.speciate();
        self.started = true;
        Ok(())
    }

    fn population(&self) -> &Population<Network> {
        &self.population
    }

    fn best(&self) -> Option<&Individual<Network>> {
        self.best.as_ref()
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

/// A builder for [`Neat`], from [`Neat::builder`], with the paper's settings by default (see
/// [the module](self)).
#[derive(Clone, Debug)]
pub struct NeatBuilder {
    inputs: usize,
    outputs: usize,
    population_size: usize,
    objective: Objective,
    compatibility: [f64; 3],
    threshold: f64,
    weight_rate: f64,
    replace_rate: f64,
    perturbation: f64,
    initial_deviation: f64,
    add_node: f64,
    add_connection: f64,
    mutation_only: f64,
    interspecies: f64,
    disable: f64,
    elitism_size: usize,
    survival: f64,
    stagnation: u64,
    activation: Activation,
    feed_forward: bool,
    initial: Initial,
    sharing: Sharing,
    seed: Option<u64>,
}

impl NeatBuilder {
    /// The number of networks, at least 1 and at most 2^24: 150 by default, the paper's.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Higher fitness is better (the default).
    pub fn maximize(mut self) -> Self {
        self.objective = Objective::Maximize;
        self
    }

    /// Lower fitness is better.
    pub fn minimize(mut self) -> Self {
        self.objective = Objective::Minimize;
        self
    }

    /// The compatibility coefficients c₁ (excess genes), c₂ (disjoint genes) and c₃ (the mean
    /// weight difference), each at least 0, and the threshold δ_t, positive: 1, 1, 0.4 and 3 by
    /// default, the paper's (it used c₃ = 3 and δ_t = 4 with a population of 1000).
    pub fn compatibility(mut self, c1: f64, c2: f64, c3: f64, threshold: f64) -> Self {
        self.compatibility = [c1, c2, c3];
        self.threshold = threshold;
        self
    }

    /// The probability that a child's weights are mutated (0.8), and that each of them is then
    /// replaced by a new random weight rather than perturbed (0.1), both in [0, 1].
    pub fn weight_mutation(mut self, rate: f64, replace: f64) -> Self {
        self.weight_rate = rate;
        self.replace_rate = replace;
        self
    }

    /// The deviation of the normal perturbation of a weight (1), and of a new weight (1), both
    /// positive and finite.
    pub fn weight_deviations(mut self, perturbation: f64, new: f64) -> Self {
        self.perturbation = perturbation;
        self.initial_deviation = new;
        self
    }

    /// The probabilities that a child gets a new node (0.03) and a new connection (0.05), in
    /// [0, 1]; the paper used 0.3 for new connections in its population of 1000.
    pub fn structural_mutation(mut self, add_node: f64, add_connection: f64) -> Self {
        self.add_node = add_node;
        self.add_connection = add_connection;
        self
    }

    /// The fraction of offspring made by mutation alone (0.25), the probability that a crossover
    /// takes its second parent from another species (0.001), and that a gene disabled in either
    /// parent is disabled in the child (0.75), each in [0, 1].
    pub fn reproduction(mut self, mutation_only: f64, interspecies: f64, disable: f64) -> Self {
        self.mutation_only = mutation_only;
        self.interspecies = interspecies;
        self.disable = disable;
        self
    }

    /// The size a species must exceed for its champion to be copied unchanged (5), and the
    /// fraction of each species, its best, that breeds (0.2, in (0, 1]).
    pub fn selection(mut self, elitism_size: usize, survival: f64) -> Self {
        self.elitism_size = elitism_size;
        self.survival = survival;
        self
    }

    /// The generations without improvement after which a species stops reproducing, at least 1:
    /// 15, the paper's. The species with the best network is never stopped.
    pub fn stagnation(mut self, generations: u64) -> Self {
        self.stagnation = generations;
        self
    }

    /// The activation of the outputs and the hidden nodes: the paper's steepened sigmoid by
    /// default.
    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Whether networks stay feed-forward (the default): a new connection never closes a cycle,
    /// so every network has a [`Network::feed_forward`] evaluator. `false` allows recurrent
    /// connections and loops.
    pub fn feed_forward(mut self, feed_forward: bool) -> Self {
        self.feed_forward = feed_forward;
        self
    }

    /// The initial networks: fully connected (the default) or unconnected.
    pub fn initial(mut self, initial: Initial) -> Self {
        self.initial = initial;
        self
    }

    /// How fitness is shared: normalized (the default) or raw, the paper's.
    pub fn sharing(mut self, sharing: Sharing) -> Self {
        self.sharing = sharing;
        self
    }

    /// The seed of the random number generator; random by default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// The algorithm.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a setting outside its range, naming it, or
    /// [`Sharing::Raw`] with minimization.
    pub fn build(self) -> Result<Neat> {
        let invalid =
            |setting: &'static str, reason: String| Err(Error::InvalidSetting { setting, reason });
        let limit = 1 << 24;
        if !(1..=limit).contains(&self.inputs) || !(1..=limit).contains(&self.outputs) {
            return invalid(
                "neat",
                format!(
                    "inputs and outputs must be 1 to 2^24, got {} and {}",
                    self.inputs, self.outputs
                ),
            );
        }
        if !(1..=limit).contains(&self.population_size) {
            return invalid(
                "population_size",
                format!("must be 1 to 2^24, got {}", self.population_size),
            );
        }
        let [c1, c2, c3] = self.compatibility;
        if ![c1, c2, c3].iter().all(|c| *c >= 0.0 && c.is_finite())
            || !(self.threshold > 0.0 && self.threshold.is_finite())
        {
            return invalid(
                "compatibility",
                format!(
                    "the coefficients must be at least 0 and the threshold positive, all \
                     finite, got {c1}, {c2}, {c3} and {}",
                    self.threshold
                ),
            );
        }
        let probabilities = [
            ("weight_mutation", self.weight_rate),
            ("weight_mutation", self.replace_rate),
            ("structural_mutation", self.add_node),
            ("structural_mutation", self.add_connection),
            ("reproduction", self.mutation_only),
            ("reproduction", self.interspecies),
            ("reproduction", self.disable),
        ];
        for (setting, p) in probabilities {
            if !(0.0..=1.0).contains(&p) {
                return invalid(setting, format!("a probability must be in [0, 1], got {p}"));
            }
        }
        for deviation in [self.perturbation, self.initial_deviation] {
            if !(deviation > 0.0 && deviation.is_finite()) {
                return invalid(
                    "weight_deviations",
                    format!("must be positive and finite, got {deviation}"),
                );
            }
        }
        if !(self.survival > 0.0 && self.survival <= 1.0) {
            return invalid(
                "selection",
                format!(
                    "the survival fraction must be in (0, 1], got {}",
                    self.survival
                ),
            );
        }
        if self.stagnation == 0 {
            return invalid("stagnation", "must be at least 1".to_string());
        }
        if self.sharing == Sharing::Raw && self.objective == Objective::Minimize {
            return invalid(
                "sharing",
                "raw fitness sharing needs maximization".to_string(),
            );
        }
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let settings = Settings {
            inputs: self.inputs as u32,
            outputs: self.outputs as u32,
            population_size: self.population_size,
            objective: self.objective,
            compatibility: self.compatibility,
            threshold: self.threshold,
            weight_rate: self.weight_rate,
            replace_rate: self.replace_rate,
            perturbation: self.perturbation,
            initial_deviation: self.initial_deviation,
            add_node: self.add_node,
            add_connection: self.add_connection,
            mutation_only: self.mutation_only,
            interspecies: self.interspecies,
            disable: self.disable,
            elitism_size: self.elitism_size,
            survival: self.survival,
            stagnation: self.stagnation,
            activation: self.activation,
            feed_forward: self.feed_forward,
            initial: self.initial,
            sharing: self.sharing,
        };
        let mut neat = Neat {
            settings,
            seed,
            rng: StreamRng::seed_from_u64(seed),
            registry: Registry::default(),
            population: Population::default(),
            species: Vec::new(),
            next_species: 0,
            pending: Vec::new(),
            started: false,
            asked: false,
            reevaluating: false,
            generation: 0,
            evaluations: 0,
            best: None,
            best_generation: 0,
        };
        neat.population = neat.initial_population();
        Ok(neat)
    }
}
