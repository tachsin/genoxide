//! A run as the Python package describes it, in JSON: the genome, the algorithm, the objectives
//! and when to stop.

use serde::Deserialize;

/// A run.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Run {
    pub genome: Genome,
    pub algorithm: Algorithm,
    pub objectives: Vec<Objective>,
    pub stop: Stop,
}

/// The genome, the search space.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Genome {
    Binary { length: usize },
    Integer { bounds: Vec<(i64, i64)> },
    Real { bounds: Vec<(f64, f64)> },
    Permutation { length: usize },
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    Maximize,
    Minimize,
}

/// The algorithm and its settings.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Algorithm {
    Ga(Ga),
    De {
        population_size: Option<usize>,
        seed: Option<u64>,
        /// L-SHADE with this many evaluations, instead of the defaults.
        l_shade: Option<u64>,
    },
    Cmaes {
        population_size: Option<usize>,
        seed: Option<u64>,
        restarts: Option<Restarts>,
        /// The initial step size, as a fraction of each gene's range.
        initial_step: Option<f64>,
    },
    Pso {
        population_size: Option<usize>,
        seed: Option<u64>,
        /// Neighbors on each side in a ring, instead of the whole swarm.
        ring: Option<usize>,
    },
    LocalSearch {
        seed: Option<u64>,
        neighbor: Mutate,
        neighbors: Option<usize>,
        acceptance: Option<Acceptance>,
        /// Iterated local search: `[patience, kicks]`.
        restart: Option<(u64, usize)>,
    },
    Nsga2 {
        population_size: usize,
        seed: Option<u64>,
        crossover: Crossover,
        mutate: Mutate,
        crossover_rate: Option<f64>,
    },
}

/// A genetic algorithm.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ga {
    pub population_size: usize,
    pub seed: Option<u64>,
    pub select: Select,
    pub crossover: Crossover,
    pub mutate: Mutate,
    pub crossover_rate: Option<f64>,
    pub mutation_rate: Option<f64>,
    pub scheme: Option<Scheme>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Select {
    Tournament { size: usize },
    Rank { pressure: f64 },
    Roulette {},
    StochasticUniversalSampling {},
    Truncation { fraction: f64 },
    Random {},
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Crossover {
    Uniform {},
    Point { points: usize },
    None {},
    SimulatedBinary { eta: f64 },
    Blend { alpha: f64 },
    Arithmetic {},
    Order {},
    PartiallyMapped {},
    Cycle {},
    EdgeRecombination {},
}

/// A mutation: `rate` changes each gene with that probability, `count` exactly that many genes.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Mutate {
    BitFlip {
        rate: Option<f64>,
        count: Option<usize>,
    },
    Uniform {
        rate: Option<f64>,
        count: Option<usize>,
    },
    Gaussian {
        rate: Option<f64>,
        count: Option<usize>,
        sigma: f64,
    },
    Polynomial {
        rate: Option<f64>,
        count: Option<usize>,
        eta: f64,
    },
    Swap {
        count: usize,
    },
    Inversion {},
    Insertion {},
    Scramble {},
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Scheme {
    Generational { elitism: usize },
    SteadyState { replacements: usize },
    MuPlusLambda { lambda: usize },
    MuCommaLambda { lambda: usize },
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Restarts {
    Never,
    Ipop,
    Bipop,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Acceptance {
    Improving {},
    NotWorse {},
    Annealing {
        initial_temperature: f64,
        cooling: f64,
    },
    Tabu {
        tenure: usize,
    },
}

/// When to stop: at the first condition met. At least one is needed.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stop {
    pub generations: Option<u64>,
    pub evaluations: Option<u64>,
    pub target: Option<f64>,
    pub seconds: Option<f64>,
    pub stagnation: Option<u64>,
}
