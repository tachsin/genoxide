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
        strategy: Option<DeStrategy>,
        control: Option<DeControl>,
        restarts: Option<DeRestarts>,
    },
    Cmaes {
        population_size: Option<usize>,
        seed: Option<u64>,
        restarts: Option<Restarts>,
        /// The initial step size, as a fraction of each gene's range.
        initial_step: Option<f64>,
        covariance: Option<Covariance>,
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
        variation: Variation,
    },
    Nsga3 {
        /// A direction per row, a value per objective.
        reference_directions: Vec<Vec<f64>>,
        population_size: Option<usize>,
        seed: Option<u64>,
        variation: Variation,
    },
    Spea2 {
        population_size: usize,
        seed: Option<u64>,
        variation: Variation,
    },
    Moead {
        /// A weight vector per row, a value per objective.
        weights: Vec<Vec<f64>>,
        neighbors: Option<usize>,
        neighbor_mating: Option<f64>,
        max_replacements: Option<usize>,
        decomposition: Option<Decomposition>,
        seed: Option<u64>,
        variation: Variation,
    },
    SmsEmoa {
        population_size: usize,
        offspring: Option<usize>,
        seed: Option<u64>,
        variation: Variation,
    },
}

/// How a multi-objective algorithm makes children.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variation {
    pub crossover: Crossover,
    pub mutate: Mutate,
    pub crossover_rate: Option<f64>,
    pub mutation_rate: Option<f64>,
    /// Whether a child equal to a member of the population or an earlier child is bred again.
    pub eliminate_duplicates: Option<bool>,
}

/// How MOEA/D turns the objectives into one value per weight vector.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Decomposition {
    Tchebycheff {},
    Pbi { theta: f64 },
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

/// CMA-ES's covariance matrix: full, or diagonal (sep-CMA-ES).
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Covariance {
    Full,
    Diagonal,
}

/// How differential evolution builds its mutant vectors: `"rand1"`, `"best1"`, `{p, archive}`
/// (current-to-pbest/1) or `{max_p, archive}` (with a random `p` per trial).
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(untagged)]
pub enum DeStrategy {
    Named(DeStrategyName),
    CurrentToPBest(CurrentToPBest),
    CurrentToPBestRandomP(CurrentToPBestRandomP),
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeStrategyName {
    Rand1,
    Best1,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentToPBest {
    pub p: f64,
    pub archive: f64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentToPBestRandomP {
    pub max_p: f64,
    pub archive: f64,
}

/// Where differential evolution's `F` and `CR` come from: `{f, cr}` (fixed), `{min_f, max_f, cr}`
/// (dither), `{c}` (JADE) or `{memory}` (SHADE).
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(untagged)]
pub enum DeControl {
    Fixed(Fixed),
    Dither(Dither),
    Jade(Jade),
    Shade(Shade),
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fixed {
    pub f: f64,
    pub cr: f64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dither {
    pub min_f: f64,
    pub max_f: f64,
    pub cr: f64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Jade {
    pub c: f64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shade {
    pub memory: usize,
}

/// When differential evolution starts over: `"never"`, or on stagnation with
/// `{tolerance, patience}`.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(untagged)]
pub enum DeRestarts {
    Named(DeRestartsName),
    OnStagnation(OnStagnation),
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeRestartsName {
    Never,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnStagnation {
    pub tolerance: f64,
    pub patience: u64,
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
