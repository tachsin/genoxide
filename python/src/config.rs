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
    Binary {
        length: usize,
    },
    Integer {
        bounds: Vec<(i64, i64)>,
    },
    Real {
        bounds: Vec<(f64, f64)>,
    },
    Permutation {
        length: usize,
    },
    /// Real genes with a step size that evolves with them.
    AdaptiveReal {
        bounds: Vec<(f64, f64)>,
        initial_step: f64,
    },
    /// NEAT's networks, of `inputs` inputs (and a bias) and `outputs` outputs.
    Network {
        inputs: usize,
        outputs: usize,
    },
    /// Trees of a genetic program.
    Gp(Box<Gp>),
}

/// Trees of a genetic program: a primitive set of genoxide's built-in primitives, the limits and
/// the initialization.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gp {
    pub primitives: genoxide::gp::PrimitiveSet<crate::trees::Op>,
    pub max_depth: usize,
    pub max_size: usize,
    pub init: Init,
}

/// How random trees are made, with the range of their depths.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Init {
    Full { depths: (usize, usize) },
    Grow { depths: (usize, usize) },
    RampedHalfAndHalf { depths: (usize, usize) },
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
    /// NEAT, on the networks of the `network` genome.
    Neat(Neat),
    De(De),
    Es {
        parents: usize,
        offspring: usize,
        recombination: Option<Recombination>,
        /// The parents per offspring, all of them by default.
        rho: Option<usize>,
        selection: Option<EsSelection>,
        step_sizes: Option<StepSizes>,
        /// The initial step size, as a fraction of each gene's range.
        initial_step: Option<f64>,
        /// Each offspring made on its own random stream, in parallel.
        parallel_breeding: Option<bool>,
        seed: Option<u64>,
    },
    /// Islands of genetic algorithms, or of differential evolutions.
    Islands {
        islands: Vec<Algorithm>,
        topology: Option<Topology>,
        interval: Option<u64>,
        migrants: Option<usize>,
        seed: Option<u64>,
    },
    Cmaes {
        population_size: Option<usize>,
        seed: Option<u64>,
        restarts: Option<Restarts>,
        /// The initial step size, as a fraction of each gene's range.
        initial_step: Option<f64>,
        covariance: Option<Covariance>,
        /// The lower bound of the step size, as a fraction of each gene's range.
        min_step: Option<f64>,
    },
    Pso {
        population_size: Option<usize>,
        seed: Option<u64>,
        /// Neighbors on each side in a ring, instead of the whole swarm.
        ring: Option<usize>,
    },
    /// OpenAI's evolution strategy.
    OpenEs {
        population_size: usize,
        /// The perturbations' standard deviation, as a fraction of each gene's range.
        sigma: Option<f64>,
        optimizer: Option<Optimizer>,
        weight_decay: Option<f64>,
        /// Whether the mean is evaluated each generation too.
        evaluate_mean: Option<bool>,
        initial_mean: Option<Vec<f64>>,
        /// The samples drawn on their own random streams, in parallel.
        parallel_breeding: Option<bool>,
        seed: Option<u64>,
    },
    LocalSearch {
        seed: Option<u64>,
        neighbor: Mutate,
        neighbors: Option<usize>,
        acceptance: Option<Acceptance>,
        /// Iterated local search: `[patience, kicks]`.
        restart: Option<(u64, usize)>,
    },
    /// Gradient descent, momentum, Nesterov, Adam or AdamW.
    FirstOrder {
        step: FirstOrderStep,
        /// Where the gradients come from.
        gradients: Option<GradientSource>,
        /// The relative step of finite differences.
        difference_step: Option<f64>,
        gradient_tolerance: Option<f64>,
        step_tolerance: Option<f64>,
        /// The number of random restarts; none by default.
        restarts: Option<u64>,
        initial_genome: Option<Vec<f64>>,
        seed: Option<u64>,
    },
    /// Bayesian optimization.
    Bo {
        /// The points of the initial design.
        initial_points: Option<usize>,
        /// Genomes evaluated first, in the initial design.
        initial_genomes: Option<Vec<Vec<f64>>>,
        acquisition: Option<Acquisition>,
        kernel: Option<KernelName>,
        noise: Option<NoiseConfig>,
        output: Option<BoOutput>,
        /// The random points at which the acquisition is evaluated before its maximization.
        raw_samples: Option<usize>,
        acquisition_starts: Option<usize>,
        hyperparameter_starts: Option<usize>,
        seed: Option<u64>,
    },
    NelderMead {
        coefficients: Option<NelderMeadCoefficients>,
        /// The size of the first simplex, as a fraction of each gene's range.
        initial_step: Option<f64>,
        /// The size of the first simplex as a distance, instead.
        initial_step_absolute: Option<f64>,
        /// The simplex size at which a run has converged, as a fraction of the initial step.
        tolerance: Option<f64>,
        /// The number of random restarts; none by default.
        restarts: Option<u64>,
        /// The reflection, expansion and both contractions evaluated in one round.
        speculative: Option<bool>,
        initial_genome: Option<Vec<f64>>,
        seed: Option<u64>,
    },
    /// The method of moving asymptotes, or its globally convergent form.
    Mma {
        method: Option<MmaMethod>,
        /// The asymptotes' distance in the first two iterations, a fraction of each range.
        asymptote_initial: Option<f64>,
        /// The factors that bring the asymptotes nearer and move them away.
        asymptote_decrease: Option<f64>,
        asymptote_increase: Option<f64>,
        /// The largest step, a fraction of each range.
        move_limit: Option<f64>,
        /// The cost of the artificial variable that relaxes a constraint.
        constraint_cost: Option<f64>,
        kkt_tolerance: Option<f64>,
        step_tolerance: Option<f64>,
        /// The restoration step after a run that converges to an infeasible point.
        restoration: Option<bool>,
        /// The dual's sums over the genes on several threads.
        parallel_sums: Option<bool>,
        initial_genome: Option<Vec<f64>>,
        seed: Option<u64>,
    },
    Lbfgsb {
        /// The correction pairs kept, m.
        memory: Option<usize>,
        /// Where the gradients come from.
        gradients: Option<GradientSource>,
        /// The relative step of finite differences.
        difference_step: Option<f64>,
        gradient_tolerance: Option<f64>,
        function_tolerance: Option<f64>,
        max_line_search: Option<usize>,
        /// The number of random restarts; none by default.
        restarts: Option<u64>,
        /// Whether a continuation's next stage that keeps the state keeps the pairs.
        keep_pairs: Option<bool>,
        initial_genome: Option<Vec<f64>>,
        seed: Option<u64>,
    },
    /// A gradient method run through stages of one problem, its state kept between them.
    Continuation {
        algorithm: Box<Algorithm>,
        stages: usize,
        /// The most generations of each stage.
        generations: Option<u64>,
        keep: Option<Keep>,
    },
    Nsga2 {
        population_size: usize,
        seed: Option<u64>,
        variation: Variation,
        /// The trees of the initial population, for a `Gp` genome.
        initial_genomes: Option<Vec<genoxide::gp::Tree>>,
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

/// What a continuation's algorithm keeps between stages.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Keep {
    State,
    Point,
}

/// Differential evolution.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct De {
    pub population_size: Option<usize>,
    pub seed: Option<u64>,
    /// L-SHADE with this many evaluations, instead of the defaults.
    pub l_shade: Option<u64>,
    pub strategy: Option<DeStrategy>,
    pub control: Option<DeControl>,
    pub restarts: Option<DeRestarts>,
    /// Each trial built on its own random stream, in parallel.
    pub parallel_breeding: Option<bool>,
}

/// How an evolution strategy combines its parents.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recombination {
    Intermediate,
    Dominant,
}

/// Which individuals of an evolution strategy survive.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EsSelection {
    Comma,
    Plus,
}

/// An evolution strategy's step sizes: one per individual, or one per gene.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepSizes {
    One,
    PerGene,
}

/// How OpenAI's evolution strategy moves its mean along the gradient estimate.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Optimizer {
    Adam {
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
    },
    Sgd {
        learning_rate: f64,
        momentum: f64,
    },
}

/// MMA's method: `"mma"` or `"gcmma"`.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MmaMethod {
    Mma,
    Gcmma,
}

/// A first-order method's step rule; Adam's settings default to Kingma and Ba's.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum FirstOrderStep {
    Gradient {
        learning_rate: f64,
    },
    Momentum {
        learning_rate: f64,
        momentum: f64,
    },
    Nesterov {
        learning_rate: f64,
        momentum: f64,
    },
    Adam {
        learning_rate: Option<f64>,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: Option<f64>,
    },
    Adamw {
        learning_rate: Option<f64>,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: Option<f64>,
        weight_decay: f64,
    },
}

/// Where the migrants of islands go.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Topology {
    Ring,
    FullyConnected,
    Random,
    Isolated,
}

/// How a multi-objective algorithm makes children.
#[derive(Clone, Debug, Deserialize)]
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

/// NEAT, with its settings as `genoxide::neat::NeatBuilder` groups them; the paper's by default.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Neat {
    pub population_size: Option<usize>,
    /// c₁, c₂, c₃ and the threshold.
    pub compatibility: Option<(f64, f64, f64, f64)>,
    /// The rate and the probability of a new weight.
    pub weight_mutation: Option<(f64, f64)>,
    /// The deviations of a perturbation and of a new weight.
    pub weight_deviations: Option<(f64, f64)>,
    /// The probabilities of a new node and of a new connection.
    pub structural_mutation: Option<(f64, f64)>,
    /// Mutation only, interspecies mating, and disabling an inherited disabled gene.
    pub reproduction: Option<(f64, f64, f64)>,
    /// The elitism size and the survival fraction.
    pub selection: Option<(usize, f64)>,
    pub stagnation: Option<u64>,
    pub activation: Option<crate::networks::ActivationName>,
    pub feed_forward: Option<bool>,
    pub initial: Option<NeatInitial>,
    pub sharing: Option<NeatSharing>,
    pub seed: Option<u64>,
}

/// NEAT's initial networks.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeatInitial {
    FullyConnected,
    Unconnected,
}

/// NEAT's fitness sharing.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeatSharing {
    Normalized,
    Raw,
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
    /// Crossover and mutation of each pair of parents on its own random stream, in parallel.
    pub parallel_breeding: Option<bool>,
    /// The trees of the initial population, for a `Gp` genome.
    pub initial_genomes: Option<Vec<genoxide::gp::Tree>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Select {
    Tournament {
        size: usize,
    },
    Rank {
        pressure: f64,
    },
    Roulette {},
    StochasticUniversalSampling {},
    Truncation {
        fraction: f64,
    },
    Random {},
    /// Of equal fitness, the smaller genome wins.
    LexicographicTournament {
        size: usize,
        bucket_ratio: Option<f64>,
    },
    /// A size tournament of two fitness tournaments' winners, or the other way around.
    DoubleTournament {
        fitness_size: usize,
        parsimony: f64,
        size_first: bool,
    },
    /// `select`, with genomes larger than the mean counted as invalid with probability `rate`.
    Tarpeian {
        select: Box<Select>,
        rate: f64,
    },
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Crossover {
    Uniform {},
    Point {
        points: usize,
    },
    None {},
    SimulatedBinary {
        eta: f64,
    },
    Blend {
        alpha: f64,
    },
    Arithmetic {},
    Order {},
    PartiallyMapped {},
    Cycle {},
    EdgeRecombination {},
    /// Trees: subtrees exchanged, at function nodes with probability `internal_rate`.
    Subtree {
        internal_rate: Option<f64>,
    },
    /// Trees: subtrees exchanged at a point of the common region.
    OnePoint {},
}

/// A mutation: `rate` changes each gene with that probability, `count` exactly that many genes.
#[derive(Clone, Debug, Deserialize)]
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
    SelfAdaptive {
        learning_rate: Option<f64>,
        min_step: Option<f64>,
    },
    /// Trees: a subtree replaced by a new one of depth at most `max_depth`.
    Subtree {
        max_depth: Option<usize>,
    },
    /// Trees: nodes replaced by others of their signature, each with probability `rate`, or
    /// `count` of them.
    Point {
        rate: Option<f64>,
        count: Option<usize>,
    },
    /// Trees: the tree replaced by one of its subtrees.
    Hoist {},
    /// Trees: a function's subtree replaced by a leaf.
    Shrink {},
    /// Trees: a constant moved by normal noise.
    Constant {
        sigma: f64,
    },
    /// Trees: one of `mutations`, `(weight, mutation)` pairs, chosen by weight.
    Mutations {
        mutations: Vec<(f64, Mutate)>,
    },
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
    Stop,
}

/// CMA-ES's covariance matrix: full, or diagonal (sep-CMA-ES).
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Covariance {
    Full,
    Diagonal,
}

/// Where a gradient-based method's gradients come from: `"auto"`, `"supplied"`, `"forward"` or
/// `"central"`.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GradientSource {
    Auto,
    Supplied,
    Forward,
    Central,
}

/// An acquisition function of Bayesian optimization.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Acquisition {
    ExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement { xi: f64 },
    UpperConfidenceBound { beta: f64 },
}

/// The kernel of a Gaussian process.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelName {
    Matern52,
    SquaredExponential,
}

/// The noise of a Gaussian process: a fixed variance, or learned from a least one, as fractions
/// of the values' variance.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum NoiseConfig {
    Fixed { variance: f64 },
    Learned { min: f64 },
}

/// What the model of Bayesian optimization fits.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoOutput {
    Standardize,
    Log,
}

/// Nelder-Mead's coefficients: `"adaptive"` (Gao and Han's), `"standard"` (Nelder and Mead's) or
/// `{reflection, expansion, contraction, shrink}`.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(untagged)]
pub enum NelderMeadCoefficients {
    Named(NelderMeadCoefficientsName),
    Custom(CustomCoefficients),
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NelderMeadCoefficientsName {
    Adaptive,
    Standard,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomCoefficients {
    pub reflection: f64,
    pub expansion: f64,
    pub contraction: f64,
    pub shrink: f64,
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
