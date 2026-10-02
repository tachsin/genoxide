//! The run file: what to optimize, how, and when to stop.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// A run file.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Run {
    pub genome: Genome,
    pub fitness: Fitness,
    pub algorithm: Algorithm,
    pub stop: Stop,
    #[serde(default)]
    pub report: Report,
    pub checkpoint: Option<Checkpoint>,
}

/// The genome, the search space.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Genome {
    Binary {
        length: usize,
    },
    Integer {
        length: Option<usize>,
        bounds: Bounds<i64>,
    },
    Real {
        length: Option<usize>,
        bounds: Bounds<f64>,
    },
    Permutation {
        length: usize,
    },
}

/// Bounds: one `[low, high]` for every gene (with a length), or one per gene.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Bounds<T> {
    Same([T; 2]),
    PerGene(Vec<[T; 2]>),
}

/// What a bound is, for the error when one isn't.
pub trait Bound {
    const WHAT: &'static str;
}

impl Bound for i64 {
    const WHAT: &'static str = "whole numbers";
}

impl Bound for f64 {
    const WHAT: &'static str = "numbers";
}

// the two forms, and an error that says what they are: serde's own for an untagged enum is
// "data did not match any variant"
impl<'de, T: Deserialize<'de> + Bound> Deserialize<'de> for Bounds<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Form<T> {
            Same([T; 2]),
            PerGene(Vec<[T; 2]>),
        }
        match Form::deserialize(deserializer) {
            Ok(Form::Same(bounds)) => Ok(Bounds::Same(bounds)),
            Ok(Form::PerGene(bounds)) => Ok(Bounds::PerGene(bounds)),
            Err(_) => Err(serde::de::Error::custom(format!(
                "`bounds` is [low, high] for every gene, or a list of [low, high], one per gene, of {}",
                T::WHAT
            ))),
        }
    }
}

impl<T: Copy> Bounds<T> {
    // the bounds of each gene, made as they are read: `Real` and `Integer` reject too many genes
    // before they are collected
    pub fn per_gene(
        &self,
        length: Option<usize>,
    ) -> Result<Box<dyn Iterator<Item = [T; 2]> + '_>, String> {
        match (self, length) {
            (Bounds::Same(bounds), Some(length)) => {
                Ok(Box::new(std::iter::repeat_n(*bounds, length)))
            }
            (Bounds::Same(_), None) => {
                Err("`genome.length` is needed with one pair of bounds for every gene".to_string())
            }
            (Bounds::PerGene(bounds), None) => Ok(Box::new(bounds.iter().copied())),
            (Bounds::PerGene(bounds), Some(length)) if bounds.len() == length => {
                Ok(Box::new(bounds.iter().copied()))
            }
            (Bounds::PerGene(bounds), Some(length)) => Err(format!(
                "`genome.length` is {length}, but there are bounds for {} genes",
                bounds.len()
            )),
        }
    }
}

/// The fitness: a program that reads genomes and writes their fitness, one per line.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fitness {
    /// The program and its arguments.
    pub command: Option<Vec<String>>,
    /// A built-in test function instead of a program, see `genoxide fitness`.
    pub builtin: Option<String>,
    /// Whether to maximize or minimize, one per objective.
    #[serde(default = "maximize")]
    pub objectives: Vec<Objective>,
    /// The number of programs evaluating at the same time.
    pub workers: Option<usize>,
    /// What a NaN from the program means.
    #[serde(default)]
    pub nan: Nan,
    /// The longest wait for an answer; `stop.time` if unset.
    #[serde(default, with = "duration")]
    pub timeout: Option<Duration>,
    /// Whether the program writes the gradient after the value: a value and then a derivative
    /// per gene on each line.
    #[serde(default)]
    pub gradient: bool,
    /// The number of inequality constraints g(x) <= 0 whose values, then Jacobian, the program
    /// writes after the gradient.
    #[serde(default)]
    pub constraints: usize,
}

fn maximize() -> Vec<Objective> {
    vec![Objective::Maximize]
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Objective {
    Maximize,
    Minimize,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Nan {
    #[default]
    Invalid,
    Error,
}

/// The algorithm and its settings.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Algorithm {
    Ga(Ga),
    SteadyGa(Ga),
    De {
        population_size: Option<usize>,
        seed: Option<u64>,
        /// L-SHADE with this many evaluations, instead of the defaults.
        l_shade: Option<u64>,
        #[serde(default, deserialize_with = "de_strategy")]
        strategy: Option<DeStrategy>,
        #[serde(default, deserialize_with = "de_control")]
        control: Option<DeControl>,
        #[serde(default, deserialize_with = "de_restarts")]
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
        #[serde(deserialize_with = "neighbor")]
        neighbor: Mutate,
        neighbors: Option<usize>,
        #[serde(default, deserialize_with = "acceptance")]
        acceptance: Option<Acceptance>,
        /// Iterated local search: `[patience, kicks]`.
        restart: Option<(u64, usize)>,
    },
    Mma {
        seed: Option<u64>,
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
    },
    Lbfgsb {
        seed: Option<u64>,
        /// The correction pairs kept.
        memory: Option<usize>,
        /// Where the gradients come from: `auto` by default.
        gradients: Option<GradientSource>,
        /// The relative step of finite differences.
        difference_step: Option<f64>,
        gradient_tolerance: Option<f64>,
        function_tolerance: Option<f64>,
        max_line_search: Option<usize>,
        /// Random restarts after convergence; none if unset.
        restarts: Option<u64>,
    },
    Bo {
        seed: Option<u64>,
        /// The points of the initial design; 2(n + 1) if unset.
        initial_points: Option<usize>,
        #[serde(default, deserialize_with = "bo_acquisition")]
        acquisition: Option<BoAcquisition>,
        kernel: Option<Kernel>,
        #[serde(default, deserialize_with = "bo_noise")]
        noise: Option<BoNoise>,
        output: Option<BoOutput>,
        raw_samples: Option<usize>,
        acquisition_starts: Option<usize>,
        hyperparameter_starts: Option<usize>,
    },
    NelderMead {
        seed: Option<u64>,
        #[serde(default, deserialize_with = "nelder_mead_coefficients")]
        coefficients: Option<NelderMeadCoefficients>,
        /// The size of the first simplex, as a fraction of each gene's range.
        initial_step: Option<f64>,
        /// The size of the first simplex as a distance, instead.
        initial_step_absolute: Option<f64>,
        /// The simplex size at which a run has converged, as a fraction of the initial step.
        tolerance: Option<f64>,
        /// Random restarts after convergence; none if unset.
        restarts: Option<u64>,
        /// The reflection, expansion and both contractions in one round.
        speculative: Option<bool>,
    },
    FirstOrder {
        seed: Option<u64>,
        /// The step rule; Adam with a learning rate of 0.001 if unset.
        #[serde(default, deserialize_with = "first_order_step")]
        step: Option<FirstOrderStep>,
        /// Where the gradients come from: `auto` by default.
        gradients: Option<GradientSource>,
        /// The relative step of the finite differences.
        difference_step: Option<f64>,
        gradient_tolerance: Option<f64>,
        step_tolerance: Option<f64>,
        /// Random restarts after convergence; none if unset.
        restarts: Option<u64>,
    },
    Nsga2 {
        population_size: usize,
        seed: Option<u64>,
        #[serde(deserialize_with = "crossover")]
        crossover: Crossover,
        #[serde(deserialize_with = "mutate")]
        mutate: Mutate,
        crossover_rate: Option<f64>,
    },
}

/// A genetic algorithm.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Ga {
    pub population_size: usize,
    pub seed: Option<u64>,
    #[serde(deserialize_with = "select")]
    pub select: Select,
    #[serde(deserialize_with = "crossover")]
    pub crossover: Crossover,
    #[serde(deserialize_with = "mutate")]
    pub mutate: Mutate,
    pub crossover_rate: Option<f64>,
    pub mutation_rate: Option<f64>,
    #[serde(default, deserialize_with = "scheme")]
    pub scheme: Option<Scheme>,
}

// The operators' fields, read with their name in the error. `[algorithm]` is an internally tagged
// enum, which serde reads from a buffered copy of the table: an error in an operator's own table
// (`select = { type = "tournament" }`, without its size) can only point at the `[algorithm]`
// header, so the message says which operator it is.
macro_rules! named {
    ($($function:ident: $operator:ty = $field:literal;)*) => {$(
        fn $function<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<$operator, D::Error> {
            <$operator>::deserialize(deserializer)
                .map_err(|error| serde::de::Error::custom(format!("`algorithm.{}`: {error}", $field)))
        }
    )*};
}

named! {
    select: Select = "select";
    crossover: Crossover = "crossover";
    mutate: Mutate = "mutate";
    neighbor: Mutate = "neighbor";
    scheme: Option<Scheme> = "scheme";
    acceptance: Option<Acceptance> = "acceptance";
    de_strategy: Option<DeStrategy> = "strategy";
    de_control: Option<DeControl> = "control";
    de_restarts: Option<DeRestarts> = "restarts";
    nelder_mead_coefficients: Option<NelderMeadCoefficients> = "coefficients";
    bo_acquisition: Option<BoAcquisition> = "acquisition";
    bo_noise: Option<BoNoise> = "noise";
    first_order_step: Option<FirstOrderStep> = "step";
}

/// A first-order method's step rule; Adam's settings default to Kingma and Ba's.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
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

/// How differential evolution builds its mutant vectors: `"rand1"`, `"best1"`,
/// `{ p, archive }` (current-to-pbest/1) or `{ max_p, archive }` (with a random `p` per trial).
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(
    untagged,
    expecting = "\"rand1\", \"best1\", { p, archive } or { max_p, archive }"
)]
pub enum DeStrategy {
    Named(DeStrategyName),
    CurrentToPBest(CurrentToPBest),
    CurrentToPBestRandomP(CurrentToPBestRandomP),
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeStrategyName {
    Rand1,
    Best1,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentToPBest {
    pub p: f64,
    pub archive: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentToPBestRandomP {
    pub max_p: f64,
    pub archive: f64,
}

/// Where differential evolution's `F` and `CR` come from: `{ f, cr }` (fixed),
/// `{ min_f, max_f, cr }` (dither), `{ c }` (JADE) or `{ memory }` (SHADE).
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(
    untagged,
    expecting = "{ f, cr }, { min_f, max_f, cr }, { c } or { memory }"
)]
pub enum DeControl {
    Fixed(Fixed),
    Dither(Dither),
    Jade(Jade),
    Shade(Shade),
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Fixed {
    pub f: f64,
    pub cr: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Dither {
    pub min_f: f64,
    pub max_f: f64,
    pub cr: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Jade {
    pub c: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Shade {
    pub memory: usize,
}

/// When differential evolution starts over: `"never"`, or on stagnation with
/// `{ tolerance, patience }`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(untagged, expecting = "\"never\" or { tolerance, patience }")]
pub enum DeRestarts {
    Named(DeRestartsName),
    OnStagnation(OnStagnation),
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeRestartsName {
    Never,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OnStagnation {
    pub tolerance: f64,
    pub patience: u64,
}

/// Bayesian optimization's acquisition function: `"log-ei"`, `"ei"`, `{ type = "pi", xi }` or
/// `{ type = "ucb", beta }`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(
    untagged,
    expecting = "\"log-ei\", \"ei\", { type = \"pi\", xi } or { type = \"ucb\", beta }"
)]
pub enum BoAcquisition {
    Named(BoAcquisitionName),
    Table(BoAcquisitionTable),
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BoAcquisitionName {
    LogEi,
    Ei,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum BoAcquisitionTable {
    Pi { xi: f64 },
    Ucb { beta: f64 },
}

/// The kernel of Bayesian optimization's Gaussian process.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Kernel {
    Matern52,
    SquaredExponential,
}

/// The noise of Bayesian optimization's Gaussian process: a fixed variance, a fraction of the
/// values' variance, or `{ learned = <least> }`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(
    untagged,
    expecting = "a fixed variance (a number) or { learned = <least variance> }"
)]
pub enum BoNoise {
    Fixed(f64),
    Learned { learned: f64 },
}

/// What Bayesian optimization's model fits: `"standardize"` or `"log"`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BoOutput {
    Standardize,
    Log,
}

/// MMA's method: `"mma"` or `"gcmma"`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MmaMethod {
    Mma,
    Gcmma,
}

/// Where a gradient-based method's gradients come from.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum GradientSource {
    Auto,
    Supplied,
    Forward,
    Central,
}

/// Nelder-Mead's coefficients: `"adaptive"`, `"standard"` or
/// `{ reflection, expansion, contraction, shrink }`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(
    untagged,
    expecting = "\"adaptive\", \"standard\" or { reflection, expansion, contraction, shrink }"
)]
pub enum NelderMeadCoefficients {
    Named(NelderMeadCoefficientsName),
    Custom(CustomCoefficients),
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NelderMeadCoefficientsName {
    Adaptive,
    Standard,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CustomCoefficients {
    pub reflection: f64,
    pub expansion: f64,
    pub contraction: f64,
    pub shrink: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Select {
    Tournament { size: usize },
    Rank { pressure: f64 },
    Roulette {},
    StochasticUniversal {},
    Truncation { fraction: f64 },
    Random {},
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Crossover {
    Uniform {},
    OnePoint {},
    TwoPoint {},
    KPoint { points: usize },
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
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
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
        count: Option<usize>,
    },
    Inversion {},
    Insertion {},
    Scramble {},
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Scheme {
    Generational { elitism: usize },
    SteadyState { replacements: usize },
    MuPlusLambda { lambda: usize },
    MuCommaLambda { lambda: usize },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Restarts {
    Never,
    Ipop,
    Bipop,
    Stop,
}

/// CMA-ES's covariance matrix: full, or diagonal (sep-CMA-ES).
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Covariance {
    Full,
    Diagonal,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
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
    #[serde(default, with = "duration")]
    pub time: Option<Duration>,
    pub stagnation: Option<u64>,
}

/// Progress lines on stderr: every so often (`"2s"`), every so many generations, or `"off"`.
#[derive(Debug, Deserialize)]
#[serde(
    untagged,
    expecting = "a duration like \"2s\", a number of generations, or \"off\""
)]
pub enum Report {
    Generations(u64),
    Text(String),
}

impl Default for Report {
    fn default() -> Self {
        Report::Text("1s".to_string())
    }
}

/// Checkpoints to resume the run with `--resume`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub path: PathBuf,
    pub every: u64,
}

/// Durations as text: `"500ms"`, `"30s"`, `"10m"`, `"2h"`.
pub fn parse_duration(text: &str) -> Result<Duration, String> {
    let text = text.trim();
    let split = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .ok_or_else(|| format!("`{text}` needs a unit: ms, s, m or h"))?;
    let (number, unit) = text.split_at(split);
    let number: f64 = number
        .parse()
        .map_err(|_| format!("`{text}` isn't a duration, e.g. 30s"))?;
    let seconds = match unit.trim() {
        "ms" => number / 1000.0,
        "s" => number,
        "m" | "min" => number * 60.0,
        "h" => number * 3600.0,
        unit => {
            return Err(format!(
                "unknown unit `{unit}` in `{text}`: use ms, s, m or h"
            ));
        }
    };
    Duration::try_from_secs_f64(seconds).map_err(|error| format!("`{text}`: {error}"))
}

mod duration {
    use serde::{Deserialize, Deserializer};
    use std::time::Duration;

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Duration>, D::Error> {
        let text = Option::<String>::deserialize(deserializer)?;
        text.map(|text| super::parse_duration(&text).map_err(serde::de::Error::custom))
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_parse() {
        assert_eq!(parse_duration("500ms"), Ok(Duration::from_millis(500)));
        assert_eq!(parse_duration("1.5s"), Ok(Duration::from_millis(1500)));
        assert_eq!(parse_duration(" 10m "), Ok(Duration::from_secs(600)));
        assert_eq!(parse_duration("2h"), Ok(Duration::from_secs(7200)));
        assert!(parse_duration("10").unwrap_err().contains("needs a unit"));
        assert!(
            parse_duration("5 weeks")
                .unwrap_err()
                .contains("unknown unit")
        );
        assert!(parse_duration("s").is_err());
    }
}
