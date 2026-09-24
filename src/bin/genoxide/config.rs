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
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Bounds<T> {
    Same([T; 2]),
    PerGene(Vec<[T; 2]>),
}

impl<T: Copy> Bounds<T> {
    // the bounds of each gene
    pub fn per_gene(&self, length: Option<usize>) -> Result<Vec<[T; 2]>, String> {
        match (self, length) {
            (Bounds::Same(bounds), Some(length)) => Ok(vec![*bounds; length]),
            (Bounds::Same(_), None) => {
                Err("`genome.length` is needed with one pair of bounds for every gene".to_string())
            }
            (Bounds::PerGene(bounds), None) => Ok(bounds.clone()),
            (Bounds::PerGene(bounds), Some(length)) if bounds.len() == length => Ok(bounds.clone()),
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
#[derive(Debug, Deserialize, Serialize)]
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
#[serde(untagged)]
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
