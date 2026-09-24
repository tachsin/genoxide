//! # genoxide
//!
//! Evolutionary computation for Rust: genetic algorithms, evolution strategies, multi-objective
//! optimization and more.
//!
//! **Pre-alpha:** see the [roadmap](https://github.com/tachsin/genoxide/blob/main/ROADMAP.md) for
//! what is planned.
//!
//! ```
//! use genoxide::prelude::*;
//!
//! // OneMax: find the genome with the most ones
//! let ga = Ga::builder(Binary::new(100)?)
//!     .population_size(100)
//!     .select(Tournament::new(3)?)
//!     .crossover(UniformCrossover::new())
//!     .mutate(BitFlip::per_gene(0.01)?)
//!     .seed(42)
//!     .build()?;
//! let outcome = Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
//!     .stop_when(Stop::target(100.0).or(Stop::generations(1_000)))
//!     .run()?;
//! println!("best: {:?} after {} generations", outcome.best_fitness(), outcome.generations());
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! The building blocks:
//!
//! - [`Error`]: all errors, no panics in library code
//! - [`StreamRng`]: portable, seedable random numbers with independent streams
//! - [`Fitness`] and [`Objective`]: totally ordered fitness values, with an invalid state and
//!   constraint violations ([`constraint`], Deb's feasibility rules)
//! - [`genome`]: genomes and the spaces they live in, e.g. bit-packed [`Binary`](genome::Binary)
//! - [`Individual`] and [`Population`]: genomes with their fitness and age
//! - [`operator`]: selection, crossover and mutation
//! - [`algorithm`]: algorithms as ask / tell state machines: the genetic algorithm [`Ga`],
//!   [`LocalSearch`](algorithm::LocalSearch) (hill climbing, simulated annealing, tabu search),
//!   evolution strategies [`Es`](algorithm::Es), CMA-ES [`Cmaes`](algorithm::Cmaes), differential
//!   evolution [`De`](algorithm::De) and particle swarm optimization [`Pso`](algorithm::Pso)
//! - [`Engine`]: runs an algorithm with stop conditions, parallel evaluation and cancellation
//! - [`multi`]: multi-objective optimization: NSGA-II, NSGA-III, SPEA2, MOEA/D, SMS-EMOA, the [`MultiEngine`](multi::MultiEngine), Pareto
//!   dominance and non-dominated sorting
//! - [`observer`]: statistics, hall of fame and custom callbacks
//! - [`prelude`]: everything above in one import

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod algorithm;
pub mod constraint;
pub mod engine;
pub mod error;
pub mod fitness;
pub mod genome;
pub mod individual;
mod math;
pub mod multi;
pub mod observer;
pub mod operator;
pub mod population;
pub mod prelude;
pub mod rng;

pub use algorithm::{Algorithm, Ga};
pub use engine::{Engine, Outcome, Stop, StopReason};

pub use error::{Error, Result};
pub use fitness::{Fitness, Objective};
pub use individual::Individual;
pub use population::Population;
pub use rng::StreamRng;

// the Rust code in the README and the guide for AI assistants runs as doctests, so it can't drift
// from the API
#[cfg(all(doctest, feature = "parallel"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

#[cfg(all(doctest, feature = "parallel"))]
#[doc = include_str!("../AGENTS.md")]
struct AgentsDoctests;
