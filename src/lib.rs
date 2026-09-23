//! # genoxide
//!
//! Evolutionary computation for Rust: genetic algorithms, evolution strategies, multi-objective
//! optimization and more.
//!
//! **Pre-alpha:** only the core types exist so far. See the
//! [roadmap](https://github.com/tachsin/genoxide/blob/main/ROADMAP.md) for what is planned.
//!
//! - [`Error`]: all errors, no panics in library code
//! - [`StreamRng`]: portable, seedable random numbers with independent streams
//! - [`Fitness`] and [`Objective`]: totally ordered fitness values, with an invalid state

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod fitness;
pub mod rng;

pub use error::{Error, Result};
pub use fitness::{Fitness, Objective};
pub use rng::StreamRng;
