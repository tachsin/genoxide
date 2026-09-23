//! Everything needed for most programs, in one import.
//!
//! ```
//! use genoxide::prelude::*;
//! ```

pub use crate::algorithm::{Algorithm, Ga, Scheme};
pub use crate::engine::{Engine, FitnessFunction, NanPolicy, Outcome, Stop, StopReason};
pub use crate::genome::{
    Binary, Bits, Genome, Integer, Integers, Order, Permutation, Real, Reals, Representation,
};
pub use crate::observer::{HallOfFame, Observer, Statistics};
pub use crate::operator::{
    BitFlip, Crossover, GaussianMutation, Mutate, NoCrossover, PointCrossover, PolynomialMutation,
    RandomSelection, Rank, Roulette, Select, StochasticUniversalSampling, SwapMutation, Tournament,
    Truncation, UniformCrossover, UniformMutation,
};
pub use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
