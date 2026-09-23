//! Everything needed for most programs, in one import.
//!
//! ```
//! use genoxide::prelude::*;
//! ```

pub use crate::algorithm::{Acceptance, Algorithm, Ga, LocalSearch, Scheme};
pub use crate::engine::{Engine, FitnessFunction, NanPolicy, Outcome, Stop, StopReason};
pub use crate::genome::{
    Binary, Bits, Genome, Integer, Integers, Order, Permutation, Real, Reals, Representation,
};
pub use crate::observer::{HallOfFame, Observer, Statistics};
pub use crate::operator::{
    ArithmeticCrossover, BitFlip, BlendCrossover, Crossover, CycleCrossover,
    EdgeRecombinationCrossover, GaussianMutation, InsertionMutation, InversionMutation, Mutate,
    NoCrossover, OrderCrossover, PartiallyMappedCrossover, PointCrossover, PolynomialMutation,
    RandomSelection, Rank, Roulette, ScrambleMutation, Select, SimulatedBinaryCrossover,
    StochasticUniversalSampling, SwapMutation, Tournament, Truncation, UniformCrossover,
    UniformMutation,
};
pub use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
