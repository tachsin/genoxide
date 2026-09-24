//! Multi-objective optimization: solutions scored on several objectives at once, and the Pareto
//! front of the best trade-offs between them.
//!
//! A solution with [`Scores`] `a` dominates one with `b` if it's at least as good in every
//! objective and better in at least one ([`dominates`], with constraints). The solutions that no
//! other solution dominates form the Pareto front: improving one of their objectives means
//! worsening another. [`non_dominated_sort`] ranks solutions into successive fronts, and
//! [`crowding_distance`] measures how isolated a solution is within its front, and
//! [`indicator`] measures the quality of a whole front.

mod algorithm;
mod archive;
mod breed;
mod engine;
pub mod indicator;
pub mod nsga2;
pub mod nsga3;
mod pareto;
pub mod problems;
mod scores;

pub use algorithm::MultiObjectiveAlgorithm;
pub use archive::ParetoArchive;
pub use engine::{IntoScores, MultiEngine, MultiFitnessFunction, MultiOutcome, MultiSnapshot};
pub use nsga2::{Nsga2, Nsga2Builder};
pub use nsga3::{Nsga3, Nsga3Builder};
pub use pareto::{crowding_distance, dominates, non_dominated_sort};
pub use problems::das_dennis;
pub use scores::Scores;
