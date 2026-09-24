//! Multi-objective optimization: solutions scored on several objectives at once, and the Pareto
//! front of the best trade-offs between them.
//!
//! A solution with [`Scores`] `a` dominates one with `b` if it's at least as good in every
//! objective and better in at least one ([`dominates`], with constraints). The solutions that no
//! other solution dominates form the Pareto front: improving one of their objectives means
//! worsening another. [`non_dominated_sort`] ranks solutions into successive fronts, and
//! [`crowding_distance`] measures how isolated a solution is within its front.

mod pareto;
mod scores;

pub use pareto::{crowding_distance, dominates, non_dominated_sort};
pub use scores::Scores;
