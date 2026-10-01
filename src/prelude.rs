//! Everything needed for most programs, in one import.
//!
//! ```
//! use genoxide::prelude::*;
//! ```

pub use crate::algorithm::cmaes::{self, Cmaes};
pub use crate::algorithm::de::{self, De};
pub use crate::algorithm::es::{self, Es};
pub use crate::algorithm::local;
pub use crate::algorithm::nelder_mead::{self, NelderMead};
pub use crate::algorithm::open_es::{self, OpenEs};
pub use crate::algorithm::pso::{self, Pso};
pub use crate::algorithm::{
    Acceptance, Algorithm, Ga, Incremental, Islands, LocalSearch, Migrate, Reevaluate, Scheme,
    SteadyGa,
};
pub use crate::constraint::{self, Penalty};
pub use crate::engine::{
    AsyncEngine, Batch, Engine, Evaluated, FitnessFunction, NanPolicy, Outcome, Stop, StopReason,
};
pub use crate::genome::{
    AdaptiveReal, AdaptiveReals, Binary, Bits, Genome, Integer, Integers, Order, Permutation, Real,
    Reals, Representation,
};
pub use crate::multi::{
    self, Moead, MultiEngine, MultiObjectiveAlgorithm, Nsga2, Nsga3, Scores, SmsEmoa, Spea2,
};
pub use crate::observer::{HallOfFame, Observer, Report, Statistics};
pub use crate::operator::{
    ArithmeticCrossover, BitFlip, BlendCrossover, Crossover, CycleCrossover, DoubleTournament,
    EdgeRecombinationCrossover, GaussianMutation, InsertionMutation, InversionMutation,
    LexicographicTournament, Mutate, NoCrossover, OrderCrossover, PartiallyMappedCrossover,
    PointCrossover, PolynomialMutation, RandomSelection, Rank, Roulette, ScrambleMutation, Select,
    SelfAdaptiveMutation, SimulatedBinaryCrossover, StochasticUniversalSampling, SwapMutation,
    Tarpeian, Tournament, Truncation, UniformCrossover, UniformMutation,
};
pub use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
