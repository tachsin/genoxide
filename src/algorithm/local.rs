//! Settings shared by the local methods, which improve one point until it converges:
//! [`NelderMead`](super::NelderMead), [`Lbfgsb`](super::Lbfgsb) and
//! [`FirstOrder`](super::FirstOrder).

use crate::{Error, Result};

/// Whether a local method starts again when it has converged.
///
/// A local method converges to the minimum of the basin it starts in, and stops there
/// ([`StopReason::Converged`](crate::StopReason::Converged)). Restarts from other points find other
/// minima; the best of all the runs is the result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Restarts {
    /// No restarts: the method finishes when it first converges (the default).
    #[default]
    Never,
    /// Up to `times` restarts, each from a new random point in the bounds: for multimodal
    /// functions, or to make sure of a minimum.
    Random {
        /// The number of restarts, at least 1.
        times: u64,
    },
}

impl Restarts {
    pub(crate) fn validate(self) -> Result<()> {
        if let Restarts::Random { times: 0 } = self {
            return Err(Error::InvalidSetting {
                setting: "restarts",
                reason: "times must be at least 1 (`Restarts::Never` for none)".to_string(),
            });
        }
        Ok(())
    }

    // the number of restarts allowed
    pub(crate) fn times(self) -> u64 {
        match self {
            Restarts::Never => 0,
            Restarts::Random { times } => times,
        }
    }
}
