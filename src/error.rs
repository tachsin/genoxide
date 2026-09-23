//! Errors returned by genoxide.

use std::fmt;

/// Errors returned by genoxide.
///
/// Library code doesn't panic on invalid input: invalid settings and values are reported as an
/// `Error`, before a run starts where possible.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A required setting was not provided.
    MissingSetting {
        /// The name of the setting, e.g. `"population_size"`.
        setting: &'static str,
    },
    /// A setting has an invalid value.
    InvalidSetting {
        /// The name of the setting, e.g. `"crossover_rate"`.
        setting: &'static str,
        /// Why the value is invalid, e.g. `"must be between 0 and 1, got 1.5"`.
        reason: String,
    },
    /// A fitness value is NaN, see [`Fitness::try_new`](crate::Fitness::try_new).
    NanFitness,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::MissingSetting { setting } => write!(f, "missing setting `{setting}`"),
            Error::InvalidSetting { setting, reason } => {
                write!(f, "invalid setting `{setting}`: {reason}")
            }
            Error::NanFitness => write!(f, "fitness value is NaN"),
        }
    }
}

impl std::error::Error for Error {}

/// Shorthand for a [`Result`](std::result::Result) with [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        assert_eq!(
            Error::MissingSetting {
                setting: "population_size"
            }
            .to_string(),
            "missing setting `population_size`"
        );
        assert_eq!(
            Error::InvalidSetting {
                setting: "crossover_rate",
                reason: "must be between 0 and 1, got 1.5".to_string()
            }
            .to_string(),
            "invalid setting `crossover_rate`: must be between 0 and 1, got 1.5"
        );
        assert_eq!(Error::NanFitness.to_string(), "fitness value is NaN");
    }

    #[test]
    fn is_std_error() {
        fn assert_error<E: std::error::Error + Send + Sync + 'static>() {}
        assert_error::<Error>();
    }
}
