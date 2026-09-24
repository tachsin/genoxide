//! Stop conditions.

use super::Progress;
use crate::{Error, Fitness, Result};
use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

/// When to stop a run: a condition, or a combination with [`or`](Stop::or) and
/// [`and`](Stop::and).
///
/// Conditions are checked after every generation, including the initial population, so a run
/// always evaluates at least the initial population, and may use up to one generation of
/// evaluations more than [`evaluations`](Stop::evaluations) allows.
///
/// ```
/// use genoxide::Stop;
/// use std::time::Duration;
///
/// let stop = Stop::target(0.0)
///     .or(Stop::generations(1_000))
///     .or(Stop::time(Duration::from_secs(10)));
/// ```
#[derive(Clone)]
pub struct Stop {
    condition: Condition,
}

#[derive(Clone)]
enum Condition {
    Target(f64),
    Generations(u64),
    Evaluations(u64),
    Time(Duration),
    Stagnation(u64),
    Custom(Arc<dyn Fn(&Progress) -> bool + Send + Sync>),
    Any(Vec<Stop>),
    All(Vec<Stop>),
}

/// Why a run stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StopReason {
    /// The best fitness reached the target.
    Target,
    /// The number of generations was reached.
    Generations,
    /// The number of evaluations was reached.
    Evaluations,
    /// The time ran out.
    Time,
    /// The best fitness didn't improve for the number of generations.
    Stagnation,
    /// A custom condition was met.
    Custom,
    /// The abort flag was set.
    Aborted,
}

impl Stop {
    /// Stops when the best fitness is at least as good as `score`: at least `score` when
    /// maximizing, at most `score` when minimizing.
    pub fn target(score: f64) -> Self {
        Self::new(Condition::Target(score))
    }

    /// Stops when the algorithm has completed `generations` generations, counted from the start of
    /// the algorithm, not of this run. 0 evaluates only the initial population.
    pub fn generations(generations: u64) -> Self {
        Self::new(Condition::Generations(generations))
    }

    /// Stops when the algorithm has used at least `evaluations` fitness evaluations, counted from
    /// the start of the algorithm.
    pub fn evaluations(evaluations: u64) -> Self {
        Self::new(Condition::Evaluations(evaluations))
    }

    /// Stops when this run has taken at least `time`.
    pub fn time(time: Duration) -> Self {
        Self::new(Condition::Time(time))
    }

    /// Stops when the best fitness hasn't improved for `generations` generations, at least 1.
    pub fn stagnation(generations: u64) -> Self {
        Self::new(Condition::Stagnation(generations))
    }

    /// Stops when `condition` returns true.
    ///
    /// ```
    /// use genoxide::Stop;
    ///
    /// // stop once half of the generations brought no improvement
    /// let stop = Stop::custom(|progress| {
    ///     progress.stagnant_generations() * 2 > progress.generation().max(100)
    /// });
    /// ```
    pub fn custom<F: Fn(&Progress) -> bool + Send + Sync + 'static>(condition: F) -> Self {
        Self::new(Condition::Custom(Arc::new(condition)))
    }

    /// Stops when this condition or `other` is met.
    pub fn or(self, other: Stop) -> Self {
        match self.condition {
            Condition::Any(mut stops) => {
                stops.push(other);
                Self::new(Condition::Any(stops))
            }
            condition => Self::new(Condition::Any(vec![Self::new(condition), other])),
        }
    }

    /// Stops when both this condition and `other` are met.
    pub fn and(self, other: Stop) -> Self {
        match self.condition {
            Condition::All(mut stops) => {
                stops.push(other);
                Self::new(Condition::All(stops))
            }
            condition => Self::new(Condition::All(vec![Self::new(condition), other])),
        }
    }

    fn new(condition: Condition) -> Self {
        Self { condition }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        match &self.condition {
            Condition::Target(score) if score.is_nan() => Err(Error::InvalidSetting {
                setting: "stop_when",
                reason: "the target is NaN".to_string(),
            }),
            Condition::Stagnation(0) => Err(Error::InvalidSetting {
                setting: "stop_when",
                reason: "stagnation must be at least 1 generation".to_string(),
            }),
            Condition::Any(stops) | Condition::All(stops) => {
                stops.iter().try_for_each(Stop::validate)
            }
            _ => Ok(()),
        }
    }

    /// Why to stop now, or `None` to go on. For [`or`](Stop::or), the first condition that is
    /// met; for [`and`](Stop::and), the last one.
    pub fn check(&self, progress: &Progress) -> Option<StopReason> {
        let met = |met: bool, reason| met.then_some(reason);
        match &self.condition {
            Condition::Target(score) => met(
                progress.best().is_some_and(|best| {
                    progress.objective().compare(best, Fitness::new(*score)) != Ordering::Less
                }),
                StopReason::Target,
            ),
            Condition::Generations(generations) => met(
                progress.generation() >= *generations,
                StopReason::Generations,
            ),
            Condition::Evaluations(evaluations) => met(
                progress.evaluations() >= *evaluations,
                StopReason::Evaluations,
            ),
            Condition::Time(time) => met(progress.elapsed() >= *time, StopReason::Time),
            Condition::Stagnation(generations) => met(
                progress.stagnant_generations() >= *generations,
                StopReason::Stagnation,
            ),
            Condition::Custom(condition) => met(condition(progress), StopReason::Custom),
            Condition::Any(stops) => stops.iter().find_map(|stop| stop.check(progress)),
            Condition::All(stops) => stops
                .iter()
                .map(|stop| stop.check(progress))
                .try_fold(None, |_, reason| reason.map(Some))
                .flatten(),
        }
    }

    // whether a condition needs a single objective (a target)
    pub(crate) fn has_target(&self) -> bool {
        match &self.condition {
            Condition::Target(_) => true,
            Condition::Any(stops) | Condition::All(stops) => stops.iter().any(Stop::has_target),
            _ => false,
        }
    }
}

impl fmt::Debug for Stop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.condition {
            Condition::Target(score) => write!(f, "Target({score})"),
            Condition::Generations(generations) => write!(f, "Generations({generations})"),
            Condition::Evaluations(evaluations) => write!(f, "Evaluations({evaluations})"),
            Condition::Time(time) => write!(f, "Time({time:?})"),
            Condition::Stagnation(generations) => write!(f, "Stagnation({generations})"),
            Condition::Custom(_) => write!(f, "Custom"),
            Condition::Any(stops) => f.debug_tuple("Any").field(stops).finish(),
            Condition::All(stops) => f.debug_tuple("All").field(stops).finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective;

    fn progress(generation: u64, best: Option<f64>, objective: Objective) -> Progress {
        Progress {
            generation,
            evaluations: generation * 10,
            elapsed: Duration::from_secs(generation),
            best: best.map(Fitness::new),
            objective,
            best_generation: 0,
        }
    }

    #[test]
    fn target_respects_the_objective() {
        let stop = Stop::target(5.0);
        let reached = |best, objective| stop.check(&progress(1, Some(best), objective));
        assert_eq!(reached(5.0, Objective::Maximize), Some(StopReason::Target));
        assert_eq!(reached(6.0, Objective::Maximize), Some(StopReason::Target));
        assert_eq!(reached(4.0, Objective::Maximize), None);
        assert_eq!(reached(4.0, Objective::Minimize), Some(StopReason::Target));
        assert_eq!(reached(6.0, Objective::Minimize), None);
        assert_eq!(stop.check(&progress(1, None, Objective::Maximize)), None);
    }

    #[test]
    fn conditions() {
        let at = |generation| progress(generation, Some(0.0), Objective::Maximize);
        assert_eq!(
            Stop::generations(3).check(&at(3)),
            Some(StopReason::Generations)
        );
        assert_eq!(Stop::generations(3).check(&at(2)), None);
        assert_eq!(
            Stop::evaluations(30).check(&at(3)),
            Some(StopReason::Evaluations)
        );
        assert_eq!(
            Stop::time(Duration::from_secs(2)).check(&at(2)),
            Some(StopReason::Time)
        );
        assert_eq!(
            Stop::stagnation(4).check(&at(4)),
            Some(StopReason::Stagnation)
        );
        assert_eq!(Stop::stagnation(4).check(&at(3)), None);
        let custom = Stop::custom(|progress| progress.generation() == 7);
        assert_eq!(custom.check(&at(7)), Some(StopReason::Custom));
        assert_eq!(custom.check(&at(8)), None);
    }

    #[test]
    fn combinations() {
        let at = |generation| progress(generation, Some(0.0), Objective::Maximize);
        let any = Stop::generations(5).or(Stop::evaluations(30));
        assert_eq!(any.check(&at(3)), Some(StopReason::Evaluations));
        assert_eq!(any.check(&at(2)), None);
        let all = Stop::generations(5).and(Stop::evaluations(30));
        assert_eq!(all.check(&at(3)), None);
        assert_eq!(all.check(&at(5)), Some(StopReason::Evaluations));
        let nested = Stop::generations(10).or(Stop::generations(2).and(Stop::target(0.0)));
        assert_eq!(nested.check(&at(2)), Some(StopReason::Target));
    }

    #[test]
    fn validation() {
        assert!(Stop::target(f64::NAN).validate().is_err());
        assert!(Stop::stagnation(0).validate().is_err());
        assert!(
            Stop::generations(1)
                .or(Stop::stagnation(0))
                .validate()
                .is_err()
        );
        assert!(Stop::generations(0).validate().is_ok());
        assert_eq!(
            format!("{:?}", Stop::target(1.0).or(Stop::custom(|_| true))),
            "Any([Target(1), Custom])"
        );
    }
}
