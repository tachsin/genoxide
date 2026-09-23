//! Fitness values and the optimization objective.

use crate::error::{Error, Result};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// Whether a higher or a lower fitness score is better.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Objective {
    /// Higher scores are better.
    #[default]
    Maximize,
    /// Lower scores are better.
    Minimize,
}

impl Objective {
    /// Compares two fitness values: [`Ordering::Greater`] if `a` is better than `b`.
    ///
    /// This is a total order. [`Fitness::invalid`] is worse than every score, for both
    /// objectives, and equal to another invalid fitness.
    ///
    /// ```
    /// use genoxide::{Fitness, Objective};
    /// use std::cmp::Ordering;
    ///
    /// let (low, high) = (Fitness::new(1.0), Fitness::new(2.0));
    /// assert_eq!(Objective::Maximize.compare(high, low), Ordering::Greater);
    /// assert_eq!(Objective::Minimize.compare(high, low), Ordering::Less);
    /// assert_eq!(Objective::Minimize.compare(low, Fitness::invalid()), Ordering::Greater);
    /// ```
    pub fn compare(self, a: Fitness, b: Fitness) -> Ordering {
        match (a.score, b.score) {
            (Some(a), Some(b)) => match self {
                Objective::Maximize => a.total_cmp(&b),
                Objective::Minimize => b.total_cmp(&a),
            },
            (Some(_), None) => Ordering::Greater,
            (None, Some(_)) => Ordering::Less,
            (None, None) => Ordering::Equal,
        }
    }

    /// Whether `a` is strictly better than `b`.
    pub fn is_better(self, a: Fitness, b: Fitness) -> bool {
        self.compare(a, b) == Ordering::Greater
    }
}

/// The fitness of a solution: a score, or invalid.
///
/// A score is any `f64` except NaN (infinities are allowed), and `-0.0` is the same as `0.0`.
/// An invalid fitness marks a solution that can't be scored, e.g. one that violates a hard
/// constraint. Invalid is worse than every score, for both objectives, and it's a result of its
/// own, not "not evaluated yet".
///
/// Fitness values are compared through the [`Objective`], so "better" always takes maximizing or
/// minimizing into account.
///
/// ```
/// use genoxide::{Error, Fitness};
///
/// assert_eq!(Fitness::new(1.5).score(), Some(1.5));
/// assert!(!Fitness::new(f64::NAN).is_valid()); // NaN is invalid
/// assert_eq!(Fitness::try_new(f64::NAN), Err(Error::NanFitness)); // or an error
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Fitness {
    // never NaN, never -0.0
    score: Option<f64>,
}

impl Fitness {
    /// A fitness with this score. NaN gives an invalid fitness, see [`try_new`](Fitness::try_new)
    /// to get an error instead.
    pub fn new(score: f64) -> Self {
        Self::try_new(score).unwrap_or(Self::invalid())
    }

    /// A fitness with this score, or [`Error::NanFitness`] for NaN.
    pub fn try_new(score: f64) -> Result<Self> {
        if score.is_nan() {
            Err(Error::NanFitness)
        } else {
            // + 0.0 turns -0.0 into 0.0 and leaves every other value unchanged
            Ok(Self {
                score: Some(score + 0.0),
            })
        }
    }

    /// The fitness of a solution that can't be scored.
    pub fn invalid() -> Self {
        Self { score: None }
    }

    /// The score, or `None` for an invalid fitness.
    pub fn score(self) -> Option<f64> {
        self.score
    }

    /// Whether this fitness has a score.
    pub fn is_valid(self) -> bool {
        self.score.is_some()
    }
}

impl PartialEq for Fitness {
    fn eq(&self, other: &Self) -> bool {
        Objective::Maximize.compare(*self, *other) == Ordering::Equal
    }
}

impl Eq for Fitness {}

impl Hash for Fitness {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.score.map(f64::to_bits).hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn negative_zero_is_zero() {
        assert_eq!(Fitness::new(-0.0), Fitness::new(0.0));
        assert!(Fitness::new(-0.0).score().unwrap().is_sign_positive());
    }

    #[test]
    fn nan() {
        assert_eq!(Fitness::new(f64::NAN), Fitness::invalid());
        assert_eq!(Fitness::try_new(f64::NAN), Err(Error::NanFitness));
    }

    #[test]
    fn infinities_are_scores() {
        let objective = Objective::Maximize;
        assert!(objective.is_better(Fitness::new(f64::INFINITY), Fitness::new(f64::MAX)));
        assert!(objective.is_better(Fitness::new(f64::NEG_INFINITY), Fitness::invalid()));
    }

    fn any_fitness() -> impl Strategy<Value = Fitness> {
        prop_oneof![
            1 => Just(Fitness::invalid()),
            1 => Just(Fitness::new(f64::NAN)),
            1 => Just(Fitness::new(-0.0)),
            1 => Just(Fitness::new(f64::INFINITY)),
            1 => Just(Fitness::new(f64::NEG_INFINITY)),
            10 => any::<f64>().prop_map(Fitness::new),
        ]
    }

    fn any_objective() -> impl Strategy<Value = Objective> {
        prop_oneof![Just(Objective::Maximize), Just(Objective::Minimize)]
    }

    proptest! {
        #[test]
        fn compare_is_a_total_order(
            objective in any_objective(),
            a in any_fitness(),
            b in any_fitness(),
            c in any_fitness(),
        ) {
            // antisymmetric
            prop_assert_eq!(objective.compare(a, b), objective.compare(b, a).reverse());
            // transitive
            if objective.compare(a, b) != Ordering::Less && objective.compare(b, c) != Ordering::Less {
                prop_assert_ne!(objective.compare(a, c), Ordering::Less);
            }
            // consistent with Eq
            prop_assert_eq!(objective.compare(a, b) == Ordering::Equal, a == b);
        }

        #[test]
        fn invalid_is_worst(objective in any_objective(), score in any::<f64>().prop_filter("not NaN", |v| !v.is_nan())) {
            prop_assert!(objective.is_better(Fitness::new(score), Fitness::invalid()));
            prop_assert!(!objective.is_better(Fitness::invalid(), Fitness::new(score)));
        }

        #[test]
        fn minimize_mirrors_maximize(a in any_fitness(), b in any_fitness()) {
            if a.is_valid() && b.is_valid() {
                prop_assert_eq!(
                    Objective::Minimize.compare(a, b),
                    Objective::Maximize.compare(a, b).reverse()
                );
            }
        }

        #[test]
        fn equal_fitness_hashes_equal(a in any_fitness(), b in any_fitness()) {
            use std::collections::hash_map::DefaultHasher;
            if a == b {
                let hash = |f: Fitness| { let mut h = DefaultHasher::new(); f.hash(&mut h); h.finish() };
                prop_assert_eq!(hash(a), hash(b));
            }
        }
    }
}
