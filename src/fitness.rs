//! Fitness values and the optimization objective.

use crate::error::{Error, Result};
use std::cmp::Ordering;
use std::fmt::{self, Write};
use std::hash::{Hash, Hasher};

/// Whether a higher or a lower fitness score is better.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// This is a total order, with Deb's feasibility rules for [constrained](Fitness::constrained)
    /// fitness values:
    ///
    /// 1. [`Fitness::invalid`] is worse than everything else, for both objectives, and equal to
    ///    another invalid fitness.
    /// 2. A feasible fitness (no constraint violation) is better than an infeasible one.
    /// 3. Between feasible fitness values, the better score wins.
    /// 4. Between infeasible ones, the smaller violation wins; the score breaks ties.
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
    #[inline]
    pub fn compare(self, a: Fitness, b: Fitness) -> Ordering {
        // invalid (a NaN score) is the worst
        match (a.score.is_nan(), b.score.is_nan()) {
            (false, false) => {}
            (true, true) => return Ordering::Equal,
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
        }
        // a smaller violation is better, and feasible (0) beats every infeasible fitness
        if a.violation != b.violation {
            return b.violation.total_cmp(&a.violation);
        }
        match self {
            Objective::Maximize => a.score.total_cmp(&b.score),
            Objective::Minimize => b.score.total_cmp(&a.score),
        }
    }

    /// Whether `a` is strictly better than `b`.
    #[inline]
    pub fn is_better(self, a: Fitness, b: Fitness) -> bool {
        self.compare(a, b) == Ordering::Greater
    }
}

/// The fitness of a solution: a score, optionally with a constraint violation, or invalid.
///
/// A score is any `f64` except NaN (infinities are allowed), and `-0.0` is the same as `0.0`.
///
/// - A [constrained](Fitness::constrained) fitness also has a constraint violation: 0 for a
///   feasible solution, and how far it is from feasible otherwise. Feasible solutions are better
///   than infeasible ones, and infeasible ones are ranked by their violation (Deb's feasibility
///   rules, see [`Objective::compare`]), so the search is guided towards the feasible region.
/// - An invalid fitness marks a solution that can't be scored at all. Invalid is worse than every
///   other fitness, for both objectives, and it's a result of its own, not "not evaluated yet".
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
#[derive(Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Fitness {
    // f64::NAN for an invalid fitness, otherwise never NaN and never -0.0; with the violation, the
    // fitness is 16 bytes, like a score alone, which keeps comparisons and copies cheap
    score: f64,
    // 0 when feasible or invalid, otherwise positive (possibly infinite); never NaN or -0.0
    violation: f64,
}

impl Fitness {
    /// A fitness with this score. NaN gives an invalid fitness, see [`try_new`](Fitness::try_new)
    /// to get an error instead.
    pub fn new(score: f64) -> Self {
        Self::try_new(score).unwrap_or(Self::invalid())
    }

    /// A fitness with this score.
    ///
    /// # Errors
    ///
    /// [`Error::NanFitness`] for NaN.
    pub fn try_new(score: f64) -> Result<Self> {
        if score.is_nan() {
            Err(Error::NanFitness)
        } else {
            // + 0.0 turns -0.0 into 0.0 and leaves every other value unchanged
            Ok(Self {
                score: score + 0.0,
                violation: 0.0,
            })
        }
    }

    /// A fitness with this score and constraint violation: 0 for a feasible solution, positive
    /// for how far it is from feasible (e.g. the sum of the amounts by which its constraints are
    /// exceeded, see [`constraint`](crate::constraint)).
    ///
    /// A NaN score or violation, or a negative violation, gives an invalid fitness; see
    /// [`try_constrained`](Fitness::try_constrained) to get an error instead.
    ///
    /// ```
    /// use genoxide::{Fitness, Objective};
    ///
    /// let feasible = Fitness::constrained(10.0, 0.0);
    /// let slightly_off = Fitness::constrained(50.0, 0.1);
    /// let far_off = Fitness::constrained(90.0, 3.0);
    /// // feasible first, then by violation, whatever the scores
    /// assert!(Objective::Maximize.is_better(feasible, slightly_off));
    /// assert!(Objective::Maximize.is_better(slightly_off, far_off));
    /// assert_eq!(slightly_off.to_string(), "50 (violation 0.1)");
    /// ```
    pub fn constrained(score: f64, violation: f64) -> Self {
        Self::try_constrained(score, violation).unwrap_or(Self::invalid())
    }

    /// A fitness with this score and constraint violation.
    ///
    /// # Errors
    ///
    /// [`Error::NanFitness`] for a NaN score or violation, and [`Error::InvalidFitness`] for a
    /// negative violation.
    pub fn try_constrained(score: f64, violation: f64) -> Result<Self> {
        if violation.is_nan() {
            return Err(Error::NanFitness);
        }
        if violation < 0.0 {
            return Err(Error::InvalidFitness {
                reason: format!("a constraint violation can't be negative, got {violation}"),
            });
        }
        let fitness = Self::try_new(score)?;
        Ok(Self {
            // + 0.0 turns -0.0 into 0.0
            violation: violation + 0.0,
            ..fitness
        })
    }

    /// The fitness of a solution that can't be scored.
    pub fn invalid() -> Self {
        Self {
            score: f64::NAN,
            violation: 0.0,
        }
    }

    /// The score, or `None` for an invalid fitness.
    pub fn score(self) -> Option<f64> {
        (!self.score.is_nan()).then_some(self.score)
    }

    /// Whether this fitness has a score.
    pub fn is_valid(self) -> bool {
        !self.score.is_nan()
    }

    /// The constraint violation: 0 for a feasible solution (and for an invalid one).
    pub fn violation(self) -> f64 {
        self.violation
    }

    /// Whether this fitness has a score and no constraint violation.
    pub fn is_feasible(self) -> bool {
        self.is_valid() && self.violation == 0.0
    }
}

impl fmt::Debug for Fitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.score() {
            None => write!(f, "Fitness(invalid)"),
            Some(score) if self.violation > 0.0 => {
                write!(f, "Fitness({score:?}, violation {:?})", self.violation)
            }
            Some(score) => write!(f, "Fitness({score:?})"),
        }
    }
}

impl fmt::Display for Fitness {
    /// The score, formatted like an `f64` (precision applies to it), and its constraint
    /// violation if any, or `invalid`. Width, fill and alignment apply to the whole text.
    ///
    /// ```
    /// use genoxide::Fitness;
    ///
    /// assert_eq!(format!("{:.2}", Fitness::new(1.0 / 3.0)), "0.33");
    /// assert_eq!(Fitness::invalid().to_string(), "invalid");
    /// assert_eq!(format!("{:>18}", Fitness::constrained(1.0, 0.5)), " 1 (violation 0.5)");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(score) = self.score() else {
            return f.pad("invalid");
        };
        if self.violation == 0.0 {
            return fmt::Display::fmt(&score, f);
        }
        let text = match f.precision() {
            Some(precision) => format!("{score:.precision$} (violation {})", self.violation),
            None => format!("{score} (violation {})", self.violation),
        };
        // padded by hand: `pad` would also cut the text to the precision
        let width = f.width().unwrap_or(0);
        let missing = width.saturating_sub(text.chars().count());
        let (before, after) = match f.align() {
            Some(fmt::Alignment::Left) => (0, missing),
            Some(fmt::Alignment::Center) => (missing / 2, missing - missing / 2),
            _ => (missing, 0),
        };
        let fill = f.fill();
        (0..before).try_for_each(|_| f.write_char(fill))?;
        f.write_str(&text)?;
        (0..after).try_for_each(|_| f.write_char(fill))
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
        // every invalid fitness has the same NaN
        self.score.to_bits().hash(state);
        self.violation.to_bits().hash(state);
    }
}

// validated like `try_constrained`; a NaN score is the invalid fitness
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Fitness {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename = "Fitness")]
        struct Raw {
            score: f64,
            violation: f64,
        }
        let raw = Raw::deserialize(deserializer)?;
        if raw.score.is_nan() {
            return Ok(Self::invalid());
        }
        Self::try_constrained(raw.score, raw.violation).map_err(serde::de::Error::custom)
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
            // constrained: small sets of scores and violations, to get ties
            10 => ((-2i32..2), (0i32..3)).prop_map(|(score, violation)| {
                Fitness::constrained(f64::from(score), f64::from(violation) / 2.0)
            }),
            2 => Just(Fitness::constrained(1.0, f64::INFINITY)),
        ]
    }

    #[test]
    fn compact() {
        assert_eq!(std::mem::size_of::<Fitness>(), 16);
        assert_eq!(format!("{:?}", Fitness::invalid()), "Fitness(invalid)");
        assert_eq!(format!("{:?}", Fitness::new(1.5)), "Fitness(1.5)");
        assert_eq!(
            format!("{:?}", Fitness::constrained(1.0, 0.5)),
            "Fitness(1.0, violation 0.5)"
        );
    }

    #[test]
    fn deb_feasibility_rules() {
        let objective = Objective::Minimize;
        let feasible = Fitness::constrained(100.0, 0.0);
        let barely = Fitness::constrained(1.0, 0.5);
        let far = Fitness::constrained(-100.0, 2.0);
        assert!(objective.is_better(feasible, barely));
        assert!(objective.is_better(barely, far));
        assert!(objective.is_better(far, Fitness::invalid()));
        // equal violations: the score decides
        assert!(objective.is_better(
            Fitness::constrained(1.0, 0.5),
            Fitness::constrained(2.0, 0.5)
        ));
        // a zero violation is just a score
        assert_eq!(Fitness::constrained(3.0, 0.0), Fitness::new(3.0));
        assert_eq!(Fitness::constrained(3.0, -0.0), Fitness::new(3.0));
        assert!(barely.is_valid() && !barely.is_feasible() && feasible.is_feasible());
        assert!(!Fitness::invalid().is_feasible());
    }

    #[test]
    fn invalid_constraint_violations() {
        assert_eq!(
            Fitness::try_constrained(1.0, f64::NAN),
            Err(Error::NanFitness)
        );
        assert_eq!(
            Fitness::try_constrained(f64::NAN, 1.0),
            Err(Error::NanFitness)
        );
        assert!(matches!(
            Fitness::try_constrained(1.0, -1.0),
            Err(Error::InvalidFitness { .. })
        ));
        assert_eq!(Fitness::constrained(1.0, -1.0), Fitness::invalid());
        assert_eq!(
            Fitness::constrained(1.0, f64::INFINITY).violation(),
            f64::INFINITY
        );
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
                if a.violation() == b.violation() {
                    // the scores decide, in opposite directions
                    prop_assert_eq!(
                        Objective::Minimize.compare(a, b),
                        Objective::Maximize.compare(a, b).reverse()
                    );
                } else {
                    // the violations decide, whatever the objective
                    prop_assert_eq!(Objective::Minimize.compare(a, b), Objective::Maximize.compare(a, b));
                }
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
