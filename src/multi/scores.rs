//! The fitness of a multi-objective solution.

use crate::{Error, Result};
use std::fmt;
use std::hash::{Hash, Hasher};

/// The fitness of a solution with `M` objectives: `M` objective values, optionally with a
/// constraint violation, or invalid. The multi-objective counterpart of
/// [`Fitness`](crate::Fitness).
///
/// A value is any `f64` except NaN (infinities are allowed), and `-0.0` is the same as `0.0`.
///
/// - A [constrained](Scores::constrained) solution also has a constraint violation: 0 when it's
///   feasible, and how far it is from feasible otherwise. Feasible solutions dominate infeasible
///   ones (see [`dominates`](crate::multi::dominates)).
/// - An invalid solution can't be scored at all, e.g. because the simulation behind the fitness
///   function failed. Every valid solution dominates it.
///
/// ```
/// use genoxide::multi::Scores;
///
/// let scores = Scores::new([1.5, -2.0]);
/// assert_eq!(scores.values(), Some([1.5, -2.0]));
/// assert!(!Scores::new([1.0, f64::NAN]).is_valid()); // NaN is invalid
/// assert_eq!(Scores::constrained([1.0, 2.0], 0.5).violation(), 0.5);
/// ```
#[derive(Clone, Copy)]
pub struct Scores<const M: usize> {
    // all NaN for an invalid solution, otherwise never NaN and never -0.0
    values: [f64; M],
    // 0 when feasible or invalid, otherwise positive (possibly infinite); never NaN or -0.0
    violation: f64,
}

impl<const M: usize> Scores<M> {
    /// Scores with these objective values. A NaN value gives invalid scores, see
    /// [`try_new`](Scores::try_new) to get an error instead.
    pub fn new(values: [f64; M]) -> Self {
        Self::try_new(values).unwrap_or(Self::invalid())
    }

    /// Scores with these objective values, or [`Error::NanFitness`] if one is NaN.
    pub fn try_new(values: [f64; M]) -> Result<Self> {
        if values.iter().any(|value| value.is_nan()) {
            return Err(Error::NanFitness);
        }
        // + 0.0 turns -0.0 into 0.0 and leaves every other value unchanged
        Ok(Self {
            values: values.map(|value| value + 0.0),
            violation: 0.0,
        })
    }

    /// Scores with these objective values and constraint violation: 0 for a feasible solution,
    /// positive for how far it is from feasible (see [`constraint`](crate::constraint)).
    ///
    /// A NaN value or violation, or a negative violation, gives invalid scores; see
    /// [`try_constrained`](Scores::try_constrained) to get an error instead.
    pub fn constrained(values: [f64; M], violation: f64) -> Self {
        Self::try_constrained(values, violation).unwrap_or(Self::invalid())
    }

    /// Scores with these objective values and constraint violation, or [`Error::NanFitness`] for
    /// a NaN value or violation, and [`Error::InvalidFitness`] for a negative violation.
    pub fn try_constrained(values: [f64; M], violation: f64) -> Result<Self> {
        if violation.is_nan() {
            return Err(Error::NanFitness);
        }
        if violation < 0.0 {
            return Err(Error::InvalidFitness {
                reason: format!("a constraint violation can't be negative, got {violation}"),
            });
        }
        let scores = Self::try_new(values)?;
        Ok(Self {
            violation: violation + 0.0,
            ..scores
        })
    }

    /// The scores of a solution that can't be scored.
    pub fn invalid() -> Self {
        Self {
            values: [f64::NAN; M],
            violation: 0.0,
        }
    }

    /// The objective values, or `None` for invalid scores.
    pub fn values(&self) -> Option<[f64; M]> {
        self.is_valid().then_some(self.values)
    }

    /// Whether these scores have objective values. With no objectives (`M` = 0), always.
    pub fn is_valid(&self) -> bool {
        self.values.first().is_none_or(|value| !value.is_nan())
    }

    /// The constraint violation: 0 for a feasible solution (and for an invalid one).
    pub fn violation(&self) -> f64 {
        self.violation
    }

    /// Whether these scores are valid and have no constraint violation.
    pub fn is_feasible(&self) -> bool {
        self.is_valid() && self.violation == 0.0
    }

    // the objective values, NaN for invalid scores
    pub(crate) fn raw(&self) -> &[f64; M] {
        &self.values
    }
}

impl<const M: usize> fmt::Debug for Scores<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.values() {
            None => write!(f, "Scores(invalid)"),
            Some(values) if self.violation > 0.0 => {
                write!(f, "Scores({values:?}, violation {:?})", self.violation)
            }
            Some(values) => write!(f, "Scores({values:?})"),
        }
    }
}

impl<const M: usize> fmt::Display for Scores<M> {
    /// The objective values in parentheses, each formatted like an `f64` (precision applies), or
    /// `invalid`.
    ///
    /// ```
    /// use genoxide::multi::Scores;
    ///
    /// assert_eq!(format!("{:.1}", Scores::new([1.0, 2.25])), "(1.0, 2.2)");
    /// assert_eq!(Scores::constrained([1.0, 2.0], 0.5).to_string(), "(1, 2) (violation 0.5)");
    /// assert_eq!(Scores::<2>::invalid().to_string(), "invalid");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(values) = self.values() else {
            return f.pad("invalid");
        };
        write!(f, "(")?;
        for (index, value) in values.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            fmt::Display::fmt(value, f)?;
        }
        write!(f, ")")?;
        if self.violation > 0.0 {
            write!(f, " (violation {})", self.violation)?;
        }
        Ok(())
    }
}

impl<const M: usize> PartialEq for Scores<M> {
    /// Equal objective values and violations; every invalid solution is equal to every other.
    fn eq(&self, other: &Self) -> bool {
        match (self.is_valid(), other.is_valid()) {
            (true, true) => self.values == other.values && self.violation == other.violation,
            (valid, other_valid) => valid == other_valid,
        }
    }
}

impl<const M: usize> Eq for Scores<M> {}

impl<const M: usize> Hash for Scores<M> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // every invalid solution has the same NaNs, and there is no -0.0
        for value in &self.values {
            value.to_bits().hash(state);
        }
        self.violation.to_bits().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn construction() {
        assert_eq!(Scores::try_new([1.0, f64::NAN]), Err(Error::NanFitness));
        assert_eq!(
            Scores::try_constrained([1.0, 2.0], f64::NAN),
            Err(Error::NanFitness)
        );
        assert!(matches!(
            Scores::try_constrained([1.0, 2.0], -1.0),
            Err(Error::InvalidFitness { .. })
        ));
        assert!(!Scores::constrained([1.0, 2.0], -1.0).is_valid());
        let scores = Scores::constrained([-0.0, f64::INFINITY], 2.0);
        assert_eq!(scores.values(), Some([0.0, f64::INFINITY]));
        assert!(scores.values().unwrap()[0].is_sign_positive());
        assert!(scores.is_valid() && !scores.is_feasible());
        assert!(Scores::new([1.0]).is_feasible());
        assert_eq!(Scores::<3>::invalid().values(), None);
        assert_eq!(Scores::<3>::invalid().violation(), 0.0);
        // no objectives
        assert!(Scores::<0>::new([]).is_valid());
    }

    #[test]
    fn equality_and_hashing() {
        assert_eq!(Scores::new([0.0, 1.0]), Scores::new([-0.0, 1.0]));
        assert_ne!(
            Scores::new([0.0, 1.0]),
            Scores::constrained([0.0, 1.0], 1.0)
        );
        assert_eq!(Scores::new([f64::NAN, 1.0]), Scores::<2>::invalid());
        assert_ne!(Scores::new([1.0, 1.0]), Scores::<2>::invalid());
        let set: HashSet<Scores<2>> = [
            Scores::new([-0.0, 1.0]),
            Scores::new([0.0, 1.0]),
            Scores::invalid(),
            Scores::new([f64::NAN, 2.0]),
        ]
        .into_iter()
        .collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn formatting() {
        assert_eq!(
            format!("{:?}", Scores::new([1.0, 2.0])),
            "Scores([1.0, 2.0])"
        );
        assert_eq!(
            format!("{:?}", Scores::constrained([1.0, 2.0], 0.5)),
            "Scores([1.0, 2.0], violation 0.5)"
        );
        assert_eq!(format!("{:?}", Scores::<2>::invalid()), "Scores(invalid)");
    }
}
