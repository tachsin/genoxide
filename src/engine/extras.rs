//! What a fitness function gives besides the fitness, such as a gradient, and how the engine
//! hands it to an algorithm.
//!
//! - A fitness function declares what it gives with [`FitnessFunction::provides`], and writes it
//!   into the buffers of [`Extras`] in [`evaluate_with`](FitnessFunction::evaluate_with) (or of
//!   [`BatchExtras`] in [`evaluate_batch_with`](FitnessFunction::evaluate_batch_with)).
//! - An algorithm checks what it gets once per run in [`Algorithm::prepare`], says what each ask
//!   needs with [`Algorithm::wants`], and reads it in
//!   [`Algorithm::tell_evaluations`] from [`Evaluations`].
//!
//! When an algorithm wants nothing, which is the default, the [`Engine`](crate::Engine) evaluates
//! as it always has, with [`evaluate`](FitnessFunction::evaluate) and
//! [`tell`](Algorithm::tell).
//!
//! [`FitnessFunction::provides`]: crate::engine::FitnessFunction::provides
//! [`FitnessFunction`]: crate::engine::FitnessFunction
//! [`Algorithm::prepare`]: crate::Algorithm::prepare
//! [`Algorithm::wants`]: crate::Algorithm::wants
//! [`Algorithm::tell_evaluations`]: crate::Algorithm::tell_evaluations
//! [`Algorithm`]: crate::Algorithm

#[cfg(doc)]
use crate::Algorithm;
#[cfg(doc)]
use crate::engine::FitnessFunction;
use crate::{Error, Fitness, Result};

/// What a fitness function gives besides the fitness, for every genome: the answer of
/// [`FitnessFunction::provides`].
///
/// Nothing by default. More extras (constraint values, residuals, their Jacobians) come in later
/// versions, so the struct is `#[non_exhaustive]`: build it from [`Provided::NOTHING`] or
/// [`Provided::GRADIENT`].
///
/// ```
/// use genoxide::engine::Provided;
///
/// assert!(Provided::NOTHING.is_empty());
/// assert!(Provided::NOTHING.with_gradient().gradient);
/// assert_eq!(Provided::default(), Provided::NOTHING);
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Provided {
    /// The gradient of the score, `∂score / ∂xᵢ`, one value per gene: of the score as the fitness
    /// function returns it, whether it's minimized or maximized.
    pub gradient: bool,
}

impl Provided {
    /// Nothing besides the fitness.
    pub const NOTHING: Self = Self { gradient: false };

    /// The gradient of the score.
    pub const GRADIENT: Self = Self { gradient: true };

    /// These extras and the gradient.
    #[must_use]
    pub const fn with_gradient(self) -> Self {
        Self { gradient: true }
    }

    /// Whether nothing is provided besides the fitness.
    pub const fn is_empty(self) -> bool {
        !self.gradient
    }
}

/// What an algorithm wants besides the fitness, for the genomes of its next ask: the answer of
/// [`Algorithm::wants`]. A subset of what the fitness function [provides](Provided), which the
/// algorithm checks in [`Algorithm::prepare`].
///
/// ```
/// use genoxide::engine::{Provided, Wanted};
///
/// assert!(Wanted::NOTHING.is_empty());
/// assert_eq!(Wanted::GRADIENT.missing_from(Provided::NOTHING), Some("gradient"));
/// assert_eq!(Wanted::GRADIENT.missing_from(Provided::GRADIENT), None);
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Wanted {
    /// The gradient of the score, one value per gene.
    pub gradient: bool,
}

impl Wanted {
    /// Nothing besides the fitness: the [`Engine`](crate::Engine) evaluates as it would without
    /// extras.
    pub const NOTHING: Self = Self { gradient: false };

    /// The gradient of the score.
    pub const GRADIENT: Self = Self { gradient: true };

    /// These extras and the gradient.
    #[must_use]
    pub const fn with_gradient(self) -> Self {
        Self { gradient: true }
    }

    /// Whether nothing is wanted besides the fitness.
    #[inline]
    pub const fn is_empty(self) -> bool {
        !self.gradient
    }

    /// The name of the first extra that is wanted and not provided, or `None` if everything
    /// wanted is provided.
    pub const fn missing_from(self, provided: Provided) -> Option<&'static str> {
        if self.gradient && !provided.gradient {
            Some("gradient")
        } else {
            None
        }
    }
}

/// The buffers for the extras of one genome, which
/// [`FitnessFunction::evaluate_with`] fills: only the wanted ones are there.
///
/// ```
/// use genoxide::engine::{Extras, FitnessFunction};
/// use genoxide::genome::Reals;
/// use genoxide::problems::Sphere;
///
/// let x = Reals::from(vec![1.0, -2.0]);
/// let mut gradient = [0.0; 2];
/// let value = Sphere::new(2).evaluate_with(&x, &mut Extras::with_gradient(&mut gradient));
/// assert_eq!(value, 5.0);
/// assert_eq!(gradient, [2.0, -4.0]);
/// ```
#[derive(Debug, Default)]
pub struct Extras<'a> {
    gradient: Option<&'a mut [f64]>,
}

impl<'a> Extras<'a> {
    /// No extras: [`evaluate_with`](FitnessFunction::evaluate_with) computes the fitness only.
    pub fn none() -> Self {
        Self { gradient: None }
    }

    /// A buffer for the gradient, one value per gene of the genome. A fitness function may panic
    /// for another length, as the test problems do.
    pub fn with_gradient(gradient: &'a mut [f64]) -> Self {
        Self {
            gradient: Some(gradient),
        }
    }

    /// What's wanted: the buffers there are.
    pub fn wanted(&self) -> Wanted {
        Wanted {
            gradient: self.gradient.is_some(),
        }
    }

    /// The buffer for the gradient of the score, `∂score / ∂xᵢ` into element `i`, if it's wanted.
    /// The engine zeroes it before each evaluation.
    pub fn gradient(&mut self) -> Option<&mut [f64]> {
        self.gradient.as_deref_mut()
    }
}

/// The buffers for the extras of a batch of genomes, which
/// [`FitnessFunction::evaluate_batch_with`] fills: flat and row-major, a row per genome in the
/// batch's order, of [`dimensions`](BatchExtras::dimensions) values each.
#[derive(Debug, Default)]
pub struct BatchExtras<'a> {
    gradients: Option<&'a mut [f64]>,
    dimensions: usize,
}

impl<'a> BatchExtras<'a> {
    /// No extras.
    pub fn none() -> Self {
        Self {
            gradients: None,
            dimensions: 0,
        }
    }

    /// A buffer for the gradients: a row of `dimensions` values per genome, one after the other.
    pub fn with_gradients(gradients: &'a mut [f64], dimensions: usize) -> Self {
        Self {
            gradients: Some(gradients),
            dimensions,
        }
    }

    /// What's wanted: the buffers there are.
    pub fn wanted(&self) -> Wanted {
        Wanted {
            gradient: self.gradients.is_some(),
        }
    }

    /// The length of a row: the number of genes of each genome.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// The buffer for the gradients of all genomes, row-major, if they're wanted: the gradient of
    /// the genome at position `p` is `gradients[p * dimensions..(p + 1) * dimensions]`.
    pub fn gradients(&mut self) -> Option<&mut [f64]> {
        self.gradients.as_deref_mut()
    }

    /// The buffers of the genome at `position`, as one evaluation's [`Extras`]: none for a
    /// position beyond the rows.
    pub fn get(&mut self, position: usize) -> Extras<'_> {
        let dimensions = self.dimensions;
        let gradient = self.gradients.as_deref_mut().and_then(|gradients| {
            let start = position.checked_mul(dimensions)?;
            gradients.get_mut(start..start.checked_add(dimensions)?)
        });
        Extras { gradient }
    }
}

/// The fitness and the wanted extras of the genomes of an ask, in its order, for
/// [`Algorithm::tell_evaluations`]: flat, row-major buffers that the engine reuses from one
/// generation to the next.
///
/// The gradient of an invalid fitness is unspecified: with [`NanPolicy::Invalid`], a NaN in a
/// gradient makes the whole evaluation invalid, and the algorithm should not use its gradient.
///
/// ```
/// use genoxide::Fitness;
/// use genoxide::engine::Evaluations;
///
/// let fitness = [Fitness::new(1.0), Fitness::new(4.0)];
/// let gradients = [2.0, 0.0, 4.0, 0.0];
/// let evaluations = Evaluations::with_gradients(&fitness, &gradients, 2)?;
/// assert_eq!(evaluations.gradient(1), Some(&[4.0, 0.0][..]));
/// assert_eq!(Evaluations::new(&fitness).gradient(1), None);
/// # Ok::<(), genoxide::Error>(())
/// ```
///
/// [`NanPolicy::Invalid`]: crate::engine::NanPolicy::Invalid
#[derive(Clone, Copy, Debug)]
pub struct Evaluations<'a> {
    fitness: &'a [Fitness],
    gradients: Option<&'a [f64]>,
    dimensions: usize,
}

impl<'a> Evaluations<'a> {
    /// The fitness values alone, without extras.
    pub fn new(fitness: &'a [Fitness]) -> Self {
        Self {
            fitness,
            gradients: None,
            dimensions: 0,
        }
    }

    /// The fitness values with their gradients: a row of `dimensions` values per fitness, one
    /// after the other.
    ///
    /// # Errors
    ///
    /// [`Error::FitnessCount`] if `gradients` doesn't have `dimensions` values per fitness.
    pub fn with_gradients(
        fitness: &'a [Fitness],
        gradients: &'a [f64],
        dimensions: usize,
    ) -> Result<Self> {
        let expected = fitness.len().saturating_mul(dimensions);
        if gradients.len() != expected {
            return Err(Error::FitnessCount {
                expected,
                got: gradients.len(),
            });
        }
        Ok(Self::from_parts(fitness, Some(gradients), dimensions))
    }

    // the engine's buffers, whose lengths it has made right
    pub(crate) fn from_parts(
        fitness: &'a [Fitness],
        gradients: Option<&'a [f64]>,
        dimensions: usize,
    ) -> Self {
        Self {
            fitness,
            gradients,
            dimensions,
        }
    }

    /// The number of evaluated genomes.
    pub fn len(&self) -> usize {
        self.fitness.len()
    }

    /// Whether no genome was evaluated.
    pub fn is_empty(&self) -> bool {
        self.fitness.is_empty()
    }

    /// The fitness values, in the order of the ask.
    pub fn fitness(&self) -> &'a [Fitness] {
        self.fitness
    }

    /// What the evaluations have besides the fitness.
    pub fn provided(&self) -> Provided {
        Provided {
            gradient: self.gradients.is_some(),
        }
    }

    /// The length of a gradient's row: the number of genes; 0 without gradients.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// The gradients of all genomes, row-major, if they were wanted.
    pub fn gradients(&self) -> Option<&'a [f64]> {
        self.gradients
    }

    /// The gradient of the genome at `position`, if gradients were wanted and the position is in
    /// bounds.
    pub fn gradient(&self, position: usize) -> Option<&'a [f64]> {
        if position >= self.fitness.len() {
            return None;
        }
        let start = position * self.dimensions;
        self.gradients?.get(start..start + self.dimensions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_rows() {
        let mut buffer = [0.0; 6];
        let mut extras = BatchExtras::with_gradients(&mut buffer, 2);
        assert_eq!(extras.wanted(), Wanted::GRADIENT);
        extras.get(1).gradient().unwrap()[1] = 3.0;
        assert!(extras.get(3).gradient().is_none());
        assert!(extras.get(usize::MAX).gradient().is_none());
        assert_eq!(buffer, [0.0, 0.0, 0.0, 3.0, 0.0, 0.0]);
        assert!(BatchExtras::none().get(0).gradient().is_none());
        assert_eq!(BatchExtras::none().wanted(), Wanted::NOTHING);
    }

    #[test]
    fn evaluations_check_their_lengths() {
        let fitness = [Fitness::new(1.0); 3];
        assert_eq!(
            Evaluations::with_gradients(&fitness, &[0.0; 5], 2).unwrap_err(),
            Error::FitnessCount {
                expected: 6,
                got: 5
            }
        );
        let gradients = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let evaluations = Evaluations::with_gradients(&fitness, &gradients, 2).unwrap();
        assert_eq!(evaluations.len(), 3);
        assert_eq!(evaluations.provided(), Provided::GRADIENT);
        assert_eq!(evaluations.gradient(2), Some(&[5.0, 6.0][..]));
        assert_eq!(evaluations.gradient(3), None);
        assert_eq!(Evaluations::new(&fitness).provided(), Provided::NOTHING);
        assert!(Evaluations::new(&[]).is_empty());
    }

    #[test]
    fn wanted_and_provided() {
        assert!(Wanted::default().is_empty());
        assert!(!Wanted::NOTHING.with_gradient().is_empty());
        assert_eq!(Wanted::NOTHING.missing_from(Provided::NOTHING), None);
        assert!(Provided::GRADIENT == Provided::NOTHING.with_gradient());
        let mut gradient = [0.0];
        assert_eq!(
            Extras::with_gradient(&mut gradient).wanted(),
            Wanted::GRADIENT
        );
        assert_eq!(Extras::none().wanted(), Wanted::NOTHING);
    }
}
