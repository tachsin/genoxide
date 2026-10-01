//! What a fitness function gives besides the fitness, such as a gradient or constraint values,
//! and how the engine hands it to an algorithm.
//!
//! - A fitness function declares what it gives with [`FitnessFunction::provides`], and writes it
//!   into the buffers of [`Extras`] in [`evaluate_with`](FitnessFunction::evaluate_with) (or of
//!   [`BatchExtras`] in [`evaluate_batch_with`](FitnessFunction::evaluate_batch_with)).
//! - An algorithm checks what it gets once per run in [`Algorithm::prepare`], says what each ask
//!   needs with [`Algorithm::wants`], and reads it in
//!   [`Algorithm::tell_evaluations`] from [`Evaluations`].
//!
//! The extras are the gradient of the score, the values `gᵢ(x)` of inequality constraints
//! `gᵢ(x) ≤ 0`, and their Jacobian `∂gᵢ / ∂xⱼ`.
//! [`Differentiable`](crate::gradient::Differentiable) and
//! [`Constrained`](crate::constraint::Constrained) give them from closures.
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
/// Nothing by default. More extras (equality constraints, residuals) come in later versions, so
/// the struct is `#[non_exhaustive]`: build it from [`Provided::NOTHING`] or
/// [`Provided::GRADIENT`] and the `with_` methods.
///
/// ```
/// use genoxide::engine::Provided;
///
/// assert!(Provided::NOTHING.is_empty());
/// assert!(Provided::NOTHING.with_gradient().gradient);
/// assert_eq!(Provided::default(), Provided::NOTHING);
/// // a score with its gradient, 2 inequality constraints g(x) <= 0 and their Jacobian
/// let all = Provided::GRADIENT.with_inequalities(2).with_constraint_jacobian();
/// assert_eq!(all.inequalities, 2);
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Provided {
    /// The gradient of the score, `∂score / ∂xᵢ`, one value per gene: of the score as the fitness
    /// function returns it, whether it's minimized or maximized.
    pub gradient: bool,
    /// The number of inequality constraints `gᵢ(x) ≤ 0` whose values `gᵢ(x)` the function gives,
    /// in the same order for every genome; 0 for none. The fitness's violation should then be
    /// `Σ max(0, gᵢ(x))`, as [`constraint::at_most`](crate::constraint::at_most) measures it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub inequalities: usize,
    /// The Jacobian of the inequality constraints: `∂gᵢ / ∂xⱼ`, a row of one value per gene for
    /// each constraint.
    #[cfg_attr(feature = "serde", serde(default))]
    pub constraint_jacobian: bool,
}

impl Provided {
    /// Nothing besides the fitness.
    pub const NOTHING: Self = Self {
        gradient: false,
        inequalities: 0,
        constraint_jacobian: false,
    };

    /// The gradient of the score.
    pub const GRADIENT: Self = Self::NOTHING.with_gradient();

    /// These extras and the gradient.
    #[must_use]
    pub const fn with_gradient(self) -> Self {
        Self {
            gradient: true,
            ..self
        }
    }

    /// These extras and the values of `count` inequality constraints.
    #[must_use]
    pub const fn with_inequalities(self, count: usize) -> Self {
        Self {
            inequalities: count,
            ..self
        }
    }

    /// These extras and the Jacobian of the inequality constraints.
    #[must_use]
    pub const fn with_constraint_jacobian(self) -> Self {
        Self {
            constraint_jacobian: true,
            ..self
        }
    }

    /// Whether nothing is provided besides the fitness.
    pub const fn is_empty(self) -> bool {
        !self.gradient && self.inequalities == 0 && !self.constraint_jacobian
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
/// let constrained = Wanted::GRADIENT.with_inequalities().with_constraint_jacobian();
/// let provided = Provided::GRADIENT.with_inequalities(2);
/// assert_eq!(constrained.missing_from(provided), Some("constraint Jacobian"));
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Wanted {
    /// The gradient of the score, one value per gene.
    pub gradient: bool,
    /// The values of the inequality constraints, as many as the fitness function
    /// [provides](Provided::inequalities).
    #[cfg_attr(feature = "serde", serde(default))]
    pub inequalities: bool,
    /// The Jacobian of the inequality constraints, a row per constraint.
    #[cfg_attr(feature = "serde", serde(default))]
    pub constraint_jacobian: bool,
}

impl Wanted {
    /// Nothing besides the fitness: the [`Engine`](crate::Engine) evaluates as it would without
    /// extras.
    pub const NOTHING: Self = Self {
        gradient: false,
        inequalities: false,
        constraint_jacobian: false,
    };

    /// The gradient of the score.
    pub const GRADIENT: Self = Self::NOTHING.with_gradient();

    /// These extras and the gradient.
    #[must_use]
    pub const fn with_gradient(self) -> Self {
        Self {
            gradient: true,
            ..self
        }
    }

    /// These extras and the values of the inequality constraints.
    #[must_use]
    pub const fn with_inequalities(self) -> Self {
        Self {
            inequalities: true,
            ..self
        }
    }

    /// These extras and the Jacobian of the inequality constraints.
    #[must_use]
    pub const fn with_constraint_jacobian(self) -> Self {
        Self {
            constraint_jacobian: true,
            ..self
        }
    }

    /// Whether nothing is wanted besides the fitness.
    #[inline]
    pub const fn is_empty(self) -> bool {
        !self.gradient && !self.inequalities && !self.constraint_jacobian
    }

    /// The name of the first extra that is wanted and not provided, or `None` if everything
    /// wanted is provided.
    pub const fn missing_from(self, provided: Provided) -> Option<&'static str> {
        if self.gradient && !provided.gradient {
            Some("gradient")
        } else if self.inequalities && provided.inequalities == 0 {
            Some("inequality constraint values")
        } else if self.constraint_jacobian
            && !(provided.constraint_jacobian && provided.inequalities > 0)
        {
            Some("constraint Jacobian")
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
    inequalities: Option<&'a mut [f64]>,
    constraint_jacobian: Option<&'a mut [f64]>,
}

impl<'a> Extras<'a> {
    /// No extras: [`evaluate_with`](FitnessFunction::evaluate_with) computes the fitness only.
    pub fn none() -> Self {
        Self::default()
    }

    /// A buffer for the gradient, one value per gene of the genome. A fitness function may panic
    /// for another length, as the test problems do.
    pub fn with_gradient(gradient: &'a mut [f64]) -> Self {
        Self::new(Some(gradient), None, None)
    }

    /// Buffers for any of the extras: the gradient (a value per gene), the values of the
    /// inequality constraints (a value per constraint) and their Jacobian (row-major: a row of a
    /// value per gene for each constraint, `∂gᵢ / ∂xⱼ` at `i * genes + j`). A fitness function
    /// may panic for other lengths.
    ///
    /// ```
    /// use genoxide::constraint::Constrained;
    /// use genoxide::engine::{Extras, FitnessFunction};
    /// use genoxide::genome::Reals;
    ///
    /// // x₀ + x₁, subject to 1 − x₀ x₁ <= 0
    /// let f = Constrained::differentiable(1, |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
    ///     gradient.copy_from_slice(&[1.0, 1.0]);
    ///     g[0] = 1.0 - x[0] * x[1];
    ///     jacobian.copy_from_slice(&[-x[1], -x[0]]);
    ///     x[0] + x[1]
    /// });
    /// let (mut gradient, mut g, mut jacobian) = ([0.0; 2], [0.0], [0.0; 2]);
    /// let mut extras = Extras::new(Some(&mut gradient), Some(&mut g), Some(&mut jacobian));
    /// let (score, violation) = f.evaluate_with(&Reals::from(vec![0.5, 1.0]), &mut extras);
    /// assert_eq!((score, violation), (1.5, 0.5));
    /// assert_eq!((g, jacobian), ([0.5], [-1.0, -0.5]));
    /// ```
    pub fn new(
        gradient: Option<&'a mut [f64]>,
        inequalities: Option<&'a mut [f64]>,
        constraint_jacobian: Option<&'a mut [f64]>,
    ) -> Self {
        Self {
            gradient,
            inequalities,
            constraint_jacobian,
        }
    }

    /// What's wanted: the buffers there are.
    pub fn wanted(&self) -> Wanted {
        Wanted {
            gradient: self.gradient.is_some(),
            inequalities: self.inequalities.is_some(),
            constraint_jacobian: self.constraint_jacobian.is_some(),
        }
    }

    /// The buffer for the gradient of the score, `∂score / ∂xᵢ` into element `i`, if it's wanted.
    /// The engine zeroes it before each evaluation.
    pub fn gradient(&mut self) -> Option<&mut [f64]> {
        self.gradient.as_deref_mut()
    }

    /// The buffer for the values of the inequality constraints, `gᵢ(x)` into element `i`, if
    /// they're wanted. The engine zeroes it before each evaluation.
    pub fn inequalities(&mut self) -> Option<&mut [f64]> {
        self.inequalities.as_deref_mut()
    }

    /// The buffer for the Jacobian of the inequality constraints, row-major: `∂gᵢ / ∂xⱼ` into
    /// element `i * genes + j`, if it's wanted. The engine zeroes it before each evaluation.
    pub fn constraint_jacobian(&mut self) -> Option<&mut [f64]> {
        self.constraint_jacobian.as_deref_mut()
    }

    // the three buffers at once: the gradient, the inequalities and the Jacobian
    #[allow(clippy::type_complexity)]
    pub(crate) fn buffers(
        &mut self,
    ) -> (Option<&mut [f64]>, Option<&mut [f64]>, Option<&mut [f64]>) {
        (
            self.gradient.as_deref_mut(),
            self.inequalities.as_deref_mut(),
            self.constraint_jacobian.as_deref_mut(),
        )
    }
}

// the row at `position` of a flat buffer of rows of `width` values, if there's one
fn row_mut<'b>(
    buffer: &'b mut Option<&mut [f64]>,
    position: usize,
    width: usize,
) -> Option<&'b mut [f64]> {
    let start = position.checked_mul(width)?;
    buffer
        .as_deref_mut()?
        .get_mut(start..start.checked_add(width)?)
}

/// The buffers for the extras of a batch of genomes, which
/// [`FitnessFunction::evaluate_batch_with`] fills: flat and row-major, a row per genome in the
/// batch's order. A gradient's row has [`dimensions`](BatchExtras::dimensions) values, the
/// inequalities' [`constraints`](BatchExtras::constraints), and the Jacobian's
/// `constraints × dimensions`.
#[derive(Debug, Default)]
pub struct BatchExtras<'a> {
    gradients: Option<&'a mut [f64]>,
    inequalities: Option<&'a mut [f64]>,
    constraint_jacobians: Option<&'a mut [f64]>,
    dimensions: usize,
    constraints: usize,
}

impl<'a> BatchExtras<'a> {
    /// No extras.
    pub fn none() -> Self {
        Self::default()
    }

    /// A buffer for the gradients: a row of `dimensions` values per genome, one after the other.
    pub fn with_gradients(gradients: &'a mut [f64], dimensions: usize) -> Self {
        Self::new(Some(gradients), None, None, dimensions, 0)
    }

    /// Buffers for any of the extras of genomes of `dimensions` genes and `constraints`
    /// inequality constraints: per genome, a row of `dimensions` values of `gradients`, of
    /// `constraints` values of `inequalities`, and of `constraints × dimensions` values of
    /// `constraint_jacobians` (itself row-major, a row per constraint).
    pub fn new(
        gradients: Option<&'a mut [f64]>,
        inequalities: Option<&'a mut [f64]>,
        constraint_jacobians: Option<&'a mut [f64]>,
        dimensions: usize,
        constraints: usize,
    ) -> Self {
        Self {
            gradients,
            inequalities,
            constraint_jacobians,
            dimensions,
            constraints,
        }
    }

    /// What's wanted: the buffers there are.
    pub fn wanted(&self) -> Wanted {
        Wanted {
            gradient: self.gradients.is_some(),
            inequalities: self.inequalities.is_some(),
            constraint_jacobian: self.constraint_jacobians.is_some(),
        }
    }

    /// The length of a gradient's row: the number of genes of each genome.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// The number of inequality constraints: the length of a row of their values.
    pub fn constraints(&self) -> usize {
        self.constraints
    }

    /// The buffer for the gradients of all genomes, row-major, if they're wanted: the gradient of
    /// the genome at position `p` is `gradients[p * dimensions..(p + 1) * dimensions]`.
    pub fn gradients(&mut self) -> Option<&mut [f64]> {
        self.gradients.as_deref_mut()
    }

    /// The buffer for the inequality constraints' values of all genomes, if they're wanted: those
    /// of the genome at position `p` are `inequalities[p * constraints..(p + 1) * constraints]`.
    pub fn inequalities(&mut self) -> Option<&mut [f64]> {
        self.inequalities.as_deref_mut()
    }

    /// The buffer for the constraint Jacobians of all genomes, if they're wanted: that of the
    /// genome at position `p` is the `constraints × dimensions` values from
    /// `p * constraints * dimensions` on, a row per constraint.
    pub fn constraint_jacobians(&mut self) -> Option<&mut [f64]> {
        self.constraint_jacobians.as_deref_mut()
    }

    /// The buffers of the genome at `position`, as one evaluation's [`Extras`]: none for a
    /// position beyond the rows.
    pub fn get(&mut self, position: usize) -> Extras<'_> {
        let (dimensions, constraints) = (self.dimensions, self.constraints);
        let jacobian_width = constraints.saturating_mul(dimensions);
        Extras {
            gradient: row_mut(&mut self.gradients, position, dimensions),
            inequalities: row_mut(&mut self.inequalities, position, constraints),
            constraint_jacobian: row_mut(&mut self.constraint_jacobians, position, jacobian_width),
        }
    }
}

/// The fitness and the wanted extras of the genomes of an ask, in its order, for
/// [`Algorithm::tell_evaluations`]: flat, row-major buffers that the engine reuses from one
/// generation to the next.
///
/// The extras of an invalid fitness are unspecified: with [`NanPolicy::Invalid`], a NaN in an
/// extra makes the whole evaluation invalid, and the algorithm should not use its extras.
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
    inequalities: Option<&'a [f64]>,
    constraint_jacobians: Option<&'a [f64]>,
    dimensions: usize,
    constraints: usize,
}

impl<'a> Evaluations<'a> {
    /// The fitness values alone, without extras.
    pub fn new(fitness: &'a [Fitness]) -> Self {
        Self::from_parts(fitness, None, None, None, 0, 0)
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
        Self::with_extras(fitness, Some(gradients), None, None, dimensions, 0)
    }

    /// The fitness values with any of the extras, for genomes of `dimensions` genes with
    /// `constraints` inequality constraints: per fitness, a row of `dimensions` values of
    /// `gradients`, of `constraints` values of `inequalities`, and of `constraints × dimensions`
    /// values of `constraint_jacobians` (a row per constraint).
    ///
    /// ```
    /// use genoxide::Fitness;
    /// use genoxide::engine::Evaluations;
    ///
    /// // one genome of 2 genes with 1 constraint
    /// let fitness = [Fitness::constrained(1.5, 0.5)];
    /// let (gradient, g, jacobian) = ([1.0, 1.0], [0.5], [-1.0, -0.5]);
    /// let evaluations =
    ///     Evaluations::with_extras(&fitness, Some(&gradient), Some(&g), Some(&jacobian), 2, 1)?;
    /// assert_eq!(evaluations.inequalities(0), Some(&[0.5][..]));
    /// assert_eq!(evaluations.constraint_jacobian(0), Some(&[-1.0, -0.5][..]));
    /// # Ok::<(), genoxide::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::FitnessCount`] if a buffer doesn't have the length its rows need.
    pub fn with_extras(
        fitness: &'a [Fitness],
        gradients: Option<&'a [f64]>,
        inequalities: Option<&'a [f64]>,
        constraint_jacobians: Option<&'a [f64]>,
        dimensions: usize,
        constraints: usize,
    ) -> Result<Self> {
        let jacobian_width = constraints.saturating_mul(dimensions);
        for (buffer, width) in [
            (gradients, dimensions),
            (inequalities, constraints),
            (constraint_jacobians, jacobian_width),
        ] {
            let Some(buffer) = buffer else { continue };
            let expected = fitness.len().saturating_mul(width);
            if buffer.len() != expected {
                return Err(Error::FitnessCount {
                    expected,
                    got: buffer.len(),
                });
            }
        }
        Ok(Self::from_parts(
            fitness,
            gradients,
            inequalities,
            constraint_jacobians,
            dimensions,
            constraints,
        ))
    }

    // the engine's buffers, whose lengths it has made right
    pub(crate) fn from_parts(
        fitness: &'a [Fitness],
        gradients: Option<&'a [f64]>,
        inequalities: Option<&'a [f64]>,
        constraint_jacobians: Option<&'a [f64]>,
        dimensions: usize,
        constraints: usize,
    ) -> Self {
        Self {
            fitness,
            gradients,
            inequalities,
            constraint_jacobians,
            dimensions,
            constraints,
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
            inequalities: if self.inequalities.is_some() {
                self.constraints
            } else {
                0
            },
            constraint_jacobian: self.constraint_jacobians.is_some(),
        }
    }

    /// The length of a gradient's row: the number of genes; 0 without extras.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// The number of inequality constraints: the length of a row of their values; 0 without
    /// them.
    pub fn constraints(&self) -> usize {
        self.constraints
    }

    /// The gradients of all genomes, row-major, if they were wanted.
    pub fn gradients(&self) -> Option<&'a [f64]> {
        self.gradients
    }

    /// The gradient of the genome at `position`, if gradients were wanted and the position is in
    /// bounds.
    pub fn gradient(&self, position: usize) -> Option<&'a [f64]> {
        self.row(self.gradients, position, self.dimensions)
    }

    /// The values `gᵢ(x)` of the inequality constraints of the genome at `position`, if they
    /// were wanted and the position is in bounds.
    pub fn inequalities(&self, position: usize) -> Option<&'a [f64]> {
        self.row(self.inequalities, position, self.constraints)
    }

    /// The Jacobian of the inequality constraints of the genome at `position`, row-major
    /// (`∂gᵢ / ∂xⱼ` at `i * dimensions + j`), if it was wanted and the position is in bounds.
    pub fn constraint_jacobian(&self, position: usize) -> Option<&'a [f64]> {
        let width = self.constraints.saturating_mul(self.dimensions);
        self.row(self.constraint_jacobians, position, width)
    }

    fn row(&self, buffer: Option<&'a [f64]>, position: usize, width: usize) -> Option<&'a [f64]> {
        if position >= self.fitness.len() {
            return None;
        }
        let start = position * width;
        buffer?.get(start..start + width)
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
    fn batch_rows_of_constraints() {
        // 2 genomes of 3 genes with 2 constraints
        let (mut g, mut jacobians) = ([0.0; 4], [0.0; 12]);
        let mut extras = BatchExtras::new(None, Some(&mut g), Some(&mut jacobians), 3, 2);
        assert_eq!(
            extras.wanted(),
            Wanted::NOTHING
                .with_inequalities()
                .with_constraint_jacobian()
        );
        assert_eq!((extras.dimensions(), extras.constraints()), (3, 2));
        let mut second = extras.get(1);
        assert!(second.gradient().is_none());
        second.inequalities().unwrap()[1] = 1.0;
        second.constraint_jacobian().unwrap()[5] = 2.0;
        assert!(extras.get(2).inequalities().is_none());
        assert!(extras.get(2).constraint_jacobian().is_none());
        assert_eq!(g, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(jacobians[11], 2.0);
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
        assert_eq!(evaluations.inequalities(0), None);
        assert_eq!(Evaluations::new(&fitness).provided(), Provided::NOTHING);
        assert!(Evaluations::new(&[]).is_empty());
        // 3 genomes of 2 genes with 1 constraint: a value and a row of 2 per genome
        let g = [0.1, 0.2, 0.3];
        let jacobians = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            Evaluations::with_extras(&fitness, None, Some(&g[..2]), None, 2, 1).unwrap_err(),
            Error::FitnessCount {
                expected: 3,
                got: 2
            }
        );
        assert!(Evaluations::with_extras(&fitness, None, None, Some(&g), 2, 1).is_err());
        let evaluations =
            Evaluations::with_extras(&fitness, None, Some(&g), Some(&jacobians), 2, 1).unwrap();
        assert_eq!(
            evaluations.provided(),
            Provided::NOTHING
                .with_inequalities(1)
                .with_constraint_jacobian()
        );
        assert_eq!(evaluations.constraints(), 1);
        assert_eq!(evaluations.inequalities(2), Some(&[0.3][..]));
        assert_eq!(evaluations.constraint_jacobian(1), Some(&[3.0, 4.0][..]));
        assert_eq!(evaluations.constraint_jacobian(3), None);
        assert_eq!(evaluations.gradient(0), None);
    }

    #[test]
    fn wanted_and_provided() {
        assert!(Wanted::default().is_empty());
        assert!(!Wanted::NOTHING.with_gradient().is_empty());
        assert!(!Wanted::NOTHING.with_inequalities().is_empty());
        assert!(!Wanted::NOTHING.with_constraint_jacobian().is_empty());
        assert_eq!(Wanted::NOTHING.missing_from(Provided::NOTHING), None);
        assert!(Provided::GRADIENT == Provided::NOTHING.with_gradient());
        assert!(!Provided::NOTHING.with_inequalities(1).is_empty());
        assert!(!Provided::NOTHING.with_constraint_jacobian().is_empty());
        let values = Wanted::NOTHING.with_inequalities();
        assert_eq!(
            values.missing_from(Provided::GRADIENT),
            Some("inequality constraint values")
        );
        assert_eq!(
            values.missing_from(Provided::NOTHING.with_inequalities(3)),
            None
        );
        // a Jacobian of no constraints isn't one
        let jacobian = Wanted::NOTHING.with_constraint_jacobian();
        let empty = Provided::NOTHING.with_constraint_jacobian();
        assert_eq!(jacobian.missing_from(empty), Some("constraint Jacobian"));
        assert_eq!(jacobian.missing_from(empty.with_inequalities(1)), None);
        let mut gradient = [0.0];
        assert_eq!(
            Extras::with_gradient(&mut gradient).wanted(),
            Wanted::GRADIENT
        );
        assert_eq!(Extras::none().wanted(), Wanted::NOTHING);
        let mut g = [0.0];
        let mut extras = Extras::new(None, Some(&mut g), None);
        assert_eq!(extras.wanted(), Wanted::NOTHING.with_inequalities());
        assert!(extras.constraint_jacobian().is_none());
        extras.inequalities().unwrap()[0] = 1.0;
        assert_eq!(g, [1.0]);
    }
}
