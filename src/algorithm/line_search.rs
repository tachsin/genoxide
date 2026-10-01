//! Line searches: a step along a descent direction that satisfies the strong Wolfe conditions.
//!
//! Given a point `x` and a descent direction `p`, a line search looks at φ(α) = f(x + α p) for
//! α ≥ 0, with φ′(0) < 0, and finds a step α that satisfies the *strong Wolfe conditions*
//! (Moré and Thuente, 1994, eq. 1.1 and 1.2):
//!
//! - sufficient decrease: φ(α) ≤ φ(0) + μ α φ′(0),
//! - curvature: |φ′(α)| ≤ η |φ′(0)|,
//!
//! for constants 0 < μ, η < 1 (c₁ and c₂ in Nocedal and Wright), with α in `[α_min, α_max]`
//! (eq. 1.4). Typical constants are μ = 10⁻⁴, and η = 0.9 for quasi-Newton methods or 0.1 for
//! nonlinear conjugate gradients (Nocedal and Wright, 2006, section 3.1):
//! [`Settings::QUASI_NEWTON`] and [`Settings::CONJUGATE_GRADIENT`].
//!
//! # The method
//!
//! [`MoreThuente`] is the search of Moré and Thuente (1994). It keeps an interval with endpoints
//! α_l (the endpoint with the least value) and α_u, which contains a step that satisfies the
//! conditions once the search has *bracketed* one, and refines it with each trial step α_t:
//!
//! - **The updating algorithm** (section 2, cases U1-U3): U1, a larger value at α_t: α_u = α_t;
//!   U2, a value no larger, with the derivative still descending toward α_t: α_l = α_t; U3, a
//!   value no larger with the derivative pointing back: α_u = α_l, α_l = α_t. The values are of
//!   the auxiliary function ψ(α) = φ(α) − φ(0) − μ φ′(0) α (whose minimizers satisfy both
//!   conditions when μ ≤ η, Theorem 2.1) until a trial has ψ(α) ≤ 0 and φ′(α) > 0, and of φ
//!   itself from then on (the modified updating algorithm of section 3, Theorem 3.3), so that
//!   the search finds a step for any μ and η.
//! - **The trial steps** (section 4): from the values and derivatives at α_l and α_t, by four
//!   cases. Case 1, a higher value: the minimizer of the cubic that interpolates both values and
//!   derivatives, α_c, if it's closer to α_l than the minimizer of the quadratic through both
//!   values and the derivative at α_l, α_q, else their midpoint. Case 2, a lower value and
//!   derivatives of opposite signs: α_c or the secant step α_s (the zero of the line through the
//!   two derivatives), whichever is farther from α_t. Case 3, a lower value, derivatives of the
//!   same sign and a smaller derivative: an extrapolation by α_c or α_s, held to
//!   α_t + 0.66 (α_u − α_t) once bracketed. Case 4, a derivative no smaller: the cubic through
//!   α_t and α_u once bracketed, else the farthest extrapolation allowed.
//! - **Safeguards** (section 2): until bracketed, the next trial lies in
//!   `[α_t + 1.1 (α_t − α_l), α_t + 4 (α_t − α_l)]` (eq. 2.2 and the text after it), so the
//!   search reaches α_max if it must; once bracketed, a bisection step replaces the trial when the
//!   interval hasn't shrunk by a factor of 0.66 in two trials, so its length goes to 0.
//! - **Termination** (sections 2 and 3): at a step that satisfies both conditions; at α_max if
//!   ψ(α_max) ≤ 0 and ψ′(α_max) < 0 (a step that satisfies both conditions lies beyond it; ≤ 0
//!   here, as in the authors' implementation, so that a search at α_max always ends); at α_min
//!   if ψ(α_min) > 0 or ψ′(α_min) ≥ 0 (one lies below it).
//!
//! A [`MoreThuente`] is resumable, the ask / tell of genoxide's algorithms on one dimension:
//! [`next`](MoreThuente::next) gives the step to evaluate next, and [`tell`](MoreThuente::tell)
//! takes φ and φ′ there and returns a [`Status`]: the next step to try, the step that satisfies
//! the conditions, or a [`Warning`] with the step the search stopped at. It never panics.
//!
//! # Where the implementation follows the authors' code
//!
//! The paper's tables come from the authors' implementation (MINPACK-2's `dcsrch` and `dcstep`,
//! by Moré and Thuente with Averick and Carter; read to compare behavior, in the copy distributed
//! with L-BFGS-B 3.0 under the BSD-3-Clause license). It differs from the paper's text in two
//! places, and with the text's rules the counts of evaluations of 11 of the 24 runs of tables
//! 5.1-5.6 differ from the paper's, so genoxide follows the implementation there:
//!
//! - ψ is used before the switch to φ only for a trial with a value of φ no larger than at α_l
//!   but without sufficient decrease (ψ(α_t) > 0); for the other trials, φ (section 4 of the
//!   text: ψ for all of them).
//! - In case 3 before bracketing, the trial is whichever of α_c and α_s is *farther* from α_t
//!   (the text: closer), and a cubic without a minimizer beyond α_t is replaced by the farthest
//!   extrapolation allowed (the text: by α_s).
//!
//! Smaller choices: the preprint's case 4 breaks off before saying what it does before
//! bracketing, and the farthest extrapolation allowed is the implementation's (and what eq. 2.2
//! needs); derivatives of equal magnitude at α_l and α_t fall in case 4, as in the implementation
//! (the text puts them in case 3, where α_s doesn't exist); the switch to φ follows section 3's
//! test, ψ(α) ≤ 0 and φ′(α) > 0 (section 4 states ψ′(α) ≥ 0; the implementation φ′(α) ≥ 0, which
//! differs from section 3's only at a step that satisfies both conditions); and the bisection
//! compares with the interval of two trials before, infinite before bracketing (I₀ = [0, ∞]; the
//! implementation starts from α_max − α_min). The interpolants are genoxide's own algebra, in a
//! scaled form that avoids overflow, and ψ drops the constant φ(0), which changes no interpolant.
//!
//! # Beyond the paper
//!
//! - **Failed steps.** A value of φ or φ′ that isn't finite (NaN or infinite: the function isn't
//!   defined there) makes the step a failed one: it never becomes an endpoint, no step beyond it
//!   is tried again, and the next trial is halfway between α_l and it.
//! - **Stops** ([`Warning`]): after `max_trials` evaluations; when rounding errors leave no step
//!   strictly inside the interval; when the interval's relative width is at most `xtol` (the
//!   authors' implementation has the last two too); and when an extrapolation held at α_max
//!   would try α_max again.
//!
//! # References
//!
//! - Moré, J. J. and Thuente, D. J. (1994). Line search algorithms with guaranteed sufficient
//!   decrease. *ACM Transactions on Mathematical Software* 20(3): 286-307.
//!   doi:10.1145/192115.192132. Read in its preprint, Argonne MCS-P330-1092 (1992).
//! - Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., chapter 3.
//!   Springer. doi:10.1007/978-0-387-40065-5.

// L-BFGS-B (batch A2 of docs/optimization-plan.md) uses the line search next.
#![allow(dead_code)]

use crate::{Error, Result};

/// δ of the paper, 0.66: the factor by which the interval must shrink in two trials, else a
/// bisection step (section 2), and the share of the way to α_u a case 3 trial may go once
/// bracketed (section 4).
const DELTA: f64 = 0.66;
/// The smallest and largest extrapolation, α⁺ − α_t over α_t − α_l, before bracketing
/// (section 2, after eq. 2.2: δ ∈ [1.1, 4]).
const EXTRAPOLATION: [f64; 2] = [1.1, 4.0];
/// genoxide's default for `xtol`: an order of magnitude below the relative width of the smallest
/// set of acceptable steps among the paper's tests (about 2.5 × 10⁻⁹ at α ≈ 1.6, section 5 on
/// table 5.2), so it never stops a search that can still succeed there, and far above rounding.
const XTOL: f64 = 1e-10;
/// genoxide's default for `max_trials`: half again the most any of the paper's runs needs (13,
/// tables 5.3 and 5.6).
const MAX_TRIALS: usize = 20;

/// The constants and bounds of a line search.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Settings {
    /// μ (c₁): the sufficient decrease condition is φ(α) ≤ φ(0) + μ α φ′(0). Between 0 and 1,
    /// exclusive.
    pub(crate) ftol: f64,
    /// η (c₂): the curvature condition is |φ′(α)| ≤ η |φ′(0)|. Between 0 and 1, exclusive.
    pub(crate) gtol: f64,
    /// The search stops with [`Warning::IntervalTolerance`] once it has bracketed a step and the
    /// interval's width is at most `xtol` times its larger end. At least 0.
    pub(crate) xtol: f64,
    /// α_min: the smallest step tried. At least 0.
    pub(crate) min_step: f64,
    /// α_max: the largest step tried. Finite, at least `min_step`.
    pub(crate) max_step: f64,
    /// The search stops with [`Warning::TooManyTrials`] after this many evaluations. At least 1.
    pub(crate) max_trials: usize,
}

impl Settings {
    /// For quasi-Newton methods (BFGS, L-BFGS): μ = 10⁻⁴ and η = 0.9 (Nocedal and Wright, 2006,
    /// section 3.1), `xtol` 10⁻¹⁰, steps in `[0, f64::MAX]`, at most 20 trials.
    pub(crate) const QUASI_NEWTON: Settings = Settings {
        ftol: 1e-4,
        gtol: 0.9,
        xtol: XTOL,
        min_step: 0.0,
        max_step: f64::MAX,
        max_trials: MAX_TRIALS,
    };

    /// For nonlinear conjugate gradients: as [`QUASI_NEWTON`](Settings::QUASI_NEWTON), with
    /// η = 0.1 (Nocedal and Wright, 2006, section 3.1).
    pub(crate) const CONJUGATE_GRADIENT: Settings = Settings {
        gtol: 0.1,
        ..Settings::QUASI_NEWTON
    };

    fn validate(&self) -> Result<()> {
        let invalid = |setting, reason: String| Err(Error::InvalidSetting { setting, reason });
        for (setting, value) in [("ftol", self.ftol), ("gtol", self.gtol)] {
            if !(value > 0.0 && value < 1.0) {
                return invalid(
                    setting,
                    format!("must be between 0 and 1 (exclusive), got {value}"),
                );
            }
        }
        if !(self.xtol >= 0.0 && self.xtol.is_finite()) {
            return invalid(
                "xtol",
                format!("must be finite and at least 0, got {}", self.xtol),
            );
        }
        if !(self.min_step >= 0.0 && self.min_step.is_finite()) {
            return invalid(
                "min_step",
                format!("must be finite and at least 0, got {}", self.min_step),
            );
        }
        if !(self.max_step >= self.min_step && self.max_step.is_finite()) {
            return invalid(
                "max_step",
                format!(
                    "must be finite and at least min_step ({}), got {}",
                    self.min_step, self.max_step
                ),
            );
        }
        if self.max_trials == 0 {
            return invalid("max_trials", "must be at least 1, got 0".to_string());
        }
        Ok(())
    }
}

/// A step with the value and derivative of φ there.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Point {
    /// α.
    pub(crate) step: f64,
    /// φ(α).
    pub(crate) value: f64,
    /// φ′(α).
    pub(crate) derivative: f64,
}

/// What a [`MoreThuente`] says after a trial.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum Status {
    /// Evaluate φ and φ′ at this step and [`tell`](MoreThuente::tell) them.
    Evaluate(f64),
    /// This step, the last one told, satisfies the sufficient decrease and curvature
    /// conditions.
    Converged(f64),
    /// The search stopped without a step that satisfies both conditions. `step` is the last one
    /// told for [`Warning::MaxStep`] and [`Warning::MinStep`], and the best step evaluated (the
    /// endpoint α_l, 0 if no trial lowered φ) for the others.
    Stopped {
        /// The step the search ends at.
        step: f64,
        /// Why it stopped.
        warning: Warning,
    },
}

/// Why a [`MoreThuente`] stopped without a step that satisfies both conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum Warning {
    /// The step is α_max, with sufficient decrease and φ still decreasing: ψ′(α_max) ≤ 0, so a
    /// step that satisfies both conditions may lie only beyond it (section 2); or φ′(α_max) < 0
    /// after a trial there that the search compared on φ.
    MaxStep,
    /// The step is α_min, with ψ(α_min) > 0 or ψ′(α_min) ≥ 0: a step that satisfies both
    /// conditions lies below it (section 2).
    MinStep,
    /// Rounding errors leave no step strictly inside the interval: the conditions can't be met to
    /// the precision of `f64` (e.g. φ isn't smooth there, or η is too small).
    RoundingErrors,
    /// The interval's width is at most `xtol` times its larger end.
    IntervalTolerance,
    /// The search evaluated `max_trials` steps.
    TooManyTrials,
    /// φ or φ′ wasn't finite at the trial steps, and no step is left between the best one and
    /// the failed ones, within the bounds.
    InvalidValues,
}

/// The line search of Moré and Thuente (1994), resumable: see the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct MoreThuente {
    settings: Settings,
    /// α = 0, φ(0) and φ′(0).
    origin: Point,
    /// α_l, the endpoint with the least value of the function used to update the interval.
    lower: Point,
    /// α_u, the other endpoint (meaningful once bracketed).
    upper: Point,
    /// Whether the interval brackets a step that satisfies the conditions: α_u is finite.
    bracketed: bool,
    /// Whether the modified updating algorithm of section 3 (on φ) is in use.
    modified: bool,
    /// The lengths of the interval after the last update and the one before, once bracketed
    /// (an interval with α_u = ∞ before).
    length: Option<f64>,
    previous_length: Option<f64>,
    /// The range of the trial step: the interval once bracketed, the extrapolation range before.
    range: [f64; 2],
    /// The step to evaluate next.
    trial: f64,
    /// The steps told.
    trials: usize,
    /// The smallest failed step above α_l, and the largest below it.
    failed_above: Option<f64>,
    failed_below: Option<f64>,
    /// The final status, once the search has stopped.
    stopped: Option<Status>,
}

impl MoreThuente {
    /// A search from φ(0) = `value` and φ′(0) = `derivative`, starting with `initial_step`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] if `value` or `derivative` isn't finite, if `derivative` isn't
    /// negative (the direction isn't a descent direction), if `initial_step` isn't positive and
    /// within `[min_step, max_step]`, or for invalid [`Settings`].
    pub(crate) fn new(
        value: f64,
        derivative: f64,
        initial_step: f64,
        settings: Settings,
    ) -> Result<Self> {
        settings.validate()?;
        if !value.is_finite() {
            return Err(Error::InvalidSetting {
                setting: "value",
                reason: format!("φ(0) must be finite, got {value}"),
            });
        }
        if !(derivative < 0.0 && derivative.is_finite()) {
            return Err(Error::InvalidSetting {
                setting: "derivative",
                reason: format!(
                    "φ′(0) must be finite and negative (a descent direction), got {derivative}"
                ),
            });
        }
        if !(initial_step > 0.0
            && initial_step >= settings.min_step
            && initial_step <= settings.max_step)
        {
            return Err(Error::InvalidSetting {
                setting: "initial_step",
                reason: format!(
                    "must be positive and between min_step ({}) and max_step ({}), got {initial_step}",
                    settings.min_step, settings.max_step
                ),
            });
        }
        let origin = Point {
            step: 0.0,
            value,
            derivative,
        };
        let mut search = MoreThuente {
            settings,
            origin,
            lower: origin,
            upper: origin,
            bracketed: false,
            modified: false,
            length: None,
            previous_length: None,
            range: [0.0, 0.0],
            trial: initial_step,
            trials: 0,
            failed_above: None,
            failed_below: None,
            stopped: None,
        };
        search.range = search.extrapolation(initial_step);
        Ok(search)
    }

    /// The step to evaluate next, or `None` once the search has stopped.
    pub(crate) fn next(&self) -> Option<f64> {
        match self.stopped {
            None => Some(self.trial),
            Some(_) => None,
        }
    }

    /// The number of steps told so far.
    pub(crate) fn trials(&self) -> usize {
        self.trials
    }

    /// α_l with φ and φ′ there: the best step evaluated (the origin before any trial lowers φ).
    pub(crate) fn best(&self) -> Point {
        self.lower
    }

    /// Takes φ(α) = `value` and φ′(α) = `derivative` at the step of [`next`](Self::next), and
    /// returns what to do next. Once the search has stopped, returns its final status again
    /// and changes nothing.
    pub(crate) fn tell(&mut self, value: f64, derivative: f64) -> Status {
        if let Some(status) = self.stopped {
            return status;
        }
        self.trials += 1;
        let trial = Point {
            step: self.trial,
            value,
            derivative,
        };
        if !(value.is_finite() && derivative.is_finite()) {
            return self.fail(trial.step);
        }
        let settings = self.settings;
        let slope = settings.ftol * self.origin.derivative; // μ φ′(0) = ψ′(α) − φ′(α)
        let sufficient = value <= self.origin.value + trial.step * slope; // ψ(α) ≤ 0
        if sufficient && derivative.abs() <= settings.gtol * -self.origin.derivative {
            return self.stop(Status::Converged(trial.step));
        }
        if trial.step == settings.max_step && sufficient && derivative <= slope {
            return self.stop_at(trial.step, Warning::MaxStep);
        }
        if trial.step == settings.min_step && (!sufficient || derivative >= slope) {
            return self.stop_at(trial.step, Warning::MinStep);
        }
        if sufficient && derivative > 0.0 {
            self.modified = true; // section 3: from now on, φ
        }
        // ψ before the switch, for a lower value of φ without sufficient decrease (the authors'
        // implementation, see the module docs); φ otherwise. ψ without the constant φ(0).
        let psi = !self.modified && !sufficient && value <= self.lower.value;
        let function = |point: Point| {
            if psi {
                Point {
                    step: point.step,
                    value: point.value - point.step * slope,
                    derivative: point.derivative - slope,
                }
            } else {
                point
            }
        };
        let (lower, upper, at) = (function(self.lower), function(self.upper), function(trial));
        let mut next = self.select(lower, upper, at);

        // the updating algorithm: cases U1-U3 (a-c of section 3, on φ)
        if at.value > lower.value {
            self.upper = trial;
            self.bracketed = true;
        } else {
            if opposite(at.derivative, lower.step - at.step) {
                self.upper = self.lower;
                self.bracketed = true;
            }
            self.lower = trial;
        }
        if self.trials >= settings.max_trials {
            return self.stop_at(self.lower.step, Warning::TooManyTrials);
        }

        if self.bracketed {
            let length = (self.upper.step - self.lower.step).abs();
            if self
                .previous_length
                .is_some_and(|previous| length >= DELTA * previous)
            {
                next = self.lower.step + 0.5 * (self.upper.step - self.lower.step);
            }
            self.previous_length = self.length;
            self.length = Some(length);
            self.range = [
                self.lower.step.min(self.upper.step),
                self.lower.step.max(self.upper.step),
            ];
        } else {
            self.range = self.extrapolation(next);
        }
        let Some(next) = self.admissible(next) else {
            return self.stop_at(self.lower.step, Warning::InvalidValues);
        };
        if self.bracketed {
            let [low, high] = self.range;
            if high - low <= settings.xtol * high {
                return self.stop_at(self.lower.step, Warning::IntervalTolerance);
            }
            if !(next > low && next < high) {
                return self.stop_at(self.lower.step, Warning::RoundingErrors);
            }
        } else if next <= self.lower.step {
            // an extrapolation held at α_max, after a step there with a lower value and φ′ < 0
            // (compared on φ, so ψ′(α_max) > 0 is possible): trying it again changes nothing
            return self.stop_at(self.lower.step, Warning::MaxStep);
        }
        self.trial = next;
        Status::Evaluate(next)
    }

    /// The trial step after `at` (section 4), from the endpoints before the update, all in the
    /// function that updates the interval.
    fn select(&self, lower: Point, upper: Point, at: Point) -> f64 {
        let [low, high] = self.range;
        if at.value > lower.value {
            // case 1: a higher value
            let quadratic = quadratic_minimizer(lower, at);
            match Cubic::new(lower, at).minimizer(true) {
                Some(cubic) if (cubic - lower.step).abs() < (quadratic - lower.step).abs() => cubic,
                Some(cubic) => cubic + 0.5 * (quadratic - cubic),
                None => quadratic,
            }
        } else if opposite(at.derivative, lower.derivative) {
            // case 2: a lower value, derivatives of opposite signs
            let secant = secant(lower, at);
            match Cubic::new(lower, at).minimizer(true) {
                Some(cubic) if (cubic - at.step).abs() >= (secant - at.step).abs() => cubic,
                _ => secant,
            }
        } else if at.derivative.abs() < lower.derivative.abs() {
            // case 3: a lower value, derivatives of the same sign, the derivative smaller
            let forward = at.step > lower.step;
            let secant = secant(lower, at);
            // the cubic's minimizer, if it tends to infinity in the direction of the step and
            // its minimum is beyond α_t
            let cubic = Cubic::new(lower, at);
            let cubic = cubic.minimizer(false).filter(|&step| {
                cubic.rises()
                    && (if forward {
                        step > at.step
                    } else {
                        step < at.step
                    })
            });
            if self.bracketed {
                let step = match cubic {
                    Some(cubic) if (cubic - at.step).abs() < (secant - at.step).abs() => cubic,
                    _ => secant,
                };
                let limit = at.step + DELTA * (upper.step - at.step);
                if forward {
                    step.min(limit)
                } else {
                    step.max(limit)
                }
            } else {
                let cubic = cubic.unwrap_or(if forward { high } else { low });
                let step = if (cubic - at.step).abs() > (secant - at.step).abs() {
                    cubic
                } else {
                    secant
                };
                step.max(low).min(high)
            }
        } else if self.bracketed {
            // case 4: a lower value, derivatives of the same sign, the derivative no smaller
            Cubic::new(at, upper)
                .minimizer(true)
                .unwrap_or(at.step + 0.5 * (upper.step - at.step))
        } else if at.step > lower.step {
            high
        } else {
            low
        }
    }

    /// The extrapolation range after a trial at `step` from α_l (section 2, after eq. 2.2).
    fn extrapolation(&self, step: f64) -> [f64; 2] {
        let distance = step - self.lower.step;
        [
            step + EXTRAPOLATION[0] * distance,
            step + EXTRAPOLATION[1] * distance,
        ]
    }

    /// `step` away from the failed steps (halfway from α_l to the nearest one it reaches) and
    /// within `[min_step, max_step]`, or `None` if that leaves no step strictly between the
    /// failed steps.
    fn admissible(&self, step: f64) -> Option<f64> {
        let best = self.lower.step;
        let step = match (self.failed_above, self.failed_below) {
            (Some(above), _) if step >= above => best + 0.5 * (above - best),
            (_, Some(below)) if step <= below => best + 0.5 * (below - best),
            _ => step,
        };
        // `max` and `min` also replace a NaN from an interpolation by a bound
        let step = step.max(self.settings.min_step).min(self.settings.max_step);
        let below_failures = self.failed_above.is_none_or(|above| step < above);
        let above_failures = self.failed_below.is_none_or(|below| step > below);
        (below_failures && above_failures).then_some(step)
    }

    /// A trial where φ or φ′ isn't finite: never an endpoint; the next trial is halfway back to
    /// α_l.
    fn fail(&mut self, step: f64) -> Status {
        let best = self.lower.step;
        if step > best {
            self.failed_above = Some(self.failed_above.map_or(step, |above| above.min(step)));
        } else {
            self.failed_below = Some(self.failed_below.map_or(step, |below| below.max(step)));
        }
        if self.trials >= self.settings.max_trials {
            return self.stop_at(best, Warning::TooManyTrials);
        }
        let next = self
            .admissible(best + 0.5 * (step - best))
            .filter(|&next| next != best);
        let Some(next) = next else {
            return self.stop_at(best, Warning::InvalidValues);
        };
        if !self.bracketed {
            self.range = self.extrapolation(next);
        }
        self.trial = next;
        Status::Evaluate(next)
    }

    fn stop_at(&mut self, step: f64, warning: Warning) -> Status {
        self.stop(Status::Stopped { step, warning })
    }

    fn stop(&mut self, status: Status) -> Status {
        self.stopped = Some(status);
        status
    }
}

/// Whether `a` and `b` have opposite signs, neither 0 (the sign of a product, without its
/// underflow).
fn opposite(a: f64, b: f64) -> bool {
    (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0)
}

/// α_q: the minimizer of the quadratic that interpolates φ at `lower` and `at` and φ′ at `lower`
/// (case 1, where it exists: a higher value at `at`, and the derivative at `lower` descending
/// toward it).
fn quadratic_minimizer(lower: Point, at: Point) -> f64 {
    let h = at.step - lower.step;
    let slope = (lower.value - at.value) / h + lower.derivative;
    lower.step + 0.5 * (lower.derivative / slope) * h
}

/// α_s: the zero of the line through φ′ at `lower` and `at`, the minimizer of the quadratic that
/// interpolates both derivatives.
fn secant(lower: Point, at: Point) -> f64 {
    lower.step + lower.derivative / (lower.derivative - at.derivative) * (at.step - lower.step)
}

/// The cubic that interpolates φ and φ′ at two steps a and b, in s = (α − a) / (b − a):
/// p(s) = φ(a) + d s + c₂ s² + c₃ s³, with d = φ′(a) (b − a), so p(1) = φ(b) and p′(1) =
/// φ′(b) (b − a).
struct Cubic {
    start: f64,
    width: f64,
    d: f64,
    c2: f64,
    c3: f64,
}

impl Cubic {
    fn new(a: Point, b: Point) -> Self {
        let width = b.step - a.step;
        let d = a.derivative * width;
        let e = b.derivative * width;
        let rise = b.value - a.value;
        Cubic {
            start: a.step,
            width,
            d,
            c2: 3.0 * rise - 2.0 * d - e,
            c3: d + e - 2.0 * rise,
        }
    }

    /// Whether the cubic tends to +∞ in the direction from a to b.
    fn rises(&self) -> bool {
        self.c3 > 0.0
    }

    /// The local minimizer, where p′(s) = d + 2 c₂ s + 3 c₃ s² = 0 and p″(s) = 2 √D ≥ 0, with
    /// D = c₂² − 3 c₃ d; `None` if there is none. `exists`: the minimizer exists in theory (cases
    /// 1, 2 and 4), so a negative D is rounding and counts as 0.
    fn minimizer(&self, exists: bool) -> Option<f64> {
        let scale = self.c2.abs().max(self.c3.abs()).max(self.d.abs());
        if !(scale > 0.0 && scale.is_finite()) {
            return None;
        }
        let (c2, c3, d) = (self.c2 / scale, self.c3 / scale, self.d / scale);
        let discriminant = c2 * c2 - 3.0 * c3 * d;
        if discriminant < 0.0 && !exists {
            return None;
        }
        let root = discriminant.max(0.0).sqrt();
        // (−c₂ + √D) / (3 c₃), in the form without cancellation
        let s = if c2 >= 0.0 {
            -d / (c2 + root)
        } else {
            (root - c2) / (3.0 * c3)
        };
        let step = self.start + s * self.width;
        step.is_finite().then_some(step)
    }
}

#[cfg(test)]
mod tests;
