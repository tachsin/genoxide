//! Acquisition functions: how much a point is worth evaluating, from a surrogate model's
//! predictive mean `μ` and standard deviation `σ` there, and the best value observed so far.
//!
//! Every function here is higher for a point more worth evaluating, for both objectives: when
//! minimizing, an improvement is a value below the best, and the functions mirror themselves. With
//! `z = (μ − best) / σ` when maximizing (`(best − μ) / σ` when minimizing), the standardized
//! improvement:
//!
//! | Function | Value | Reference |
//! |---|---|---|
//! | [`expected_improvement`] | `σ (φ(z) + z Φ(z))`, the expected amount by which the point beats the best | Močkus (1975); Jones, Schonlau and Welch (1998) |
//! | [`log_expected_improvement`] | its logarithm, finite and smooth where the expected improvement underflows to 0 | Ament, Daulton, Eriksson, Balandat and Bakshy (2023) |
//! | [`probability_of_improvement`] | `Φ(z − ξ / σ)`, the probability that the point beats the best by `ξ` | Kushner (1964) |
//! | [`upper_confidence_bound`] | `μ + √β σ` (`−μ + √β σ` when minimizing), an optimistic value | Srinivas, Krause, Kakade and Seeger (2010) |
//!
//! `φ` and `Φ` are the standard normal density and distribution function. With `σ = 0` (a
//! point the model is sure of) they take their limits: the improvement itself, or 0. A negative
//! or NaN `σ`, or a NaN mean or best, gives NaN. Every function uses [`math`](crate::math)'s
//! portable functions only, so it gives the same bits on every platform.
//!
//! ```
//! use genoxide::Objective;
//! use genoxide::algorithm::bo::acquisition::{expected_improvement, log_expected_improvement};
//!
//! // a model that predicts 1.0 ± 0.5 where the best so far is 1.2, minimizing
//! let ei = expected_improvement(1.0, 0.5, 1.2, Objective::Minimize);
//! assert!((log_expected_improvement(1.0, 0.5, 1.2, Objective::Minimize) - ei.ln()).abs() < 1e-14);
//! // far from any improvement, the expected improvement is 0, but its logarithm still ranks
//! // points
//! assert_eq!(expected_improvement(100.0, 0.5, 1.2, Objective::Minimize), 0.0);
//! let far = log_expected_improvement(100.0, 0.5, 1.2, Objective::Minimize);
//! let farther = log_expected_improvement(200.0, 0.5, 1.2, Objective::Minimize);
//! assert!(far.is_finite() && farther < far);
//! ```
//!
//! References: Močkus, J. (1975). On Bayesian methods for seeking the extremum. *Optimization
//! Techniques IFIP Technical Conference 1974*, LNCS 27: 400-404. Jones, D. R., Schonlau, M. and
//! Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of
//! Global Optimization* 13(4): 455-492. Ament, S., Daulton, S., Eriksson, D., Balandat, M. and
//! Bakshy, E. (2023). Unexpected improvements to expected improvement for Bayesian optimization.
//! *NeurIPS 2023*, arXiv:2310.20708. Kushner, H. J. (1964). A new method of locating the maximum
//! point of an arbitrary multipeak curve in the presence of noise. *Journal of Basic Engineering*
//! 86(1): 97-106. Srinivas, N., Krause, A., Kakade, S. and Seeger, M. (2010). Gaussian process
//! optimization in the bandit setting: no regret and experimental design. *ICML 2010*,
//! arXiv:0912.3995.

use crate::Objective;
use crate::math::{erfc, erfcx, exp, exp_m1, ln, ln_1p};
use std::f64::consts::{FRAC_1_SQRT_2, LN_2};

/// The expected improvement of a point over `best`: `σ (φ(z) + z Φ(z))`, at least 0. It
/// underflows to 0 when `z` is below about −38; [`log_expected_improvement`] doesn't.
///
/// With `sd = 0`, the improvement itself: `max(μ − best, 0)` when maximizing.
#[must_use]
pub fn expected_improvement(mean: f64, sd: f64, best: f64, objective: Objective) -> f64 {
    let improvement = improvement(mean, best, objective);
    if invalid(sd) || improvement.is_nan() {
        return f64::NAN;
    }
    if sd == 0.0 {
        return improvement.max(0.0);
    }
    let z = improvement / sd;
    (sd * (normal_pdf(z) + z * normal_cdf(z))).max(0.0)
}

/// The natural logarithm of the [expected improvement](expected_improvement), computed so that it
/// stays finite, accurate and smooth where the expected improvement underflows to 0: Ament et
/// al.'s (2023) `LogEI`, `ln σ + log_h(z)` (their eq. 8), with `log_h` as their eq. 9.
///
/// For `z > −1`, `ln(φ(z) + z Φ(z))` directly. For `−1/√ε < z ≤ −1` (ε the machine epsilon),
/// `−z²/2 − c₁ + log1mexp(ln(erfcx(−z/√2) |z|) + c₂)`, with `c₁ = ln(2π)/2`, `c₂ = ln(π/2)/2` and
/// `log1mexp(x) = ln(1 − eˣ)` (their eq. 13). Below, the asymptote `−z²/2 − c₁ − 2 ln |z|`.
///
/// With `sd = 0`, the logarithm of the improvement: `−∞` without one.
#[must_use]
pub fn log_expected_improvement(mean: f64, sd: f64, best: f64, objective: Objective) -> f64 {
    let improvement = improvement(mean, best, objective);
    if invalid(sd) || improvement.is_nan() {
        return f64::NAN;
    }
    if sd == 0.0 {
        return if improvement > 0.0 {
            ln(improvement)
        } else {
            f64::NEG_INFINITY
        };
    }
    ln(sd) + log_h(improvement / sd)
}

/// The probability that a point improves on `best` by more than `xi` (`ξ ≥ 0`, often 0.01 of the
/// values' range, to prefer larger improvements): `Φ(z − ξ / σ)`.
///
/// With `sd = 0`, 1 if the improvement is larger than `xi`, else 0. NaN for a negative or NaN
/// `xi`.
#[must_use]
pub fn probability_of_improvement(
    mean: f64,
    sd: f64,
    best: f64,
    xi: f64,
    objective: Objective,
) -> f64 {
    let improvement = improvement(mean, best, objective) - xi;
    if invalid(sd) || invalid(xi) || improvement.is_nan() {
        return f64::NAN;
    }
    if sd == 0.0 {
        return if improvement > 0.0 { 1.0 } else { 0.0 };
    }
    normal_cdf(improvement / sd)
}

/// The upper confidence bound `μ + √β σ` when maximizing, and the lower one, negated, `−μ + √β σ`,
/// when minimizing: higher is better either way. `β ≥ 0` weighs exploration (`σ`) against
/// exploitation (`μ`); Srinivas et al.'s schedule raises it slowly with the number of evaluations.
///
/// NaN for a negative or NaN `sd` or `beta`.
#[must_use]
pub fn upper_confidence_bound(mean: f64, sd: f64, beta: f64, objective: Objective) -> f64 {
    if invalid(sd) || invalid(beta) {
        return f64::NAN;
    }
    let mean = match objective {
        Objective::Maximize => mean,
        Objective::Minimize => -mean,
    };
    mean + beta.sqrt() * sd
}

// a negative or NaN standard deviation or parameter
fn invalid(x: f64) -> bool {
    x.is_nan() || x < 0.0
}

// how much `mean` beats `best`, positive for an improvement
fn improvement(mean: f64, best: f64, objective: Objective) -> f64 {
    match objective {
        Objective::Maximize => mean - best,
        Objective::Minimize => best - mean,
    }
}

// ln(2π) / 2 and ln(π/2) / 2, Ament et al.'s c₁ and c₂
const C1: f64 = 0.918_938_533_204_672_8;
const C2: f64 = 0.225_791_352_644_727_43;

// 1 / √(2π)
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

// the standard normal density
fn normal_pdf(z: f64) -> f64 {
    FRAC_1_SQRT_2PI * exp(-0.5 * z * z)
}

// the standard normal distribution function, Φ(z) = erfc(−z / √2) / 2, accurate in both tails
fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z * FRAC_1_SQRT_2)
}

// ln(φ(z) + z Φ(z)), Ament et al.'s eq. 9
fn log_h(z: f64) -> f64 {
    // 1 / √ε: below, erfcx(−z/√2) |z| √(π/2) rounds to 1, and the asymptote is exact
    const ASYMPTOTE: f64 = 67_108_864.0; // 2^26 = 1 / √(2^-52)
    if z > -1.0 {
        ln(normal_pdf(z) + z * normal_cdf(z))
    } else if z > -ASYMPTOTE {
        -0.5 * z * z - C1 + log1mexp(ln(erfcx(-z * FRAC_1_SQRT_2) * -z) + C2)
    } else {
        -0.5 * z * z - C1 - 2.0 * ln(-z)
    }
}

// ln(1 − eˣ) for x ≤ 0, accurate near 0 and for very negative x (Mächler, 2012, cited by Ament
// et al. as their eq. 13)
fn log1mexp(x: f64) -> f64 {
    if x > -LN_2 {
        ln(-exp_m1(x))
    } else {
        ln_1p(-exp(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Objective::{Maximize, Minimize};
    use std::f64::consts::PI;

    // ulps between a and b
    fn ulps(a: f64, b: f64) -> u64 {
        let key = |x: f64| {
            let bits = x.to_bits() as i64;
            if bits < 0 { i64::MIN - bits } else { bits }
        };
        key(a).abs_diff(key(b))
    }

    #[test]
    fn expected_improvement_closed_forms() {
        // at μ = best and σ = 1, EI = φ(0) = 1 / √(2π)
        let phi_0 = 0.398_942_280_401_432_7;
        assert_eq!(phi_0, 1.0 / (2.0 * PI).sqrt());
        // correctly rounded, from mpmath: computed in f64, ln(2π) / 2 is 1 ulp below
        assert_eq!(C1, 0.9189385332046728);
        assert!(ulps(C1, ln(2.0 * PI) / 2.0) <= 1);
        assert_eq!(C2, 0.22579135264472744);
        assert!(ulps(C2, ln(PI / 2.0) / 2.0) <= 1);
        assert_eq!(expected_improvement(3.0, 1.0, 3.0, Maximize), phi_0);
        assert_eq!(expected_improvement(3.0, 1.0, 3.0, Minimize), phi_0);
        // σ scales it
        assert_eq!(expected_improvement(3.0, 2.0, 3.0, Maximize), 2.0 * phi_0);
        // no uncertainty: the improvement itself
        assert_eq!(expected_improvement(5.0, 0.0, 3.0, Maximize), 2.0);
        assert_eq!(expected_improvement(5.0, 0.0, 3.0, Minimize), 0.0);
        assert_eq!(expected_improvement(1.0, 0.0, 3.0, Minimize), 2.0);
        // far better than the best: about the improvement
        let sure = expected_improvement(13.0, 1.0, 3.0, Maximize);
        assert!((sure - 10.0).abs() < 1e-12);
        for invalid in [
            expected_improvement(1.0, -1.0, 0.0, Maximize),
            expected_improvement(1.0, f64::NAN, 0.0, Maximize),
            expected_improvement(f64::NAN, 1.0, 0.0, Maximize),
        ] {
            assert!(invalid.is_nan());
        }
    }

    #[test]
    fn minimizing_mirrors_maximizing() {
        for (mean, sd, best) in [(0.3, 0.7, 1.0), (-2.0, 0.1, 5.0), (4.0, 3.0, -1.0)] {
            assert_eq!(
                expected_improvement(mean, sd, best, Maximize),
                expected_improvement(-mean, sd, -best, Minimize)
            );
            assert_eq!(
                log_expected_improvement(mean, sd, best, Maximize),
                log_expected_improvement(-mean, sd, -best, Minimize)
            );
            assert_eq!(
                probability_of_improvement(mean, sd, best, 0.01, Maximize),
                probability_of_improvement(-mean, sd, -best, 0.01, Minimize)
            );
            assert_eq!(
                upper_confidence_bound(mean, sd, 2.0, Maximize),
                upper_confidence_bound(-mean, sd, 2.0, Minimize)
            );
        }
    }

    #[test]
    fn log_h_against_mpmath() {
        // tests/reference/special_functions.py: ln(φ(z) + z Φ(z)) to 50 digits, rounded
        let reference = [
            (5.0, 1.6094379231264313),
            (1.0, 0.08002621884930694),
            (0.0, -0.9189385332046728),
            (-0.5, -1.6205162643873199),
            (-0.999, -2.4832171154475855),
            (-1.0, -2.4851210257126413),
            (-1.001, -2.4870256579553893),
            (-2.0, -4.768783523917114),
            (-5.0, -16.74430116266099),
            (-10.0, -55.55312203612235),
            (-30.0, -457.724653760598),
            (-40.0, -808.29856835662),
            (-1000.0, -500014.73445209116),
            (-10000000.0, -50000000000033.16),
            (-67000000.0, -2244500000000037.0),
            (-67200000.0, -2257920000000037.0),
            (-100000000.0, -5000000000000038.0),
            (-10000000000.0, -5e+19),
            (-1e+100, -5e+199),
        ];
        for (z, expected) in reference {
            let value = log_h(z);
            assert!(
                ulps(value, expected) <= 8,
                "log_h({z}) = {value}, not {expected}"
            );
        }
    }

    #[test]
    fn log_expected_improvement_is_the_logarithm_where_it_can_be() {
        for z in (-300..=300).map(|k| f64::from(k) / 10.0) {
            for sd in [1e-3, 1.0, 50.0] {
                let mean = 2.0 + z * sd;
                let ei = expected_improvement(mean, sd, 2.0, Maximize);
                let log_ei = log_expected_improvement(mean, sd, 2.0, Maximize);
                assert!(
                    (log_ei - ln(ei)).abs() <= 1e-12 * ln(ei).abs().max(1.0),
                    "z = {z}, σ = {sd}: {log_ei} against ln {ei}"
                );
            }
        }
    }

    #[test]
    fn log_expected_improvement_keeps_ranking_where_it_underflows() {
        let mut previous = f64::INFINITY;
        for k in 0..400 {
            let z = -40.0 * 1.06_f64.powi(k);
            let log_ei = log_expected_improvement(z, 1.0, 0.0, Maximize);
            assert_eq!(expected_improvement(z, 1.0, 0.0, Maximize), 0.0);
            assert!(log_ei.is_finite() || z < -1e154, "z = {z}");
            assert!(log_ei < previous, "z = {z}");
            previous = log_ei;
        }
        assert_eq!(
            log_expected_improvement(0.0, 0.0, 1.0, Maximize),
            f64::NEG_INFINITY
        );
        assert_eq!(log_expected_improvement(3.0, 0.0, 1.0, Maximize), LN_2);
    }

    #[test]
    fn probability_of_improvement_and_upper_confidence_bound() {
        assert_eq!(
            probability_of_improvement(1.0, 2.0, 1.0, 0.0, Maximize),
            0.5
        );
        // one standard deviation above the best
        let one_sigma = probability_of_improvement(3.0, 2.0, 1.0, 0.0, Maximize);
        assert!((one_sigma - 0.841_344_746_068_542_9).abs() < 1e-15);
        // ξ asks for more
        assert!(probability_of_improvement(3.0, 2.0, 1.0, 1.0, Maximize) < one_sigma);
        assert_eq!(
            probability_of_improvement(3.0, 0.0, 1.0, 1.0, Maximize),
            1.0
        );
        assert_eq!(
            probability_of_improvement(3.0, 0.0, 1.0, 2.0, Maximize),
            0.0
        );
        assert!(probability_of_improvement(3.0, 1.0, 1.0, -1.0, Maximize).is_nan());
        assert_eq!(upper_confidence_bound(1.0, 2.0, 4.0, Maximize), 5.0);
        assert_eq!(upper_confidence_bound(1.0, 2.0, 4.0, Minimize), 3.0);
        assert!(upper_confidence_bound(1.0, 2.0, -1.0, Maximize).is_nan());
    }

    #[test]
    fn portable_values() {
        let values = [
            expected_improvement(0.3, 0.7, 1.0, Maximize),
            log_expected_improvement(0.3, 0.7, 1.0, Maximize),
            log_expected_improvement(-30.0, 0.7, 1.0, Maximize),
            probability_of_improvement(0.3, 0.7, 1.0, 0.01, Maximize),
        ];
        assert_eq!(
            values.map(f64::to_bits),
            [
                4588565738335155431,  // 0.05832082941138044
                13836953611289785667, // -2.8417959696513733
                13875286729078217892, // -989.4707096078387
                4594760526153066046,  // 0.15522321934831668
            ],
            "the acquisition functions changed, which breaks reproducibility"
        );
    }
}
