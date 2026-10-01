use super::*;
use crate::math::sin_cos;
use std::f64::consts::PI;

/// A search run to its end, every trial evaluated by `phi` (φ and φ′).
struct Run {
    status: Status,
    steps: Vec<f64>,
    /// φ′ at the last step told.
    derivative: f64,
}

/// Runs a search, and checks the strong Wolfe conditions at the step of every `Converged`.
fn line_search(phi: impl Fn(f64) -> (f64, f64), initial_step: f64, settings: Settings) -> Run {
    let (value, derivative) = phi(0.0);
    let mut search = MoreThuente::new(value, derivative, initial_step, settings).unwrap();
    let mut steps = Vec::new();
    loop {
        let step = search.next().unwrap();
        assert_eq!(search.next(), Some(step), "next() changes nothing");
        let (phi_step, dphi_step) = phi(step);
        steps.push(step);
        let status = search.tell(phi_step, dphi_step);
        assert_eq!(search.trials(), steps.len());
        if let Status::Evaluate(next) = status {
            assert_eq!(search.next(), Some(next));
            continue;
        }
        if let Status::Converged(at) = status {
            assert_eq!(at, step);
            assert!(
                phi_step <= value + settings.ftol * step * derivative,
                "sufficient decrease at {step}"
            );
            assert!(
                dphi_step.abs() <= settings.gtol * derivative.abs(),
                "curvature at {step}"
            );
        }
        assert_eq!(search.next(), None);
        assert_eq!(
            search.tell(0.0, 0.0),
            status,
            "a stopped search stays stopped"
        );
        assert_eq!(search.trials(), steps.len());
        return Run {
            status,
            steps,
            derivative: dphi_step,
        };
    }
}

fn settings(ftol: f64, gtol: f64) -> Settings {
    Settings {
        ftol,
        gtol,
        ..Settings::QUASI_NEWTON
    }
}

// The test functions of Moré and Thuente (1994), section 5, with φ′ by hand.

/// Eq. 5.1, β = 2: concave right of its minimizer √2.
fn phi1(a: f64) -> (f64, f64) {
    let beta = 2.0;
    let d = a * a + beta;
    (-a / d, (a * a - beta) / (d * d))
}

/// Eq. 5.2, β = 0.004: concave left of its minimizer 1.596.
fn phi2(a: f64) -> (f64, f64) {
    let x = a + 0.004;
    let (x3, x4) = (x * x * x, x * x * x * x);
    (x4 * x - 2.0 * x4, 5.0 * x4 - 8.0 * x3)
}

/// Eq. 5.3, Plassmann's function, β = 0.01, l = 39: φ′(0) = −β, and oscillations away from the
/// minimizer 1.
fn phi3(a: f64) -> (f64, f64) {
    let (beta, l) = (0.01, 39.0);
    let (value, derivative) = if a <= 1.0 - beta {
        (1.0 - a, -1.0)
    } else if a >= 1.0 + beta {
        (a - 1.0, 1.0)
    } else {
        (
            (a - 1.0) * (a - 1.0) / (2.0 * beta) + 0.5 * beta,
            (a - 1.0) / beta,
        )
    };
    let (sin, cos) = sin_cos(l * PI / 2.0 * a);
    (
        value + 2.0 * (1.0 - beta) / (l * PI) * sin,
        derivative + (1.0 - beta) * cos,
    )
}

/// Eq. 5.4, Yanai, Ozawa and Kaneko's convex functions.
fn phi4(beta1: f64, beta2: f64) -> impl Fn(f64) -> (f64, f64) {
    let gamma = |beta: f64| (1.0 + beta * beta).sqrt() - beta;
    let (g1, g2) = (gamma(beta1), gamma(beta2));
    move |a: f64| {
        let left = ((1.0 - a) * (1.0 - a) + beta2 * beta2).sqrt();
        let right = (a * a + beta1 * beta1).sqrt();
        (
            g1 * left + g2 * right,
            -g1 * (1.0 - a) / left + g2 * a / right,
        )
    }
}

/// φ: a step to φ and φ′ there.
type Phi = dyn Fn(f64) -> (f64, f64);

const STARTS: [f64; 4] = [1e-3, 1e-1, 1e1, 1e3];

/// A row of the paper's tables: m (evaluations), α_m and φ′(α_m) as printed.
struct Row {
    evaluations: usize,
    step: f64,
    /// The unit of the last printed digit of α_m.
    unit: f64,
    derivative: (f64, i32),
}

const fn row(evaluations: usize, step: f64, unit: f64, derivative: (f64, i32)) -> Row {
    Row {
        evaluations,
        step,
        unit,
        derivative,
    }
}

/// Checks a table: the evaluations exactly, α_m and φ′(α_m) to the printed digits (within half a
/// unit of the last one, the ties of the printed rounding included). `except`: rows whose φ′
/// isn't compared.
fn check_table(
    table: &str,
    phi: impl Fn(f64) -> (f64, f64),
    settings: Settings,
    rows: [Row; 4],
    except: &[usize],
) {
    let near = |got: f64, printed: f64, unit: f64| (got - printed).abs() <= 0.5 * unit * 1.000_001;
    for (i, (start, row)) in STARTS.iter().zip(rows).enumerate() {
        let run = line_search(&phi, *start, settings);
        let at = *run.steps.last().unwrap();
        assert_eq!(
            run.status,
            Status::Converged(at),
            "table {table}, α₀ = {start}"
        );
        assert_eq!(
            run.steps.len(),
            row.evaluations,
            "table {table}, α₀ = {start}: evaluations"
        );
        assert!(
            near(at, row.step, row.unit),
            "table {table}, α₀ = {start}: α_m = {at}, printed {}",
            row.step
        );
        if !except.contains(&i) {
            let (mantissa, exponent) = row.derivative;
            let scale = crate::math::powi(10.0, exponent);
            assert!(
                near(run.derivative, mantissa * scale, 0.1 * scale),
                "table {table}, α₀ = {start}: φ′(α_m) = {}, printed {mantissa}e{exponent}",
                run.derivative
            );
        }
    }
}

/// Table 5.1: φ₁ (eq. 5.1, β = 2), μ = 0.001, η = 0.1.
#[test]
fn table_5_1() {
    check_table(
        "5.1",
        phi1,
        settings(0.001, 0.1),
        [
            row(6, 1.4, 0.1, (-9.2, -3)),
            row(3, 1.4, 0.1, (4.7, -3)),
            row(1, 10.0, 1.0, (9.4, -3)),
            row(4, 37.0, 1.0, (7.3, -4)),
        ],
        &[],
    );
}

/// Table 5.2: φ₂ (eq. 5.2, β = 0.004), μ = η = 0.1.
///
/// φ′(α_m) from α₀ = 10⁻³ is 3.8e-9, the paper's 7.1e-9: every trial agrees with the paper's
/// (the extrapolations 0.001, 0.005, ..., 5.461 of section 5, then the same interval), and the
/// last is a cubic step between trials 1.7e-3 apart whose values of φ differ in the 7th digit,
/// so it lands 1.6e-10 from the paper's: rounding, the same with φ₂ evaluated in other orders.
#[test]
fn table_5_2() {
    check_table(
        "5.2",
        phi2,
        settings(0.1, 0.1),
        [
            row(12, 1.6, 0.1, (7.1, -9)),
            row(8, 1.6, 0.1, (1.0, -10)),
            row(8, 1.6, 0.1, (-5.0, -9)),
            row(11, 1.6, 0.1, (-2.3, -8)),
        ],
        &[0],
    );
}

/// Table 5.3: φ₃ (eq. 5.3, β = 0.01, l = 39), μ = η = 0.1.
#[test]
fn table_5_3() {
    check_table(
        "5.3",
        phi3,
        settings(0.1, 0.1),
        [
            row(12, 1.0, 0.1, (-5.1, -5)),
            row(12, 1.0, 0.1, (-1.9, -4)),
            row(10, 1.0, 0.1, (-2.0, -6)),
            row(13, 1.0, 0.1, (-1.6, -5)),
        ],
        &[],
    );
}

/// Table 5.4: φ (eq. 5.4) with β₁ = β₂ = 0.001, μ = η = 0.001. From α₀ = 10⁻³, α_m = 0.085
/// exactly (the extrapolations 0.001, 0.005, 0.021, 0.085), printed 0.08.
#[test]
fn table_5_4() {
    check_table(
        "5.4",
        phi4(0.001, 0.001),
        settings(0.001, 0.001),
        [
            row(4, 0.08, 0.01, (-6.9, -5)),
            row(1, 0.10, 0.01, (-4.9, -5)),
            row(3, 0.35, 0.01, (-2.9, -6)),
            row(4, 0.83, 0.01, (1.6, -5)),
        ],
        &[],
    );
}

/// Table 5.5: φ (eq. 5.4) with β₁ = 0.01, β₂ = 0.001, μ = η = 0.001.
#[test]
fn table_5_5() {
    check_table(
        "5.5",
        phi4(0.01, 0.001),
        settings(0.001, 0.001),
        [
            row(6, 0.075, 0.001, (1.9, -4)),
            row(3, 0.078, 0.001, (7.4, -4)),
            row(7, 0.073, 0.001, (-2.6, -4)),
            row(8, 0.076, 0.001, (4.5, -4)),
        ],
        &[],
    );
}

/// Table 5.6: φ (eq. 5.4) with β₁ = 0.001, β₂ = 0.01, μ = η = 0.001.
#[test]
fn table_5_6() {
    check_table(
        "5.6",
        phi4(0.001, 0.01),
        settings(0.001, 0.001),
        [
            row(13, 0.93, 0.01, (5.2, -4)),
            row(11, 0.93, 0.01, (8.4, -5)),
            row(8, 0.92, 0.01, (-2.4, -4)),
            row(11, 0.92, 0.01, (-3.2, -4)),
        ],
        &[],
    );
}

fn evaluations(phi: impl Fn(f64) -> (f64, f64), settings: Settings) -> Vec<usize> {
    STARTS
        .iter()
        .map(|&start| {
            let run = line_search(&phi, start, settings);
            assert!(matches!(run.status, Status::Converged(_)));
            run.steps.len()
        })
        .collect()
}

/// The other settings of section 5's text. Its α_k counts the evaluations (it names α₄ ≈ 37 for
/// the m = 4 of table 5.1), except in the extrapolations α₁ = 0.005, ..., α₅ = 1.365.
#[test]
fn section_5_text() {
    // "the extrapolation process generates iterates ... α₁ = 0.005, α₂ = 0.021, α₃ = 0.085,
    // α₄ = 0.341, α₅ = 1.365": table 5.1 from α₀ = 10⁻³
    let run = line_search(phi1, 1e-3, settings(0.001, 0.1));
    assert_eq!(run.steps, [0.001, 0.005, 0.021, 0.085, 0.341, 1.365]);

    // μ = η = 0.1: "terminates with α₃ ≈ 1.6 when α₀ = 10 and with α₇ ≈ 1.6 when α₀ = 10⁺³.
    // There is no change in the behavior of the algorithm from the other two starting points."
    assert_eq!(evaluations(phi1, settings(0.1, 0.1)), [6, 3, 3, 7]);
    for start in [1e1, 1e3] {
        let step = *run_steps(phi1, start, settings(0.1, 0.1)).last().unwrap();
        assert!((step - 1.6).abs() < 0.05, "{step}");
    }

    // η = 0.001: "six function evaluations for α₀ = 10 and ten ... for α₀ = 10⁺³ ... for
    // α₀ = 10⁻³ and α₀ = 10⁻¹ is, respectively, 8 and 4", with α_m near α* = √2. From 10⁻³
    // genoxide needs 9, with μ = 0.1 or 0.001 (the text's "leave μ unchanged" is ambiguous),
    // under every reading of the paper and of the authors' implementation tried: the six
    // extrapolations to 1.365 (where |φ′| = 9.2e-3 > η |φ′(0)| = 5e-4), a seventh that brackets,
    // and two interpolations.
    for ftol in [0.1, 0.001] {
        assert_eq!(evaluations(phi1, settings(ftol, 0.001)), [9, 4, 6, 10]);
        for start in STARTS {
            let step = *run_steps(phi1, start, settings(ftol, 0.001))
                .last()
                .unwrap();
            assert!((step - 2f64.sqrt()).abs() < 0.01, "{step}");
        }
    }

    // table 5.6's function with μ = 0.001 and η = 0.1: "2, 1, 3, 4"
    assert_eq!(
        evaluations(phi4(0.001, 0.01), settings(0.001, 0.1)),
        [2, 1, 3, 4]
    );

    // table 5.2: "these results remain unchanged if we set η = 0.1 and choose any μ < η"
    for ftol in [1e-4, 1e-3, 1e-2, 0.05, 0.09] {
        assert_eq!(evaluations(phi2, settings(ftol, 0.1)), [12, 8, 8, 11]);
    }
}

fn run_steps(phi: impl Fn(f64) -> (f64, f64), start: f64, settings: Settings) -> Vec<f64> {
    line_search(phi, start, settings).steps
}

/// The final steps to the bit, on every platform: the search and the functions use only +, −,
/// ×, ÷, `sqrt` and `genoxide::math`. (`f64`'s `Debug` prints the shortest decimal that reads
/// back to the same bits.)
#[test]
fn bit_exact() {
    let last =
        |phi: &Phi, start, settings| *line_search(phi, start, settings).steps.last().unwrap();
    let got = [
        last(&phi2, 1e-3, settings(0.1, 0.1)),
        last(&phi2, 1e3, settings(0.1, 0.1)),
        last(&phi3, 1e-1, settings(0.1, 0.1)),
        last(&phi3, 1e3, settings(0.1, 0.1)),
        last(&phi4(0.001, 0.01), 1e-3, settings(0.001, 0.001)),
        last(&phi4(0.01, 0.001), 1e1, settings(0.001, 0.001)),
    ];
    let expected = [
        1.5960000001860755,
        1.595999998872531,
        0.9999988033548208,
        0.9999999017146364,
        0.9279032286386139,
        0.07314201106897261,
    ];
    assert_eq!(got.map(f64::to_bits), expected.map(f64::to_bits), "{got:?}");
}

/// The strong Wolfe conditions at every step a search converges to (checked by `line_search`), and
/// every search converges, on the paper's functions, convex quadratics and a quartic, from
/// starts 10⁻⁶ to 10⁶, with the quasi-Newton and conjugate gradient constants.
#[test]
fn strong_wolfe_conditions() {
    let quadratic =
        |curvature: f64| move |a: f64| (curvature * a * a - a, 2.0 * curvature * a - 1.0);
    let quartic = |a: f64| {
        let x = a - 3.0;
        (x * x * x * x + 0.1 * a, 4.0 * x * x * x + 0.1)
    };
    let functions: Vec<Box<Phi>> = vec![
        Box::new(phi1),
        Box::new(phi2),
        Box::new(phi3),
        Box::new(phi4(0.001, 0.001)),
        Box::new(phi4(0.01, 0.001)),
        Box::new(phi4(0.001, 0.01)),
        Box::new(quadratic(1e-4)),
        Box::new(quadratic(1.0)),
        Box::new(quadratic(1e4)),
        Box::new(quartic),
    ];
    let mut converged = 0;
    for phi in &functions {
        for settings in [
            Settings::QUASI_NEWTON,
            Settings::CONJUGATE_GRADIENT,
            settings(0.1, 0.1),
            settings(0.001, 0.001),
        ] {
            for exponent in -12..=12 {
                let start = crate::math::pow(10.0, f64::from(exponent) / 2.0);
                let run = line_search(phi, start, settings);
                assert!(
                    matches!(run.status, Status::Converged(_)),
                    "{:?} from {start}",
                    run.status
                );
                converged += 1;
            }
        }
    }
    assert_eq!(converged, functions.len() * 4 * 25);
}

#[test]
fn a_descent_direction_is_required() {
    for derivative in [0.0, 1.0, f64::NAN, f64::NEG_INFINITY] {
        let error = MoreThuente::new(1.0, derivative, 1.0, Settings::QUASI_NEWTON).unwrap_err();
        assert!(
            matches!(
                error,
                Error::InvalidSetting {
                    setting: "derivative",
                    ..
                }
            ),
            "{error}"
        );
    }
    for value in [f64::NAN, f64::INFINITY] {
        assert!(MoreThuente::new(value, -1.0, 1.0, Settings::QUASI_NEWTON).is_err());
    }
}

#[test]
fn invalid_settings() {
    let base = Settings::QUASI_NEWTON;
    let invalid = [
        ("ftol", Settings { ftol: 0.0, ..base }),
        ("ftol", Settings { ftol: 1.0, ..base }),
        (
            "gtol",
            Settings {
                gtol: f64::NAN,
                ..base
            },
        ),
        ("xtol", Settings { xtol: -1.0, ..base }),
        (
            "min_step",
            Settings {
                min_step: -1.0,
                ..base
            },
        ),
        (
            "max_step",
            Settings {
                min_step: 2.0,
                max_step: 1.0,
                ..base
            },
        ),
        (
            "max_step",
            Settings {
                max_step: f64::INFINITY,
                ..base
            },
        ),
        (
            "max_trials",
            Settings {
                max_trials: 0,
                ..base
            },
        ),
    ];
    for (name, settings) in invalid {
        let error = MoreThuente::new(0.0, -1.0, 1.0, settings).unwrap_err();
        assert!(
            matches!(error, Error::InvalidSetting { setting, .. } if setting == name),
            "{name}: {error}"
        );
    }
    let bounded = Settings {
        min_step: 1.0,
        max_step: 2.0,
        ..base
    };
    for step in [0.0, 0.5, 3.0, f64::NAN] {
        let error = MoreThuente::new(0.0, -1.0, step, bounded).unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidSetting {
                setting: "initial_step",
                ..
            }
        ));
    }
    assert!(MoreThuente::new(0.0, -1.0, 1.0, bounded).is_ok());
}

/// φ(α) = −α decreases without bound: the search extrapolates to α_max and stops there.
#[test]
fn stops_at_the_largest_step() {
    let settings = Settings {
        max_step: 10.0,
        ..Settings::QUASI_NEWTON
    };
    let run = line_search(|a| (-a, -1.0), 1.0, settings);
    assert_eq!(run.steps, [1.0, 5.0, 10.0]);
    assert_eq!(
        run.status,
        Status::Stopped {
            step: 10.0,
            warning: Warning::MaxStep
        }
    );
}

/// A step at α_max with sufficient decrease and φ′ < 0 but ψ′ > 0, compared on φ: trying α_max
/// again would change nothing.
#[test]
fn stops_at_the_largest_step_compared_on_phi() {
    // φ′(0) = −1, so ψ′(α) = φ′(α) + 0.1 with μ = 0.1; at α = 1, φ′ = −0.05
    let phi = |a: f64| (-a + 0.475 * a * a, -1.0 + 0.95 * a);
    let settings = Settings {
        ftol: 0.1,
        gtol: 0.01,
        max_step: 1.0,
        ..Settings::QUASI_NEWTON
    };
    let run = line_search(phi, 1.0, settings);
    assert_eq!(run.steps, [1.0]);
    assert_eq!(
        run.status,
        Status::Stopped {
            step: 1.0,
            warning: Warning::MaxStep
        }
    );
}

/// A steep rise: no step with sufficient decrease from α_min up, so the search stops at α_min.
#[test]
fn stops_at_the_smallest_step() {
    let settings = Settings {
        min_step: 1.0,
        ..Settings::QUASI_NEWTON
    };
    let run = line_search(|a| (-a + 100.0 * a * a, -1.0 + 200.0 * a), 1.0, settings);
    assert_eq!(run.steps, [1.0]);
    assert_eq!(
        run.status,
        Status::Stopped {
            step: 1.0,
            warning: Warning::MinStep
        }
    );
}

/// φ isn't defined from 1.8 on (NaN or infinite): failed steps shrink toward the best step, and
/// later steps stay below the failures.
#[test]
fn failed_steps_shrink() {
    for undefined in [f64::NAN, f64::INFINITY] {
        let phi = |a: f64| {
            if a < 1.8 {
                (a * a - 3.0 * a, 2.0 * a - 3.0)
            } else {
                (undefined, 2.0 * a - 3.0)
            }
        };
        let run = line_search(phi, 10.0, Settings::QUASI_NEWTON);
        assert_eq!(run.steps, [10.0, 5.0, 2.5, 1.25]);
        assert_eq!(run.status, Status::Converged(1.25));

        // η = 0.1: 1.25 isn't enough, and the extrapolation from it stops short of 2.5
        let run = line_search(phi, 10.0, Settings::CONJUGATE_GRADIENT);
        assert_eq!(run.steps, [10.0, 5.0, 2.5, 1.25, 1.875, 1.5625]);
        assert_eq!(run.status, Status::Converged(1.5625));
    }
    // a NaN derivative fails a step too
    let run = line_search(
        |a: f64| {
            (
                a * a - 3.0 * a,
                if a < 1.8 { 2.0 * a - 3.0 } else { f64::NAN },
            )
        },
        2.0,
        Settings::QUASI_NEWTON,
    );
    assert_eq!(run.steps, [2.0, 1.0]);
    assert_eq!(run.status, Status::Converged(1.0));
}

/// φ defined nowhere but at 0: the search halves its steps until it stops, at the origin.
#[test]
fn invalid_everywhere() {
    let phi = |a: f64| {
        if a == 0.0 {
            (0.0, -1.0)
        } else {
            (f64::NAN, f64::NAN)
        }
    };
    let run = line_search(phi, 1.0, Settings::QUASI_NEWTON);
    assert_eq!(run.steps.len(), 20);
    assert_eq!(run.steps[1..4], [0.5, 0.25, 0.125]);
    assert_eq!(
        run.status,
        Status::Stopped {
            step: 0.0,
            warning: Warning::TooManyTrials
        }
    );

    // with α_min, the halving ends at it
    let settings = Settings {
        min_step: 0.5,
        ..Settings::QUASI_NEWTON
    };
    let run = line_search(phi, 1.0, settings);
    assert_eq!(run.steps, [1.0, 0.5]);
    assert_eq!(
        run.status,
        Status::Stopped {
            step: 0.0,
            warning: Warning::InvalidValues
        }
    );
}

#[test]
fn too_many_trials() {
    let settings = Settings {
        max_trials: 3,
        ..settings(0.1, 0.1)
    };
    let run = line_search(phi2, 1e-3, settings);
    assert_eq!(run.steps, [0.001, 0.005, 0.021]);
    assert_eq!(
        run.status,
        Status::Stopped {
            step: 0.021,
            warning: Warning::TooManyTrials
        }
    );
}

/// φ(α) = |α − 1| has |φ′| = 1 everywhere, so the curvature condition never holds: the interval
/// closes on the kink until rounding, or `xtol`, stops it.
#[test]
fn rounding_errors_and_xtol() {
    let kink = |a: f64| ((a - 1.0).abs(), if a < 1.0 { -1.0 } else { 1.0 });
    let settings = Settings {
        xtol: 0.0,
        max_trials: 1000,
        ..Settings::QUASI_NEWTON
    };
    let run = line_search(kink, 0.5, settings);
    let Status::Stopped {
        step,
        warning: Warning::RoundingErrors,
    } = run.status
    else {
        panic!("{:?}", run.status);
    };
    assert!((step - 1.0).abs() <= 4.0 * f64::EPSILON, "{step}");

    let settings = Settings {
        xtol: 1e-6,
        ..Settings::QUASI_NEWTON
    };
    let run = line_search(kink, 0.5, settings);
    let Status::Stopped {
        step,
        warning: Warning::IntervalTolerance,
    } = run.status
    else {
        panic!("{:?}", run.status);
    };
    assert!((step - 1.0).abs() <= 1e-5, "{step}");
}

/// A search saved and restored mid-way continues as the original.
#[cfg(feature = "serde")]
#[test]
fn resumes_from_a_checkpoint() {
    let (value, derivative) = phi3(0.0);
    let mut search = MoreThuente::new(value, derivative, 1e-3, settings(0.1, 0.1)).unwrap();
    for _ in 0..5 {
        let (value, derivative) = phi3(search.next().unwrap());
        search.tell(value, derivative);
    }
    let mut bytes = Vec::new();
    crate::checkpoint::save(&search, &mut bytes).unwrap();
    let mut restored: MoreThuente = crate::checkpoint::load(bytes.as_slice()).unwrap();
    assert_eq!(restored, search);
    while let Some(step) = search.next() {
        let (value, derivative) = phi3(step);
        assert_eq!(restored.next(), Some(step));
        assert_eq!(
            restored.tell(value, derivative),
            search.tell(value, derivative)
        );
    }
    assert_eq!(restored.next(), None);
}
