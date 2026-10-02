use super::*;
use crate::genome::Integer;

// a model of a smooth function of 2 genes, from 10 points, with its best value standardized
fn model(f: impl Fn(&[f64]) -> f64, seed: u64) -> (GaussianProcess, f64) {
    let real = Real::new([-1.0..=2.0, 0.0..=3.0]).unwrap();
    let points = real
        .latin_hypercube(10, &mut StreamRng::seed_from_u64(2))
        .unwrap();
    let values: Vec<f64> = points.iter().map(|x| f(x)).collect();
    let scaling = Scaling::new(&real);
    let mut x = vec![0.0; 20];
    for (point, unit) in points.iter().zip(x.as_chunks_mut::<2>().0) {
        scaling.to_unit(point, unit);
    }
    let settings = gp::Settings {
        kernel: Kernel::Matern52,
        noise: Noise::default(),
        starts: 3,
        seed,
    };
    let model = GaussianProcess::fit_unit(settings, scaling, x, &values, None).unwrap();
    let (mean, scale) = model.standardization();
    let best = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    (model, (best - mean) / scale)
}

fn smooth(x: &[f64]) -> f64 {
    (2.0 * x[0]).sin() + 0.5 * x[1] * x[1] - 0.3 * x[0] * x[1]
}

// a constraint g(x) = x₀ + x₁ − 2.5, feasible below the diagonal, standardized
fn constraint() -> (GaussianProcess, f64) {
    let (model, _) = model(|x| x[0] + x[1] - 2.5 + 0.1 * (3.0 * x[0]).cos(), 4);
    let (mean, scale) = model.standardization();
    (model, (0.0 - mean) / scale)
}

// the gradient of `search` at `u` against central differences
fn check_gradient(search: &Search<'_>, u: [f64; 2], name: &str) {
    let mut gradient = [0.0; 2];
    let value = search.value(&u, Some(&mut gradient));
    assert_eq!(value.to_bits(), search.value(&u, None).to_bits());
    for i in 0..2 {
        let h = 1e-6;
        let (mut plus, mut minus) = (u, u);
        plus[i] += h;
        minus[i] -= h;
        let numeric = (search.value(&plus, None) - search.value(&minus, None)) / (2.0 * h);
        assert!(
            (gradient[i] - numeric).abs() <= 1e-5 * numeric.abs().max(1.0),
            "{name} at {u:?}, gene {i}: {} against {numeric}",
            gradient[i]
        );
    }
}

const POINTS: [[f64; 2]; 4] = [[0.3, 0.6], [0.9, 0.1], [0.55, 0.45], [0.05, 0.95]];

fn unconstrained(objective: GaussianProcess, best: f64) -> Surrogate {
    Surrogate {
        objective,
        constraints: Vec::new(),
        thresholds: Vec::new(),
        best: Some(best),
    }
}

#[test]
fn the_acquisitions_gradients_match_central_differences() {
    let (objective, best) = model(smooth, 1);
    let acquisitions = [
        Acquisition::ExpectedImprovement,
        Acquisition::LogExpectedImprovement,
        Acquisition::ProbabilityOfImprovement { xi: 0.01 },
        Acquisition::UpperConfidenceBound { beta: 2.0 },
    ];
    let surrogate = unconstrained(objective.clone(), best);
    for acquisition in acquisitions {
        let search = Search {
            surrogate: &surrogate,
            acquisition,
            scale: objective.standardization().1,
        };
        for u in POINTS {
            check_gradient(&search, u, &format!("{acquisition:?}"));
        }
    }
}

#[test]
fn the_probability_of_feasibility_and_its_gradient() {
    let (objective, best) = model(smooth, 1);
    let (constraint, threshold) = constraint();
    let mut surrogate = Surrogate {
        objective: objective.clone(),
        constraints: vec![constraint.clone(), constraint.clone()],
        thresholds: vec![threshold, threshold + 0.3],
        best: Some(best),
    };
    let scale = objective.standardization().1;
    // weighing EI by P, and log-EI and log-PI by ln P, and before a feasible point, ln P alone
    for acquisition in [
        Acquisition::ExpectedImprovement,
        Acquisition::LogExpectedImprovement,
        Acquisition::ProbabilityOfImprovement { xi: 0.01 },
    ] {
        for best in [Some(best), None] {
            surrogate.best = best;
            let search = Search {
                surrogate: &surrogate,
                acquisition,
                scale,
            };
            for u in POINTS {
                check_gradient(&search, u, &format!("{acquisition:?} with {best:?}"));
            }
        }
    }
    // the value is the unconstrained one times P, P = Π Φ((tᵢ − μᵢ)/σᵢ)
    surrogate.best = Some(best);
    let plain = unconstrained(objective, best);
    for u in POINTS {
        let mut p = 1.0;
        for &t in &surrogate.thresholds {
            let (mean, variance) = constraint.predict_unit(&u, None);
            let sd = variance.max(VARIANCE_FLOOR).sqrt();
            p *= 0.5 * crate::math::erfc(-(t - mean) / sd / std::f64::consts::SQRT_2);
        }
        let ei = |surrogate| {
            Search {
                surrogate,
                acquisition: Acquisition::ExpectedImprovement,
                scale,
            }
            .value(&u, None)
        };
        let weighed = ei(&surrogate);
        assert!((weighed - ei(&plain) * p).abs() <= 1e-12 * weighed.abs().max(1e-300));
    }
}

#[test]
fn a_believer_keeps_the_mean_and_removes_the_uncertainty() {
    let (objective, best) = model(smooth, 1);
    let base = unconstrained(objective, best);
    let u = [0.37, 0.71];
    let (mean, variance) = base.objective.predict_unit(&u, None);
    assert!(variance > 1e-4);
    let mut fantasies = Fantasies::default();
    fantasies.push(&u, mean, &[]);
    let believed = condition(&base, &fantasies).unwrap();
    // the mean is the same everywhere (the model conditioned on its own prediction), and the
    // variance at the point is gone
    for v in POINTS.iter().chain([&u]) {
        let (before, _) = base.objective.predict_unit(v, None);
        let (after, _) = believed.objective.predict_unit(v, None);
        assert!(
            (before - after).abs() < 1e-9,
            "{v:?}: {before} against {after}"
        );
    }
    assert!(believed.objective.predict_unit(&u, None).1 < 1e-9);
    // a lie above the mean pulls the mean up around the point, and lowers nothing below the best
    let mut lie = Fantasies::default();
    lie.push(&u, mean + 2.0, &[]);
    let lied = condition(&base, &lie).unwrap();
    assert!((lied.objective.predict_unit(&u, None).0 - (mean + 2.0)).abs() < 1e-6);
    assert_eq!(lied.best, Some(best));
    // a fantasy below the best is the best
    let mut low = Fantasies::default();
    low.push(&u, best - 1.0, &[]);
    assert_eq!(condition(&base, &low).unwrap().best, Some(best - 1.0));
}

#[test]
fn the_log_transform_keeps_the_order_and_imputes_the_worst() {
    let values = [Some(3.0), None, Some(10.0), Some(3.5), Some(1e6), Some(4.0)];
    let targets = transform(&values, Output::Log).unwrap();
    // δ is the first quartile of the distances above the best, 0.5 here: the ⌊5/4⌋-th of
    // 0, 0.5, 1, 7 and 999997
    assert_eq!(targets[0], 0.5_f64.ln());
    assert_eq!(targets[3], 1.0_f64.ln());
    assert!(targets[0] < targets[3] && targets[3] < targets[5] && targets[5] < targets[2]);
    assert_eq!(targets[1], targets[4]);
    let targets = transform(&values, Output::Standardize).unwrap();
    assert_eq!(targets, [3.0, 1e6, 10.0, 3.5, 1e6, 4.0]);
    assert_eq!(transform(&[None, None], Output::Log), None);
    // every value the best: δ is 1
    assert_eq!(
        transform(&[Some(2.0), Some(2.0)], Output::Log),
        Some(vec![0.0, 0.0])
    );
}

#[test]
fn a_small_lattice_is_searched_exhaustively() {
    // 12 points: the design takes 8, then every point once, then the search has finished
    let integer = Integer::new([0..=3, -1..=1]).unwrap();
    let mut bo = Bo::builder(integer).minimize().seed(1).build().unwrap();
    let f = |x: &crate::genome::Integers| ((x[0] - 2) * (x[0] - 2) + x[1] * x[1]) as f64;
    let mut seen = Vec::new();
    while !bo.is_finished() {
        let asked: Vec<_> = bo.ask().iter().cloned().collect();
        assert!(!asked.is_empty());
        for genome in &asked {
            assert!(!seen.contains(genome));
            seen.push(genome.clone());
        }
        let fitness: Vec<Fitness> = asked.iter().map(|x| Fitness::new(f(x))).collect();
        bo.tell(&fitness).unwrap();
    }
    assert_eq!(seen.len(), 12);
    assert_eq!(bo.best().unwrap().genome()[..], [2, 0]);
    assert!(bo.ask().is_empty());
}
