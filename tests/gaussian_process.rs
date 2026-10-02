//! The Gaussian process of `model::gp`: Rasmussen and Williams's (2006) formulas on cases small
//! enough to work out by hand, interpolation, the posterior's gradients and the fit.

use genoxide::model::gp::{GaussianProcess, Hyperparameters, Kernel, Noise};
use genoxide::prelude::*;

// the kernels' correlation at distance r (Rasmussen and Williams, eq. 4.17 and 4.9)
fn correlation(kernel: Kernel, r: f64) -> f64 {
    match kernel {
        Kernel::Matern52 => {
            let s = 5.0_f64.sqrt() * r;
            (1.0 + s + s * s / 3.0) * (-s).exp()
        }
        _ => (-0.5 * r * r).exp(),
    }
}

// the posterior of two points worked out by hand: K_y = σ_f² [[1, c], [c, 1]] + σ_n² I, its
// inverse [[a, −b], [−b, a]] / det; the mean m + k*ᵀ K_y⁻¹ (y − m) (eq. 2.38), the variance
// σ_f² − k*ᵀ K_y⁻¹ k* (eq. 2.26), and the log marginal likelihood (eq. 2.30)
struct TwoPoints {
    mean: f64,
    variance: f64,
    log_likelihood: f64,
}

#[expect(clippy::too_many_arguments)]
fn two_points(
    kernel: Kernel,
    x: [f64; 2],
    y: [f64; 2],
    at: f64,
    length: f64,
    signal: f64,
    noise: f64,
    mean: f64,
) -> TwoPoints {
    let c = correlation(kernel, (x[0] - x[1]).abs() / length);
    let (a, b) = (signal + noise, signal * c);
    let det = a * a - b * b;
    let k = [
        signal * correlation(kernel, (at - x[0]).abs() / length),
        signal * correlation(kernel, (at - x[1]).abs() / length),
    ];
    let r = [y[0] - mean, y[1] - mean];
    let alpha = [(a * r[0] - b * r[1]) / det, (a * r[1] - b * r[0]) / det];
    let kinv_k = [(a * k[0] - b * k[1]) / det, (a * k[1] - b * k[0]) / det];
    TwoPoints {
        mean: mean + k[0] * alpha[0] + k[1] * alpha[1],
        variance: signal - (k[0] * kinv_k[0] + k[1] * kinv_k[1]),
        log_likelihood: -0.5 * (r[0] * alpha[0] + r[1] * alpha[1])
            - 0.5 * det.ln()
            - (2.0 * std::f64::consts::PI).ln(),
    }
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * b.abs().max(1.0)
}

#[test]
fn two_points_are_the_formulas_worked_out_by_hand() {
    for kernel in [Kernel::Matern52, Kernel::SquaredExponential] {
        // a gene in [−2, 6]: the length scale 2 is 0.25 of the range in the model's unit cube
        let real = Real::uniform(1, -2.0..=6.0).unwrap();
        let (x, y) = ([0.5, 2.0], [1.0, 4.0]);
        let (length, signal, noise, mean) = (2.0, 3.0, 0.1, 1.5);
        let points: Vec<Reals> = x.iter().map(|&x| Reals::from(vec![x])).collect();
        let gp = GaussianProcess::builder(real)
            .kernel(kernel)
            .hyperparameters(Hyperparameters::new(mean, vec![length], signal, noise))
            .fit(&points, &y)
            .unwrap();
        for at in [-2.0, 0.5, 1.0, 1.7, 3.0, 6.0] {
            let expected = two_points(kernel, x, y, at, length, signal, noise, mean);
            let prediction = gp.predict(&[at]);
            assert!(
                close(prediction.mean(), expected.mean, 1e-13),
                "{kernel:?} at {at}: mean {} against {}",
                prediction.mean(),
                expected.mean
            );
            assert!(
                close(prediction.variance(), expected.variance, 1e-12),
                "{kernel:?} at {at}: variance {} against {}",
                prediction.variance(),
                expected.variance
            );
            assert!(close(
                gp.log_marginal_likelihood(),
                expected.log_likelihood,
                1e-13
            ));
        }
        // the hyperparameters come back in the genes' and values' units
        let h = gp.hyperparameters();
        assert!(close(h.mean(), mean, 1e-15));
        assert!(close(h.length_scales()[0], length, 1e-15));
        assert!(close(h.signal_variance(), signal, 1e-15));
        assert!(close(h.noise_variance(), noise, 1e-15));
        assert_eq!(gp.jitter(), 0.0);
    }
}

// a smooth function of three genes, the third fixed
fn smooth(x: &[f64]) -> f64 {
    (2.0 * x[0]).sin() + 0.5 * x[1] * x[1] - 0.3 * x[0] * x[1] + x[2]
}

fn sample(count: usize) -> (Real, Vec<Reals>, Vec<f64>) {
    let real = Real::new([-1.0..=2.0, 0.0..=3.0, 0.5..=0.5]).unwrap();
    let points = real
        .latin_hypercube(count, &mut StreamRng::seed_from_u64(4))
        .unwrap();
    let values = points.iter().map(|x| smooth(x)).collect();
    (real, points, values)
}

#[test]
fn without_noise_the_mean_interpolates_and_the_variance_vanishes_at_the_data() {
    let (real, points, values) = sample(12);
    let gp = GaussianProcess::builder(real)
        .noise(Noise::Fixed(0.0))
        .fit(&points, &values)
        .unwrap();
    for (point, &value) in points.iter().zip(&values) {
        let prediction = gp.predict(point);
        assert!(
            (prediction.mean() - value).abs() <= 1e-6,
            "{} against {value}",
            prediction.mean()
        );
        assert!(prediction.variance() <= 1e-8 * gp.hyperparameters().signal_variance());
    }
    // between the points, a good model of a smooth function
    let prediction = gp.predict(&[0.4, 1.2, 0.5]);
    assert!((prediction.mean() - smooth(&[0.4, 1.2, 0.5])).abs() < 0.05);
    assert!(prediction.variance() > 0.0);
}

#[test]
fn far_from_the_data_the_variance_is_the_signal_variance() {
    let real = Real::uniform(1, 0.0..=100.0).unwrap();
    let points: Vec<Reals> = [1.0, 2.0, 3.0].map(|x| Reals::from(vec![x])).to_vec();
    let gp = GaussianProcess::builder(real)
        .hyperparameters(Hyperparameters::new(0.0, vec![0.5], 2.0, 1e-6))
        .fit(&points, &[0.3, -0.2, 0.5])
        .unwrap();
    let far = gp.predict(&[90.0]);
    assert!(close(far.variance(), 2.0, 1e-12));
    assert!(close(far.mean(), 0.0, 1e-12));
}

#[test]
fn the_posterior_gradients_match_central_differences() {
    let (real, points, values) = sample(15);
    for kernel in [Kernel::Matern52, Kernel::SquaredExponential] {
        let gp = GaussianProcess::builder(real.clone())
            .kernel(kernel)
            .fit(&points, &values)
            .unwrap();
        for at in [[0.1, 0.4, 0.5], [1.7, 2.9, 0.5], [-0.8, 1.5, 0.5]] {
            let (mut dmean, mut dvariance) = ([0.0; 3], [0.0; 3]);
            let prediction = gp.predict_with_gradient(&at, &mut dmean, &mut dvariance);
            assert_eq!(prediction, gp.predict(&at));
            // the fixed gene takes no part
            assert_eq!((dmean[2], dvariance[2]), (0.0, 0.0));
            for i in 0..2 {
                let h = 1e-6;
                let (mut plus, mut minus) = (at, at);
                plus[i] += h;
                minus[i] -= h;
                let (p, m) = (gp.predict(&plus), gp.predict(&minus));
                let numeric_mean = (p.mean() - m.mean()) / (2.0 * h);
                let numeric_variance = (p.variance() - m.variance()) / (2.0 * h);
                assert!(
                    close(dmean[i], numeric_mean, 1e-6),
                    "{kernel:?} at {at:?}, gene {i}: {} against {numeric_mean}",
                    dmean[i]
                );
                assert!(
                    (dvariance[i] - numeric_variance).abs() <= 1e-6,
                    "{kernel:?} at {at:?}, gene {i}: {} against {numeric_variance}",
                    dvariance[i]
                );
            }
        }
    }
}

#[test]
fn the_fit_maximizes_the_likelihood_and_is_reproducible() {
    let (real, points, values) = sample(20);
    let fit = |seed| {
        GaussianProcess::builder(real.clone())
            .seed(seed)
            .fit(&points, &values)
            .unwrap()
    };
    let gp = fit(0);
    let again = fit(0);
    assert_eq!(gp.hyperparameters(), again.hyperparameters());
    assert_eq!(
        gp.log_marginal_likelihood().to_bits(),
        again.log_marginal_likelihood().to_bits()
    );
    // better than the fixed first start, and than nearby hyperparameters
    let h = gp.hyperparameters();
    let at = |length_factor: f64, signal_factor: f64| {
        let lengths: Vec<f64> = h
            .length_scales()
            .iter()
            .map(|l| l * length_factor)
            .collect();
        GaussianProcess::builder(real.clone())
            .hyperparameters(Hyperparameters::new(
                h.mean(),
                lengths,
                h.signal_variance() * signal_factor,
                h.noise_variance(),
            ))
            .fit(&points, &values)
            .unwrap()
            .log_marginal_likelihood()
    };
    let best = gp.log_marginal_likelihood();
    assert!(close(at(1.0, 1.0), best, 1e-9));
    for (length, signal) in [(1.1, 1.0), (0.9, 1.0), (1.0, 1.2), (1.0, 0.8)] {
        assert!(at(length, signal) < best);
    }
}

#[test]
fn repeated_points_factor_with_jitter() {
    let real = Real::uniform(2, 0.0..=1.0).unwrap();
    let points: Vec<Reals> = [[0.2, 0.3], [0.2, 0.3], [0.7, 0.1], [0.4, 0.9]]
        .map(|x| Reals::from(x.to_vec()))
        .to_vec();
    let values = [1.0, 1.0, 0.0, 2.0];
    // without noise, a singular kernel matrix: it factors with a jitter
    let gp = GaussianProcess::builder(real.clone())
        .hyperparameters(Hyperparameters::new(1.0, vec![0.5, 0.5], 1.0, 0.0))
        .fit(&points, &values)
        .unwrap();
    assert!(gp.jitter() > 0.0 && gp.jitter() < 1e-6, "{}", gp.jitter());
    assert!((gp.predict(&[0.2, 0.3]).mean() - 1.0).abs() < 1e-6);
    // fitted, with or without noise
    for noise in [Noise::Fixed(0.0), Noise::default()] {
        let gp = GaussianProcess::builder(real.clone())
            .noise(noise)
            .fit(&points, &values)
            .unwrap();
        assert!((gp.predict(&[0.2, 0.3]).mean() - 1.0).abs() < 1e-3);
    }
}

#[test]
fn invalid_settings_are_errors() {
    let real = Real::uniform(2, 0.0..=1.0).unwrap();
    let points = vec![Reals::from(vec![0.1, 0.2]), Reals::from(vec![0.5, 0.5])];
    let builder = GaussianProcess::builder(real);
    let setting = |result: Result<GaussianProcess>| match result {
        Err(Error::InvalidSetting { setting, .. }) => setting,
        other => panic!("not an invalid setting: {other:?}"),
    };
    assert_eq!(setting(builder.fit(&[], &[])), "points");
    assert_eq!(setting(builder.fit(&points, &[1.0])), "values");
    assert_eq!(setting(builder.fit(&points, &[1.0, f64::NAN])), "values");
    let short = vec![Reals::from(vec![0.1])];
    assert_eq!(setting(builder.fit(&short, &[1.0])), "points");
    let infinite = vec![Reals::from(vec![0.1, f64::INFINITY])];
    assert_eq!(setting(builder.fit(&infinite, &[1.0])), "points");
    assert_eq!(
        setting(builder.clone().starts(0).fit(&points, &[1.0, 2.0])),
        "starts"
    );
    for noise in [
        Noise::Learned { min: 0.0 },
        Noise::Learned { min: 1.0 },
        Noise::Fixed(-1.0),
        Noise::Fixed(f64::NAN),
    ] {
        let result = builder.clone().noise(noise).fit(&points, &[1.0, 2.0]);
        assert_eq!(setting(result), "noise");
    }
    for h in [
        Hyperparameters::new(0.0, vec![1.0], 1.0, 0.0),
        Hyperparameters::new(0.0, vec![1.0, 0.0], 1.0, 0.0),
        Hyperparameters::new(0.0, vec![1.0, 1.0], -1.0, 0.0),
        Hyperparameters::new(0.0, vec![1.0, 1.0], 1.0, -1.0),
        Hyperparameters::new(f64::NAN, vec![1.0, 1.0], 1.0, 0.0),
    ] {
        let result = builder.clone().hyperparameters(h).fit(&points, &[1.0, 2.0]);
        assert_eq!(setting(result), "hyperparameters");
    }
}

#[test]
#[should_panic(expected = "a Gaussian process of 2 genes")]
fn a_point_of_the_wrong_length_panics() {
    let real = Real::uniform(2, 0.0..=1.0).unwrap();
    let points = vec![Reals::from(vec![0.1, 0.2])];
    let gp = GaussianProcess::builder(real).fit(&points, &[1.0]).unwrap();
    let _ = gp.predict(&[0.5]);
}
