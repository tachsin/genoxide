use super::*;

// a fixed pseudo-random sequence in [0, 1), so the tests don't depend on a generator
fn sequence(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}

#[test]
fn kernels_are_their_formulas() {
    for r in [0.0, 1e-8, 0.3, 1.0, 2.5, 7.0] {
        let r2 = r * r;
        // Rasmussen and Williams, eq. 4.17 and 4.9
        let sqrt5 = 5.0_f64.sqrt();
        let matern = (1.0 + sqrt5 * r + 5.0 * r * r / 3.0) * (-sqrt5 * r).exp();
        let (k, _) = Kernel::Matern52.eval(r2);
        assert!(
            (k - matern).abs() <= 1e-15,
            "Matérn at r = {r}: {k} against {matern}"
        );
        let (k, _) = Kernel::SquaredExponential.eval(r2);
        assert!((k - (-0.5 * r2).exp()).abs() <= 1e-15);
    }
    // the derivatives with respect to r², against central differences
    for kernel in [Kernel::Matern52, Kernel::SquaredExponential] {
        for r2 in [0.01, 0.4, 1.0, 3.0] {
            let h = 1e-6;
            let numeric = (kernel.eval(r2 + h).0 - kernel.eval(r2 - h).0) / (2.0 * h);
            let (_, analytic) = kernel.eval(r2);
            assert!(
                (numeric - analytic).abs() <= 1e-8,
                "{kernel:?} at r² = {r2}: {analytic} against {numeric}"
            );
        }
        // at 0: the correlation 1, and a finite slope
        assert_eq!(kernel.eval(0.0).0, 1.0);
        assert!(kernel.eval(0.0).1.is_finite());
    }
}

// standardized values of a smooth function at `count` points of `dims` genes
fn data(count: usize, dims: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let x = sequence(count * dims, seed);
    let values: Vec<f64> = x
        .chunks_exact(dims)
        .map(|p| {
            p.iter()
                .enumerate()
                .map(|(i, &v)| (3.0 * v + i as f64).sin())
                .sum()
        })
        .collect();
    let (_, _, y) = standardize(&values);
    (x, y)
}

#[test]
fn log_likelihood_gradient_matches_central_differences() {
    for kernel in [Kernel::Matern52, Kernel::SquaredExponential] {
        for noise in [Noise::Learned { min: 1e-6 }, Noise::Fixed(1e-3)] {
            let (count, dims) = (9, 3);
            let (x, y) = data(count, dims, 7);
            let data = Data {
                x: &x,
                y: &y,
                count,
                dims,
                kernel,
                noise,
            };
            let mut workspace = Workspace::new(count);
            let points = [
                vec![-0.7, 0.2, -1.2, 0.3, -5.0],
                vec![0.5, -0.4, 0.1, -0.8, -2.0],
                vec![-1.5, -1.0, 0.0, 1.0, -9.0],
            ];
            for (case, point) in points.into_iter().enumerate() {
                let parameters = match noise {
                    Noise::Learned { .. } => point,
                    Noise::Fixed(_) => point[..dims + 1].to_vec(),
                };
                let mut gradient = vec![0.0; parameters.len()];
                let value = data.log_likelihood(&parameters, &mut workspace, Some(&mut gradient));
                assert!(value.is_finite());
                for j in 0..parameters.len() {
                    let h = 1e-5;
                    let mut plus = parameters.clone();
                    plus[j] += h;
                    let mut minus = parameters.clone();
                    minus[j] -= h;
                    let numeric = (data.log_likelihood(&plus, &mut workspace, None)
                        - data.log_likelihood(&minus, &mut workspace, None))
                        / (2.0 * h);
                    let error = (numeric - gradient[j]).abs();
                    assert!(
                        error <= 1e-6 * gradient[j].abs().max(1.0),
                        "{kernel:?}, {noise:?}, case {case}, parameter {j}: {} against {numeric}",
                        gradient[j]
                    );
                }
            }
        }
    }
}

#[test]
fn the_mean_maximizes_the_likelihood() {
    // the generalized least squares mean: the likelihood with any other mean is lower
    let (count, dims) = (8, 2);
    let (x, y) = data(count, dims, 3);
    let data = Data {
        x: &x,
        y: &y,
        count,
        dims,
        kernel: Kernel::Matern52,
        noise: Noise::Fixed(1e-4),
    };
    let mut workspace = Workspace::new(count);
    let (length_scales, signal, noise) = unpack(&[-0.5, 0.1, 0.3], dims, data.noise);
    data.factor(&length_scales, signal, noise, &mut workspace)
        .unwrap();
    let (mean, best) = data.solve(&mut workspace, None);
    for offset in [-0.1, -1e-3, 1e-3, 0.1] {
        let (_, other) = data.solve(&mut workspace, Some(mean + offset));
        assert!(
            other < best,
            "the mean {mean} + {offset}: {other} against {best}"
        );
    }
    // the same likelihood as the fixed mean at the estimate
    let (_, fixed) = data.solve(&mut workspace, Some(mean));
    assert!((fixed - best).abs() <= 1e-12 * best.abs());
}
