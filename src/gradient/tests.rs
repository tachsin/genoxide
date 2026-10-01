use super::*;
use crate::engine::BatchExtras;
use crate::math;
use proptest::prelude::*;

#[test]
fn the_steps_are_powers_of_epsilon() {
    assert_eq!(FORWARD_STEP, f64::EPSILON.sqrt());
    assert_eq!(FORWARD_STEP, 2f64.powi(-26));
    assert_eq!(CENTRAL_STEP, math::cbrt(f64::EPSILON));
}

// f(x) = exp(x₀) sin(x₁) + x₀³, and its gradient
fn smooth(x: &[f64]) -> f64 {
    math::exp(x[0]) * math::sin(x[1]) + x[0] * x[0] * x[0]
}

fn smooth_gradient(x: &[f64]) -> [f64; 2] {
    [
        math::exp(x[0]) * math::sin(x[1]) + 3.0 * x[0] * x[0],
        math::exp(x[0]) * math::cos(x[1]),
    ]
}

// the gradient of `f` at `x` by `stencil`, evaluated point by point
fn estimate(stencil: &mut Stencil, f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    stencil.set_center(x).unwrap();
    let mut point = vec![0.0; x.len()];
    let values: Vec<f64> = (0..stencil.len())
        .map(|k| {
            stencil.write_point(k, &mut point).unwrap();
            f(&point)
        })
        .collect();
    let mut gradient = vec![0.0; x.len()];
    stencil.gradient(f(x), &values, &mut gradient).unwrap();
    gradient
}

// the largest error of the gradient by finite differences with relative steps s, s/2, s/4, ...
fn errors(real: &Real, central: bool, x: &[f64]) -> Vec<f64> {
    let truth = smooth_gradient(x);
    (0..4)
        .map(|halvings| {
            let step = Some(1e-2 / f64::from(1 << halvings));
            let gradients = if central {
                Gradients::Central { step }
            } else {
                Gradients::Forward { step }
            };
            let mut stencil = Stencil::new(real, gradients).unwrap();
            let gradient = estimate(&mut stencil, smooth, x);
            gradient
                .iter()
                .zip(truth)
                .fold(0.0, |largest: f64, (g, t)| largest.max((g - t).abs()))
        })
        .collect()
}

// the error halves with the step: O(h) (forward), or quarters: O(h²) (central)
fn assert_order(errors: &[f64], ratio: f64) {
    for pair in errors.windows(2) {
        let measured = pair[0] / pair[1];
        assert!(
            (0.9 * ratio..=1.1 * ratio).contains(&measured),
            "{errors:?}: ratio {measured}, expected {ratio}"
        );
    }
}

#[test]
fn forward_differences_are_first_order() {
    let real = Real::uniform(2, -2.0..=2.0).unwrap();
    assert_order(&errors(&real, false, &[0.3, 0.7]), 2.0);
    // backwards from the upper bound of x₀
    assert_order(&errors(&real, false, &[2.0, 0.7]), 2.0);
}

#[test]
fn central_differences_are_second_order() {
    let real = Real::uniform(2, -2.0..=2.0).unwrap();
    assert_order(&errors(&real, true, &[0.3, 0.7]), 4.0);
    // one-sided at the bounds, and next to them
    assert_order(&errors(&real, true, &[2.0, -2.0]), 4.0);
    assert_order(&errors(&real, true, &[2.0 - 1e-3, -2.0 + 1e-3]), 4.0);
}

// a quadratic with small integer coefficients: Σ (i + 1) xᵢ² + x₀x₁ − 3x₂ + 0.5
fn quadratic(x: &[f64]) -> f64 {
    let squares: f64 = x
        .iter()
        .enumerate()
        .map(|(i, xi)| (i + 1) as f64 * xi * xi)
        .sum();
    squares + x[0] * x[1] - 3.0 * x[2] + 0.5
}

fn quadratic_gradient(x: &[f64]) -> Vec<f64> {
    let mut gradient: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, xi)| 2.0 * (i + 1) as f64 * xi)
        .collect();
    gradient[0] += x[1];
    gradient[1] += x[0];
    gradient[2] -= 3.0;
    gradient
}

#[test]
fn central_differences_of_a_quadratic_are_exact() {
    let real = Real::uniform(4, -1.0..=1.0).unwrap();
    let mut rng = crate::StreamRng::seed_from_u64(1);
    for _ in 0..100 {
        // genes on a grid of 2^-8, and a step of 2^-10, so that every value is exact: the parabola
        // through three points of a quadratic is the quadratic, with no truncation error, and so
        // no error at all, at the bounds (one-sided) too
        let mut x: Vec<f64> = (0..4)
            .map(|_| (rng.unit_f64() * 512.0 - 256.0).round() / 256.0)
            .collect();
        x[3] = if rng.unit_f64() < 0.5 { -1.0 } else { 1.0 };
        let exact = Gradients::Central {
            step: Some(2f64.powi(-10)),
        };
        let mut stencil = Stencil::new(&real, exact).unwrap();
        assert_eq!(
            estimate(&mut stencil, quadratic, &x),
            quadratic_gradient(&x)
        );
        // with the default step, exact up to the rounding of the values
        let mut stencil = Stencil::new(&real, Gradients::Central { step: None }).unwrap();
        let gradient = estimate(&mut stencil, quadratic, &x);
        for (g, t) in gradient.iter().zip(quadratic_gradient(&x)) {
            assert!((g - t).abs() <= 1e-9 * t.abs().max(1.0), "{g} {t}");
        }
    }
}

#[test]
fn fixed_genes_are_skipped() {
    let real = Real::new([0.0..=1.0, 2.0..=2.0, -1.0..=1.0, 5.0..=5.0]).unwrap();
    let forward = Stencil::new(&real, Gradients::Forward { step: None }).unwrap();
    assert_eq!(forward.len(), 2);
    let mut central = Stencil::new(&real, Gradients::Central { step: None }).unwrap();
    assert_eq!(central.len(), 4);
    assert!(central.is_central() && !forward.is_central());
    central.set_center(&[0.5, 2.0, 0.0, 5.0]).unwrap();
    let genes: Vec<usize> = (0..4).map(|k| central.point(k).unwrap().0).collect();
    assert_eq!(genes, [0, 0, 2, 2]);
    let mut gradient = [f64::NAN; 4];
    central
        .gradient(1.0, &[2.0, 0.0, 1.0, 1.0], &mut gradient)
        .unwrap();
    assert!(gradient[0] > 0.0);
    assert_eq!(gradient[1], 0.0);
    assert_eq!(gradient[2], 0.0);
    assert_eq!(gradient[3], 0.0);
    // no gene to move
    let fixed = Real::new([1.0..=1.0]).unwrap();
    let mut stencil = Stencil::new(&fixed, Gradients::Forward { step: None }).unwrap();
    assert!(stencil.is_empty());
    stencil.set_center(&[1.0]).unwrap();
    let mut gradient = [f64::NAN];
    stencil.gradient(3.0, &[], &mut gradient).unwrap();
    assert_eq!(gradient, [0.0]);
}

#[test]
fn stencils_report_their_misuse() {
    let real = Real::uniform(2, -1.0..=1.0).unwrap();
    let mut stencil = Stencil::new(&real, Gradients::Forward { step: None }).unwrap();
    let mut point = [0.0; 2];
    // no center yet
    assert!(matches!(
        stencil.write_point(0, &mut point),
        Err(Error::InvalidGenome { .. })
    ));
    assert!(stencil.point(0).is_none());
    assert!(matches!(
        stencil.gradient(0.0, &[0.0, 0.0], &mut [0.0; 2]),
        Err(Error::InvalidGenome { .. })
    ));
    assert!(matches!(
        stencil.set_center(&[0.0]),
        Err(Error::InvalidGenome { .. })
    ));
    stencil.set_center(&[0.0, 0.0]).unwrap();
    assert_eq!(stencil.center(), [0.0, 0.0]);
    assert!(stencil.write_point(2, &mut point).is_err());
    assert!(stencil.write_point(0, &mut [0.0; 3]).is_err());
    assert_eq!(
        stencil.gradient(0.0, &[0.0], &mut [0.0; 2]),
        Err(Error::FitnessCount {
            expected: 2,
            got: 1
        })
    );
    assert!(stencil.gradient(0.0, &[0.0, 0.0], &mut [0.0; 3]).is_err());
    for gradients in [Gradients::Auto, Gradients::Supplied] {
        assert!(matches!(
            Stencil::new(&real, gradients),
            Err(Error::InvalidSetting {
                setting: "gradients",
                ..
            })
        ));
    }
}

#[test]
fn steps_must_be_positive_and_finite() {
    let real = Real::uniform(1, 0.0..=1.0).unwrap();
    for step in [0.0, -1e-8, f64::NAN, f64::INFINITY] {
        for gradients in [
            Gradients::Forward { step: Some(step) },
            Gradients::Central { step: Some(step) },
        ] {
            assert!(gradients.validate().is_err(), "{gradients:?}");
            assert!(Stencil::new(&real, gradients).is_err());
            assert!(gradients.resolve(Provided::GRADIENT, &real).is_err());
        }
    }
    assert!(Gradients::Forward { step: Some(1e-300) }.validate().is_ok());
}

#[test]
fn resolution() {
    let real = Real::uniform(3, -1.0..=1.0).unwrap();
    let central = Gradients::Central { step: Some(1e-4) };
    for (gradients, provided, resolved) in [
        (Gradients::Auto, Provided::GRADIENT, Gradients::Supplied),
        (
            Gradients::Auto,
            Provided::NOTHING,
            Gradients::Forward { step: None },
        ),
        (Gradients::Supplied, Provided::GRADIENT, Gradients::Supplied),
        (central, Provided::GRADIENT, central),
        (central, Provided::NOTHING, central),
    ] {
        assert_eq!(gradients.resolve(provided, &real), Ok(resolved));
    }
    let Err(Error::InvalidSetting { setting, reason }) =
        Gradients::Supplied.resolve(Provided::NOTHING, &real)
    else {
        panic!("Supplied without a gradient");
    };
    assert_eq!(setting, "gradients");
    assert!(reason.contains("Differentiable"), "{reason}");
    assert!(Gradients::Supplied.is_supplied() && !Gradients::Auto.is_supplied());
    assert_eq!(Gradients::Supplied.extra_evaluations(10), 0);
    assert_eq!(Gradients::Auto.extra_evaluations(10), 10);
    assert_eq!(central.extra_evaluations(10), 20);
}

#[test]
fn auto_needs_a_supplied_gradient_above_the_limit() {
    let large = Real::uniform(AUTO_LIMIT + 1, -1.0..=1.0).unwrap();
    let Err(Error::InvalidSetting { setting, reason }) =
        Gradients::Auto.resolve(Provided::NOTHING, &large)
    else {
        panic!("forward differences for 10001 genes");
    };
    assert_eq!(setting, "gradients");
    assert!(reason.contains("Differentiable"), "{reason}");
    // with a gradient, or finite differences asked for, any size
    assert_eq!(
        Gradients::Auto.resolve(Provided::GRADIENT, &large),
        Ok(Gradients::Supplied)
    );
    let forward = Gradients::Forward { step: None };
    assert_eq!(forward.resolve(Provided::NOTHING, &large), Ok(forward));
    // fixed genes cost nothing, and don't count
    let mut bounds = vec![0.0..=1.0; AUTO_LIMIT];
    bounds.push(1.0..=1.0);
    let fixed = Real::new(bounds).unwrap();
    assert_eq!(
        Gradients::Auto.resolve(Provided::NOTHING, &fixed),
        Ok(forward)
    );
}

// bounds with fixed and narrow genes, and a center in them, at and near the bounds
fn box_and_center() -> impl Strategy<Value = (Vec<(f64, f64)>, Vec<f64>)> {
    let gene = prop_oneof![
        // a fixed gene
        (-10.0..10.0f64).prop_map(|x| ((x, x), x)),
        // at a bound, next to one, or anywhere, in a range of any width (as narrow as 1e-12)
        (-10.0..10.0f64, -12.0..2.0f64, 0..6u8, 0.0..1.0f64).prop_map(
            |(low, log_width, place, t)| {
                let high = low + 10f64.powf(log_width);
                let x = match place {
                    0 => low,
                    1 => high,
                    2 => low.next_up(),
                    3 => high.next_down(),
                    4 => (high - 1e-9).max(low),
                    _ => low + t * (high - low),
                };
                ((low, high), x.clamp(low, high))
            }
        ),
    ];
    prop::collection::vec(gene, 1..6).prop_map(|genes| genes.into_iter().unzip())
}

fn any_differences() -> impl Strategy<Value = Gradients> {
    let step = prop_oneof![
        Just(None),
        (-14.0..-1.0f64).prop_map(|e| Some(10f64.powf(e)))
    ];
    (any::<bool>(), step).prop_map(|(central, step)| {
        if central {
            Gradients::Central { step }
        } else {
            Gradients::Forward { step }
        }
    })
}

proptest! {
    #[test]
    fn stencil_points_stay_in_the_box(
        (bounds, x) in box_and_center(),
        gradients in any_differences(),
    ) {
        let real = Real::new(bounds.iter().map(|&(low, high)| low..=high)).unwrap();
        let mut stencil = Stencil::new(&real, gradients).unwrap();
        let variable = bounds.iter().filter(|(low, high)| low < high).count();
        let per_gene = if stencil.is_central() { 2 } else { 1 };
        prop_assert_eq!(stencil.len(), per_gene * variable);
        stencil.set_center(&x).unwrap();
        let mut point = vec![0.0; x.len()];
        for k in 0..stencil.len() {
            stencil.write_point(k, &mut point).unwrap();
            let (gene, value) = stencil.point(k).unwrap();
            prop_assert_eq!(point[gene], value);
            // one gene moved, which isn't fixed, within its bounds
            let (low, high) = bounds[gene];
            prop_assert!(low < high);
            prop_assert!((low..=high).contains(&value), "{} not in {:?}", value, (low, high));
            prop_assert!(value != x[gene]);
            for (other, (&p, &c)) in point.iter().zip(&x).enumerate() {
                if other != gene {
                    prop_assert_eq!(p.to_bits(), c.to_bits());
                }
            }
            // the step is exact: x + h − x = h
            let h = value - x[gene];
            prop_assert_eq!((x[gene] + h) - x[gene], h);
            prop_assert_eq!(x[gene] + h, value);
        }
        // a linear function's gradient is finite wherever the points are, and 0 for fixed genes
        let slopes: Vec<f64> = (0..x.len()).map(|i| i as f64 - 1.5).collect();
        let linear = |x: &[f64]| x.iter().zip(&slopes).map(|(x, s)| x * s).sum::<f64>();
        let gradient = estimate(&mut stencil, linear, &x);
        for (&g, &(low, high)) in gradient.iter().zip(&bounds) {
            if low == high {
                prop_assert_eq!(g, 0.0);
            } else {
                prop_assert!(g.is_finite(), "{:?}", gradient);
            }
        }
    }

    #[test]
    fn rounded_steps_are_exact(x in -1e6..1e6f64, log_step in -16.0..0.0f64, up in any::<bool>()) {
        let h = 10f64.powf(log_step) * x.abs().max(1.0);
        let step = rounded_step(x, if up { h } else { -h });
        prop_assert!(step != 0.0);
        prop_assert_eq!(step > 0.0, up);
        prop_assert_eq!((x + step) - x, step);
    }
}

#[test]
fn check_finds_wrong_gradients() {
    let rosenbrock = |x: &Reals, gradient: &mut [f64]| {
        let (a, b) = (x[1] - x[0] * x[0], 1.0 - x[0]);
        gradient[0] = -400.0 * x[0] * a - 2.0 * b;
        gradient[1] = 200.0 * a;
        100.0 * a * a + b * b
    };
    let x = Reals::from(vec![-1.2, 1.0]);
    let right = check(&Differentiable(rosenbrock), &x).unwrap();
    assert!(right.largest() < 1e-9, "{right:?}");
    assert_eq!(right.supplied().len(), 2);
    assert_eq!(right.estimated().len(), 2);
    // 1% off in x₁
    let wrong = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        let value = rosenbrock(x, gradient);
        gradient[1] *= 1.01;
        value
    });
    let check = check(&wrong, &x).unwrap();
    assert!(check.largest() > 5e-3, "{check:?}");
    assert_eq!(check.worst_gene(), Some(1));
    assert!(check.errors()[0] < 1e-9);
}

#[test]
fn check_needs_a_gradient() {
    let plain = |x: &Reals| x[0];
    assert!(matches!(
        check(&plain, &Reals::from(vec![1.0])),
        Err(Error::InvalidSetting { .. })
    ));
    // an invalid value is NaN, an error of the fitness function an error
    let nowhere = Differentiable(|_: &Reals, _: &mut [f64]| None::<f64>);
    let result = check(&nowhere, &Reals::from(vec![1.0, 2.0])).unwrap();
    assert!(result.largest().is_nan());
    assert_eq!(result.worst_gene(), Some(0));
    let negative = Differentiable(|_: &Reals, _: &mut [f64]| (1.0, -1.0));
    assert!(matches!(
        check(&negative, &Reals::from(vec![1.0])),
        Err(Error::InvalidFitness { .. })
    ));
    let empty = Check {
        supplied: Vec::new(),
        estimated: Vec::new(),
        errors: Vec::new(),
    };
    assert_eq!(empty.largest(), 0.0);
    assert_eq!(empty.worst_gene(), None);
}

#[test]
fn differentiable_gives_the_same_value_with_and_without_a_gradient() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = AtomicUsize::new(0);
    let sphere = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(gradient.len(), x.len());
        assert!(gradient.iter().all(|&g| g == 0.0), "zeroed");
        for (g, xi) in gradient.iter_mut().zip(x.iter()) {
            *g = 2.0 * xi;
        }
        x.iter().map(|xi| xi * xi).sum::<f64>()
    });
    let x = Reals::from(vec![1.5, -2.0, 0.25]);
    let mut gradient = [0.0; 3];
    let value = sphere.evaluate_with(&x, &mut Extras::with_gradient(&mut gradient));
    assert_eq!(gradient, [3.0, -4.0, 0.5]);
    assert_eq!(value, sphere.evaluate(&x));
    assert_eq!(value, sphere.evaluate_with(&x, &mut Extras::none()));
    // the scratch gradient is zeroed again for the next call, of another length
    assert_eq!(sphere.evaluate(&Reals::from(vec![2.0])), 4.0);
    assert_eq!(calls.load(Ordering::Relaxed), 4);
}

#[test]
fn a_differentiable_function_can_evaluate_another() {
    let inner = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        gradient[0] = 1.0;
        x[0]
    });
    let outer = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        gradient[0] = 7.0;
        let value = inner.evaluate(x);
        // the inner evaluation didn't write into this gradient
        assert_eq!(gradient[0], 7.0);
        value * 2.0
    });
    assert_eq!(outer.evaluate(&Reals::from(vec![3.0])), 6.0);
}

#[test]
fn batches_of_differentiable_functions() {
    // the sphere, a row of gradients per genome
    let batch = Batch(Differentiable(|xs: &[&Reals], gradients: &mut [f64]| {
        let n = xs.first().map_or(0, |x| x.len());
        assert_eq!(gradients.len(), n * xs.len());
        xs.iter()
            .zip(gradients.chunks_mut(n.max(1)))
            .map(|(x, row)| {
                for (g, xi) in row.iter_mut().zip(x.iter()) {
                    *g = 2.0 * xi;
                }
                x.iter().map(|xi| xi * xi).sum::<f64>()
            })
            .collect::<Vec<_>>()
    }));
    assert!(batch.is_batch());
    assert!(batch.provides().gradient);
    let (a, b) = (Reals::from(vec![1.0, 2.0]), Reals::from(vec![-3.0, 0.5]));
    let mut gradients = [0.0; 4];
    let values = batch.evaluate_batch_with(
        &[&a, &b],
        &mut BatchExtras::with_gradients(&mut gradients, 2),
    );
    assert_eq!(values, [5.0, 9.25]);
    assert_eq!(gradients, [2.0, 4.0, -6.0, 1.0]);
    assert_eq!(batch.evaluate_batch(&[&a, &b]), values);
    assert_eq!(
        batch.evaluate_batch_with(&[&a, &b], &mut BatchExtras::none()),
        values
    );
    assert!(batch.evaluate_batch(&[]).is_empty());
    assert_eq!(batch.evaluate(&b), 9.25);
    let mut gradient = [0.0; 2];
    assert_eq!(
        batch.evaluate_with(&a, &mut Extras::with_gradient(&mut gradient)),
        5.0
    );
    assert_eq!(gradient, [2.0, 4.0]);
    assert_eq!(batch.evaluate_with(&a, &mut Extras::none()), 5.0);
    assert!(check(&batch, &b).unwrap().largest() < 1e-9);
}

// a fitness function that supplies no gradient keeps its batch evaluation when asked with no
// extras, and ignores a gradient buffer
#[test]
fn plain_functions_ignore_extras() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = AtomicUsize::new(0);
    let batch = Batch(|xs: &[&Reals]| {
        calls.fetch_add(1, Ordering::Relaxed);
        xs.iter().map(|x| x[0]).collect::<Vec<_>>()
    });
    let (a, b) = (Reals::from(vec![1.0]), Reals::from(vec![2.0]));
    assert_eq!(
        batch.evaluate_batch_with(&[&a, &b], &mut BatchExtras::none()),
        [1.0, 2.0]
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    let plain = |x: &Reals| x[0] * 2.0;
    assert!(plain.provides().is_empty());
    let mut gradient = [5.0];
    assert_eq!(
        plain.evaluate_with(&a, &mut Extras::with_gradient(&mut gradient)),
        2.0
    );
    assert_eq!(gradient, [5.0]);
}

#[test]
fn the_slope_of_a_parabola() {
    // f(t) = 1 + 2t + 3t², slope 2 at 0, from any two points
    let f = |t: f64| 1.0 + 2.0 * t + 3.0 * t * t;
    for (a, b) in [(0.5, -0.5), (0.25, 0.5), (-0.25, -0.5), (0.5, -0.25)] {
        assert_eq!(parabola_slope(f(0.0), a, f(a), b, f(b)), 2.0, "{a} {b}");
    }
    // one point twice: the forward difference
    assert_eq!(parabola_slope(f(0.0), 0.5, f(0.5), 0.5, f(0.5)), 3.5);
}
