//! Adam with a learning-rate schedule: smooth 100,000 noisy points into a curve, the curve's
//! 100,000 values the parameters, by penalized least squares with its gradient.
//!
//! The data are made so that the exact answer is known, and the run ends at it: Adam's learning
//! rate is halved every 500 steps by `control`, and the run stops when the gradient vanishes.
//! Then the same number of steps with the learning rate kept constant, which hovers around the
//! answer instead.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example adam
//! ```

mod trace;

use genoxide::algorithm::first_order::Step;
use genoxide::math;
use genoxide::prelude::*;
use std::f64::consts::TAU;

// the number of points, and of the curve's values
const POINTS: usize = 100_000;
// the weight of the roughness against the misfit
const LAMBDA: f64 = 50.0;
// Adam's first learning rate, halved every `HALVING` steps
const RATE: f64 = 0.05;
const HALVING: u64 = 500;
// a row of the table every this many steps
const EVERY: u64 = 100;

fn main() -> Result<()> {
    let (exact, data) = problem()?;
    // the misfit to the data and the roughness, with its gradient
    let smoothing = Differentiable(|x: &Reals, gradient: &mut [f64]| loss(x, &data, gradient));
    let adam = || {
        FirstOrder::builder(Real::uniform(POINTS, -10.0..=10.0)?)
            .step(Step::adam(RATE))
            .initial_genome(Reals::from(vec![0.0; POINTS]))
            .gradient_tolerance(1e-9)
            .minimize()
            .build()
    };

    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let mut rows = Vec::new();
    let mut engine = Engine::new(adam()?, smoothing)
        .stop_when(Stop::generations(10_000))
        .control(|adam: &mut FirstOrder, progress| {
            let point = &adam.population()[0];
            let distance = distance(point.genome(), &exact);
            let rate = adam.step().learning_rate();
            trace.scheduled(progress.generation(), distance, rate);
            rows.push(Row {
                step: progress.generation(),
                rate,
                loss: point.fitness().and_then(Fitness::score).expect("valid"),
                gradient: adam.gradient_norm(),
                distance,
            });
            // the learning rate of the next step: halved every `HALVING` steps
            let halvings = (progress.generation() / HALVING) as i32;
            adam.set_learning_rate(RATE * 0.5f64.powi(halvings))
        });
    let outcome = engine.run()?;
    let steps = engine.algorithm().iterations();
    drop(engine);

    println!("Smoothing {POINTS} noisy points: the misfit plus {LAMBDA} times the roughness");
    println!("Adam from a flat curve, its learning rate {RATE} halved every {HALVING} steps");
    println!("  step  learning rate  loss         largest gradient  distance to the answer");
    let last = rows.len() - 1;
    for (index, row) in rows.iter().enumerate() {
        if row.step.is_multiple_of(EVERY) || index == last {
            println!(
                "{:>6}  {:>13}  {:>11}  {:>16}  {:>22}",
                row.step,
                scientific(row.rate, 1),
                scientific(row.loss, 7),
                scientific(row.gradient, 1),
                scientific(row.distance, 1)
            );
        }
    }
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    let end = &rows[last];
    println!(
        "converged after {steps} steps and {} evaluations: every value within {} of the answer",
        outcome.evaluations(),
        scientific(end.distance, 1)
    );

    // the same number of steps, the learning rate constant
    let mut constant = Vec::new();
    Engine::new(adam()?, smoothing)
        .stop_when(Stop::generations(steps))
        .control(|adam: &mut FirstOrder, progress| {
            let distance = distance(adam.population()[0].genome(), &exact);
            trace.constant(progress.generation(), distance, adam.step().learning_rate());
            constant.push(distance);
            Ok(())
        })
        .run()?;
    println!(
        "the learning rate kept at {RATE}, after the same {steps} steps: within {} of the answer",
        scientific(constant[constant.len() - 1], 1)
    );
    trace.write();
    Ok(())
}

// a row of the table
struct Row {
    step: u64,
    rate: f64,
    loss: f64,
    gradient: f64,
    distance: f64,
}

// the exact answer x*, a smooth curve with a little noise of its own, and the data that make it
// the answer: d = x* + λ L x*, where L is the Laplacian of the chain of points, so that
// (I + λ L) x* = d, the normal equations of the loss
fn problem() -> Result<(Vec<f64>, Vec<f64>)> {
    let noise =
        Real::uniform(POINTS, -0.01..=0.01)?.random_genome(&mut StreamRng::seed_from_u64(1));
    let exact: Vec<f64> = (0..POINTS)
        .map(|i| {
            let t = i as f64 / (POINTS - 1) as f64;
            math::sin(TAU * t) + 0.3 * math::sin(5.0 * TAU * t) + noise[i]
        })
        .collect();
    let mut data = exact.clone();
    for i in 0..POINTS - 1 {
        let step = exact[i + 1] - exact[i];
        data[i + 1] += LAMBDA * step;
        data[i] -= LAMBDA * step;
    }
    Ok((exact, data))
}

// Σ (xᵢ − dᵢ)² + λ Σ (xᵢ₊₁ − xᵢ)², and its gradient into `gradient`
fn loss(x: &[f64], data: &[f64], gradient: &mut [f64]) -> f64 {
    let mut value = 0.0;
    for ((g, &xi), &di) in gradient.iter_mut().zip(x).zip(data) {
        let misfit = xi - di;
        *g = 2.0 * misfit;
        value += misfit * misfit;
    }
    let weight = 2.0 * LAMBDA;
    for i in 0..x.len() - 1 {
        let step = x[i + 1] - x[i];
        gradient[i + 1] += weight * step;
        gradient[i] -= weight * step;
        value += LAMBDA * step * step;
    }
    value
}

// the largest difference between the curve and the answer
fn distance(x: &[f64], exact: &[f64]) -> f64 {
    x.iter()
        .zip(exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

// `digits` digits after the point, e.g. 1.2e-7
fn scientific(value: f64, digits: usize) -> String {
    format!("{value:.digits$e}")
}
