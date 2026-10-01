//! Continuation: a tilted Rastrigin function in 10 dimensions, Σ (xᵢ − aᵢ)² + 10 (1 − cos 2πxᵢ),
//! minimized through 6 stages of its Gaussian smoothing, σ from 0.6 (convex) to 0 (the function
//! itself), each stage from the last, with L-BFGS-B.
//!
//! The function is separable, so its global minimum is computed exactly, gene by gene, and the
//! last stage ends at it. Then, as contrasts, σ = 0 from the same start, which a local method
//! can't take out of the nearest basin, and the stages with L-BFGS-B's curvature pairs kept.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example continuation
//! ```

mod trace;

use genoxide::math;
use genoxide::prelude::*;
use std::f64::consts::{PI, TAU};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// the centers of the quadratic, off the cosine's lattice, and the cosine's amplitude
const CENTERS: [f64; 10] = [1.3, -0.7, 2.2, -1.6, 0.35, 3.25, -2.8, 0.7, -0.3, 1.8];
const A: f64 = 10.0;
// the stages' smoothing: 0.6 is convex (above 0.517), 0 the function itself
const SIGMAS: [f64; 6] = [0.6, 0.4, 0.3, 0.2, 0.1, 0.0];
// every gene starts here
const START: f64 = -3.0;
// L-BFGS-B's largest projected gradient component at which a stage has converged
const TOLERANCE: f64 = 1e-10;

// what a run did: per stage, its σ, rounds, evaluations, best value and distance to the global
// minimum; and in all, its rounds, evaluations and last point
struct Run {
    stages: Vec<(f64, u64, u64, f64, f64)>,
    rounds: u64,
    evaluations: u64,
    point: Vec<f64>,
}

fn main() -> Result<()> {
    let exact = global_minimum();
    let minimum = value(&exact, 0.0);
    let mut trace = trace::Trace::from_env();

    println!(
        "A tilted Rastrigin function in {} dimensions, Σ (xᵢ − aᵢ)² + {A} (1 − cos 2πxᵢ), from \
         xᵢ = {START}",
        CENTERS.len()
    );
    println!(
        "L-BFGS-B through 6 stages of the function smoothed by a Gaussian of σ, each from the last"
    );
    let staged = run(&SIGMAS, false, &exact, &mut trace, 0)?;
    println!(
        " stage     σ  rounds  evaluations  value of the stage  distance to the global minimum"
    );
    for (index, &(sigma, rounds, evaluations, best, distance)) in staged.stages.iter().enumerate() {
        println!(
            "{:>6}  {sigma:>4.2}  {rounds:>6}  {evaluations:>11}  {best:>18.10}  {:>29}",
            index + 1,
            scientific(distance)
        );
    }
    let error = distance(&staged.point, &exact);
    println!(
        "converged after {} rounds and {} evaluations: f = {:.14}, within {} of the global \
         minimum",
        staged.rounds,
        staged.evaluations,
        value(&staged.point, 0.0),
        scientific(error)
    );
    println!("the global minimum, gene by gene by bisection: f* = {minimum:.14}");
    assert!(error < 1e-12);

    // the contrasts: σ = 0 from the same start, and the stages keeping the curvature pairs
    let cold = run(&SIGMAS[5..], false, &exact, &mut trace, 1)?;
    let trapped = value(&cold.point, 0.0);
    println!(
        "σ = 0 from the start: {} rounds and {} evaluations, trapped at f = {trapped:.8}, {:.8} \
         above f*, {} from the global minimum",
        cold.rounds,
        cold.evaluations,
        trapped - minimum,
        scientific(distance(&cold.point, &exact))
    );
    let paired = run(&SIGMAS, true, &exact, &mut trace, 2)?;
    println!(
        "the stages with L-BFGS-B's curvature pairs kept: {} rounds and {} evaluations, within {}",
        paired.rounds,
        paired.evaluations,
        scientific(distance(&paired.point, &exact))
    );
    trace.write();
    Ok(())
}

// L-BFGS-B through the stages of `sigmas` from the same start, keeping its pairs between them
// or not, until the last stage has converged; each round recorded in the trace as run `line`
fn run(
    sigmas: &[f64],
    keep_pairs: bool,
    exact: &[f64],
    trace: &mut trace::Trace,
    line: usize,
) -> Result<Run> {
    // the stage's σ, shared with the fitness function
    let sigma = Arc::new(AtomicU64::new(sigmas[0].to_bits()));
    let shared = Arc::clone(&sigma);
    let smoothed = Differentiable(move |x: &Reals, gradient: &mut [f64]| {
        let s = f64::from_bits(shared.load(Ordering::Relaxed));
        smoothed(x, s, gradient)
    });
    let lbfgsb = Lbfgsb::builder(Real::uniform(CENTERS.len(), -5.0..=5.0)?)
        .initial_genome(Reals::from(vec![START; CENTERS.len()]))
        .gradient_tolerance(TOLERANCE)
        .keep_pairs(keep_pairs)
        .minimize()
        .build()?;
    // each stage's distance to the global minimum when it ends
    let distances = Arc::new(Mutex::new(Vec::new()));
    let (ended, minimum) = (Arc::clone(&distances), exact.to_vec());
    let stages = sigmas.to_vec();
    let continuation = Continuation::builder(lbfgsb)
        .stages(sigmas.len())
        .on_stage(move |stage, _| {
            sigma.store(stages[stage].to_bits(), Ordering::Relaxed);
            Ok(())
        })
        .on_stage_finished(move |_, lbfgsb| {
            let point = lbfgsb.population()[0].genome();
            let mut ended = ended.lock().expect("not poisoned");
            ended.push(distance(point, &minimum));
        })
        .build()?;
    let mut engine = Engine::new(continuation, smoothed)
        .stop_when(Stop::evaluations(10_000))
        .control(|staged: &mut Continuation<Lbfgsb>, progress| {
            let point = staged.population()[0].genome();
            let s = sigmas[staged.stage()];
            trace.record(line, progress.generation(), distance(point, exact), s);
            Ok(())
        });
    let outcome = engine.run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    let distances = distances.lock().expect("not poisoned");
    let stages = engine
        .algorithm()
        .stages()
        .iter()
        .zip(distances.iter())
        .map(|(stage, &distance)| {
            let best = stage.best().score().expect("valid");
            let s = sigmas[stage.index()];
            (s, stage.generations(), stage.evaluations(), best, distance)
        })
        .collect();
    Ok(Run {
        stages,
        rounds: outcome.generations(),
        evaluations: outcome.evaluations(),
        point: engine.algorithm().population()[0].genome().to_vec(),
    })
}

// the function smoothed by a Gaussian of σ, E[f(x + σz)] for z standard normal, and its gradient
// into `gradient`. E[cos 2π(x + σz)] = e^(−2π²σ²) cos 2πx and E[(x + σz − a)²] = (x − a)² + σ²,
// so each gene's term is (x − a)² + σ² + A (1 − e^(−2π²σ²) cos 2πx), convex once
// 4π²A e^(−2π²σ²) < 2. With genoxide's portable cos, sin and exp, the same bits everywhere.
fn smoothed(x: &[f64], s: f64, gradient: &mut [f64]) -> f64 {
    let e = math::exp(-2.0 * PI * PI * s * s);
    let mut sum = 0.0;
    for ((g, &xi), &a) in gradient.iter_mut().zip(x).zip(&CENTERS) {
        let d = xi - a;
        sum += d * d + s * s + A * (1.0 - e * math::cos(TAU * xi));
        *g = 2.0 * d + 2.0 * PI * A * e * math::sin(TAU * xi);
    }
    sum
}

// the function smoothed by σ, without the gradient
fn value(x: &[f64], s: f64) -> f64 {
    smoothed(x, s, &mut vec![0.0; x.len()])
}

// the global minimum, gene by gene: in each basin around an integer k near the center a, where
// the term is convex (|x − k| ≤ 1/4), the root of its derivative 2(x − a) + 2πA sin 2πx by
// bisection to the last bit, and of those the lowest
fn global_minimum() -> Vec<f64> {
    let mut minimum = Vec::with_capacity(CENTERS.len());
    for &a in &CENTERS {
        let derivative = |x: f64| 2.0 * (x - a) + 2.0 * PI * A * math::sin(TAU * x);
        let term = |x: f64| (x - a) * (x - a) + A * (1.0 - math::cos(TAU * x));
        let mut best: Option<(f64, f64)> = None;
        for k in (a.floor() as i64 - 3)..=(a.ceil() as i64 + 3) {
            let (mut low, mut high) = (k as f64 - 0.25, k as f64 + 0.25);
            for _ in 0..100 {
                let middle = 0.5 * (low + high);
                if derivative(middle) > 0.0 {
                    high = middle;
                } else {
                    low = middle;
                }
            }
            let x = 0.5 * (low + high);
            if best.is_none_or(|(lowest, _)| term(x) < lowest) {
                best = Some((term(x), x));
            }
        }
        minimum.push(best.expect("a basin").1);
    }
    minimum
}

// the largest difference of a gene from the global minimum's
fn distance(x: &[f64], exact: &[f64]) -> f64 {
    x.iter()
        .zip(exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

// one digit after the point, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
}
