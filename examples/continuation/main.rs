//! Continuation: the smallest ball around 420 points in 10 dimensions, its center found by
//! minimizing a smoothed largest distance, (Σ dᵢ^2p)^(1/2p), for p = 2, 4, 8 and 16, each stage
//! from the last, with Adam's state kept between them.
//!
//! The points are made so that the center is known exactly, and the last stage ends at it. Then
//! the same stages keeping only the point, and p = 16 from the start, as contrasts.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example continuation
//! ```

mod trace;

use genoxide::algorithm::continuation::Keep;
use genoxide::algorithm::first_order::Step;
use genoxide::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

// the dimensions, and the points near the center
const D: usize = 10;
const NEAR: usize = 400;
// how far from the center the near points are, at most
const SPREAD: f64 = 0.3;
// the stages' p = 2^k, for k = 1 to 4
const POWERS: [u32; 4] = [1, 2, 3, 4];
// Adam's learning rate, and the largest component of the gradient at which a stage has converged
const RATE: f64 = 0.05;
const TOLERANCE: f64 = 1e-11;

// the problem: the center, the points, and the cell with the current stage's k, which the fitness
// function reads
struct Problem {
    center: Vec<f64>,
    points: Arc<Vec<Vec<f64>>>,
    power: Arc<AtomicU32>,
}

// what a run did: per stage, its p, steps, evaluations, smoothed distance and distance to the
// center; and in all, its steps, evaluations and last point
struct Run {
    stages: Vec<(u32, u64, u64, f64, f64)>,
    steps: u64,
    evaluations: u64,
    point: Vec<f64>,
}

fn main() -> Result<()> {
    let problem = Problem::new()?;
    let mut trace = trace::Trace::from_env();

    println!(
        "The smallest ball around {} points in {D} dimensions: {} at distance 1 from its center, \
         {NEAR} within {SPREAD} of it",
        2 * D + NEAR,
        2 * D
    );
    println!("Adam through 4 stages of the smoothed largest distance, each from the last");
    let kept = problem.run(Keep::State, &POWERS, &mut trace, 0)?;
    println!(" stage   p  steps  evaluations  smoothed distance  distance to the center");
    for (index, &(p, steps, evaluations, smoothed, distance)) in kept.stages.iter().enumerate() {
        println!(
            "{:>6}  {p:>2}  {steps:>5}  {evaluations:>11}  {smoothed:>17.10}  {:>22}",
            index + 1,
            scientific(distance)
        );
    }
    let error = problem.distance(&kept.point);
    println!(
        "converged after {} steps and {} evaluations: the center within {}, the largest \
         distance 1 + {}",
        kept.steps,
        kept.evaluations,
        scientific(error),
        scientific(problem.largest(&kept.point) - 1.0)
    );
    assert!(error < 1e-10);

    // the contrasts: the same stages keeping only the point, and p = 16 alone
    let point = problem.run(Keep::Point, &POWERS, &mut trace, 1)?;
    println!(
        "the point only kept between stages: {} steps and {} evaluations, the center within {}",
        point.steps,
        point.evaluations,
        scientific(problem.distance(&point.point))
    );
    let cold = problem.run(Keep::State, &POWERS[3..], &mut trace, 2)?;
    println!(
        "p = 16 from the start: {} steps and {} evaluations, the center within {}",
        cold.steps,
        cold.evaluations,
        scientific(problem.distance(&cold.point))
    );
    trace.write();
    Ok(())
}

impl Problem {
    // the center c, and the points: c ± eᵢ on every axis i, the farthest from c, and NEAR points
    // within SPREAD of it, each coordinate moved by up to SPREAD / √D
    fn new() -> Result<Self> {
        let center: Vec<f64> = (0..D).map(|i| (i + 1) as f64 / D as f64 - 0.5).collect();
        let mut points = Vec::with_capacity(2 * D + NEAR);
        for i in 0..D {
            for sign in [1.0, -1.0] {
                let mut point = center.clone();
                point[i] += sign;
                points.push(point);
            }
        }
        let offsets =
            Real::uniform(NEAR * D, 0.0..=1.0)?.random_genome(&mut StreamRng::seed_from_u64(1));
        let scale = SPREAD / (D as f64).sqrt();
        for offset in offsets.as_chunks::<D>().0 {
            points.push(
                center
                    .iter()
                    .zip(offset)
                    .map(|(c, u)| c + scale * u)
                    .collect(),
            );
        }
        Ok(Self {
            center,
            points: Arc::new(points),
            power: Arc::new(AtomicU32::new(POWERS[0])),
        })
    }

    // Adam through the stages of `powers` from the same start, keeping `keep` between them, until
    // the last stage has converged; each step recorded in the trace as run `line`
    fn run(
        &self,
        keep: Keep,
        powers: &[u32],
        trace: &mut trace::Trace,
        line: usize,
    ) -> Result<Run> {
        let (points, power) = (Arc::clone(&self.points), Arc::clone(&self.power));
        let smoothed = Differentiable(move |x: &Reals, gradient: &mut [f64]| {
            smoothed(x, &points, power.load(Ordering::Relaxed), gradient)
        });
        let adam = FirstOrder::builder(Real::uniform(D, -2.0..=2.0)?)
            .step(Step::adam(RATE))
            .initial_genome(Reals::from(vec![-1.5; D]))
            .gradient_tolerance(TOLERANCE)
            .minimize()
            .build()?;
        // each stage's p in the shared cell, and its distance to the center when it ends
        let (cell, stages) = (Arc::clone(&self.power), powers.to_vec());
        let distances = Arc::new(Mutex::new(Vec::new()));
        let (ended, center) = (Arc::clone(&distances), self.center.clone());
        let continuation = Continuation::builder(adam)
            .stages(powers.len())
            .keep(keep)
            .on_stage(move |stage, _| {
                cell.store(stages[stage], Ordering::Relaxed);
                Ok(())
            })
            .on_stage_finished(move |_, adam| {
                let point = adam.population()[0].genome();
                let mut ended = ended.lock().expect("not poisoned");
                ended.push(distance(point, &center));
            })
            .build()?;
        let mut engine = Engine::new(continuation, smoothed)
            .stop_when(Stop::generations(100_000))
            .control(|staged: &mut Continuation<FirstOrder>, progress| {
                let point = staged.population()[0].genome();
                let p = 1u32 << powers[staged.stage()];
                trace.record(line, progress.generation(), self.distance(point), p);
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
                let p = 1u32 << powers[stage.index()];
                let smoothed = stage.best().score().expect("valid").sqrt();
                (
                    p,
                    stage.generations(),
                    stage.evaluations(),
                    smoothed,
                    distance,
                )
            })
            .collect();
        Ok(Run {
            stages,
            steps: outcome.generations(),
            evaluations: outcome.evaluations(),
            point: engine.algorithm().population()[0].genome().to_vec(),
        })
    }

    // the largest difference of a coordinate from the center's
    fn distance(&self, x: &[f64]) -> f64 {
        distance(x, &self.center)
    }

    // the largest distance from x to a point
    fn largest(&self, x: &[f64]) -> f64 {
        let squared = self.points.iter().map(|point| squared(x, point));
        squared.fold(0.0, f64::max).sqrt()
    }
}

fn distance(x: &[f64], center: &[f64]) -> f64 {
    x.iter()
        .zip(center)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

// |x − a|², summed in the order of the genes
fn squared(x: &[f64], a: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (xi, ai) in x.iter().zip(a) {
        sum += (xi - ai) * (xi - ai);
    }
    sum
}

// the smoothed largest squared distance (Σ gᵢ^p)^(1/p), gᵢ = |x − aᵢ|², p = 2^k, and its gradient
// into `gradient`. With M the largest gᵢ, it's M (Σ rᵢ^p)^(1/p) with rᵢ = gᵢ / M ≤ 1, which can't
// overflow; the powers are k squarings and the root k square roots, which round the same on every
// platform. The gradient is (Σ rᵢ^p)^(1/p − 1) Σ rᵢ^(p−1) 2 (x − aᵢ).
fn smoothed(x: &[f64], points: &[Vec<f64>], k: u32, gradient: &mut [f64]) -> f64 {
    let g: Vec<f64> = points.iter().map(|point| squared(x, point)).collect();
    let most = g.iter().copied().fold(0.0, f64::max);
    let mut sum = 0.0;
    let mut weights = Vec::with_capacity(g.len());
    for &gi in &g {
        let r = gi / most;
        let mut power = r;
        for _ in 0..k {
            power *= power;
        }
        sum += power;
        weights.push(if r > 0.0 { power / r } else { 0.0 });
    }
    let mut root = sum;
    for _ in 0..k {
        root = root.sqrt();
    }
    gradient.fill(0.0);
    for (point, &weight) in points.iter().zip(&weights) {
        for j in 0..x.len() {
            gradient[j] += 2.0 * weight * (x[j] - point[j]);
        }
    }
    let factor = root / sum;
    for gj in gradient.iter_mut() {
        *gj *= factor;
    }
    most * root
}

// one digit after the point, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
}
