//! Two spirals: evolve the 2,545 weights of a neural network that tells two interleaved spirals
//! apart, by OpenAI's evolution strategy.
//!
//! Lang and Witbrock's (1988) benchmark: 194 points on two spirals that wind three times around
//! the origin, 97 each, one the mirror image of the other through the origin. A network with two
//! hidden layers of 48 tanh units (2,545 weights with the biases) outputs a value in [−1, 1] for
//! a point; its sign is the spiral. The fitness is the mean squared error to the targets 1 and −1,
//! minimized by `OpenEs` from small random weights, until the network classifies every point.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example two_spirals
//! ```

mod trace;

use genoxide::math::{cos, sin};
use genoxide::nn::{Activation, Mlp};
use genoxide::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// the points of each spiral
const PER_SPIRAL: u32 = 97;
// the largest radius, to which the coordinates are scaled: the points lie in [−1, 1]²
const RADIUS: f64 = 6.5;

// Lang and Witbrock's points, scaled to [−1, 1]², with the targets 1 and −1: point i of the first
// spiral at the angle i π / 16 and the radius 6.5 (104 − i) / 104, and its mirror image through
// the origin on the second
pub fn points() -> Vec<([f64; 2], f64)> {
    let mut points = Vec::new();
    for i in 0..PER_SPIRAL {
        let angle = f64::from(i) * std::f64::consts::PI / 16.0;
        let radius = RADIUS * f64::from(104 - i) / 104.0;
        let (x, y) = (radius * sin(angle) / RADIUS, radius * cos(angle) / RADIUS);
        points.push(([x, y], 1.0));
        points.push(([-x, -y], -1.0));
    }
    points
}

// 2 inputs, two hidden layers of 48 tanh units and a tanh output, with biases
pub fn network() -> Result<Mlp> {
    Ok(Mlp::new([2, 48, 48, 1], Activation::Tanh)?.output_activation(Activation::Tanh))
}

// the points on the right side of 0
fn classified(network: &Mlp, weights: &[f64], points: &[([f64; 2], f64)]) -> usize {
    let mut network = network.with(weights).expect("the network's weights");
    let mut output = [0.0];
    let right = |(input, target): &&([f64; 2], f64)| {
        network.forward(input, &mut output);
        output[0] * target > 0.0
    };
    points.iter().filter(right).count()
}

fn main() -> Result<()> {
    let points = points();
    let mlp = network()?;
    // the mean squared error to the targets
    let error = |weights: &Reals| -> Option<f64> {
        let mut network = mlp.with(weights).ok()?;
        let mut output = [0.0];
        let squares = points.iter().map(|(input, target)| {
            network.forward(input, &mut output);
            (output[0] - target) * (output[0] - target)
        });
        Some(squares.sum::<f64>() / points.len() as f64)
    };
    // small random weights to start from: large ones saturate the tanh units
    let initial = Real::uniform(mlp.parameters(), -0.25..=0.25)?
        .random_genome(&mut StreamRng::seed_from_u64(1));
    let open_es = OpenEs::builder(mlp.representation(-3.0..=3.0)?)
        .population_size(100)
        .sigma(0.01)
        .optimizer(open_es::Optimizer::adam(0.01))
        .evaluate_mean(true)
        .initial_mean(initial)
        .parallel_breeding(true)
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env(&mlp);
    let solved = Arc::new(AtomicBool::new(false));
    let stop = Arc::clone(&solved);
    let mut solution = None;
    let outcome = Engine::new(open_es, error)
        .parallel(true)
        .stop_when(Stop::custom(move |_| stop.load(Ordering::Relaxed)))
        .stop_when(Stop::evaluations(1_000_000))
        .on_generation(|snapshot| trace.record(snapshot))
        // the generation's best network, if it classifies every point
        .on_generation(|snapshot| {
            let best = snapshot.population().best(Objective::Minimize);
            let Some(best) = best else { return };
            if classified(&mlp, best.genome(), &points) == points.len() {
                let progress = snapshot.progress();
                let error = best.fitness().and_then(Fitness::score);
                solution = Some((best.genome().clone(), error, progress.evaluations()));
                solved.store(true, Ordering::Relaxed);
            }
        })
        .run()?;

    let Some((weights, error, evaluations)) = solution else {
        let best = outcome.best_genome();
        println!(
            "not solved after {} evaluations: {} of {} points classified",
            outcome.evaluations(),
            classified(&mlp, best, &points),
            points.len()
        );
        return Ok(());
    };
    println!(
        "all {} points classified after {evaluations} evaluations in {} generations, by a \
         network of {} weights",
        points.len(),
        outcome.generations(),
        mlp.parameters()
    );
    println!("mean squared error {:.6}", error.unwrap_or(f64::NAN));
    println!();
    // the network's decision over [−1, 1]², a character per cell: # for the first spiral's side
    // and . for the second's, the points as A and B
    let mut network = mlp.with(&weights)?;
    let (columns, rows) = (61, 31);
    let mut map = vec![vec![' '; columns]; rows];
    let mut output = [0.0];
    for (row, line) in map.iter_mut().enumerate() {
        let y = 1.0 - 2.0 * row as f64 / (rows - 1) as f64;
        for (column, cell) in line.iter_mut().enumerate() {
            let x = -1.0 + 2.0 * column as f64 / (columns - 1) as f64;
            network.forward(&[x, y], &mut output);
            *cell = if output[0] > 0.0 { '#' } else { '.' };
        }
    }
    for ([x, y], target) in &points {
        let column = ((x + 1.0) / 2.0 * (columns - 1) as f64).round() as usize;
        let row = ((1.0 - y) / 2.0 * (rows - 1) as f64).round() as usize;
        map[row][column] = if *target > 0.0 { 'A' } else { 'B' };
    }
    for line in map {
        println!("{}", line.into_iter().collect::<String>());
    }
    trace.write();
    Ok(())
}
