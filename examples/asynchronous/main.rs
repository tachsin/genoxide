//! Asynchronous evaluation for a fitness function that takes a varying time, like a simulation:
//! a generational GA evaluating in parallel waits for the slowest evaluation of every generation,
//! while a steady-state GA with asynchronous evaluation gives every worker a new genome as soon as
//! it's done.
//!
//! cargo run --release --example asynchronous

use genoxide::prelude::*;
use std::f64::consts::TAU;
use std::time::{Duration, Instant};

const DIMENSIONS: usize = 6;
const EVALUATIONS: u64 = 2_000;

// Rastrigin, taking 1 to 8 ms depending on the genome
fn simulation(x: &Reals) -> f64 {
    let value = 10.0 * x.len() as f64
        + x.iter()
            .map(|xi| xi * xi - 10.0 * (TAU * xi).cos())
            .sum::<f64>();
    let millis = 1 + (x[0].to_bits() % 8);
    std::thread::sleep(Duration::from_millis(millis));
    value
}

fn main() -> genoxide::Result<()> {
    let workers = rayon::current_num_threads();
    let real = || Real::uniform(DIMENSIONS, -5.12..=5.12);
    println!("{EVALUATIONS} evaluations of 1 to 8 ms, {workers} at a time\n");

    let ga = Ga::builder(real()?)
        .population_size(40)
        .select(Tournament::new(3)?)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / DIMENSIONS as f64, 20.0)?)
        .minimize()
        .seed(1)
        .build()?;
    let start = Instant::now();
    let outcome = Engine::new(ga, simulation)
        .parallel(true)
        .stop_when(Stop::evaluations(EVALUATIONS))
        .run()?;
    report("generational, parallel", &outcome, start.elapsed());

    let steady = Ga::builder(real()?)
        .population_size(40)
        .select(Tournament::new(3)?)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / DIMENSIONS as f64, 20.0)?)
        .minimize()
        .seed(1)
        .build_steady()?;
    let start = Instant::now();
    let outcome = AsyncEngine::new(steady, simulation)
        .workers(workers)
        .stop_when(Stop::evaluations(EVALUATIONS))
        .run()?;
    report("steady-state, asynchronous", &outcome, start.elapsed());
    Ok(())
}

fn report(name: &str, outcome: &Outcome<Reals>, elapsed: Duration) {
    println!(
        "{name:>27}: {:.2} s, {:.0} evaluations/s, best {:.3}",
        elapsed.as_secs_f64(),
        outcome.evaluations() as f64 / elapsed.as_secs_f64(),
        outcome.best_fitness()
    );
}
