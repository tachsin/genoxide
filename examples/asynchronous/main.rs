//! Asynchronous evaluation for a fitness function that takes a varying time, like a simulation:
//! a generational GA evaluating in parallel waits for the slowest evaluation of every generation,
//! while a steady-state GA with asynchronous evaluation gives every worker a new genome as soon as
//! it's done. The steady-state run goes on until it finds the minimum, and the generational run
//! then gets as many evaluations.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! cargo run --release --example asynchronous

mod trace;

use genoxide::prelude::*;
use std::f64::consts::TAU;
use std::time::{Duration, Instant};

const DIMENSIONS: usize = 4;

// Rastrigin, taking 0.25 to 2 ms depending on the genome
fn simulation(x: &Reals) -> f64 {
    let value = 10.0 * x.len() as f64
        + x.iter()
            .map(|xi| xi * xi - 10.0 * (TAU * xi).cos())
            .sum::<f64>();
    let quarters = 1 + (x[0].to_bits() % 8);
    std::thread::sleep(Duration::from_micros(250 * quarters));
    value
}

fn main() -> genoxide::Result<()> {
    let workers = rayon::current_num_threads();
    let real = || Real::uniform(DIMENSIONS, -5.12..=5.12);
    println!("Rastrigin in {DIMENSIONS}-D, evaluations of 0.25 to 2 ms, {workers} at a time\n");

    let steady = Ga::builder(real()?)
        .population_size(80)
        .select(Tournament::new(3)?)
        .crossover(SimulatedBinaryCrossover::new(1.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / DIMENSIONS as f64, 50.0)?)
        .minimize()
        .seed(1)
        .build_steady()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env(workers);
    let start = Instant::now();
    let asynchronous = AsyncEngine::new(steady, trace.timed(simulation))
        .workers(workers)
        .stop_when(Stop::target(1e-6).or(Stop::evaluations(50_000)))
        .observe(&mut trace)
        .run()?;
    let asynchronous_time = start.elapsed();

    let ga = Ga::builder(real()?)
        .population_size(80)
        .select(Tournament::new(3)?)
        .crossover(SimulatedBinaryCrossover::new(1.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / DIMENSIONS as f64, 50.0)?)
        .minimize()
        .seed(1)
        .build()?;
    let start = Instant::now();
    let generational = Engine::new(ga, simulation)
        .parallel(true)
        .stop_when(Stop::evaluations(asynchronous.evaluations()))
        .run()?;
    report("generational, parallel", &generational, start.elapsed());
    report(
        "steady-state, asynchronous",
        &asynchronous,
        asynchronous_time,
    );
    trace.write();
    Ok(())
}

fn report(name: &str, outcome: &Outcome<Reals>, elapsed: Duration) {
    println!(
        "{name:>27}: {:.2} s, {} evaluations, {:.0} evaluations/s, best {:.8}",
        elapsed.as_secs_f64(),
        outcome.evaluations(),
        outcome.evaluations() as f64 / elapsed.as_secs_f64(),
        outcome.best_fitness()
    );
}
