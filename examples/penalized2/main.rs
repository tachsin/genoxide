//! Penalized 2: minimize Yao, Liu and Lin's second penalized function in 30 dimensions.
//!
//! Runs CMA-ES without and with IPOP restarts (a population that doubles at each restart),
//! differential evolution (SHADE), particle swarm optimization and a real-coded genetic algorithm
//! from 10 seeds each, and counts the runs that reach the minimum, 0 at (1, …, 1), to within
//! 1e-8. The function is genoxide's `problems::Penalized2`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of a run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example penalized2
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Penalized2, Problem};

const DIMENSIONS: usize = 30;
const SEEDS: u64 = 10;
const BUDGET: u64 = 10_000 * DIMENSIONS as u64;
// a run stops once its error to the minimum is at most this
const ERROR: f64 = 1e-8;
const ALGORITHMS: [&str; 5] = ["CMA-ES", "CMA-ES with IPOP", "DE", "PSO", "GA"];

fn main() -> Result<()> {
    let problem = Penalized2::new(DIMENSIONS);
    let minimum = problem.optimum().expect("known").value();
    println!(
        "Penalized 2 in {DIMENSIONS} dimensions, {SEEDS} seeds, {BUDGET} evaluations at most per run"
    );
    println!("algorithm         at min  evaluations  median error");
    for algorithm in ALGORITHMS {
        // the evaluations of the runs that reach the minimum, and every run's best error
        let mut evaluations = Vec::new();
        let mut errors = Vec::new();
        for seed in 1..=SEEDS {
            let outcome = run(algorithm, problem, seed, minimum + ERROR)?;
            if outcome.stop_reason() == StopReason::Target {
                evaluations.push(outcome.evaluations() as f64);
            }
            // rounding can put a solution a few ulps below the minimum
            let best = outcome.best_fitness().score().expect("valid");
            errors.push((best - minimum).max(0.0));
        }
        let reached = format!("{}/{SEEDS}", evaluations.len());
        let evaluations = median(evaluations).map_or("-".to_string(), |e| format!("{e:.0}"));
        let error = median(errors).expect("a run");
        println!(
            "{algorithm:<16}  {reached:>6}  {evaluations:>11}  {:>12}",
            format!("{error:.1e}")
        );
    }
    println!("evaluations: the median of the runs that reach the minimum");

    // with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate
    // run in 2 dimensions: the plot is the function's contour
    trace::record_small()?;
    Ok(())
}

// a run of `algorithm` from `seed`, until its best is at most `target` or it has used BUDGET
// evaluations
fn run(algorithm: &str, problem: Penalized2, seed: u64, target: f64) -> Result<Outcome<Reals>> {
    let real = problem.representation();
    let stop = Stop::target(target).or(Stop::evaluations(BUDGET));
    match algorithm {
        "CMA-ES" | "CMA-ES with IPOP" => {
            let restarts = if algorithm == "CMA-ES" {
                cmaes::Restarts::Never
            } else {
                cmaes::Restarts::Ipop
            };
            let cmaes = Cmaes::builder(real)
                .restarts(restarts)
                .minimize()
                .seed(seed)
                .build()?;
            Engine::new(cmaes, problem).stop_when(stop).run()
        }
        "DE" => {
            let de = De::builder(real).minimize().seed(seed).build()?;
            Engine::new(de, problem).stop_when(stop).run()
        }
        "PSO" => {
            let pso = Pso::builder(real)
                .population_size(40)
                .minimize()
                .seed(seed)
                .build()?;
            Engine::new(pso, problem).stop_when(stop).run()
        }
        _ => {
            let ga = Ga::builder(real)
                .population_size(100)
                .select(Tournament::new(3)?)
                .crossover(SimulatedBinaryCrossover::new(15.0)?)
                .mutate(PolynomialMutation::per_gene(1.0 / DIMENSIONS as f64, 20.0)?)
                .minimize()
                .seed(seed)
                .build()?;
            Engine::new(ga, problem).stop_when(stop).run()
        }
    }
}

// the median of `values`, None without any
fn median(mut values: Vec<f64>) -> Option<f64> {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    match values.len() {
        0 => None,
        n if n % 2 == 1 => Some(values[middle]),
        _ => Some((values[middle - 1] + values[middle]) / 2.0),
    }
}
