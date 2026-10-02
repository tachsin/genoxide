//! Katsuura: minimize Katsuura's function, rugged everywhere, in 10 dimensions.
//!
//! Runs CMA-ES with IPOP restarts (a population that doubles at each restart) from 10 seeds, with
//! a budget of 500,000 evaluations per run, and counts the runs that reach the minimum, 0 at the
//! origin, to within 1e-8. Then, as contrasts with the budget of 100,000 evaluations of the other
//! functions' pages: CMA-ES without restarts, differential evolution (SHADE), particle swarm
//! optimization, which reaches the minimum only by stopping at the bounds, and a real-coded
//! genetic algorithm. The function is genoxide's `problems::Katsuura`. The runs evaluate in
//! parallel, with the same results on any number of threads.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of a run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example katsuura
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Katsuura, Problem};

const DIMENSIONS: usize = 10;
const SEEDS: u64 = 10;
// a run stops once its error to the minimum is at most this
const ERROR: f64 = 1e-8;
// the algorithms and their budgets of evaluations per run: the main method's, enough for every
// seed, then the contrasts', the 10,000 per dimension of the other functions' pages
const ALGORITHMS: [(&str, u64); 5] = [
    ("CMA-ES with IPOP", 50_000 * DIMENSIONS as u64),
    ("CMA-ES", 10_000 * DIMENSIONS as u64),
    ("DE", 10_000 * DIMENSIONS as u64),
    ("PSO", 10_000 * DIMENSIONS as u64),
    ("GA", 10_000 * DIMENSIONS as u64),
];

fn main() -> Result<()> {
    let problem = Katsuura::new(DIMENSIONS);
    let minimum = problem.optimum().expect("known").value();
    println!("Katsuura in {DIMENSIONS} dimensions, {SEEDS} seeds");
    println!("algorithm          budget  at min  evaluations  median error");
    let mut corners = 0;
    let mut main_range = (f64::NAN, f64::NAN);
    for (algorithm, budget) in ALGORITHMS {
        // the evaluations of the runs that reach the minimum, and every run's best error
        let mut evaluations = Vec::new();
        let mut errors = Vec::new();
        for seed in 1..=SEEDS {
            let outcome = run(algorithm, problem, seed, minimum + ERROR, budget)?;
            if outcome.stop_reason() == StopReason::Target {
                evaluations.push(outcome.evaluations() as f64);
            }
            // rounding can put a solution a few ulps below the minimum
            let best = outcome.best_fitness().score().expect("valid");
            errors.push((best - minimum).max(0.0));
            // a corner of the box, where every gene is at a bound
            if algorithm == "PSO" && outcome.best_genome().iter().all(|x| x.abs() == 5.0) {
                corners += 1;
            }
        }
        let reached = format!("{}/{SEEDS}", evaluations.len());
        if algorithm == "CMA-ES with IPOP" {
            // the main method's fewest and most evaluations to the minimum
            let fewest = evaluations.iter().copied().fold(f64::INFINITY, f64::min);
            let most = evaluations.iter().copied().fold(0.0, f64::max);
            main_range = (fewest, most);
        }
        let evaluations = median(evaluations).map_or("-".to_string(), |e| format!("{e:.0}"));
        let error = median(errors).expect("a run");
        println!(
            "{algorithm:<16}  {budget:>7}  {reached:>6}  {evaluations:>11}  {:>12}",
            format!("{error:.1e}")
        );
    }
    println!("evaluations: the median of the runs that reach the minimum");
    let (fewest, most) = main_range;
    println!("CMA-ES with IPOP: from {fewest:.0} to {most:.0} evaluations to the minimum");
    println!(
        "PSO: {corners} of its {SEEDS} runs end on a corner of the box, every gene at a bound"
    );

    // with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate
    // run in 2 dimensions: the plot is the function's contour
    trace::record_small()?;
    Ok(())
}

// a run of `algorithm` from `seed`, until its best is at most `target` or it has used `budget`
// evaluations
fn run(
    algorithm: &str,
    problem: Katsuura,
    seed: u64,
    target: f64,
    budget: u64,
) -> Result<Outcome<Reals>> {
    let real = problem.representation();
    let stop = Stop::target(target).or(Stop::evaluations(budget));
    match algorithm {
        "CMA-ES" | "CMA-ES with IPOP" => {
            // without restarts, the run ends once it has converged: sampling on around its point
            // wouldn't change its best
            let restarts = if algorithm == "CMA-ES" {
                cmaes::Restarts::Stop
            } else {
                cmaes::Restarts::Ipop
            };
            let cmaes = Cmaes::builder(real)
                .restarts(restarts)
                .minimize()
                .seed(seed)
                .build()?;
            Engine::new(cmaes, problem)
                .stop_when(stop)
                .parallel(true)
                .run()
        }
        "DE" => {
            let de = De::builder(real).minimize().seed(seed).build()?;
            Engine::new(de, problem)
                .stop_when(stop)
                .parallel(true)
                .run()
        }
        "PSO" => {
            let pso = Pso::builder(real)
                .population_size(40)
                .minimize()
                .seed(seed)
                .build()?;
            Engine::new(pso, problem)
                .stop_when(stop)
                .parallel(true)
                .run()
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
            Engine::new(ga, problem)
                .stop_when(stop)
                .parallel(true)
                .run()
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
