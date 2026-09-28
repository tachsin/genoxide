//! Schwefel 1.2: minimize the sum of the squared partial sums of 30 genes, which interact.
//!
//! Compares how fast CMA-ES, with a full and with a diagonal covariance matrix (sep-CMA-ES),
//! particle swarm optimization and a real-coded genetic algorithm close in on the minimum, 0 at
//! the origin: the evaluations each takes until its error is at most 1, 1e-2, 1e-4, 1e-6 and
//! 1e-8. The function is genoxide's `problems::Schwefel1_2`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of a run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example schwefel_1_2
//! ```

mod trace;

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::{Problem, Schwefel1_2};

const DIMENSIONS: usize = 30;
const BUDGET: u64 = 10_000 * DIMENSIONS as u64;
// the errors at which the table gives each run's evaluations
const ERRORS: [f64; 5] = [1e0, 1e-2, 1e-4, 1e-6, 1e-8];
const COLUMNS: [&str; 5] = ["1", "1e-2", "1e-4", "1e-6", "1e-8"];

fn main() -> Result<()> {
    let problem = Schwefel1_2::new(DIMENSIONS);
    let stop = || Stop::target(1e-8).or(Stop::evaluations(BUDGET));

    println!("Schwefel 1.2 in {DIMENSIONS} dimensions, {BUDGET} evaluations at most");
    println!("Evaluations until the error is at most");
    print!("{:<10}", "algorithm");
    COLUMNS.iter().for_each(|column| print!("{column:>9}"));
    println!("{:>9}", "best");

    for (name, covariance) in [
        ("CMA-ES", cmaes::Covariance::Full),
        ("sep-CMA-ES", cmaes::Covariance::Diagonal),
    ] {
        let cmaes = Cmaes::builder(problem.representation())
            .covariance(covariance)
            .minimize()
            .seed(1)
            .build()?;
        let mut reached = Reached::default();
        let outcome = Engine::new(cmaes, problem)
            .stop_when(stop())
            .on_generation(|snapshot| reached.record(snapshot))
            .run()?;
        reached.print(name, &outcome);
    }

    let pso = Pso::builder(problem.representation())
        .population_size(40)
        .minimize()
        .seed(1)
        .build()?;
    let mut reached = Reached::default();
    let outcome = Engine::new(pso, problem)
        .stop_when(stop())
        .on_generation(|snapshot| reached.record(snapshot))
        .run()?;
    reached.print("PSO", &outcome);

    let ga = Ga::builder(problem.representation())
        .population_size(100)
        .select(Tournament::new(3)?)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / DIMENSIONS as f64, 20.0)?)
        .minimize()
        .seed(1)
        .build()?;
    let mut reached = Reached::default();
    let outcome = Engine::new(ga, problem)
        .stop_when(stop())
        .on_generation(|snapshot| reached.record(snapshot))
        .run()?;
    reached.print("GA", &outcome);

    // with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate
    // run in 2 dimensions: the plot is the function's contour
    trace::record_2d()?;
    Ok(())
}

// the evaluations after the first generation whose best error was at most each of ERRORS
#[derive(Default)]
struct Reached([Option<u64>; 5]);

impl Reached {
    fn record(&mut self, snapshot: &Snapshot<'_, Reals>) {
        let progress = snapshot.progress();
        let Some(best) = progress.best().and_then(Fitness::score) else {
            return;
        };
        for (reached, error) in self.0.iter_mut().zip(ERRORS) {
            if reached.is_none() && best <= error {
                *reached = Some(progress.evaluations());
            }
        }
    }

    // a row of the table: the evaluations, "-" for an error not reached, and the best error
    fn print(&self, name: &str, outcome: &Outcome<Reals>) {
        print!("{name:<10}");
        for reached in self.0 {
            let reached = reached.map_or("-".to_string(), |evaluations| evaluations.to_string());
            print!("{reached:>9}");
        }
        let best = outcome.best_fitness().score().expect("valid");
        println!("{:>9}", format!("{best:.1e}"));
    }
}
