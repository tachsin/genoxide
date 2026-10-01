//! High-conditioned elliptic: minimize an ellipsoid whose axes range from 1 to 1000 in length, in 30
//! dimensions, as it is and shifted and rotated, as CEC 2005's F3.
//!
//! Compares how fast CMA-ES, with a full and with a diagonal covariance matrix (sep-CMA-ES),
//! differential evolution, particle swarm optimization and a real-coded genetic algorithm close in
//! on the minimum, 0 at the origin: the evaluations each takes until its error is at most 1, 1e-2, 1e-4, 1e-6 and 1e-8.
//! The function is genoxide's `problems::HighConditionedElliptic`. Then the same on the function shifted and rotated, with genoxide's `problems::Shifted`
//! and `problems::Rotated`, as the CEC and BBOB suites transform it.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of a run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example high_conditioned_elliptic
//! ```

mod trace;

use genoxide::observer::Snapshot;
use genoxide::prelude::*;
use genoxide::problems::{HighConditionedElliptic, Problem, Rotated, Shifted};

const DIMENSIONS: usize = 30;
const BUDGET: u64 = 10_000 * DIMENSIONS as u64;
// the errors at which the table gives each run's evaluations
const ERRORS: [f64; 5] = [1e0, 1e-2, 1e-4, 1e-6, 1e-8];
const COLUMNS: [&str; 5] = ["1", "1e-2", "1e-4", "1e-6", "1e-8"];

fn main() -> Result<()> {
    println!("High-conditioned elliptic in {DIMENSIONS} dimensions, {BUDGET} evaluations at most");
    compare(
        "High-conditioned elliptic",
        &HighConditionedElliptic::new(DIMENSIONS),
    )?;
    // CEC 2005's F3, with genoxide's own shift and rotation
    let rotated = Rotated::new(Shifted::new(HighConditionedElliptic::new(DIMENSIONS), 1), 1);
    compare("Shifted and rotated (seed 1)", &rotated)?;
    // with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate
    // run in 2 dimensions: the plot is the function's contour
    trace::record_small()?;
    Ok(())
}

// the table of the five algorithms on `problem`, after a line that names it
fn compare<P>(name: &str, problem: &P) -> Result<()>
where
    P: Problem<Representation = Real> + FitnessFunction<Reals, Output = f64> + Clone,
{
    let minimum = problem.optimum().expect("known").value();
    let stop = || Stop::target(minimum + 1e-8).or(Stop::evaluations(BUDGET));
    println!("{name}: evaluations until the error is at most");
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
        let mut reached = Reached::new(minimum);
        let outcome = Engine::new(cmaes, problem.clone())
            .stop_when(stop())
            .on_generation(|snapshot| reached.record(snapshot))
            .run()?;
        reached.print(name, &outcome);
    }

    let de = De::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    let mut reached = Reached::new(minimum);
    let outcome = Engine::new(de, problem.clone())
        .stop_when(stop())
        .on_generation(|snapshot| reached.record(snapshot))
        .run()?;
    reached.print("DE", &outcome);

    let pso = Pso::builder(problem.representation())
        .population_size(40)
        .minimize()
        .seed(1)
        .build()?;
    let mut reached = Reached::new(minimum);
    let outcome = Engine::new(pso, problem.clone())
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
    let mut reached = Reached::new(minimum);
    let outcome = Engine::new(ga, problem.clone())
        .stop_when(stop())
        .on_generation(|snapshot| reached.record(snapshot))
        .run()?;
    reached.print("GA", &outcome);
    Ok(())
}

// the evaluations after the first generation whose best error was at most each of ERRORS, for a
// function whose minimum is `minimum`
struct Reached {
    minimum: f64,
    evaluations: [Option<u64>; 5],
}

impl Reached {
    fn new(minimum: f64) -> Self {
        let evaluations = [None; 5];
        Self {
            minimum,
            evaluations,
        }
    }

    fn record(&mut self, snapshot: &Snapshot<'_, Reals>) {
        let progress = snapshot.progress();
        let Some(best) = progress.best().and_then(Fitness::score) else {
            return;
        };
        let error = best - self.minimum;
        for (reached, bound) in self.evaluations.iter_mut().zip(ERRORS) {
            if reached.is_none() && error <= bound {
                *reached = Some(progress.evaluations());
            }
        }
    }

    // a row of the table: the evaluations, "-" for an error not reached, and the best error
    fn print(&self, name: &str, outcome: &Outcome<Reals>) {
        print!("{name:<10}");
        for reached in self.evaluations {
            let reached = reached.map_or("-".to_string(), |evaluations| evaluations.to_string());
            print!("{reached:>9}");
        }
        // rounding can put a solution a few ulps below the minimum
        let best = outcome.best_fitness().score().expect("valid");
        let error = (best - self.minimum).max(0.0);
        println!("{:>9}", format!("{error:.1e}"));
    }
}
