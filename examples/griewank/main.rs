//! Griewank: minimize a wide bowl with ripples that couple the genes, in 2 to 50 dimensions.
//!
//! Runs CMA-ES with 10 seeds in each dimension, without restarts and with IPOP restarts (a
//! population that doubles at each restart), and counts the runs that reach the global minimum,
//! 0 at the origin. Without restarts, CMA-ES reaches it more often in more dimensions. The
//! function is genoxide's `problems::Griewank`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example griewank
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Griewank, Problem};

const DIMENSIONS: [usize; 6] = [2, 5, 10, 20, 30, 50];
const EVALUATIONS_PER_DIMENSION: u64 = 10_000;
// the budget in few dimensions, where 10,000 per dimension is too little for IPOP's restarts
const MINIMUM_BUDGET: u64 = 100_000;
const SEEDS: u64 = 10;

fn main() -> Result<()> {
    println!(
        "Runs of CMA-ES within 1e-8 of the minimum, of {SEEDS}, with \
         {EVALUATIONS_PER_DIMENSION} evaluations per dimension, {MINIMUM_BUDGET} at least"
    );
    println!("{:>10}{:>14}{:>14}", "dimensions", "no restarts", "IPOP");
    for dimensions in DIMENSIONS {
        let problem = Griewank::new(dimensions);
        let never = solved(problem, cmaes::Restarts::Never)?;
        let ipop = solved(problem, cmaes::Restarts::Ipop)?;
        println!(
            "{dimensions:>10}{:>14}{:>14}",
            format!("{never}/{SEEDS}"),
            format!("{ipop}/{SEEDS}")
        );
    }

    // with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate
    // run in 2 dimensions: the plot is the function's contour
    trace::record_2d()?;
    Ok(())
}

// the runs of the seeds that reach the target
fn solved(problem: Griewank, restarts: cmaes::Restarts) -> Result<u64> {
    let target = problem.optimum().expect("known").value() + 1e-8;
    let budget = (EVALUATIONS_PER_DIMENSION * problem.dimensions() as u64).max(MINIMUM_BUDGET);
    let mut solved = 0;
    for seed in 1..=SEEDS {
        let cmaes = Cmaes::builder(problem.representation())
            .restarts(restarts)
            .minimize()
            .seed(seed)
            .build()?;
        let outcome = Engine::new(cmaes, problem)
            .stop_when(Stop::target(target).or(Stop::evaluations(budget)))
            .run()?;
        if outcome.stop_reason() == StopReason::Target {
            solved += 1;
        }
    }
    Ok(solved)
}
