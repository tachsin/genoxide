//! Michalewicz: minimize a function of steep narrow valleys on flat plateaus, in 10 dimensions.
//!
//! Compares CMA-ES with IPOP restarts (a population that doubles at each restart) and L-SHADE
//! (differential evolution with a population that shrinks over the budget). The global minimum,
//! about −9.66 in 10 dimensions, is computed a gene at a time. The function is genoxide's
//! `problems::Michalewicz`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example michalewicz
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Michalewicz, Optimum, Problem};

const DIMENSIONS: usize = 10;
const BUDGET: u64 = 10_000 * DIMENSIONS as u64;

fn main() -> Result<()> {
    let problem = Michalewicz::new(DIMENSIONS);
    let optimum = problem.optimum().expect("known");
    let target = optimum.value() + 1e-8;
    let stop = || Stop::target(target).or(Stop::evaluations(BUDGET));
    println!(
        "Michalewicz in {DIMENSIONS} dimensions: minimum {:.4}, {BUDGET} evaluations at most",
        optimum.value()
    );

    let ipop = Cmaes::builder(problem.representation())
        .restarts(cmaes::Restarts::Ipop)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(ipop, problem).stop_when(stop()).run()?;
    report("CMA-ES with IPOP", &outcome, &optimum);

    let l_shade = De::l_shade(problem.representation(), BUDGET)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(l_shade, problem).stop_when(stop()).run()?;
    report("L-SHADE", &outcome, &optimum);

    // with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate
    // run in 2 dimensions: the plot is the function's contour
    trace::record_2d()?;
    Ok(())
}

// the error to the minimum, what stopped the run, and how many genes are more than 0.01 from the
// minimum's. Not the evaluations: the function calls the platform's sin, and a run that meets the
// target takes a slightly different number of them on each operating system
fn report(name: &str, outcome: &Outcome<Reals>, optimum: &Optimum<Reals>) {
    let best = outcome.best_fitness().score().expect("valid");
    // rounding can put a solution a few ulps below the minimum
    let error = (best - optimum.value()).max(0.0);
    let solution = &optimum.solutions()[0];
    let genes = outcome.best().genome().iter().zip(solution.iter());
    let off = genes
        .filter(|(x, minimum)| (*x - *minimum).abs() > 0.01)
        .count();
    let stop = match outcome.stop_reason() {
        StopReason::Target => "at the target",
        _ => "with the budget spent",
    };
    print!("{name}: error {error:.4} {stop}, ");
    println!("{off} of {DIMENSIONS} genes off by more than 0.01");
}
