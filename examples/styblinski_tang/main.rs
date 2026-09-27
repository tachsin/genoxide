//! Styblinski-Tang: minimize a separable function with a deceptive local minimum in each gene, in
//! 30 dimensions.
//!
//! Compares CMA-ES without restarts, CMA-ES with IPOP restarts (a population that doubles at each
//! restart) and L-SHADE (differential evolution with a population that shrinks over the budget).
//! The global minimum is −39.166 n, at xᵢ ≈ −2.9035. The function is genoxide's
//! `problems::StyblinskiTang`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example styblinski_tang
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Optimum, Problem, StyblinskiTang};

const DIMENSIONS: usize = 30;
const BUDGET: u64 = 10_000 * DIMENSIONS as u64;

fn main() -> Result<()> {
    let problem = StyblinskiTang::new(DIMENSIONS);
    let optimum = problem.optimum().expect("known");
    let target = optimum.value() + 1e-8;
    let stop = || Stop::target(target).or(Stop::evaluations(BUDGET));
    println!(
        "Styblinski-Tang in {DIMENSIONS} dimensions: minimum {:.4}, {BUDGET} evaluations at most",
        optimum.value()
    );

    let cmaes = Cmaes::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(cmaes, problem).stop_when(stop()).run()?;
    report("CMA-ES", &outcome, &optimum);

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

// the error to the minimum, the evaluations, and how many genes are more than 0.01 from the
// minimum's
fn report(name: &str, outcome: &Outcome<Reals>, optimum: &Optimum<Reals>) {
    let best = outcome.best_fitness().score().expect("valid");
    // rounding can put a solution a few ulps below the minimum
    let error = (best - optimum.value()).max(0.0);
    let solution = &optimum.solutions()[0];
    let genes = outcome.best().genome().iter().zip(solution.iter());
    let off = genes
        .filter(|(x, minimum)| (*x - *minimum).abs() > 0.01)
        .count();
    let evaluations = outcome.evaluations();
    print!("{name}: error {error:.4} after {evaluations} evaluations, ");
    println!("{off} of {DIMENSIONS} genes off by more than 0.01");
}
