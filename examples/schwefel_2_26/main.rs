//! Schwefel 2.26: minimize a deceptive function whose best local minima are far apart, in 30
//! dimensions.
//!
//! L-SHADE (differential evolution with a population that shrinks over the budget) reaches the
//! global minimum, −418.98 per gene, where every gene is 420.97, near the upper bound. CMA-ES with
//! IPOP restarts and particle swarm optimization with a ring topology, for contrast, stay far from
//! it. The function is genoxide's `problems::Schwefel2_26`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example schwefel_2_26
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Problem, Schwefel2_26};

const DIMENSIONS: usize = 30;
const BUDGET: u64 = 10_000 * DIMENSIONS as u64;

fn main() -> Result<()> {
    let problem = Schwefel2_26::new(DIMENSIONS);
    let optimum = problem.optimum().expect("known");
    let minimum = optimum.value();
    println!("minimum: {minimum:.2}, {BUDGET} evaluations at most");
    let target = minimum + 1e-8;
    let stop = || Stop::target(target).or(Stop::evaluations(BUDGET));
    // the global minimizer of each gene, 420.97
    let best_gene = optimum.solutions()[0][0];
    let report = |name: &str, outcome: &Outcome<Reals>| {
        let best = outcome.best_fitness().score().expect("valid");
        // rounding can put a solution a few ulps below the minimum
        let error = (best - minimum).max(0.0);
        let genome = outcome.best().genome();
        let near = genome.iter().filter(|x| (*x - best_gene).abs() < 1.0);
        println!(
            "{name}: {best:.2}, error {error:.1e}, {} of {DIMENSIONS} genes within 1 of \
             {best_gene:.2}",
            near.count()
        );
    };

    let l_shade = De::l_shade(problem.representation(), BUDGET)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(l_shade, problem).stop_when(stop()).run()?;
    report("L-SHADE", &outcome);

    println!("for contrast, two algorithms that stay far from it:");
    let cmaes = Cmaes::builder(problem.representation())
        .restarts(cmaes::Restarts::Ipop)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(cmaes, problem).stop_when(stop()).run()?;
    report("CMA-ES with IPOP", &outcome);

    let pso = Pso::builder(problem.representation())
        .population_size(40)
        .topology(pso::Topology::Ring { neighbors: 1 })
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(pso, problem).stop_when(stop()).run()?;
    report("PSO on a ring", &outcome);

    // with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate
    // run in 2 dimensions: the plot is the function's contour
    trace::record_2d()?;
    Ok(())
}
