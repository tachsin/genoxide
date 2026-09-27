//! Ackley: minimize a function with a deep central hole in a nearly flat, rippled plain, in 30
//! dimensions.
//!
//! Compares CMA-ES with IPOP restarts and particle swarm optimization with two topologies: every
//! particle following the whole swarm's best, and every particle following the best of its two
//! neighbors on a ring. The global minimum is 0, at the origin. The function is genoxide's
//! `problems::Ackley`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example ackley
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Ackley, Problem};

const DIMENSIONS: usize = 30;
const BUDGET: u64 = 10_000 * DIMENSIONS as u64;

fn main() -> Result<()> {
    let problem = Ackley::new(DIMENSIONS);
    let target = problem.optimum().expect("known").value() + 1e-8;
    let stop = || Stop::target(target).or(Stop::evaluations(BUDGET));

    let cmaes = Cmaes::builder(problem.representation())
        .restarts(cmaes::Restarts::Ipop)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(cmaes, problem).stop_when(stop()).run()?;
    report("CMA-ES", &outcome);

    let topologies = [
        ("PSO, global", pso::Topology::Global),
        ("PSO, ring", pso::Topology::Ring { neighbors: 1 }),
    ];
    for (name, topology) in topologies {
        let pso = Pso::builder(problem.representation())
            .population_size(40)
            .topology(topology)
            .minimize()
            .seed(1)
            .build()?;
        let outcome = Engine::new(pso, problem).stop_when(stop()).run()?;
        report(name, &outcome);
    }

    // with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate
    // run in 2 dimensions: the plot is the function's contour
    trace::record_2d()?;
    Ok(())
}

fn report(name: &str, outcome: &Outcome<Reals>) {
    println!(
        "{name}: {:.6} after {} evaluations",
        outcome.best_fitness(),
        outcome.evaluations()
    );
}
