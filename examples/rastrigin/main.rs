//! Rastrigin: minimize a real-valued function with many local minima, in 30 dimensions.
//!
//! Compares CMA-ES with IPOP restarts (a population that doubles at each restart) and L-SHADE
//! (differential evolution with a population that shrinks over the budget). The global minimum is
//! 0, at the origin. The function is genoxide's `problems::Rastrigin`.
//!
//! ```text
//! cargo run --release --example rastrigin
//! ```

use genoxide::prelude::*;
use genoxide::problems::{Problem, Rastrigin};

const DIMENSIONS: usize = 30;
const BUDGET: u64 = 1_000_000;

fn main() -> Result<()> {
    let problem = Rastrigin::new(DIMENSIONS);
    let target = problem.optimum().expect("known").value() + 1e-8;
    let stop = || Stop::target(target).or(Stop::evaluations(BUDGET));

    let cmaes = Cmaes::builder(problem.representation())
        .restarts(cmaes::Restarts::Ipop)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(cmaes, problem).stop_when(stop()).run()?;
    report("CMA-ES", &outcome);

    let l_shade = De::l_shade(problem.representation(), BUDGET)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(l_shade, problem).stop_when(stop()).run()?;
    report("L-SHADE", &outcome);
    Ok(())
}

fn report(name: &str, outcome: &Outcome<Reals>) {
    println!(
        "{name}: {:.6} after {} evaluations",
        outcome.best_fitness(),
        outcome.evaluations()
    );
}
