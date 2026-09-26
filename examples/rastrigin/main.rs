//! Rastrigin: minimize a real-valued function with many local minima, in 30 dimensions.
//!
//! Compares CMA-ES with IPOP restarts (a population that doubles at each restart) and L-SHADE
//! (differential evolution with a population that shrinks over the budget). The global minimum is
//! 0, at the origin.
//!
//! ```text
//! cargo run --release --example rastrigin
//! ```

use genoxide::prelude::*;
use std::f64::consts::PI;

const DIMENSIONS: usize = 30;
const BUDGET: u64 = 1_000_000;

fn rastrigin(x: &Reals) -> f64 {
    10.0 * x.len() as f64
        + x.iter()
            .map(|xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
            .sum::<f64>()
}

fn main() -> Result<()> {
    let real = || Real::uniform(DIMENSIONS, -5.12..=5.12);
    let stop = || Stop::target(1e-8).or(Stop::evaluations(BUDGET));

    let cmaes = Cmaes::builder(real()?)
        .restarts(cmaes::Restarts::Ipop)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(cmaes, rastrigin).stop_when(stop()).run()?;
    report("CMA-ES", &outcome);

    let l_shade = De::l_shade(real()?, BUDGET).minimize().seed(1).build()?;
    let outcome = Engine::new(l_shade, rastrigin).stop_when(stop()).run()?;
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
