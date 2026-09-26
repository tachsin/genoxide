//! ZDT1: minimize two conflicting objectives over 30 variables in [0, 1].
//!
//! Shows NSGA-II on genoxide's ZDT1 test problem, and the hypervolume of the final non-dominated
//! front.
//!
//! ```text
//! cargo run --release --example zdt1
//! ```

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{TestProblem, Zdt1};
use genoxide::prelude::*;

fn main() -> Result<()> {
    let problem = Zdt1::new(30);
    let nsga2 = Nsga2::builder(problem.real(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 30.0, 20.0)?)
        .seed(1)
        .build()?;

    let outcome = MultiEngine::new(nsga2, problem)
        .stop_when(Stop::evaluations(25_000))
        .run()?;

    // the hypervolume of the front, with the reference point (1.1, 1.1)
    let front = outcome.front_values();
    let volume = hypervolume(&front, &[1.1, 1.1], &[Minimize; 2]);
    println!(
        "{} solutions on the front, hypervolume {volume:.4} (the whole front: 0.8767)",
        front.len()
    );
    Ok(())
}
