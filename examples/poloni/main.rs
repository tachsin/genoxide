//! Poloni: minimize two objectives over x₁ and x₂ in [−π, π], with NSGA-II.
//!
//! Poloni's problem, from genoxide's `multi::problems::Poloni`. Its front is in two pieces, and
//! isn't known in closed form. Prints the size of the final front and how many of its solutions
//! are on each piece, and its hypervolume, against that of a fine grid over the box.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example poloni
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{MultiProblem, Poloni};
use genoxide::prelude::*;

// the reference point of the hypervolume: about 10% of the front's range beyond its worst point
// (16.77, 25)
const REFERENCE: [f64; 2] = [18.4, 27.5];

fn main() -> Result<()> {
    let problem = Poloni;
    // polynomial mutation at a rate of 1/2, one gene per child on average
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(0.5, 20.0)?)
        .seed(1)
        .build()?;

    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = MultiEngine::new(nsga2, problem)
        .stop_when(Stop::generations(250))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    // the first piece runs from (1, 25) down to f₂ ≈ 20.9, the second from f₂ ≈ 3.1 to 0
    let front = outcome.front_values();
    let first = front.iter().filter(|f| f[1] > 12.0).count();
    println!(
        "{} solutions on the front: {first} on the first piece, {} on the second",
        front.len(),
        front.len() - first
    );

    // the front isn't known: the non-dominated points of a 4001 × 4001 grid over the box give a
    // lower bound on its hypervolume
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 2]);
    println!("hypervolume {volume:.2} (a fine grid: 444.57)");
    trace.write();
    Ok(())
}
