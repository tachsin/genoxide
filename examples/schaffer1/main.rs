//! Schaffer 1: minimize x² and (x − 2)² over x in [−1000, 1000], with NSGA-II.
//!
//! Schaffer's first problem, from genoxide's `multi::problems::Schaffer1`. Its front is convex,
//! and its optimal solutions, x in [0, 2], are a thousandth of the interval. Prints the size of
//! the final front, its IGD+ to 500 points of the optimal front, and its hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example schaffer1
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Schaffer1};
use genoxide::prelude::*;

// the reference point of the hypervolume: 10% of the front's range beyond its worst point (4, 4)
const REFERENCE: [f64; 2] = [4.4, 4.4];

fn main() -> Result<()> {
    let problem = Schaffer1;
    // one variable: polynomial mutation changes it in every child
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0, 20.0)?)
        .seed(1)
        .build()?;

    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = MultiEngine::new(nsga2, problem)
        .stop_when(Stop::generations(250))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    let front = outcome.front_values();
    println!("{} solutions on the front", front.len());

    // IGD+ to the optimal front, x from 0 to 2
    let optimal = problem.optimal_front(500).expect("known");
    let distance = igd_plus(&front, &optimal, &[Minimize; 2]);
    println!("IGD+ to the optimal front: {distance:.4}");

    // the whole front's hypervolume is 4.4² − 8/3, the box minus the area under
    // f₂ = (√f₁ − 2)²
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 2]);
    println!("hypervolume {volume:.3} (the whole front: 16.693)");
    trace.write();
    Ok(())
}
