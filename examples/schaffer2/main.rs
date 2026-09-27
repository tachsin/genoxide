//! Schaffer 2: minimize a piecewise linear f₁ and (x − 5)² over x in [−5, 10], with NSGA-II.
//!
//! Schaffer's second problem, from genoxide's `multi::problems::Schaffer2`. Its front is in two
//! pieces. Prints the size of the final front and how many of its solutions are on each piece,
//! its IGD+ to 500 points of the optimal front, and its hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example schaffer2
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Schaffer2};
use genoxide::prelude::*;

// the reference point of the hypervolume: 10% of the front's range beyond its worst point (1, 16)
const REFERENCE: [f64; 2] = [1.2, 17.6];

fn main() -> Result<()> {
    let problem = Schaffer2;
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

    // the first piece, x in [1, 2), has f₁ in [−1, 0); the second, x in [4, 5], f₁ in [0, 1]
    let front = outcome.front_values();
    let first = front.iter().filter(|f| f[0] < 0.0).count();
    println!(
        "{} solutions on the front: {first} on the first piece, {} on the second",
        front.len(),
        front.len() - first
    );

    // IGD+ to the optimal front, both pieces
    let optimal = problem.optimal_front(500).expect("known");
    let distance = igd_plus(&front, &optimal, &[Minimize; 2]);
    println!("IGD+ to the optimal front: {distance:.4}");

    // the whole front's hypervolume is 2.2 × 17.6 − 37/3 − 1/3, the box minus the areas under
    // its two pieces
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 2]);
    println!("hypervolume {volume:.3} (the whole front: 26.053)");
    trace.write();
    Ok(())
}
