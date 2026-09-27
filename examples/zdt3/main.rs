//! ZDT3: minimize two conflicting objectives over 30 variables in [0, 1], with a Pareto front in
//! five disconnected pieces, with NSGA-II.
//!
//! Zitzler, Deb and Thiele's third problem, from genoxide's `multi::problems::Zdt3`. Prints the
//! size of the final front and how many of its solutions are on each piece, its IGD+ to 500
//! points of the optimal front, and its hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example zdt3
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Zdt3};
use genoxide::prelude::*;

// the reference point of the hypervolume, beyond the front's worst point (0.852, 1): f₂ is
// negative on much of the front, down to −0.773, but f₁ and f₂ are at most 1 there
const REFERENCE: [f64; 2] = [1.1, 1.1];

// the five pieces of the optimal front, as ranges of f₁: each ends at a local minimum of
// f₂ = 1 − √f₁ − f₁ sin(10π f₁), where the next piece's values drop below it
const PIECES: [(f64, f64); 5] = [
    (0.0, 0.0830),
    (0.1822, 0.2578),
    (0.4093, 0.4539),
    (0.6184, 0.6525),
    (0.8233, 0.8518),
];

fn main() -> Result<()> {
    let problem = Zdt3::new(30);
    // polynomial mutation at a rate of 1/30, one gene per child on average
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 30.0, 20.0)?)
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

    // the solutions on each piece: f₁ within 0.001 of the piece's range
    let counts: Vec<String> = PIECES
        .iter()
        .map(|(low, high)| {
            let on = front
                .iter()
                .filter(|[f1, _]| (low - 0.001..=high + 0.001).contains(f1));
            on.count().to_string()
        })
        .collect();
    println!("on the five pieces: {}", counts.join(", "));

    // IGD+ to the optimal front, 500 points spread over the pieces in proportion to their widths
    let optimal = problem.optimal_front(500).expect("known");
    let distance = igd_plus(&front, &optimal, &[Minimize; 2]);
    println!("IGD+ to the optimal front: {distance:.4}");

    // the whole front's hypervolume, found numerically
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 2]);
    println!("hypervolume {volume:.4} (the whole front: 1.3318)");
    trace.write();
    Ok(())
}
