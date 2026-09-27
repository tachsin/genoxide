//! Fonseca-Fleming: minimize two objectives with a concave front over 3 variables in [−4, 4],
//! with NSGA-II.
//!
//! Fonseca and Fleming's problem, from genoxide's `multi::problems::FonsecaFleming`, with 3
//! variables as in the NSGA-II paper. Prints the size of the final front, its IGD+ to 500 points
//! of the optimal front, and its hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example fonseca_fleming
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{FonsecaFleming, MultiProblem};
use genoxide::prelude::*;

// the reference point of the hypervolume: beyond the front's worst point (0.9817, 0.9817), and
// every objective value, which is below 1
const REFERENCE: [f64; 2] = [1.1, 1.1];

fn main() -> Result<()> {
    let problem = FonsecaFleming::new(3);
    // polynomial mutation at a rate of 1/3, one gene per child on average
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0)?)
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

    // IGD+ to the optimal front, x₁ = x₂ = x₃ from −1/√3 to 1/√3
    let optimal = problem.optimal_front(500).expect("known");
    let distance = igd_plus(&front, &optimal, &[Minimize; 2]);
    println!("IGD+ to the optimal front: {distance:.4}");

    // the whole front's hypervolume, the box minus the area under the curve, found numerically
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 2]);
    println!("hypervolume {volume:.4} (the whole front: 0.5521)");
    trace.write();
    Ok(())
}
