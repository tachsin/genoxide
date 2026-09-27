//! Viennet 1 (VNT1): minimize three objectives of two variables, the squared distances to three
//! points plus constants, whose Pareto front is a curved triangle.
//!
//! NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, a
//! population of 92 and 50 generations. Prints the size of the final front, its hypervolume and
//! its IGD+ to 1,035 points of the optimal front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example viennet1
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Viennet1};
use genoxide::prelude::*;

// the reference point of the hypervolume: the nadir point (4, 5, 4) plus a tenth of each
// objective's range on the front, whose ideal point is (0, 1, 2)
const REFERENCE: [f64; 3] = [4.4, 5.4, 4.2];

fn main() -> Result<()> {
    let problem = Viennet1;
    let directions = multi::das_dennis::<3>(12);
    let nsga3 = Nsga3::builder(problem.representation(), [Minimize; 3], directions)
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(0.5, 20.0)?)
        .seed(1)
        .build()?;

    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = MultiEngine::new(nsga3, problem)
        .stop_when(Stop::generations(50))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    // the hypervolume of the front, and its IGD+ to the images of 1,035 points spread evenly over
    // the triangle of optimal solutions
    let front = outcome.front_values();
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 3]);
    let optimal = problem.optimal_front(1000).expect("known");
    let distance = igd_plus(&front, &optimal, &[Minimize; 3]);
    println!(
        "{} solutions on the front, hypervolume {volume:.4} (the whole front: 33.52)",
        front.len()
    );
    println!("IGD+ to the optimal front {distance:.4}");
    trace.write();
    Ok(())
}
