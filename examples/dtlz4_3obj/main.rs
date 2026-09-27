//! DTLZ4 with 3 objectives: minimize three conflicting objectives over 12 variables in [0, 1],
//! whose Pareto front is the positive eighth of the unit sphere, with a bias that crowds
//! solutions towards its edges.
//!
//! NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, a
//! population of 92 and 600 generations, as in Deb and Jain (2014). Prints the size of the
//! final front, its hypervolume, its IGD+ to 1,035 points of the optimal front, and how far its
//! farthest solution is from the front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example dtlz4_3obj
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{Dtlz4, MultiProblem};
use genoxide::prelude::*;

const VARIABLES: usize = 12;

// the reference point of the hypervolume: 1.1 times the nadir point (1, 1, 1)
const REFERENCE: [f64; 3] = [1.1; 3];

fn main() -> Result<()> {
    let problem = Dtlz4::<3>::new(VARIABLES);
    let directions = multi::das_dennis::<3>(12);
    let nsga3 = Nsga3::builder(problem.representation(), [Minimize; 3], directions)
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / VARIABLES as f64, 20.0)?)
        .seed(1)
        .build()?;

    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = MultiEngine::new(nsga3, problem)
        .stop_when(Stop::generations(600))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    // the hypervolume of the front; the whole front's is 1.1³ minus the eighth of the unit ball,
    // π/6
    let front = outcome.front_values();
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 3]);
    println!(
        "{} solutions on the front, hypervolume {volume:.4} (the whole front: 0.8074)",
        front.len()
    );
    // IGD+ to 1,035 points spread evenly over the sphere
    let optimal = problem.optimal_front(1000).expect("known");
    let distance = igd_plus(&front, &optimal, &[Minimize; 3]);
    println!("IGD+ to the optimal front {distance:.5}");
    // the objectives are a point at distance 1 + g from the origin: g is 0 on the front
    let farthest = front
        .iter()
        .map(|f| f.iter().map(|v| v * v).sum::<f64>().sqrt() - 1.0)
        .fold(0.0, f64::max);
    println!("the largest g on the front {farthest:.5}");
    trace.write();
    Ok(())
}
