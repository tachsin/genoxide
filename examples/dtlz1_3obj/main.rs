//! DTLZ1 with 3 objectives: minimize three conflicting objectives over 7 variables in [0, 1],
//! whose Pareto front is the plane f₁ + f₂ + f₃ = 0.5, behind 11⁵ − 1 local fronts.
//!
//! NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, a
//! population of 92 and 400 generations, as in Deb and Jain (2014). Prints the size of the final
//! front, its hypervolume, its IGD+ to 1,035 points of the optimal front, and how far its
//! farthest solution is from the front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example dtlz1_3obj
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{Dtlz1, MultiProblem};
use genoxide::prelude::*;

const VARIABLES: usize = 7;

// the reference point of the hypervolume: 1.1 times the nadir point (0.5, 0.5, 0.5)
const REFERENCE: [f64; 3] = [0.55; 3];

fn main() -> Result<()> {
    let problem = Dtlz1::<3>::new(VARIABLES);
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
        .stop_when(Stop::generations(400))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    // the hypervolume of the front; the whole front's is 0.55³ minus the corner that the plane
    // cuts off, 0.5³ / 6
    let front = outcome.front_values();
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 3]);
    println!(
        "{} solutions on the front, hypervolume {volume:.4} (the whole front: 0.1455)",
        front.len()
    );
    // IGD+ to 1,035 points spread evenly over the plane
    let optimal = problem.optimal_front(1000).expect("known");
    let distance = igd_plus(&front, &optimal, &[Minimize; 3]);
    println!("IGD+ to the optimal front {distance:.5}");
    // the objectives sum to (1 + g) / 2: g is 0 on the front
    let farthest = front
        .iter()
        .map(|f| 2.0 * f.iter().sum::<f64>() - 1.0)
        .fold(0.0, f64::max);
    println!("the largest g on the front {farthest:.5}");
    trace.write();
    Ok(())
}
