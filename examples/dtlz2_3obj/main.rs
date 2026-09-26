//! DTLZ2 with 3 objectives: minimize three conflicting objectives over 12 variables in [0, 1],
//! whose Pareto front is the positive eighth of the unit sphere.
//!
//! NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, a
//! population of 92 and 250 generations, as in Deb and Jain (2014). Prints the size of the final
//! front and its hypervolume.
//!
//! ```text
//! cargo run --release --example dtlz2_3obj
//! ```

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{Dtlz2, TestProblem};
use genoxide::prelude::*;

const VARIABLES: usize = 12;

fn main() -> Result<()> {
    let problem = Dtlz2::<3>::new(VARIABLES);
    let directions = multi::das_dennis::<3>(12);
    let nsga3 = Nsga3::builder(problem.real(), [Minimize; 3], directions)
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / VARIABLES as f64, 20.0)?)
        .seed(1)
        .build()?;

    let outcome = MultiEngine::new(nsga3, problem)
        .stop_when(Stop::generations(250))
        .run()?;

    // the hypervolume of the front, with the reference point (1.1, 1.1, 1.1); the whole front's
    // is 1.1³ minus the eighth of the unit ball, π/6
    let front = outcome.front_values();
    let volume = hypervolume(&front, &[1.1; 3], &[Minimize; 3]);
    println!(
        "{} solutions on the front, hypervolume {volume:.4} (the whole front: 0.8074)",
        front.len()
    );
    Ok(())
}
