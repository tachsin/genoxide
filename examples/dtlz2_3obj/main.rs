//! DTLZ2 with 3 objectives: minimize three conflicting objectives over 12 variables in [0, 1],
//! whose Pareto front is the positive eighth of the unit sphere.
//!
//! NSGA-III twice: with Deb and Jain's (2014) settings, 91 reference directions from Das and
//! Dennis's method with 12 divisions and a population of 92; and with 703 directions, 36
//! divisions, and as many solutions. Both run for 250 generations. Prints each final front's
//! size, hypervolume and IGD+ to 1,035 points of the optimal front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of the second run for the plot on the
//! example's page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example dtlz2_3obj
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{Dtlz2, MultiProblem};
use genoxide::prelude::*;

const VARIABLES: usize = 12;

fn main() -> Result<()> {
    // Deb and Jain's settings: 91 directions and a population of 92, the multiple of 4 above
    run("91 directions", 12, Some(92), |_| {})?;

    // with GENOXIDE_TRACE=<file>, a trace of this run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // 703 directions, and a solution for each
    run("703 directions", 36, None, |snapshot| {
        trace.record(snapshot)
    })?;

    // the whole front's hypervolume, with the reference point (1.1, 1.1, 1.1): 1.1³ minus the
    // eighth of the unit ball, π/6
    println!("the whole front: hypervolume 0.8074");
    trace.write();
    Ok(())
}

// runs NSGA-III with the directions of Das and Dennis's method with `divisions` for 250
// generations, and reports its front
fn run(
    name: &str,
    divisions: usize,
    population: Option<usize>,
    record: impl FnMut(&MultiSnapshot<'_, Reals, 3>),
) -> Result<()> {
    let problem = Dtlz2::<3>::new(VARIABLES);
    let directions = multi::das_dennis::<3>(divisions);
    let mut builder = Nsga3::builder(problem.representation(), [Minimize; 3], directions)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / VARIABLES as f64, 20.0)?)
        .seed(1);
    if let Some(size) = population {
        builder = builder.population_size(size);
    }
    let outcome = MultiEngine::new(builder.build()?, problem)
        .stop_when(Stop::generations(250))
        .on_generation(record)
        .run()?;

    let front = outcome.front_values();
    let volume = hypervolume(&front, &[1.1; 3], &[Minimize; 3]);
    // IGD+ to 1,035 points spread evenly over the sphere
    let optimal = problem.optimal_front(1000).expect("known");
    let distance = igd_plus(&front, &optimal, &[Minimize; 3]);
    println!(
        "{name:<14} {} solutions, hypervolume {volume:.4}, IGD+ {distance:.4}",
        front.len()
    );
    Ok(())
}
