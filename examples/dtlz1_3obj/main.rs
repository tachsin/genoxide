//! DTLZ1 with 3 objectives: minimize three conflicting objectives over 7 variables in [0, 1],
//! whose Pareto front is the plane f₁ + f₂ + f₃ = 0.5, behind 11⁵ − 1 local fronts.
//!
//! NSGA-III twice: with Deb and Jain's (2014) settings, 91 reference directions from Das and
//! Dennis's method with 12 divisions, a population of 92 and 400 generations; and with 861
//! directions, 40 divisions, as many solutions, and 300 generations. Prints each final front's
//! size, hypervolume, IGD+ to 1,035 points of the optimal front, and how far its farthest solution
//! is from the front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of the second run for the plot on the
//! example's page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example dtlz1_3obj
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{Dtlz1, MultiProblem};
use genoxide::prelude::*;

const VARIABLES: usize = 7;

// the reference point of the hypervolume: 1.1 times the nadir point (0.5, 0.5, 0.5)
const REFERENCE: [f64; 3] = [0.55; 3];

fn main() -> Result<()> {
    // Deb and Jain's settings: 91 directions and a population of 92, the multiple of 4 above
    run("91 directions", 12, Some(92), 400, |_| {})?;

    // with GENOXIDE_TRACE=<file>, a trace of this run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // 861 directions, and a solution for each
    run("861 directions", 40, None, 300, |snapshot| {
        trace.record(snapshot)
    })?;

    // the whole front's hypervolume: 0.55³ minus the corner that the plane cuts off, 0.5³ / 6
    println!("the whole front: hypervolume 0.1455");
    trace.write();
    Ok(())
}

// runs NSGA-III with the directions of Das and Dennis's method with `divisions`, and reports its
// front after `generations`
fn run(
    name: &str,
    divisions: usize,
    population: Option<usize>,
    generations: u64,
    record: impl FnMut(&MultiSnapshot<'_, Reals, 3>),
) -> Result<()> {
    let problem = Dtlz1::<3>::new(VARIABLES);
    let directions = multi::das_dennis::<3>(divisions);
    let mut builder = Nsga3::builder(problem.representation(), [Minimize; 3], directions)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / VARIABLES as f64, 20.0)?)
        .seed(1);
    if let Some(size) = population {
        builder = builder.population_size(size);
    }
    let outcome = MultiEngine::new(builder.build()?, problem)
        .stop_when(Stop::generations(generations))
        .on_generation(record)
        .run()?;

    let front = outcome.front_values();
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 3]);
    // IGD+ to 1,035 points spread evenly over the plane, and scaled to the front's range: each
    // objective spans 0.5 on the front
    let optimal = problem.optimal_front(1000).expect("known");
    let distance = igd_plus(&front, &optimal, &[Minimize; 3]);
    // the objectives sum to (1 + g) / 2: g is 0 on the front
    let farthest = front
        .iter()
        .map(|f| 2.0 * f.iter().sum::<f64>() - 1.0)
        .fold(0.0, f64::max);
    println!(
        "{name:<14} {} solutions, hypervolume {volume:.4}, IGD+ {distance:.5} (scaled {:.4}), \
         largest g {farthest:.5}",
        front.len(),
        distance / 0.5
    );
    Ok(())
}
