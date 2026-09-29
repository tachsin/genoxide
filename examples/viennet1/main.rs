//! Viennet 1 (VNT1): minimize three objectives of two variables, the squared distances to three
//! points plus constants, whose Pareto front is a curved triangle.
//!
//! NSGA-III twice, for 50 generations: with the 91 reference directions of Das and Dennis's method
//! with 12 divisions and a population of 92, and with 496 directions, 30 divisions, and as many
//! solutions. Prints each final front's size, hypervolume and IGD+ to 1,035 points of the optimal
//! front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of the second run for the plot on the
//! example's page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example viennet1
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::MultiSnapshot;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Viennet1};
use genoxide::prelude::*;

// the reference point of the hypervolume: the nadir point (4, 5, 4) plus a tenth of each
// objective's range on the front, whose ideal point is (0, 1, 2)
const REFERENCE: [f64; 3] = [4.4, 5.4, 4.2];

fn main() -> Result<()> {
    // 91 directions and a population of 92, the multiple of 4 above
    run("91 directions", 12, Some(92), |_| {})?;

    // with GENOXIDE_TRACE=<file>, a trace of this run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // 496 directions, and a solution for each
    run("496 directions", 30, None, |snapshot| {
        trace.record(snapshot)
    })?;

    // the whole front's hypervolume, from a 4,001 × 4,001 grid of the variables
    println!("the whole front: hypervolume 33.52");
    trace.write();
    Ok(())
}

// runs NSGA-III with the directions of Das and Dennis's method with `divisions` for 50
// generations, and reports its front
fn run(
    name: &str,
    divisions: usize,
    population: Option<usize>,
    record: impl FnMut(&MultiSnapshot<'_, Reals, 3>),
) -> Result<()> {
    let problem = Viennet1;
    let directions = multi::das_dennis::<3>(divisions);
    let mut builder = Nsga3::builder(problem.representation(), [Minimize; 3], directions)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(0.5, 20.0)?)
        .seed(1);
    if let Some(size) = population {
        builder = builder.population_size(size);
    }
    let outcome = MultiEngine::new(builder.build()?, problem)
        .stop_when(Stop::generations(50))
        .on_generation(record)
        .run()?;

    // the hypervolume of the front, and its IGD+ to the images of 1,035 points spread evenly over
    // the triangle of optimal solutions; scaled, with each objective scaled to [0, 1] over its
    // range on the front, from the ideal point (0, 1, 2) to the nadir point (4, 5, 4)
    let front = outcome.front_values();
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 3]);
    let optimal = problem.optimal_front(1000).expect("known");
    let distance = igd_plus(&front, &optimal, &[Minimize; 3]);
    let (ideal, nadir) = (problem.ideal_point(), problem.nadir_point());
    let (ideal, nadir) = (ideal.expect("known"), nadir.expect("known"));
    let scale = |points: &[[f64; 3]]| -> Vec<[f64; 3]> {
        let scaled =
            |p: &[f64; 3]| std::array::from_fn(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j]));
        points.iter().map(scaled).collect()
    };
    let scaled = igd_plus(&scale(&front), &scale(&optimal), &[Minimize; 3]);
    println!(
        "{name:<14} {} solutions, hypervolume {volume:.4}, IGD+ {distance:.4} (scaled \
         {scaled:.4})",
        front.len()
    );
    Ok(())
}
