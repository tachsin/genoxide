//! Kursawe: minimize two objectives whose Pareto front is in disconnected pieces, with SPEA2 and
//! NSGA-II.
//!
//! Kursawe's problem in 3 variables, from genoxide's `multi::problems::Kursawe`. Its front isn't
//! known in closed form, so the example compares the two algorithms' fronts by their hypervolume,
//! and counts the pieces each finds: a new piece starts where two neighbors on the front are
//! more than 0.5 apart.
//!
//! ```text
//! cargo run --release --example kursawe
//! ```

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{Kursawe, MultiProblem};
use genoxide::prelude::*;

const REFERENCE: [f64; 2] = [-14.0, 1.0];

fn main() -> Result<()> {
    let problem = Kursawe::new(3);
    let spea2 = Spea2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0)?)
        .seed(1)
        .build()?;
    let spea2 = MultiEngine::new(spea2, problem)
        .stop_when(Stop::generations(250))
        .run()?;
    report("SPEA2", &spea2.front_values());

    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0)?)
        .seed(1)
        .build()?;
    let nsga2 = MultiEngine::new(nsga2, problem)
        .stop_when(Stop::generations(250))
        .run()?;
    report("NSGA-II", &nsga2.front_values());
    Ok(())
}

// the size of the front, its pieces and its hypervolume
fn report(name: &str, front: &[[f64; 2]]) {
    let mut sorted = front.to_vec();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let pieces = 1 + sorted
        .windows(2)
        .filter(|pair| {
            let (d1, d2) = (pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]);
            (d1 * d1 + d2 * d2).sqrt() > 0.5
        })
        .count();
    let volume = hypervolume(front, &REFERENCE, &[Minimize; 2]);
    println!(
        "{name:<8} {} solutions in {pieces} pieces, hypervolume {volume:.4}",
        front.len()
    );
}
