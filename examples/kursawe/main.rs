//! Kursawe: minimize two objectives whose Pareto front is in disconnected pieces, with SPEA2 and
//! NSGA-II.
//!
//! Kursawe's problem in 3 variables, from genoxide's `multi::problems::Kursawe`. Its front isn't
//! known in closed form, so the example compares the two algorithms' fronts by their hypervolume,
//! and with that of a reference front from much longer runs, and counts the pieces each finds: a
//! new piece starts where f₁ grows by more than 0.2 between two neighbors on the front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example kursawe
//! ```

mod trace;

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
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let spea2 = MultiEngine::new(spea2, problem)
        .stop_when(Stop::generations(250))
        .on_generation(trace.fronts("SPEA2"))
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
        .on_generation(trace.fronts("NSGA-II"))
        .run()?;
    report("NSGA-II", &nsga2.front_values());

    // the non-dominated solutions of 16 runs of NSGA-II, 500 solutions for 2,000 generations each:
    // a reference front of 309,166 points, since the true front isn't known
    println!("a reference front, from much longer runs: hypervolume 37.3489");
    trace.write();
    Ok(())
}

// the size of the front, its pieces and its hypervolume
fn report(name: &str, front: &[[f64; 2]]) {
    let mut sorted = front.to_vec();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]));
    // along a piece, f₁ grows by at most about 0.13 between neighbors, and f₂ falls; the gaps
    // between the pieces of the true front are 0.25 to 0.92 wide in f₁, with f₂ nearly unchanged
    let pieces = 1 + sorted
        .windows(2)
        .filter(|pair| pair[1][0] - pair[0][0] > 0.2)
        .count();
    let volume = hypervolume(front, &REFERENCE, &[Minimize; 2]);
    println!(
        "{name:<8} {} solutions in {pieces} pieces, hypervolume {volume:.4}",
        front.len()
    );
}
