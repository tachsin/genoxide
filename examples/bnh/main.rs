//! BNH: minimize two objectives subject to two constraints, with NSGA-II and Deb's rules.
//!
//! Binh and Korn's problem, from genoxide's `multi::problems::Bnh`, whose fitness is the two
//! objectives and the constraint violation. Prints how many solutions of the final front are
//! feasible, their IGD+ to 500 points of the optimal front, and the front's hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example bnh
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{Bnh, MultiProblem};
use genoxide::prelude::*;

fn main() -> Result<()> {
    let problem = Bnh;
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(0.5, 20.0)?)
        .seed(1)
        .build()?;

    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = MultiEngine::new(nsga2, problem)
        .stop_when(Stop::generations(250))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    let front = outcome.front();
    let feasible = front
        .iter()
        .filter(|individual| {
            individual
                .fitness()
                .is_some_and(|scores| scores.is_feasible())
        })
        .count();
    println!(
        "{} solutions on the front, {feasible} feasible",
        front.len()
    );

    // IGD+ to the optimal front, x₁ = x₂ from 0 to 5
    let values = outcome.front_values();
    let optimal = problem.optimal_front(500).expect("known");
    let distance = igd_plus(&values, &optimal, &[Minimize; 2]);
    println!("IGD+ to the optimal front: {distance:.4}");

    // the hypervolume with the reference point (210, 55); the whole front's is
    // 210 × 55 − 5000/3, the area above f₂ = 2 (√(f₁/8) − 5)²
    let volume = hypervolume(&values, &[210.0, 55.0], &[Minimize; 2]);
    println!("hypervolume {volume:.2} (the whole front: 9883.33)");
    trace.write();
    Ok(())
}
