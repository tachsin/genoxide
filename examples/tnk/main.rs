//! TNK: minimize two objectives subject to two constraints, with NSGA-II and Deb's rules.
//!
//! Tanaka's problem, from genoxide's `multi::problems::Tnk`, whose fitness is the two objectives
//! and the constraint violation. Prints how many solutions of the final front are feasible, how
//! many lie on each of the five pieces of the optimal front, their IGD+ to 500 points of the
//! optimal front, and the front's hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example tnk
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Tnk};
use genoxide::prelude::*;

fn main() -> Result<()> {
    let problem = Tnk;
    // the settings of the NSGA-II paper: a mutation rate of 1/n for n genes
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

    // the five pieces of the optimal front, symmetric in f₁ and f₂: the first ends at
    // f₁ = 0.1996, the second at f₁ = 0.6147, and the middle one begins at f₁ = 0.6202
    let values = outcome.front_values();
    let mut pieces = [0; 5];
    for [f1, f2] in &values {
        let piece = match () {
            _ if *f1 < 0.3 => 0,
            _ if *f1 < 0.6175 => 1,
            _ if *f2 < 0.3 => 4,
            _ if *f2 < 0.6175 => 3,
            _ => 2,
        };
        pieces[piece] += 1;
    }
    let [a, b, c, d, e] = pieces;
    println!("on the five pieces of the front, from f1 = 0.04 to 1.04: {a}, {b}, {c}, {d}, {e}");

    // IGD+ to the optimal front
    let optimal = problem.optimal_front(500).expect("known");
    let distance = igd_plus(&values, &optimal, &[Minimize; 2]);
    println!("IGD+ to the optimal front: {distance:.6}");

    // the hypervolume with the reference point (1.2, 1.2); the whole front's, from 2,000,000 of
    // its points, is 0.6551
    let volume = hypervolume(&values, &[1.2, 1.2], &[Minimize; 2]);
    println!("hypervolume {volume:.4} (the whole front: 0.6551)");
    trace.write();
    Ok(())
}
