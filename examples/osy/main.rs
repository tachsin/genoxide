//! OSY: minimize two objectives of six variables subject to six constraints, with NSGA-II and
//! Deb's rules.
//!
//! Osyczka and Kundu's problem, from genoxide's `multi::problems::Osy`, whose fitness is the two
//! objectives and the constraint violation. Prints how many solutions of the final front are
//! feasible, how many lie along each of the five pieces of the optimal front, their IGD+ to 500
//! points of the optimal front, and the front's hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example osy
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Osy};
use genoxide::prelude::*;

fn main() -> Result<()> {
    let problem = Osy;
    // the settings of the NSGA-II paper: a mutation rate of 1/n for n genes
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 6.0, 20.0)?)
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

    // the five pieces of the optimal front meet at f₁ = −258, −242, −123.46 and −116
    let values = outcome.front_values();
    let ends = [f64::MIN, -258.0, -242.0, -123.46, -116.0, f64::MAX];
    let pieces: Vec<String> = ends
        .windows(2)
        .map(|end| {
            let within = |point: &&[f64; 2]| end[0] <= point[0] && point[0] < end[1];
            values.iter().filter(within).count().to_string()
        })
        .collect();
    println!(
        "along the five pieces of the front, from f1 = -274 to -42: {}",
        pieces.join(", ")
    );

    // IGD+ to the optimal front
    let optimal = problem.optimal_front(500).expect("known");
    let distance = igd_plus(&values, &optimal, &[Minimize; 2]);
    println!("IGD+ to the optimal front: {distance:.4}");

    // the hypervolume with the reference point (−20, 85); the whole front's, from 2,000,000 of its
    // points, is 16546.1
    let volume = hypervolume(&values, &[-20.0, 85.0], &[Minimize; 2]);
    println!("hypervolume {volume:.1} (the whole front: 16546.1)");
    trace.write();
    Ok(())
}
