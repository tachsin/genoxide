//! SRN: minimize two objectives subject to two constraints, with NSGA-II and Deb's rules.
//!
//! Srinivas and Deb's problem, from genoxide's `multi::problems::Srn`, whose fitness is the two
//! objectives and the constraint violation. Prints how many solutions of the final front are
//! feasible, how many lie along each of the three pieces of the optimal front, their IGD+ to 500
//! points of the optimal front, and the front's hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example srn
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Srn};
use genoxide::prelude::*;

fn main() -> Result<()> {
    let problem = Srn;
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

    // the pieces of the optimal front meet at f₁ = 24.5, at x = (−2.5, 2.5), and at
    // f₁ = 212.42, at x = (−2.5, √218.75) on the circle
    let values = outcome.front_values();
    let between = |low: f64, high: f64| {
        let within = |point: &&[f64; 2]| low <= point[0] && point[0] < high;
        values.iter().filter(within).count()
    };
    println!(
        "along the line x1 = 3 x2 - 10: {}, along x1 = -2.5: {}, along the circle: {}",
        between(f64::MIN, 24.5),
        between(24.5, 212.42),
        between(212.42, f64::MAX),
    );

    // IGD+ to the optimal front, its three pieces
    let optimal = problem.optimal_front(500).expect("known");
    let distance = igd_plus(&values, &optimal, &[Minimize; 2]);
    println!("IGD+ to the optimal front: {distance:.4}");

    // the hypervolume with the reference point (245, 25); the whole front's, from 2,000,000 of its
    // points, is 35478.6
    let volume = hypervolume(&values, &[245.0, 25.0], &[Minimize; 2]);
    println!("hypervolume {volume:.1} (the whole front: 35478.6)");
    trace.write();
    Ok(())
}
