//! CONSTR: minimize two objectives subject to two constraints, with NSGA-II and Deb's rules.
//!
//! Deb's problem, from genoxide's `multi::problems::Constr`, whose fitness is the two objectives
//! and the constraint violation. Prints how many solutions of the final front are feasible, how
//! many lie along each of the two pieces of the optimal front, their IGD+ to 500 points of the
//! optimal front, and the front's hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example constr
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{Constr, MultiProblem};
use genoxide::prelude::*;

fn main() -> Result<()> {
    let problem = Constr;
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

    // the two pieces of the optimal front meet at f₁ = x₁ = 2/3
    let values = outcome.front_values();
    let boundary = values.iter().filter(|[f1, _]| *f1 < 2.0 / 3.0).count();
    println!(
        "along the boundary x2 = 6 - 9 x1: {boundary}, along x2 = 0: {}",
        values.len() - boundary
    );

    // IGD+ to the optimal front
    let optimal = problem.optimal_front(500).expect("known");
    let distance = igd_plus(&values, &optimal, &[Minimize; 2]);
    println!("IGD+ to the optimal front: {distance:.6}");

    // the hypervolume with the reference point (1.1, 10); the whole front's is
    // 19 · 5/18 − 7 ln(12/7) + 10/3 − ln(3/2) + 0.9 = 5.3327
    let volume = hypervolume(&values, &[1.1, 10.0], &[Minimize; 2]);
    println!("hypervolume {volume:.4} (the whole front: 5.3327)");
    trace.write();
    Ok(())
}
