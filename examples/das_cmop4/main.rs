//! DAS-CMOP4: minimize two objectives over 30 variables subject to 11 constraints, whose front is pieces of a concave curve, with
//! NSGA-II.
//!
//! Fan et al.'s DAS-CMOP4, from genoxide's `multi::problems::DasCmop4`, with the difficulty triplet of
//! the paper's figure 6, (0.5, 0.5, 0.5). Runs NSGA-II twice with the paper's settings, a population
//! of 300 for 1,000 generations (300,000 evaluations): with polynomial mutation with η = 20, as
//! the paper's, and with η = 5. Prints each final front's size, how many of its solutions are
//! feasible, and its IGD+ to 500 points of the optimal front and hypervolume, with the objectives
//! normalized by the front's ideal and nadir points; then the whole front's hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example das_cmop4
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::MultiFitnessFunction;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{DasCmop4, MultiProblem};
use genoxide::prelude::*;

// the reference point of the hypervolume, with the objectives normalized by the front's ideal and
// nadir points: 1.1 times the nadir point
const REFERENCE: [f64; 2] = [1.1, 1.1];

// the paper's population and evaluations: 300 for 1,000 generations
const POPULATION: usize = 300;
const GENERATIONS: u64 = 1_000;

fn main() -> Result<()> {
    let problem = DasCmop4::default();
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // the paper's settings: polynomial mutation with η = 20
    run(&problem, "η = 20", 20.0, &mut trace)?;
    // polynomial mutation with η = 5, whose steps are larger
    run(&problem, "η = 5", 5.0, &mut trace)?;
    // the hypervolume of the whole front, from 20,000 of its points
    let whole = normalized(&problem, &problem.optimal_front(20_000).expect("known"));
    let volume = hypervolume(&whole, &REFERENCE, &[Minimize; 2]);
    println!("the whole front: hypervolume {volume:.4}");
    trace.write(&problem);
    Ok(())
}

// runs NSGA-II with the paper's population, simulated binary crossover with η = 20 at genoxide's
// default rate of 0.9, and polynomial mutation with the distribution index `eta` at a rate of 1/n
// per gene; prints its final front's size, how many of it are feasible, and its IGD+ to 500
// points of the optimal front and hypervolume, with normalized objectives
fn run(problem: &DasCmop4, name: &'static str, eta: f64, trace: &mut trace::Trace) -> Result<()> {
    let rate = 1.0 / problem.variables() as f64;
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(POPULATION)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(rate, eta)?)
        .seed(1)
        .build()?;
    let mut record = trace.fronts(name);
    let outcome = MultiEngine::new(nsga2, *problem)
        .stop_when(Stop::generations(GENERATIONS))
        .on_generation(|snapshot| record(snapshot))
        .run()?;
    let scores: Vec<([f64; 2], f64)> = outcome
        .front()
        .iter()
        .map(|x| problem.evaluate(x.genome()))
        .collect();
    let feasible: Vec<[f64; 2]> = scores.iter().filter(|s| s.1 == 0.0).map(|s| s.0).collect();
    let noun = if scores.len() == 1 {
        "solution"
    } else {
        "solutions"
    };
    print!(
        "NSGA-II, {name}, {GENERATIONS} generations: {} {noun}, ",
        scores.len()
    );
    if feasible.is_empty() {
        let least = scores.iter().map(|s| s.1).fold(f64::INFINITY, f64::min);
        println!("none feasible, the least violation {least:.4}");
        return Ok(());
    }
    let found = normalized(problem, &feasible);
    let optimal = normalized(problem, &problem.optimal_front(500).expect("known"));
    let distance = igd_plus(&found, &optimal, &[Minimize; 2]);
    let volume = hypervolume(&found, &REFERENCE, &[Minimize; 2]);
    println!(
        "{} feasible, IGD+ {distance:.4}, hypervolume {volume:.4}",
        feasible.len()
    );
    Ok(())
}

// the objectives normalized by the front's ideal and nadir points: the front spans [0, 1] in each
pub fn normalized(problem: &DasCmop4, points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let ideal = problem.ideal_point().expect("known");
    let nadir = problem.nadir_point().expect("known");
    let scale = |p: &[f64; 2]| std::array::from_fn(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j]));
    points.iter().map(scale).collect()
}
