//! MW10: minimize two objectives over 15 variables subject to three constraints, with NSGA-II;
//! the optimal front is two pieces on feasible islands.
//!
//! Ma and Wang's MW10, from genoxide's `multi::problems::Mw10`, whose fitness is the two
//! objectives and the constraint violation. Runs NSGA-II twice: with the paper's settings
//! (polynomial mutation with η = 20, 600 generations), and with η = 2 for 5,000 generations.
//! Prints each final front's size, how many of its solutions are feasible, and its IGD+ to 500
//! points of the optimal front and hypervolume, with the objectives normalized by the front's
//! ideal and nadir points; then the whole front's hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example mw10
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Mw10};
use genoxide::prelude::*;

// the reference point of the hypervolume, with the objectives normalized by the front's ideal and
// nadir points: 1.1 times the nadir point
const REFERENCE: [f64; 2] = [1.1, 1.1];

fn main() -> Result<()> {
    let problem = Mw10::default();
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // the paper's settings: polynomial mutation with η = 20, 600 generations
    run(problem, "η = 20", 20.0, 600, &mut trace)?;
    // polynomial mutation with η = 2, whose steps are larger, for 5,000 generations
    run(problem, "η = 2", 2.0, 5_000, &mut trace)?;
    // the hypervolume of the whole front, from 20,000 of its points
    let whole = normalized(&problem, &problem.optimal_front(20_000).expect("known"));
    let volume = hypervolume(&whole, &REFERENCE, &[Minimize; 2]);
    println!("the whole front: hypervolume {volume:.4}");
    trace.write(&problem);
    Ok(())
}

// runs NSGA-II with a population of 100, simulated binary crossover with η = 20 at genoxide's
// default rate of 0.9, and polynomial mutation with the distribution index `eta` at a rate of 1/n
// per gene, for `generations`; prints its final front's size, how many of it are feasible, and its
// IGD+ to 500 points of the optimal front and hypervolume, with normalized objectives
fn run(
    problem: Mw10,
    name: &'static str,
    eta: f64,
    generations: u64,
    trace: &mut trace::Trace,
) -> Result<()> {
    let rate = 1.0 / problem.variables() as f64;
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(rate, eta)?)
        .seed(1)
        .build()?;
    let mut record = trace.fronts(name);
    let outcome = MultiEngine::new(nsga2, problem)
        .stop_when(Stop::generations(generations))
        .on_generation(|snapshot| record(snapshot))
        .run()?;
    let front = outcome.front();
    let scores = front.iter().filter_map(|x| x.fitness());
    let feasible: Vec<[f64; 2]> = scores
        .clone()
        .filter(|scores| scores.is_feasible())
        .filter_map(|scores| scores.values())
        .collect();
    let noun = if front.len() == 1 {
        "solution"
    } else {
        "solutions"
    };
    print!(
        "NSGA-II, {name}, {generations} generations: {} {noun}, ",
        front.len()
    );
    if feasible.is_empty() {
        let least = scores
            .map(|scores| scores.violation())
            .fold(f64::INFINITY, f64::min);
        println!("none feasible, the least violation {least:.4}");
        return Ok(());
    }
    let found = normalized(&problem, &feasible);
    let optimal = normalized(&problem, &problem.optimal_front(500).expect("known"));
    let distance = igd_plus(&found, &optimal, &[Minimize; 2]);
    let volume = hypervolume(&found, &REFERENCE, &[Minimize; 2]);
    println!(
        "{} feasible, IGD+ {distance:.4}, hypervolume {volume:.4}",
        feasible.len()
    );
    Ok(())
}

// the objectives normalized by the front's ideal and nadir points: the front spans [0, 1] in each
fn normalized(problem: &Mw10, points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let ideal = problem.ideal_point().expect("known");
    let nadir = problem.nadir_point().expect("known");
    let scale = |p: &[f64; 2]| std::array::from_fn(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j]));
    points.iter().map(scale).collect()
}
