//! Two-bar truss: minimize the volume of a truss of two bars and the stress in them, whose front
//! has two pieces.
//!
//! NSGA-II with a population of 100, simulated binary crossover and polynomial mutation at a rate
//! of 1/3 per gene, for 250 generations. Prints how many solutions of the final front are feasible,
//! the range of each objective on it, their IGD+ to the optimal front and their hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example two_bar_truss
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::MultiProblem;
use genoxide::multi::problems::engineering::TwoBarTruss;
use genoxide::prelude::*;

// the run's length
const GENERATIONS: u64 = 250;

fn main() -> Result<()> {
    let problem = TwoBarTruss;
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0)?)
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = MultiEngine::new(nsga2, problem)
        .stop_when(Stop::generations(GENERATIONS))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;
    let (front, size) = feasible(outcome.front());
    report(&front, size);
    trace.write();
    Ok(())
}

// the objectives scaled to [0, 1] on the front, by its ideal and nadir points
pub fn scaled(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let (ideal, nadir) = (
        TwoBarTruss.ideal_point().expect("known"),
        TwoBarTruss.nadir_point().expect("known"),
    );
    let scale = |p: &[f64; 2]| [0, 1].map(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j]));
    points.iter().map(scale).collect()
}

// the feasible solutions of the run's front: how many of them, the range of each objective,
// their IGD+ to the optimal front and their hypervolume, as a share of the whole front's
fn report(front: &[[f64; 2]], size: usize) {
    let feasible = if front.len() == size {
        "all feasible".to_string()
    } else {
        format!("{} feasible", front.len())
    };
    println!("NSGA-II, {GENERATIONS} generations: {size} solutions on the front, {feasible}");
    let low = |j: usize| front.iter().map(|p| p[j]).fold(f64::INFINITY, f64::min);
    let high = |j: usize| front.iter().map(|p| p[j]).fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  volume from {:.6} to {:.6}, stress from {:.1} to {:.1}",
        low(0),
        high(0),
        low(1),
        high(1)
    );
    let found = scaled(front);
    let volume = hypervolume(&found, &[1.1, 1.1], &[Minimize; 2]);
    let optimal = TwoBarTruss.optimal_front(2000).expect("known");
    let distance = igd_plus(&found, &scaled(&optimal), &[Minimize; 2]);
    let whole = whole_front_hypervolume();
    println!(
        "  IGD+ {distance:.5}, hypervolume {volume:.4}, {:.2}% of the whole front's {whole:.4}",
        100.0 * volume / whole
    );
}

// the hypervolume of the whole optimal front, from 100,000 of its points, in scaled objectives with
// the reference point (1.1, 1.1)
pub fn whole_front_hypervolume() -> f64 {
    let front = scaled(&TwoBarTruss.optimal_front(100_000).expect("known"));
    hypervolume(&front, &[1.1, 1.1], &[Minimize; 2])
}

// the objective values of the feasible solutions of a front, and the front's size
fn feasible<G: Genome>(front: &[Individual<G, multi::Scores<2>>]) -> (Vec<[f64; 2]>, usize) {
    let values = front
        .iter()
        .filter_map(|x| {
            x.fitness()
                .filter(|s| s.is_feasible())
                .and_then(|s| s.values())
        })
        .collect();
    (values, front.len())
}
