//! Disc brake: minimize the mass of a multiple-disc brake and its stopping time, subject to five
//! constraints.
//!
//! NSGA-II with a population of 100, simulated binary crossover and polynomial mutation at a rate
//! of 1/4 per gene, for 250 generations. Prints how many solutions of the final front are feasible,
//! the range of each objective on it, and their hypervolume, as a share of that of genoxide's
//! reference front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example disc_brake
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::MultiProblem;
use genoxide::multi::problems::engineering::DiscBrake;
use genoxide::prelude::*;

// the run's length
const GENERATIONS: u64 = 250;

// the hypervolume of genoxide's reference front, in scaled objectives with the reference point
// (1.1, 1.1): the non-dominated solutions of ε-constraint runs of SHADE (see the README), a lower
// bound on the whole front's
pub const REFERENCE: f64 = 1.0853;

fn main() -> Result<()> {
    let problem = DiscBrake;
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(0.25, 20.0)?)
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
        DiscBrake.ideal_point().expect("known"),
        DiscBrake.nadir_point().expect("known"),
    );
    let scale = |p: &[f64; 2]| [0, 1].map(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j]));
    points.iter().map(scale).collect()
}

// the feasible solutions of the run's front: how many of them, the range of each objective,
// and their hypervolume, as a share of the reference front's
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
        "  mass from {:.4} to {:.4}, stopping time from {:.4} to {:.4}",
        low(0),
        high(0),
        low(1),
        high(1)
    );
    let found = scaled(front);
    let volume = hypervolume(&found, &[1.1, 1.1], &[Minimize; 2]);
    println!(
        "  hypervolume {volume:.4}, {:.2}% of the reference front's {REFERENCE}",
        100.0 * volume / REFERENCE
    );
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
