//! Conceptual marine design: the Panamax bulk carrier that carries cargo the cheapest, with the
//! lightest ship and the most cargo a year, subject to nine constraints.
//!
//! SMS-EMOA with a population of 92, for 500 generations. Prints how many solutions of the final
//! front are feasible, the range of each objective on it, and its hypervolume, as a share of that
//! of genoxide's reference front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example marine_design
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::MultiProblem;
use genoxide::multi::problems::engineering::MarineDesign;
use genoxide::prelude::*;

// the run's length
const GENERATIONS: u64 = 500;

// the front's nadir point, estimated from genoxide's reference front (see the README): with the
// ideal point, what scales the objectives to [0, 1]
pub const NADIR: [f64; 3] = [11.0662, 12435.8, -386_500.0];

// the hypervolume of genoxide's reference front, in scaled objectives with the reference point
// (1.1, 1.1, 1.1): a lower bound on the whole front's
pub const REFERENCE: f64 = 0.8624;

fn main() -> Result<()> {
    let problem = MarineDesign;
    let sms_emoa = SmsEmoa::builder(problem.representation(), [Minimize; 3])
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 6.0, 20.0)?)
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = MultiEngine::new(sms_emoa, problem)
        .stop_when(Stop::generations(GENERATIONS))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;
    let (front, size) = feasible(outcome.front());
    report(&front, size);
    trace.write();
    Ok(())
}

// the objectives scaled to [0, 1] by the front's ideal point and its estimated nadir point
pub fn scaled(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let ideal = MarineDesign.ideal_point().expect("known");
    let scale = |p: &[f64; 3]| [0, 1, 2].map(|j| (p[j] - ideal[j]) / (NADIR[j] - ideal[j]));
    points.iter().map(scale).collect()
}

// the feasible solutions of the run's front: how many of them, the range of each objective, and
// their hypervolume, as a share of the reference front's
fn report(front: &[[f64; 3]], size: usize) {
    let feasible = if front.len() == size {
        "all feasible".to_string()
    } else {
        format!("{} feasible", front.len())
    };
    println!("SMS-EMOA, {GENERATIONS} generations: {size} solutions on the front, {feasible}");
    let low = |j: usize| front.iter().map(|p| p[j]).fold(f64::INFINITY, f64::min);
    let high = |j: usize| front.iter().map(|p| p[j]).fold(f64::NEG_INFINITY, f64::max);
    // the third objective is the annual cargo, negated
    println!(
        "  transportation cost from {:.4} to {:.4} £/t, light ship weight from {:.1} to {:.1} t, \
         annual cargo from {:.0} to {:.0} t",
        low(0),
        high(0),
        low(1),
        high(1),
        -high(2),
        -low(2)
    );
    let volume = hypervolume(&scaled(front), &[1.1; 3], &[Minimize; 3]);
    println!(
        "  hypervolume {volume:.4}, {:.2}% of the reference front's {REFERENCE}",
        100.0 * volume / REFERENCE
    );
}

// the objective values of the feasible solutions of a front, and the front's size
fn feasible<G: Genome>(front: &[Individual<G, multi::Scores<3>>]) -> (Vec<[f64; 3]>, usize) {
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
