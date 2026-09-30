//! Water resource planning: minimize five costs and losses of a storm drainage system, subject to
//! seven constraints.
//!
//! SPEA2 with a population of 212, simulated binary crossover and polynomial mutation at a rate of
//! 1/3 per gene, for 500 generations. Prints how many solutions of the final front are feasible,
//! how far they are from the optimal solutions in x₃, and their IGD+ to the optimal front.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example water_resource_planning
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::igd_plus;
use genoxide::multi::problems::MultiProblem;
use genoxide::multi::problems::engineering::WaterResourcePlanning;
use genoxide::prelude::*;

// the run's length
const GENERATIONS: u64 = 500;

fn main() -> Result<()> {
    let problem = WaterResourcePlanning;
    let spea2 = Spea2::builder(problem.representation(), [Minimize; 5])
        .population_size(212)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 3.0, 20.0)?)
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = MultiEngine::new(spea2, problem)
        .stop_when(Stop::generations(GENERATIONS))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;
    let front = outcome.front();
    let size = front.len();
    let feasible: Vec<_> = front
        .iter()
        .filter(|x| x.fitness().is_some_and(|s| s.is_feasible()))
        .collect();
    let feasible_count = if feasible.len() == size {
        "all feasible".to_string()
    } else {
        format!("{} feasible", feasible.len())
    };
    println!("SPEA2, {GENERATIONS} generations: {size} solutions on the front, {feasible_count}");
    // the optimal solutions have x₃ = 0.01, its lower bound
    let x3 = feasible.iter().map(|x| x.genome()[2]);
    let highest = x3.fold(f64::NEG_INFINITY, f64::max);
    println!("  x₃ at most {highest:.5}, where the optimal solutions have 0.01");
    let values: Vec<[f64; 5]> = feasible
        .iter()
        .filter_map(|x| x.fitness().and_then(|s| s.values()))
        .collect();
    let distance = igd_plus(&scaled(&values), &optimal_front(), &[Minimize; 5]);
    println!(
        "  IGD+ {distance:.5} to {} points of the optimal front",
        optimal_front().len()
    );
    trace.write();
    Ok(())
}

// the objectives scaled to [0, 1] on the front, by its ideal and nadir points
pub fn scaled(points: &[[f64; 5]]) -> Vec<[f64; 5]> {
    let (ideal, nadir) = (
        WaterResourcePlanning.ideal_point().expect("known"),
        WaterResourcePlanning.nadir_point().expect("known"),
    );
    let scale = |p: &[f64; 5]| std::array::from_fn(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j]));
    points.iter().map(scale).collect()
}

// at least 5,000 points of the optimal front, scaled
pub fn optimal_front() -> Vec<[f64; 5]> {
    scaled(&WaterResourcePlanning.optimal_front(5_000).expect("known"))
}
