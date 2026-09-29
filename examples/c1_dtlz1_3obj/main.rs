//! C1-DTLZ1 with 3 objectives: minimize three objectives over 7 variables in [0, 1], DTLZ1 with a
//! constraint that leaves feasible only a thin wedge next to its front.
//!
//! NSGA-III with the settings of Jain and Deb (2014), for 2,000 generations. Prints how many
//! solutions of the final front are feasible, how many of the points where the 91 reference
//! directions meet the front they reach, and their IGD+ to those points and hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on
//! the example's page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example c1_dtlz1_3obj
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{C1Dtlz1, MultiProblem};
use genoxide::prelude::*;

// how close a solution must come to a target point, in scaled objectives, to reach it
const REACH: f64 = 0.02;

fn main() -> Result<()> {
    let problem = C1Dtlz1::<3>::default();
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = MultiEngine::new(nsga3(&problem)?, problem)
        .stop_when(Stop::generations(2000))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;
    let (front, size) = feasible(outcome.front());
    report("NSGA-III", 2000, &front, size);
    trace.write();
    Ok(())
}

// NSGA-III with the settings of Jain and Deb (2014): the 91 reference directions of Das and
// Dennis's method with 12 divisions, a population of 92, SBX with η = 30 at a rate of 1, and
// polynomial mutation with η = 20 at a rate of 1/n per gene for n genes
fn nsga3(problem: &C1Dtlz1<3>) -> Result<impl MultiObjectiveAlgorithm<3, Genome = Reals>> {
    let variables = problem.variables() as f64;
    Nsga3::builder(
        problem.representation(),
        [Minimize; 3],
        multi::das_dennis::<3>(12),
    )
    .population_size(92)
    .crossover(SimulatedBinaryCrossover::new(30.0)?)
    .crossover_rate(1.0)
    .mutate(PolynomialMutation::per_gene(1.0 / variables, 20.0)?)
    .seed(1)
    .build()
}

// where the reference direction w meets the optimal front, if it's feasible there: on
// the plane where the objectives sum to 1/2, all feasible
fn target(w: [f64; 3]) -> Option<[f64; 3]> {
    Some(w.map(|v| v / 2.0))
}

// the objectives scaled to [0, 1] on the optimal front, by its nadir point (its ideal point is
// the origin)
pub fn scaled(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let nadir = C1Dtlz1::<3>::default().nadir_point().expect("known");
    points
        .iter()
        .map(|p| [0, 1, 2].map(|j| p[j] / nadir[j]))
        .collect()
}

// the target points: where the 91 reference directions meet the optimal front, where feasible
pub fn targets() -> Vec<[f64; 3]> {
    let directions = multi::das_dennis::<3>(12).into_iter();
    scaled(&directions.filter_map(target).collect::<Vec<_>>())
}

// the feasible solutions of a run's front: how many of them, how many target points they reach,
// their IGD+ to the target points and their hypervolume, as a share of the target points'
fn report(name: &str, generations: u64, front: &[[f64; 3]], size: usize) {
    let feasible = if front.len() == size {
        "all feasible".to_string()
    } else {
        format!("{} feasible", front.len())
    };
    println!("{name}, {generations} generations: {size} solutions on the front, {feasible}");
    let found = scaled(front);
    let targets = targets();
    let near = |t: &[f64; 3]| {
        let distance = |f: &[f64; 3]| (0..3).map(|j| (f[j] - t[j]).powi(2)).sum::<f64>().sqrt();
        found.iter().any(|f| distance(f) <= REACH)
    };
    let reached = targets.iter().filter(|t| near(t)).count();
    println!("  target points reached: {reached} of {}", targets.len());
    let distance = igd_plus(&found, &targets, &[Minimize; 3]);
    let volume = hypervolume(&found, &[1.1; 3], &[Minimize; 3]);
    let theirs = hypervolume(&targets, &[1.1; 3], &[Minimize; 3]);
    println!(
        "  IGD+ {distance:.5}, hypervolume {volume:.4}, {:.2}% of the target points' {theirs:.4}",
        100.0 * volume / theirs
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
