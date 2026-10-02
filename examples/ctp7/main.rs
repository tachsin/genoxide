//! CTP7: minimize two objectives over two variables subject to one constraint, with
//! a front of six disconnected pieces between infeasible bands.
//!
//! NSGA-II with the settings of the paper's experiments, a population of 100 for 500
//! generations. Prints how many solutions of the final front are feasible, how many pieces of
//! the optimal front they reach, their IGD+ to it and their hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on
//! the example's page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example ctp7
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{Ctp7, MultiProblem};
use genoxide::prelude::*;

// how close a solution must come to a piece of the optimal front, in scaled objectives, to reach it
const REACH: f64 = 0.02;

fn main() -> Result<()> {
    let problem = Ctp7;
    // the settings of the paper's experiments: a population of 100 for 500 generations, SBX
    // and polynomial mutation with η = 20, crossover at 0.9 and mutation at 1/n per gene
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(20.0)?)
        .mutate(PolynomialMutation::per_gene(0.5, 20.0)?)
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = MultiEngine::new(nsga2, problem)
        .stop_when(Stop::generations(500))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;
    let (front, size) = feasible(outcome.front());
    report("NSGA-II", 500, &front, size);
    trace.write();
    Ok(())
}

// the objectives scaled to [0, 1] on the optimal front, by its ideal and nadir points
fn scaled(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let (ideal, nadir) = (
        Ctp7.ideal_point().expect("known"),
        Ctp7.nadir_point().expect("known"),
    );
    let scale = |p: &[f64; 2]| [0, 1].map(|j| (p[j] - ideal[j]) / (nadir[j] - ideal[j]));
    points.iter().map(scale).collect()
}

// the pieces of a front sorted by f₁, split where neighbors are more than 0.01 apart
fn pieces(front: &[[f64; 2]]) -> Vec<Vec<[f64; 2]>> {
    let mut pieces: Vec<Vec<[f64; 2]>> = Vec::new();
    for (i, point) in front.iter().enumerate() {
        let gap = i == 0 || {
            let last = front[i - 1];
            ((point[0] - last[0]).powi(2) + (point[1] - last[1]).powi(2)).sqrt() > 0.01
        };
        if gap {
            pieces.push(Vec::new());
        }
        pieces.last_mut().expect("a piece").push(*point);
    }
    pieces
}

// the feasible solutions of a run's front: how many of them, how many pieces of the optimal
// front they reach, their IGD+ to it and their hypervolume, as a share of the whole front's
fn report(name: &str, generations: u64, front: &[[f64; 2]], size: usize) {
    let feasible = if front.len() == size {
        "all feasible".to_string()
    } else {
        format!("{} feasible", front.len())
    };
    println!("{name}, {generations} generations: {size} solutions on the front, {feasible}");
    let optimal = Ctp7.optimal_front(2000).expect("known");
    let found = scaled(front);
    let pieces = pieces(&optimal);
    let reached = pieces
        .iter()
        .filter(|piece| {
            scaled(piece).iter().any(|p| {
                found
                    .iter()
                    .any(|f| ((f[0] - p[0]).powi(2) + (f[1] - p[1]).powi(2)).sqrt() <= REACH)
            })
        })
        .count();
    println!(
        "  pieces of the optimal front reached: {reached} of {}",
        pieces.len()
    );
    let distance = igd_plus(&found, &scaled(&optimal), &[Minimize; 2]);
    let volume = hypervolume(&found, &[1.1, 1.1], &[Minimize; 2]);
    let whole = whole_front_hypervolume();
    println!(
        "  IGD+ {distance:.5}, hypervolume {volume:.4}, {:.2}% of the whole front's {whole:.4}",
        100.0 * volume / whole
    );
}

// the hypervolume of the whole optimal front, from 100,000 of its points, in scaled objectives with
// the reference point (1.1, 1.1)
pub fn whole_front_hypervolume() -> f64 {
    let front = scaled(&Ctp7.optimal_front(100_000).expect("known"));
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
