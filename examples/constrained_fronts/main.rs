//! Constrained two-objective fronts: NSGA-II with Deb's rules on SRN, TNK, OSY and CONSTR.
//!
//! Four constrained problems of genoxide's `multi::problems`, whose fitness is the two objectives
//! and the total constraint violation. Prints, for each, the size of the final front and how many
//! of its solutions are feasible, the front's hypervolume and its IGD+ to 500 points of the
//! optimal front, both with the objectives normalized by the problem's ideal and nadir points.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example constrained_fronts
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{Constr, DynMultiProblem, Osy, Srn, Tnk, boxed};
use genoxide::prelude::*;

fn main() -> Result<()> {
    let problems = [boxed(Srn), boxed(Tnk), boxed(Osy), boxed(Constr)];
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    println!("problem  front  feasible  hypervolume  (optimal)  IGD+");
    for problem in &problems {
        let problem = problem.as_ref();
        // the settings of the NSGA-II paper: a mutation rate of 1/n for n genes
        let genes = problem.real().bounds().len();
        let nsga2 = Nsga2::builder(problem.real(), [Minimize; 2])
            .population_size(100)
            .crossover(SimulatedBinaryCrossover::new(20.0)?)
            .mutate(PolynomialMutation::per_gene(1.0 / genes as f64, 20.0)?)
            .seed(1)
            .build()?;
        let outcome = MultiEngine::new(nsga2, |x: &Reals| problem.evaluate(x))
            .stop_when(Stop::generations(200))
            .on_generation(trace.panel(problem))
            .run()?;

        let front = outcome.front();
        let feasible: Vec<[f64; 2]> = front
            .iter()
            .filter_map(|individual| individual.fitness())
            .filter(|scores| scores.is_feasible())
            .filter_map(|scores| scores.values())
            .collect();
        // the hypervolume of the feasible front, beside that of 500 points of the optimal front,
        // and the IGD+ to them, all in normalized objectives
        let optimal = problem.optimal_front(500).expect("known");
        let volume = normalized_hypervolume(problem, &feasible);
        let best = normalized_hypervolume(problem, &optimal);
        let distance = igd_plus(
            &normalized(problem, &feasible),
            &normalized(problem, &optimal),
            &[Minimize; 2],
        );
        println!(
            "{:<7}  {:>5}  {:>8}  {volume:>11.4}  {best:>9.4}  {distance:.4}",
            problem.name(),
            front.len(),
            feasible.len(),
        );
    }
    trace.write(&problems);
    Ok(())
}

/// The objectives of `front` mapped to [0, 1] by the problem's ideal and nadir points.
pub fn normalized(problem: &dyn DynMultiProblem<2>, front: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let ideal = problem.ideal_point().expect("known");
    let nadir = problem.nadir_point().expect("known");
    let scaled = |point: &[f64; 2]| [0, 1].map(|i| (point[i] - ideal[i]) / (nadir[i] - ideal[i]));
    front.iter().map(scaled).collect()
}

/// The hypervolume of `front` in normalized objectives, up to the reference point (1.1, 1.1).
pub fn normalized_hypervolume(problem: &dyn DynMultiProblem<2>, front: &[[f64; 2]]) -> f64 {
    hypervolume(&normalized(problem, front), &[1.1, 1.1], &[Minimize; 2])
}
