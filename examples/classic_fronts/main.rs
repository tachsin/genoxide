//! Classic fronts: NSGA-II on four classic two-objective problems, Schaffer's first and second,
//! Fonseca and Fleming's, and Poloni's.
//!
//! The problems come from genoxide's `multi::problems`. NSGA-II has the settings of the NSGA-II
//! paper, and the same 250 generations on each. Prints, per problem, the size of the final front,
//! its hypervolume with the objectives normalized by the problem's ideal and nadir points, and,
//! where the optimal front is known, its IGD+ to 500 points of it.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example classic_fronts
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::MultiFitnessFunction;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{
    self, DynMultiProblem, FonsecaFleming, Poloni, Schaffer1, Schaffer2,
};
use genoxide::prelude::*;

const GENERATIONS: u64 = 250;

fn main() -> Result<()> {
    let problems: [(&str, Box<dyn DynMultiProblem<2>>); 4] = [
        ("Schaffer 1", problems::boxed(Schaffer1)),
        ("Schaffer 2", problems::boxed(Schaffer2)),
        ("Fonseca-Fleming", problems::boxed(FonsecaFleming::new(3))),
        ("Poloni", problems::boxed(Poloni)),
    ];

    println!(
        "{:<16}{:>6}{:>13}{:>9}",
        "problem", "front", "hypervolume", "IGD+"
    );
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    for (name, problem) in &problems {
        // polynomial mutation at a rate of 1/n, one gene per child on average
        let genes = problem.real().bounds().len();
        let nsga2 = Nsga2::builder(problem.real(), [Minimize; 2])
            .population_size(100)
            .crossover(SimulatedBinaryCrossover::new(15.0)?)
            .mutate(PolynomialMutation::per_gene(1.0 / genes as f64, 20.0)?)
            .seed(1)
            .build()?;
        let (ideal, nadir) = match (problem.ideal_point(), problem.nadir_point()) {
            (Some(ideal), Some(nadir)) => (ideal, nadir),
            _ => poloni_extremes(),
        };
        let volume = move |front: &[[f64; 2]]| normalized_hypervolume(front, ideal, nadir);
        let optimal = problem.optimal_front(100);
        let outcome = MultiEngine::new(nsga2, |x: &Reals| problem.evaluate(x))
            .stop_when(Stop::generations(GENERATIONS))
            .on_generation(trace.fronts(name, optimal, volume))
            .run()?;

        let front = outcome.front_values();
        // IGD+ to 500 points of the optimal front, where it's known
        let distance = match problem.optimal_front(500) {
            Some(optimal) => format!("{:.4}", igd_plus(&front, &optimal, &[Minimize; 2])),
            None => "-".to_string(),
        };
        println!(
            "{name:<16}{:>6}{:>13.4}{distance:>9}",
            front.len(),
            volume(&front)
        );
    }
    trace.write();
    Ok(())
}

// the hypervolume of `front` with each objective mapped to [0, 1] by the ideal and nadir points,
// and the reference point (1.1, 1.1)
fn normalized_hypervolume(front: &[[f64; 2]], ideal: [f64; 2], nadir: [f64; 2]) -> f64 {
    let scaled: Vec<[f64; 2]> = front
        .iter()
        .map(|point| std::array::from_fn(|i| (point[i] - ideal[i]) / (nadir[i] - ideal[i])))
        .collect();
    hypervolume(&scaled, &[1.1, 1.1], &[Minimize; 2])
}

// Poloni's ideal and nadir points, which genoxide doesn't give: the ends of its front are the
// minimum of f₁, 1 at (1, 2), and the minimum of f₂, 0 at (−3, −1)
fn poloni_extremes() -> ([f64; 2], [f64; 2]) {
    let at = |x1: f64, x2: f64| Poloni.evaluate(&Reals::from(vec![x1, x2]));
    let (least_f1, least_f2) = (at(1.0, 2.0), at(-3.0, -1.0));
    ([least_f1[0], least_f2[1]], [least_f2[0], least_f1[1]])
}
