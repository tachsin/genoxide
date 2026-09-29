//! ZDT6: minimize two conflicting objectives over 10 variables in [0, 1], with a concave Pareto
//! front and a search space that crowds solutions at one end of it, with NSGA-II.
//!
//! Zitzler, Deb and Thiele's sixth problem, from genoxide's `multi::problems::Zdt6`. Prints how
//! unevenly x₁ maps to f₁, then the size and range of the final front, its IGD+ to 500 points of
//! the optimal front, and its hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example zdt6
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::MultiFitnessFunction;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::multi::problems::{MultiProblem, Zdt6};
use genoxide::prelude::*;

// the reference point of the hypervolume, beyond the front's worst point (1, 0.921)
const REFERENCE: [f64; 2] = [1.1, 1.1];

// the optimal front's f₁ runs from 0.2808 to 1; the middle of that range
const MIDDLE: f64 = 0.6404;

fn main() -> Result<()> {
    let problem = Zdt6::new(10);

    // x₁ alone sets f₁: at 1,001 evenly spaced values, with the other variables at 0, how many
    // give f₁ in the lower half of the front's range
    let lower = (0..=1000)
        .map(|i| {
            let mut x = vec![0.0; 10];
            x[0] = f64::from(i) / 1000.0;
            problem.evaluate(&Reals::from(x))[0]
        })
        .filter(|&f1| f1 < MIDDLE)
        .count();
    println!("{lower} of 1001 evenly spaced x1 give f1 below {MIDDLE}");

    // polynomial mutation at a rate of 1/10, one gene per child on average
    let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0)?)
        .mutate(PolynomialMutation::per_gene(1.0 / 10.0, 20.0)?)
        .seed(1)
        .build()?;

    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = MultiEngine::new(nsga2, problem)
        .stop_when(Stop::generations(500))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    let front = outcome.front_values();
    let below = front.iter().filter(|[f1, _]| *f1 < MIDDLE).count();
    let smallest = front
        .iter()
        .map(|[f1, _]| *f1)
        .fold(f64::INFINITY, f64::min);
    println!(
        "{} solutions on the front, {below} with f1 below {MIDDLE}, the smallest {smallest:.4}",
        front.len()
    );

    // IGD+ to the optimal front, f₂ = 1 − f₁², at 500 points evenly spaced in f₁
    let optimal = problem.optimal_front(500).expect("known");
    let distance = igd_plus(&front, &optimal, &[Minimize; 2]);
    println!("IGD+ to the optimal front: {distance:.4}");

    // the whole front's hypervolume, found from the curve's integral
    let volume = hypervolume(&front, &REFERENCE, &[Minimize; 2]);
    println!("hypervolume {volume:.4} (the whole front: 0.5079)");
    trace.write();
    Ok(())
}
