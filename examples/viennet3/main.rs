//! Viennet 3 (VNT3): minimize three objectives of two variables, two of which depend only on the
//! distance from the origin, so that the Pareto front is two curves.
//!
//! NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, a
//! population of 92 and 50 generations. Prints the size of the final front, how many of its
//! solutions are on each of the two curves, and its hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example viennet3
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{MultiProblem, Viennet3};
use genoxide::prelude::*;

// the reference point of the hypervolume: the nadir point (8.1964, 17.0370, 0.1760) plus
// a tenth of each objective's range on the front, from the ideal point (0, 15, −0.1),
// rounded to 4 decimals: (9.0160, 17.2407, 0.2036)
fn reference() -> [f64; 3] {
    let (ideal, nadir) = (Viennet3.ideal_point(), Viennet3.nadir_point());
    let (ideal, nadir) = (ideal.expect("known"), nadir.expect("known"));
    std::array::from_fn(|j| ((nadir[j] + (nadir[j] - ideal[j]) / 10.0) * 1e4).round() / 1e4)
}

fn main() -> Result<()> {
    let problem = Viennet3;
    let directions = multi::das_dennis::<3>(12);
    let nsga3 = Nsga3::builder(problem.representation(), [Minimize; 3], directions)
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(30.0)?)
        .mutate(PolynomialMutation::per_gene(0.5, 20.0)?)
        .seed(1)
        .build()?;

    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = MultiEngine::new(nsga3, problem)
        .stop_when(Stop::generations(50))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    // the optimal solutions are on two curves: one where x₁² + x₂² ≤ 1.5, near the origin,
    // and one where x₁² + x₂² ≥ 4.19
    let inner = outcome
        .front()
        .iter()
        .filter(|x| x.genome().iter().map(|xi| xi * xi).sum::<f64>() < 3.0)
        .count();
    let front = outcome.front_values();
    println!(
        "{} solutions on the front: {inner} near the origin, {} farther out",
        front.len(),
        front.len() - inner
    );
    // the hypervolume of the front; the whole front's is about 5.3255
    let volume = hypervolume(&front, &reference(), &[Minimize; 3]);
    println!("hypervolume {volume:.4} (the whole front: 5.3255)");
    trace.write();
    Ok(())
}
