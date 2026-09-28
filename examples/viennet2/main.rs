//! Viennet 2 (VNT2): minimize three convex quadratic objectives of two variables, whose Pareto
//! front is a curved triangle.
//!
//! NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, a
//! population of 92 and 50 generations. Prints the size of the final front and its hypervolume.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example viennet2
//! ```

mod trace;

use genoxide::Objective::Minimize;
use genoxide::multi::indicator::hypervolume;
use genoxide::multi::problems::{MultiProblem, Viennet2};
use genoxide::prelude::*;

// the reference point of the hypervolume: the nadir point (4.2452, −16.4766, −12.0531) plus
// a tenth of each objective's range on the front, from the ideal point (3, −17, −13),
// rounded to 4 decimals: (4.3697, −16.4242, −11.9584)
fn reference() -> [f64; 3] {
    let (ideal, nadir) = (Viennet2.ideal_point(), Viennet2.nadir_point());
    let (ideal, nadir) = (ideal.expect("known"), nadir.expect("known"));
    std::array::from_fn(|j| ((nadir[j] + (nadir[j] - ideal[j]) / 10.0) * 1e4).round() / 1e4)
}

fn main() -> Result<()> {
    let problem = Viennet2;
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

    // the hypervolume of the front; the whole front's is about 0.7744
    let front = outcome.front_values();
    let volume = hypervolume(&front, &reference(), &[Minimize; 3]);
    println!(
        "{} solutions on the front, hypervolume {volume:.4} (the whole front: 0.7744)",
        front.len()
    );
    trace.write();
    Ok(())
}
