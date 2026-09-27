//! Pressure vessel design (Sandgren, 1990): the cheapest cylindrical vessel with hemispherical
//! heads that holds 1,296,000 cubic inches, a constrained mixed discrete-continuous problem. The
//! minimum cost is 6059.714335.
//!
//! The variables are the thickness of the shell and of the heads, multiples of 0.0625 inch, and
//! the inner radius and the length of the shell. genoxide's `PressureVessel` rounds the first two
//! genes to whole plates, and its fitness is the cost and the violation of the four constraints,
//! which Deb's feasibility rules compare. SHADE, a differential evolution, searches the genes.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example pressure_vessel
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::engineering::PressureVessel;

fn main() -> Result<()> {
    let problem = PressureVessel;
    let minimum = problem.optimum().expect("known").value();
    let de = De::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = Engine::new(de, problem)
        .stop_when(Stop::evaluations(50_000))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    let best = outcome.best_fitness();
    let [shell, head, radius, length] = problem.design(outcome.best_genome());
    println!(
        "cost {:.6} after {} evaluations (the minimum: {minimum:.6})",
        best.score().unwrap_or(f64::NAN),
        outcome.evaluations()
    );
    println!("violation {:.6}", best.violation());
    println!("shell {shell:.4}, heads {head:.4}, radius {radius:.6}, length {length:.6}");
    trace.write();
    Ok(())
}
