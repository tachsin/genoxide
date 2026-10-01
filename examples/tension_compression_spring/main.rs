//! The tension/compression spring (Belegundu, 1982; Arora, 1989): the lightest coil spring whose
//! deflection, shear stress, surge frequency and outer diameter stay within their limits. A
//! constrained continuous problem, whose best known weight is 0.01266523.
//!
//! The variables are the wire diameter d, the mean coil diameter D and the number of active coils
//! N. genoxide's `TensionCompressionSpring` gives the weight and the violation of the four
//! constraints, which Deb's feasibility rules compare. SHADE, a differential evolution, searches
//! the genes, and the example prints the best design and the constraints at their limits.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example tension_compression_spring
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::engineering::TensionCompressionSpring;

// a constraint within this of 0 is at its limit: active
const ACTIVE: f64 = 1e-6;

fn main() -> Result<()> {
    let problem = TensionCompressionSpring;
    let best_known = problem.optimum().expect("known").value();
    let de = De::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = Engine::new(de, problem)
        .stop_when(Stop::target(best_known * (1.0 + 1e-10)).or(Stop::evaluations(100_000)))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    let best = outcome.best_fitness();
    let x = outcome.best_genome();
    let constraints = problem.constraints(x);
    let active: Vec<String> = (constraints.inequalities().iter().enumerate())
        .filter(|(_, g)| g.abs() <= ACTIVE)
        .map(|(i, _)| format!("g{}", i + 1))
        .collect();
    println!(
        "weight {:.7} after {} evaluations (the best known: {best_known})",
        best.score().unwrap_or(f64::NAN),
        outcome.evaluations()
    );
    println!("violation {:.6}", best.violation());
    println!("d {:.6}, D {:.6}, N {:.6}", x[0], x[1], x[2]);
    println!("active constraints: {}", active.join(", "));
    trace.write();
    Ok(())
}
