//! Golinski's speed reducer (1970, 1973): the lightest gearbox whose gear teeth and shafts stay
//! within their stress and deflection limits. A constrained mixed discrete-continuous problem,
//! whose best known weight is 2996.348165.
//!
//! The variables are the face width of the gears, the module of their teeth, the number of teeth
//! on the pinion, an integer, and the lengths and diameters of the two shafts. genoxide's
//! `SpeedReducer` rounds the number of teeth when it evaluates a genome, and its fitness is the
//! weight and the violation of the eleven constraints, which Deb's feasibility rules compare.
//! SHADE, a differential evolution, searches the genes, and the example prints the best design
//! and the constraints at their limits.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example speed_reducer
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::engineering::SpeedReducer;

// a constraint within this of 0 is at its limit: active
const ACTIVE: f64 = 1e-6;

fn main() -> Result<()> {
    let problem = SpeedReducer;
    let best_known = problem.optimum().expect("known").value();
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
    let [b, m, z, l1, l2, d1, d2] = problem.design(outcome.best_genome());
    let constraints = problem.constraints(outcome.best_genome());
    let active: Vec<String> = (constraints.inequalities().iter().enumerate())
        .filter(|(_, g)| g.abs() <= ACTIVE)
        .map(|(i, _)| format!("g{}", i + 1))
        .collect();
    println!(
        "weight {:.6} (the best known: {best_known:.6})",
        best.score().unwrap_or(f64::NAN)
    );
    println!("violation {:.6}", best.violation());
    println!("face width {b:.6}, module {m:.6}, teeth {z}");
    println!("shaft 1: length {l1:.6}, diameter {d1:.6}");
    println!("shaft 2: length {l2:.6}, diameter {d2:.6}");
    println!("active constraints: {}", active.join(", "));
    trace.write();
    Ok(())
}
