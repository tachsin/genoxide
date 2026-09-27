//! The three-bar truss (Nowacki, 1974): the least volume of a planar truss of three bars whose
//! stresses stay within the allowed stress under a load. A constrained continuous problem, whose
//! minimum volume, 100 (√2 + √6/2) ≈ 263.895843, is known in closed form.
//!
//! The variables are the cross-sections of the two outer bars and of the middle one. genoxide's
//! `ThreeBarTruss` gives the volume and the violation of the three stress constraints, which Deb's
//! feasibility rules compare. SHADE, a differential evolution, searches the genes until it's
//! within 1e-10 of the minimum, relative to it, and the example prints the best design and the
//! constraints at their limits.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example three_bar_truss
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::engineering::ThreeBarTruss;

// a constraint within this of 0 is at its limit: active
const ACTIVE: f64 = 1e-6;

fn main() -> Result<()> {
    let problem = ThreeBarTruss;
    let minimum = problem.optimum().expect("known").value();
    let de = De::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = Engine::new(de, problem)
        .stop_when(Stop::target(minimum * (1.0 + 1e-10)).or(Stop::evaluations(50_000)))
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
        "volume {:.6} after {} evaluations (the minimum: {minimum:.6})",
        best.score().unwrap_or(f64::NAN),
        outcome.evaluations()
    );
    println!("violation {:.6}", best.violation());
    println!("A1 {:.6}, A2 {:.6}", x[0], x[1]);
    println!("active constraints: {}", active.join(", "));
    trace.write();
    Ok(())
}
