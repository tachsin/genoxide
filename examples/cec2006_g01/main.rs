//! CEC 2006 g01: a quadratic in 13 variables under 9 linear inequalities, whose minimum −15 lies
//! on the boundary of a feasible region that fills about 0.01 % of the box.
//!
//! The fitness is the value and the constraint violation, which Deb's feasibility rules compare:
//! a feasible solution beats an infeasible one. SHADE, a differential evolution, has the CEC 2006
//! report's budget of 500,000 evaluations and stops once it's within 1e-8 (relative) of the
//! minimum. The example prints the best solution and the constraints active at it.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example cec2006_g01
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::cec2006::G01;

// the CEC 2006 report's budget
const BUDGET: u64 = 500_000;
// a constraint g with |g| at most this is active: the solution lies on its boundary
const ACTIVE: f64 = 1e-6;

fn main() -> Result<()> {
    let optimum = G01.optimum().expect("known").value();
    // within 1e-8 of the minimum, relative to its size
    let target = optimum + 1e-8 * optimum.abs();
    let shade = De::builder(G01.representation())
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = Engine::new(shade, G01)
        .stop_when(Stop::target(target).or(Stop::evaluations(BUDGET)))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    let best = outcome.best_fitness();
    let x = outcome.best_genome();
    let genes: Vec<String> = x.iter().map(|xi| format!("{xi:.4}")).collect();
    let active: Vec<String> = G01
        .constraints(x)
        .inequalities()
        .iter()
        .enumerate()
        .filter(|(_, g)| g.abs() <= ACTIVE)
        .map(|(i, _)| format!("g{}", i + 1))
        .collect();
    println!("CEC 2006 g01 with SHADE, seed 1: at most {BUDGET} evaluations");
    println!(
        "f {:.6}, {}, after {} evaluations (the minimum: {optimum:.6}, proven)",
        best.score().unwrap_or(f64::NAN),
        if best.is_feasible() {
            "feasible"
        } else {
            "infeasible"
        },
        outcome.evaluations()
    );
    println!("x {}", genes.join(" "));
    println!("active constraints (|g| <= 1e-6): {}", active.join(" "));
    trace.write();
    Ok(())
}
