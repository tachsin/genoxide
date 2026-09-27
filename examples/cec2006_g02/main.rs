//! CEC 2006 g02: a rugged function of 20 variables with many local optima, under a product and a
//! sum constraint. Its best known value, −0.803619, isn't proven optimal.
//!
//! The fitness is the value and the constraint violation, which Deb's feasibility rules compare:
//! a feasible solution beats an infeasible one. SHADE, a differential evolution, with a population
//! of 300 instead of its published 100, uses the CEC 2006 report's whole budget of 500,000
//! evaluations, to see whether anything beats the best known value. The example prints the best
//! solution, its gap to the best known value and the constraints active at it.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example cec2006_g02
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::cec2006::G02;

// the CEC 2006 report's budget
const BUDGET: u64 = 500_000;
// a constraint g with |g| at most this is active: the solution lies on its boundary
const ACTIVE: f64 = 1e-6;

fn main() -> Result<()> {
    let best_known = G02.optimum().expect("known").value();
    let shade = De::builder(G02.representation())
        .population_size(300)
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = Engine::new(shade, G02)
        .stop_when(Stop::evaluations(BUDGET))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    let best = outcome.best_fitness();
    let value = best.score().unwrap_or(f64::NAN);
    // the gap to the best known value, relative to its size: negative if the run beats it
    let gap = (value - best_known) / best_known.abs();
    let x = outcome.best_genome();
    let genes = |range: std::ops::Range<usize>| {
        let genes: Vec<String> = x[range].iter().map(|xi| format!("{xi:.3}")).collect();
        genes.join(" ")
    };
    let active: Vec<String> = G02
        .constraints(x)
        .inequalities()
        .iter()
        .enumerate()
        .filter(|(_, g)| g.abs() <= ACTIVE)
        .map(|(i, _)| format!("g{}", i + 1))
        .collect();
    println!("CEC 2006 g02 with SHADE (population 300), seed 1: {BUDGET} evaluations");
    println!(
        "f {value:.6}, {} (the best known: {best_known:.6}, not proven)",
        if best.is_feasible() {
            "feasible"
        } else {
            "infeasible"
        },
    );
    // the last digits of f differ between platforms: g02 calls the platform's cos
    if (0.0..1e-8).contains(&gap) {
        println!("relative gap to the best known: below 1e-8");
    } else {
        println!("relative gap to the best known: {gap:.1e}");
    }
    println!("x1-x10  {}", genes(0..10));
    println!("x11-x20 {}", genes(10..20));
    println!("active constraints (|g| <= 1e-6): {}", active.join(" "));
    trace.write();
    Ok(())
}
