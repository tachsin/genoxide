//! The car side impact (Gu et al., 2001): the lightest car body whose side withstands the
//! European side-impact test, from the thicknesses of seven parts, subject to ten limits on the
//! crash dummy's injuries and the structure's velocities. A constrained continuous problem, whose
//! best known weight is 23.585658.
//!
//! The constraints are response surfaces fitted to crash simulations. The fitness is the weight
//! and the constraint violation, which Deb's feasibility rules compare. L-SHADE, a differential
//! evolution whose population shrinks over its budget, searches the thicknesses, and the example
//! prints the best design and each response next to its limit.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example car_side_impact
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::engineering::CarSideImpact;

// L-SHADE's budget of evaluations: its population shrinks over it
const BUDGET: u64 = 20_000;

// the parts whose thicknesses are the genes, in their order
const PARTS: [&str; 7] = [
    "B-pillar inner",
    "B-pillar reinforcement",
    "floor side inner",
    "cross members",
    "door beam",
    "door beltline reinforcement",
    "roof rail",
];

// the constraints, in their order: what each limits, the limit and its unit
const LIMITS: [(&str, f64, &str); 10] = [
    ("abdomen load", 1.0, "kN"),
    ("upper chest velocity", 0.32, "m/s"),
    ("middle chest velocity", 0.32, "m/s"),
    ("lower chest velocity", 0.32, "m/s"),
    ("upper rib deflection", 32.0, "mm"),
    ("middle rib deflection", 32.0, "mm"),
    ("lower rib deflection", 32.0, "mm"),
    ("pubic force", 4.0, "kN"),
    ("B-pillar velocity", 9.9, "mm/ms"),
    ("front door velocity", 15.7, "mm/ms"),
];

fn main() -> Result<()> {
    let problem = CarSideImpact;
    let best_known = problem.optimum().expect("known").value();
    let l_shade = De::l_shade(problem.representation(), BUDGET)
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = Engine::new(l_shade, problem)
        .stop_when(Stop::evaluations(BUDGET))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    let best = outcome.best_fitness();
    let weight = best.score().unwrap_or(f64::NAN);
    println!(
        "weight {weight:.9} after {} evaluations (the best known: {best_known:.9})",
        outcome.evaluations()
    );
    println!(
        "violation {:.6}, a relative gap of {:.1e}",
        best.violation(),
        (weight - best_known) / best_known
    );
    let x = outcome.best_genome();
    let bounds = problem.representation();
    println!("{:<29}{:>9}  range", "thickness (mm)", "best");
    for ((part, xi), range) in PARTS.iter().zip(x.iter()).zip(bounds.bounds()) {
        let (low, high) = (range.start(), range.end());
        println!("{part:<29}{xi:>9.6}  {low} to {high}");
    }
    // each constraint is the response minus its limit, at most 0
    println!("{:<29}{:>9}  limit", "response", "best");
    let g = problem.constraints(x);
    for ((name, limit, unit), gi) in LIMITS.iter().zip(g.inequalities()) {
        println!("{name:<29}{:>9.4}  {limit} {unit}", gi + limit);
    }
    trace.write();
    Ok(())
}
