//! CEC 2006 g03: the largest product of 10 variables on the unit sphere, an equality constraint
//! that leaves only a thin shell of feasible solutions.
//!
//! The fitness is the value and the constraint violation, which Deb's feasibility rules compare:
//! a feasible solution beats an infeasible one. The equality counts as met within the CEC 2006
//! report's tolerance, 0.0001. CMA-ES has the report's budget of 500,000 evaluations and stops
//! once it's within 1e-8 (relative) of the minimum. The example prints the best solution and the
//! equality's value at it.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example cec2006_g03
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::cec2006::G03;

// the CEC 2006 report's budget
const BUDGET: u64 = 500_000;

fn main() -> Result<()> {
    // the report's equality tolerance, EQUALITY_TOLERANCE
    let problem = G03::default();
    let optimum = problem.optimum().expect("known").value();
    // within 1e-8 of the minimum, relative to its size
    let target = optimum + 1e-8 * optimum.abs();
    let cmaes = Cmaes::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = Engine::new(cmaes, problem)
        .stop_when(Stop::target(target).or(Stop::evaluations(BUDGET)))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    let best = outcome.best_fitness();
    let x = outcome.best_genome();
    let genes: Vec<String> = x.iter().map(|xi| format!("{xi:.4}")).collect();
    let h = problem.constraints(x).equalities()[0];
    println!("CEC 2006 g03 with CMA-ES, seed 1: at most {BUDGET} evaluations");
    println!(
        "f {:.7}, {}, after {} evaluations (the minimum: {optimum:.7}, proven)",
        best.score().unwrap_or(f64::NAN),
        if best.is_feasible() {
            "feasible"
        } else {
            "infeasible"
        },
        outcome.evaluations()
    );
    println!("x {}", genes.join(" "));
    println!("h1 = x1^2 + ... + x10^2 - 1: {h:.6} (met when |h1| <= 0.0001)");
    trace.write();
    Ok(())
}
