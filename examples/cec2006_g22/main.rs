//! CEC 2006 g22: the linear function x1 of 22 variables under 1 nonlinear inequality and 19
//! equality constraints, from the CEC 2006 special session on constrained optimization (Liang et
//! al., 2006). The best known value is 236.430975504001, with the equalities met within the
//! report's tolerance of 0.0001.
//!
//! genoxide's `G22` gives the value of a solution and its constraint violation, which Deb's
//! feasibility rules compare: a feasible solution beats an infeasible one, and two infeasible ones
//! compare by violation. L-SHADE searches the 22 variables for the report's budget of 500,000
//! evaluations, and doesn't find a feasible solution. The example prints the least violation it
//! finds, with its solution and constraints.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example cec2006_g22
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::cec2006::{EQUALITY_TOLERANCE, G22};

// the CEC 2006 report's budget of evaluations per run
const BUDGET: u64 = 500_000;
// a constraint within this of its boundary is active
const ACTIVE: f64 = 1e-6;

fn main() -> Result<()> {
    // the report's equality tolerance, EQUALITY_TOLERANCE
    let problem = G22::default();
    let optimum = problem.optimum().expect("known");
    // SHADE with a population that shrinks over the budget, from 18 · 22 = 396 to 4
    let l_shade = De::l_shade(problem.representation(), BUDGET)
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // the evaluations when the best is first feasible
    let mut feasible = None;
    let outcome = Engine::new(l_shade, problem)
        .stop_when(Stop::evaluations(BUDGET))
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            if progress.best().is_some_and(Fitness::is_feasible) {
                feasible.get_or_insert(progress.evaluations());
            }
            trace.record(snapshot);
        })
        .run()?;

    let best = outcome.best_fitness();
    let x = outcome.best_genome();
    println!("L-SHADE with Deb's feasibility rules on g22, seed 1");
    let feasibility = if best.is_feasible() {
        "feasible".to_string()
    } else {
        format!("infeasible, violation {}", significant(best.violation(), 4))
    };
    let evaluations = outcome.evaluations();
    println!("stopped after {evaluations} evaluations: {feasibility}");
    match feasible {
        Some(evaluations) => println!("first feasible after {evaluations} evaluations"),
        None => println!("no feasible solution found"),
    }
    println!(
        "f(x) {}, f* {} (best known)",
        significant(best.score().expect("valid"), 6),
        significant(optimum.value(), 6),
    );
    let genes: Vec<String> = (1..)
        .zip(&x[..])
        .map(|(i, &xi)| format!("x{i} {}", gene(xi)))
        .collect();
    println!("{}", genes.join(", "));
    // g1, then h1 to h19, each equality as its excess over the tolerance, 0 when it's met
    let constraints = problem.constraints(x);
    let inequalities = (1..)
        .zip(constraints.inequalities())
        .map(|(i, &g)| format!("g{i} {}", state(g)));
    let equalities = (1..)
        .zip(constraints.equalities())
        .map(|(i, &h)| format!("h{i} {}", state((h.abs() - EQUALITY_TOLERANCE).max(0.0))));
    let constraints: Vec<String> = inequalities.chain(equalities).collect();
    println!("{}", constraints.join(", "));
    trace.write();
    Ok(())
}

// a constraint g(x) <= 0: "active" on its boundary, else its value
fn state(g: f64) -> String {
    if g.abs() <= ACTIVE {
        "active".to_string()
    } else {
        significant(g, 4)
    }
}

// a gene: in scientific notation when it's that close to 0, e.g. 3.0e-12, else to 6 significant
// digits
fn gene(value: f64) -> String {
    if value != 0.0 && value.abs() < 1e-4 {
        format!("{value:.1e}")
    } else {
        significant(value, 6)
    }
}

// `digits` significant digits, e.g. 29.9953 or -30665.5 for 6
fn significant(value: f64, digits: i32) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let magnitude = value.abs().log10().floor() as i32;
    let decimals = (digits - 1 - magnitude).max(0) as usize;
    format!("{value:.decimals$}")
}
