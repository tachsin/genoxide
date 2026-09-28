//! CEC 2006 g20: a linear function of 24 variables under 6 nonlinear inequality and 14 equality
//! constraints, from the CEC 2006 special session on constrained optimization (Liang et al.,
//! 2006). It has no feasible solution: the report's best known, 0.2049794002, violates g1 by
//! 0.1438.
//!
//! genoxide's `G20` gives the value of a solution and its constraint violation, which Deb's
//! feasibility rules compare: with no feasible solution, the least violation wins. SHADE,
//! genoxide's default differential evolution, searches the 24 variables for the report's budget of
//! 500,000 evaluations. The example prints the least violation it finds, with its solution and
//! constraints.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example cec2006_g20
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::cec2006::{EQUALITY_TOLERANCE, G20};

// the CEC 2006 report's budget of evaluations per run
const BUDGET: u64 = 500_000;
// a constraint within this of its boundary is active
const ACTIVE: f64 = 1e-6;

fn main() -> Result<()> {
    // the report's equality tolerance, EQUALITY_TOLERANCE
    let problem = G20::default();
    let optimum = problem.optimum().expect("known");
    // the violation of the report's best known solution
    let (_, reported) = problem.evaluate(&optimum.solutions()[0]);
    let shade = De::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // the evaluations when the best is first less infeasible than the report's solution
    let mut below = None;
    let outcome = Engine::new(shade, problem)
        .stop_when(Stop::evaluations(BUDGET))
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            if progress
                .best()
                .is_some_and(|best| best.violation() < reported)
            {
                below.get_or_insert(progress.evaluations());
            }
            trace.record(snapshot);
        })
        .run()?;

    let best = outcome.best_fitness();
    let x = outcome.best_genome();
    println!("SHADE with Deb's feasibility rules on g20, seed 1");
    let feasibility = if best.is_feasible() {
        "feasible".to_string()
    } else {
        format!("infeasible, violation {}", significant(best.violation(), 6))
    };
    let evaluations = outcome.evaluations();
    println!("stopped after {evaluations} evaluations: {feasibility}");
    println!(
        "the report's best known: violation {}, less violated after {} evaluations",
        significant(reported, 6),
        count(below)
    );
    println!(
        "f(x) {}, the report's {} (best known, infeasible)",
        significant(best.score().expect("valid"), 6),
        significant(optimum.value(), 6),
    );
    let genes: Vec<String> = (1..)
        .zip(&x[..])
        .map(|(i, &xi)| format!("x{i} {}", gene(xi)))
        .collect();
    println!("{}", genes.join(", "));
    // g1 to g6, then h1 to h14, each equality as its excess over the tolerance, 0 when it's met
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

// the evaluations, or "never"
fn count(evaluations: Option<u64>) -> String {
    evaluations.map_or("never".to_string(), |evaluations| evaluations.to_string())
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
