//! CEC 2006 g05: a cubic in 4 variables with 2 linear inequality constraints and 3 nonlinear
//! equality constraints, from the CEC 2006 special session on constrained optimization (Liang et
//! al., 2006). The best known value is 5126.4967140071, with the equalities met within the
//! report's tolerance of 0.0001.
//!
//! genoxide's `G05` gives the value of a solution and its constraint violation, which Deb's
//! feasibility rules compare: a feasible solution beats an infeasible one. CMA-ES searches the 4
//! variables within the report's budget of 500,000 evaluations, and stops once the error
//! f(x) − f* is at most 1e-8. The example prints the best solution and its constraints.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example cec2006_g05
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::cec2006::{EQUALITY_TOLERANCE, G05};

// the CEC 2006 report's budget of evaluations per run
const BUDGET: u64 = 500_000;
// the run stops once its best is feasible with an error f(x) - f* at most this
const ERROR: f64 = 1e-8;
// a constraint within this of its boundary is active
const ACTIVE: f64 = 1e-6;

fn main() -> Result<()> {
    let problem = G05::default();
    let optimum = problem.optimum().expect("known");
    let f_star = optimum.value();
    let cmaes = Cmaes::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    let outcome = Engine::new(cmaes, problem)
        .stop_when(Stop::target(f_star + ERROR).or(Stop::evaluations(BUDGET)))
        .on_generation(|snapshot| trace.record(snapshot))
        .run()?;

    // the platform's sin can change the run's path from one operating system to another: the
    // example prints what doesn't depend on it, without the evaluations of a run that meets its
    // target, and the solution to 4 significant digits
    let best = outcome.best_fitness();
    let value = best.score().expect("valid");
    let x = outcome.best_genome();
    println!("CMA-ES with Deb's feasibility rules on g05, seed 1");
    let (stop, error) = if outcome.stop_reason() == StopReason::Target {
        (
            "stopped by the target".to_string(),
            format!("< {ERROR:.0e}"),
        )
    } else {
        let evaluations = outcome.evaluations();
        let stop = format!("stopped after {evaluations} evaluations");
        (stop, format!("{:.1e}", value - f_star))
    };
    let feasibility = if best.is_feasible() {
        "feasible"
    } else {
        "infeasible"
    };
    println!("{stop}: f(x) - f* {error}, {feasibility}");
    println!(
        "f(x) {}, f* {} ({})",
        significant(value, 6),
        significant(f_star, 6),
        if optimum.is_proven() {
            "proven"
        } else {
            "best known"
        }
    );
    let genes: Vec<String> = (1..)
        .zip(&x[..])
        .map(|(i, xi)| format!("x{i} {}", significant(*xi, 4)))
        .collect();
    println!("{}", genes.join(", "));
    // g1 and g2, then h3 to h5, each equality as its excess over the tolerance, 0 when it's met
    let constraints = problem.constraints(x);
    let inequalities = constraints.inequalities().iter().map(|&g| ("g", g));
    let equalities = constraints.equalities().iter();
    let equalities = equalities.map(|&h| ("h", (h.abs() - EQUALITY_TOLERANCE).max(0.0)));
    let constraints: Vec<String> = (1..)
        .zip(inequalities.chain(equalities))
        .map(|(i, (name, g))| format!("{name}{i} {}", state(g)))
        .collect();
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

// `digits` significant digits, e.g. 29.9953 or -30665.5 for 6
fn significant(value: f64, digits: i32) -> String {
    let magnitude = value.abs().log10().floor() as i32;
    let decimals = (digits - 1 - magnitude).max(0) as usize;
    format!("{value:.decimals$}")
}
