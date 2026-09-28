//! CEC 2006 g12: a quadratic in 3 variables whose feasible region is 729 disjoint spheres, from
//! the CEC 2006 special session on constrained optimization (Liang et al., 2006). The minimum is
//! −1 at (5, 5, 5), the center of one of the spheres, proven.
//!
//! genoxide's `G12` gives the value of a solution and its constraint violation, which Deb's
//! feasibility rules compare: a feasible solution beats an infeasible one. SHADE, genoxide's
//! differential evolution, searches the 3 variables within the report's budget of 500,000
//! evaluations, and stops once the error f(x) − f* is at most 1e-8. The example prints the best
//! solution, its constraint and the sphere that holds it.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example cec2006_g12
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::cec2006::G12;

// the CEC 2006 report's budget of evaluations per run
const BUDGET: u64 = 500_000;
// the run stops once its best is feasible with an error f(x) - f* at most this
const ERROR: f64 = 1e-8;
// the report counts a run as successful once its error is at most this
const SUCCESS: f64 = 1e-4;

fn main() -> Result<()> {
    let problem = G12;
    let optimum = problem.optimum().expect("known");
    let f_star = optimum.value();
    let shade = De::builder(problem.representation())
        .minimize()
        .seed(1)
        .build()?;
    // with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    // the evaluations when the best is first feasible, and when its error first meets the
    // report's criterion of success
    let (mut feasible, mut success) = (None, None);
    let outcome = Engine::new(shade, problem)
        .stop_when(Stop::target(f_star + ERROR).or(Stop::evaluations(BUDGET)))
        .on_generation(|snapshot| {
            let progress = snapshot.progress();
            let best = progress.best().filter(|best| best.is_feasible());
            let error = best.and_then(Fitness::score).map(|value| value - f_star);
            if error.is_some() {
                feasible.get_or_insert(progress.evaluations());
            }
            if error.is_some_and(|error| error <= SUCCESS) {
                success.get_or_insert(progress.evaluations());
            }
            trace.record(snapshot);
        })
        .run()?;

    let best = outcome.best_fitness();
    let value = best.score().expect("valid");
    let x = outcome.best_genome();
    println!("SHADE with Deb's feasibility rules on g12, seed 1");
    let (stop, error) = if outcome.stop_reason() == StopReason::Target {
        ("stopped by the target", format!("< {ERROR:.0e}"))
    } else {
        ("stopped", format!("{:.1e}", value - f_star))
    };
    let evaluations = outcome.evaluations();
    let feasibility = if best.is_feasible() {
        "feasible"
    } else {
        "infeasible"
    };
    println!("{stop} after {evaluations} evaluations: f(x) - f* {error}, {feasibility}");
    println!(
        "first feasible after {} evaluations, f(x) - f* <= 1e-4 after {}",
        count(feasible),
        count(success)
    );
    // both to 6 decimals, alike for f* = -1 and a value just above it
    println!(
        "f(x) {value:.6}, f* {f_star:.6} ({})",
        if optimum.is_proven() {
            "proven"
        } else {
            "best known"
        }
    );
    let genes: Vec<String> = (1..)
        .zip(&x[..])
        .map(|(i, xi)| format!("x{i} {}", significant(*xi, 6)))
        .collect();
    println!("{}", genes.join(", "));
    // the nearest of the 729 centers (p, q, r), p, q, r in 1..=9, and g, the squared distance to
    // it less 0.0625: at most 0 inside its sphere
    let center: Vec<String> = x
        .iter()
        .map(|xi| format!("{}", xi.round().clamp(1.0, 9.0)))
        .collect();
    let g = problem.constraints(x).inequalities()[0];
    println!(
        "nearest center ({}), g {:.6} (<= 0 inside its sphere)",
        center.join(", "),
        g
    );
    trace.write();
    Ok(())
}

// the evaluations, or "never"
fn count(evaluations: Option<u64>) -> String {
    evaluations.map_or("never".to_string(), |evaluations| evaluations.to_string())
}

// `digits` significant digits, e.g. 29.9953 or -30665.5 for 6
fn significant(value: f64, digits: i32) -> String {
    let magnitude = value.abs().log10().floor() as i32;
    let decimals = (digits - 1 - magnitude).max(0) as usize;
    format!("{value:.decimals$}")
}
