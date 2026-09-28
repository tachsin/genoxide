//! CEC 2006 g21: the linear function x1 of 7 variables under 1 nonlinear inequality and 5
//! nonlinear equality constraints, from the CEC 2006 special session on constrained optimization
//! (Liang et al., 2006). The best known value is 193.724510070035, with the equalities met within
//! the report's tolerance of 0.0001.
//!
//! genoxide's `G21` gives the value of a solution and its constraint violation, which Deb's
//! feasibility rules compare: a feasible solution beats an infeasible one. SHADE, genoxide's
//! default differential evolution, with a population of 50, searches the 7 variables within the
//! report's budget of 500,000 evaluations, and stops once the error f(x) − f* is at most 1e-8. The
//! example prints the best solution and its constraints.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example cec2006_g21
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::Problem;
use genoxide::problems::cec2006::{EQUALITY_TOLERANCE, G21};

// the CEC 2006 report's budget of evaluations per run
const BUDGET: u64 = 500_000;
// the run stops once its best is feasible with an error f(x) - f* at most this
const ERROR: f64 = 1e-8;
// the report counts a run as successful once its error is at most this
const SUCCESS: f64 = 1e-4;
// a constraint within this of its boundary is active
const ACTIVE: f64 = 1e-6;

fn main() -> Result<()> {
    // the report's equality tolerance, EQUALITY_TOLERANCE
    let problem = G21::default();
    let optimum = problem.optimum().expect("known");
    let f_star = optimum.value();
    // half the default population: see the README
    let shade = De::builder(problem.representation())
        .population_size(50)
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
    println!("SHADE, a population of 50, with Deb's feasibility rules on g21, seed 1");
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
        .map(|(i, &xi)| format!("x{i} {}", gene(xi)))
        .collect();
    println!("{}", genes.join(", "));
    // g1, then h1 to h5, each equality as its excess over the tolerance, 0 when it's met
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
    let magnitude = value.abs().log10().floor() as i32;
    let decimals = (digits - 1 - magnitude).max(0) as usize;
    format!("{value:.decimals$}")
}
