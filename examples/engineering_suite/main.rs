//! Engineering designs and CEC 2006: SHADE with Deb's feasibility rules on eleven constrained
//! problems, five engineering designs and the CEC 2006 problems g01 to g06.
//!
//! Each problem's fitness is its value and its constraint violation, which Deb's rules compare: a
//! feasible solution beats an infeasible one. One seeded SHADE run per problem, with a budget per
//! problem, stops early within 1e-8 (relative) of the optimum or best known value f*. The table
//! gives the best value found, f*, the relative gap (f - f*) / |f*| and whether the best is
//! feasible.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example engineering_suite
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{self, DynProblem, cec2006, engineering};

// the engineering designs' budget, and the CEC 2006 report's for its problems
const DESIGN_BUDGET: u64 = 50_000;
const CEC_BUDGET: u64 = 500_000;
// a run stops once its best is feasible and within this of f*, relative to |f*|
const TOLERANCE: f64 = 1e-8;

fn main() -> Result<()> {
    let suite: Vec<(Box<dyn DynProblem>, u64)> = vec![
        (
            problems::boxed(engineering::TensionCompressionSpring),
            DESIGN_BUDGET,
        ),
        (problems::boxed(engineering::SpeedReducer), DESIGN_BUDGET),
        (problems::boxed(engineering::ThreeBarTruss), DESIGN_BUDGET),
        (problems::boxed(engineering::CantileverBeam), DESIGN_BUDGET),
        (problems::boxed(engineering::CarSideImpact), DESIGN_BUDGET),
        (problems::boxed(cec2006::G01), CEC_BUDGET),
        (problems::boxed(cec2006::G02), CEC_BUDGET),
        (problems::boxed(cec2006::G03::default()), CEC_BUDGET),
        (problems::boxed(cec2006::G04), CEC_BUDGET),
        (problems::boxed(cec2006::G05::default()), CEC_BUDGET),
        (problems::boxed(cec2006::G06), CEC_BUDGET),
    ];

    println!("SHADE with Deb's feasibility rules, one run per problem, seed 1");
    println!(
        "{:<25}{:>8}{:>12}{:>12}{:>12}{:>10}{:>10}",
        "problem", "budget", "best found", "f*", "f* is", "gap", "feasible"
    );
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let mut trace = trace::Trace::from_env();
    for (problem, budget) in &suite {
        let optimum = problem.optimum().expect("known");
        let f_star = optimum.value();
        let target = f_star + TOLERANCE * f_star.abs();
        let shade = De::builder(problem.real()).minimize().seed(1).build()?;
        let outcome = Engine::new(shade, |x: &Reals| problem.evaluate(x))
            .stop_when(Stop::target(target).or(Stop::evaluations(*budget)))
            .on_generation(trace.errors(format!("{}/SHADE", problem.name()), f_star))
            .run()?;

        let best = outcome.best_fitness();
        let value = best.score().expect("valid");
        // the relative gap; below the tolerance, the run stopped early
        let gap = if best.is_feasible() && value <= target {
            "< 1e-8".to_string()
        } else {
            scientific((value - f_star) / f_star.abs())
        };
        println!(
            "{:<25}{:>8}{:>12}{:>12}{:>12}{:>10}{:>10}",
            problem.name(),
            budget,
            significant(value),
            significant(f_star),
            if optimum.is_proven() {
                "optimum"
            } else {
                "best known"
            },
            gap,
            if best.is_feasible() { "yes" } else { "no" }
        );
    }
    trace.write();
    Ok(())
}

// 6 significant digits, e.g. 0.0126652 or -30665.5
fn significant(value: f64) -> String {
    let digits = value.abs().log10().floor() as i32;
    let decimals = (5 - digits).max(0) as usize;
    format!("{value:.decimals$}")
}

// two significant digits, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
}
