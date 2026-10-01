//! A global method, then a local one: SHADE on Rastrigin's function in 10 dimensions for 40,000
//! evaluations, which finds the basin of the global minimum, then L-BFGS-B from SHADE's best, with
//! the analytic gradient, down to the minimum itself, f = 0, in a few evaluations.
//!
//! For contrast, SHADE alone, run on to f ≤ 1e-12.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its runs for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example polish
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Problem, Rastrigin};

const N: usize = 10;
// SHADE's evaluations before the polish
const GLOBAL: u64 = 40_000;
// a row of SHADE's table every this many evaluations
const EVERY: u64 = 5_000;

fn main() -> Result<()> {
    let problem = Rastrigin::new(N);
    let shade = || {
        De::builder(problem.representation())
            .minimize()
            .seed(1)
            .build()
    };
    // SHADE for the global search: the best value after each generation
    let mut global = Vec::new();
    let found = Engine::new(shade()?, problem)
        .stop_when(Stop::evaluations(GLOBAL))
        .on_generation(|snapshot| global.push(best(snapshot)))
        .run()?;

    // L-BFGS-B from SHADE's best, with its default tolerances
    let lbfgsb = Lbfgsb::builder(problem.representation())
        .initial_genome(found.best_genome().clone())
        .minimize()
        .build()?;
    let mut local = Vec::new();
    let mut criterion = None;
    let mut engine = Engine::new(lbfgsb, problem)
        .stop_when(Stop::evaluations(10_000))
        .on_generation(|snapshot| local.push(best(snapshot)))
        .control(|lbfgsb: &mut Lbfgsb, _| {
            criterion = lbfgsb.converged();
            Ok(())
        });
    let polished = engine.run()?;
    drop(engine);

    // SHADE alone, to f <= 1e-12, for contrast
    let mut alone = Vec::new();
    let without = Engine::new(shade()?, problem)
        .stop_when(Stop::target(1e-12).or(Stop::evaluations(1_000_000)))
        .on_generation(|snapshot| alone.push(best(snapshot)))
        .run()?;

    println!("Rastrigin's function in {N} dimensions, its minimum 0 at the origin");
    println!("SHADE, 100 individuals, seed 1, for {GLOBAL} evaluations");
    println!("evaluations  best value");
    for &(evaluations, value) in &global {
        if evaluations.is_multiple_of(EVERY) {
            println!("{evaluations:>11}  {:>10}", scientific(value));
        }
    }
    let x = found.best_genome();
    println!(
        "SHADE's best: f = {}, every gene within {} of 0: in the global minimum's basin",
        scientific(found.best_fitness().score().expect("valid")),
        scientific(largest(x))
    );
    println!("L-BFGS-B from SHADE's best, with the analytic gradient");
    println!("round  evaluations  best value");
    for (round, &(evaluations, value)) in local.iter().enumerate() {
        println!("{round:>5}  {evaluations:>11}  {:>10}", scientific(value));
    }
    assert_eq!(polished.stop_reason(), StopReason::Converged);
    assert_eq!(polished.best_fitness(), Fitness::new(0.0));
    println!(
        "L-BFGS-B: f = {:?} after {} evaluations, every gene within {} of 0 ({})",
        polished.best_fitness().score().expect("valid"),
        polished.evaluations(),
        scientific(largest(polished.best_genome())),
        match criterion {
            Some(lbfgsb::Criterion::ProjectedGradient) =>
                "converged: a projected gradient below 1e-5",
            _ => "converged",
        }
    );
    println!(
        "in all, {} evaluations to the minimum",
        GLOBAL + polished.evaluations()
    );
    assert_eq!(without.stop_reason(), StopReason::Target);
    println!(
        "SHADE alone, for contrast: f = {} after {} evaluations",
        scientific(without.best_fitness().score().expect("valid")),
        without.evaluations()
    );

    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    trace::write_runs(&alone, GLOBAL, &local);
    Ok(())
}

// the evaluations and the best value after a generation
fn best(snapshot: &genoxide::observer::Snapshot<'_, Reals>) -> (u64, f64) {
    let progress = snapshot.progress();
    let value = progress.best().and_then(Fitness::score).expect("valid");
    (progress.evaluations(), value)
}

// the largest distance of a gene from 0
fn largest(x: &Reals) -> f64 {
    x.iter().fold(0.0, |largest: f64, xi| largest.max(xi.abs()))
}

// two significant digits, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
}
