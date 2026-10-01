//! Nelder-Mead with restarts: find the four global minima of Himmelblau's function with one
//! search that starts again from a random point each time its simplex has converged.
//!
//! Each run converges to the minimum of the basin it starts in; twenty runs land in all four. The
//! known minima come from genoxide's `problems::Himmelblau`.
//!
//! With `GENOXIDE_TRACE=<file>`, it also writes a trace of its run for the plot on the example's
//! page, with `trace.rs`.
//!
//! ```text
//! cargo run --release --example nelder_mead_himmelblau
//! ```

mod trace;

use genoxide::prelude::*;
use genoxide::problems::{Himmelblau, Problem};
use std::cell::RefCell;

const RESTARTS: u64 = 19;

fn main() -> Result<()> {
    let problem = Himmelblau;
    let optimum = problem.optimum().expect("known");
    let minima = optimum.solutions();
    let nelder_mead = NelderMead::builder(problem.representation())
        .restarts(local::Restarts::Random { times: RESTARTS })
        .minimize()
        .seed(1)
        .build()?;
    // the best vertex of each run when it converged
    let mut ends: Vec<Individual<Reals>> = Vec::new();
    // with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
    let trace = RefCell::new(trace::Trace::from_env());
    let outcome = Engine::new(nelder_mead, problem)
        .stop_when(Stop::evaluations(100_000))
        .on_generation(|snapshot| trace.borrow_mut().record(snapshot))
        .control(|nelder_mead, _| {
            if nelder_mead.converged() {
                let end = nelder_mead.population()[0].clone();
                trace.borrow_mut().end(end.genome());
                ends.push(end);
            }
            Ok(())
        })
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Converged);

    // per known minimum: the runs that ended nearest it, and the best and worst values they
    // reached
    let mut found = vec![(0, f64::INFINITY, 0.0f64); minima.len()];
    for end in &ends {
        let nearest = (0..minima.len())
            .min_by(|&a, &b| {
                let (a, b) = (
                    distance(end.genome(), &minima[a]),
                    distance(end.genome(), &minima[b]),
                );
                a.total_cmp(&b)
            })
            .expect("four minima");
        let value = end.fitness().and_then(Fitness::score).expect("valid");
        let (runs, best, worst) = &mut found[nearest];
        *runs += 1;
        *best = best.min(value);
        *worst = worst.max(value);
    }

    println!(
        "{} runs of Nelder-Mead in [-5, 5] x [-5, 5]: one, then {RESTARTS} restarts from random points",
        ends.len()
    );
    println!("minimum                  runs  values reached");
    for (minimum, (runs, best, worst)) in minima.iter().zip(found) {
        println!(
            "({:>9.6}, {:>9.6})  {runs:>4}  {} to {}",
            minimum[0],
            minimum[1],
            scientific(best),
            scientific(worst)
        );
    }
    println!(
        "{} evaluations in all, {:.1} per run",
        outcome.evaluations(),
        outcome.evaluations() as f64 / ends.len() as f64
    );
    trace.into_inner().write();
    Ok(())
}

fn distance(a: &Reals, b: &Reals) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// two significant digits, e.g. 1.2e-7
fn scientific(value: f64) -> String {
    format!("{value:.1e}")
}
